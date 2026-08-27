"""Random forest + xgboost multi-salida day-ahead D+1, con tuning y diagnósticos.

Ejecutar:  .venv/bin/python -m src.arboles.run_arboles (o -m src.arboles.modelo)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.redes.datos import construir_xy
from src.metricas import ancla_mase as _ancla
from src.features.validacion import split_canonico_precovid, rolling_origin_folds

DIR_MET = RAIZ / "resultados" / "metricas"
DIR_ART = RAIZ / "resultados" / "arboles"
DIR_FIG = RAIZ / "figuras" / "arboles"

SEMILLAS = (0, 1, 2, 3, 4)
N_CONFIGS_XGB = 24 # tamaño del random search de XGB
N_ARBOLES_RF = 300
N_ARBOLES_CV = 150 # menos árboles durante el ranking por CV, para abaratarlo.
N_ARBOLES_XGB_MAX = 600 # cota del early stopping en el refit final de XGB


# Tabla MIMO (D+1). La misma que redes
def construir_tabla():
    """Tabla (datos, columnas_x, columnas_y) por día-origen D para day-ahead D+1 (24 salidas)."""
    return construir_xy(n_dias=1)


def _rejilla(tipo: str, n_features: int) -> list[dict]:
    """Rejilla base (RF 18 combos, XGB 144); la búsqueda amplia está en `tuning.py`."""
    if tipo == "rf":
        opciones_features = [max(1, n_features // 3), "sqrt"]
        return [{"max_depth": profundidad, "min_samples_leaf": hojas, "max_features": features}
                for profundidad in (8, 12, 20) for hojas in (1, 5, 20) for features in opciones_features]
    if tipo == "xgb":
        return [{"max_depth": profundidad, "learning_rate": tasa, "min_child_weight": peso_hoja,
                 "subsample": submuestra, "colsample_bytree": submuestra_cols, "multi_strategy": estrategia}
                for profundidad in (4, 6, 8) for tasa in (0.03, 0.05, 0.1) for peso_hoja in (1, 5)
                for submuestra in (0.8, 1.0) for submuestra_cols in (0.8, 1.0)
                for estrategia in ("one_output_per_tree", "multi_output_tree")]
    raise ValueError(tipo)


def _muestrear(rejilla: list[dict], n: int, semilla: int = 0, forzar: dict | None = None) -> list[dict]:
    """Submuestreo aleatorio determinista sin reemplazo; `forzar` mete una configuración concreta en la
    muestra (sustituye a la primera, sin cambiar el tamaño ni la secuencia aleatoria). Función ÚNICA:
    `tuning.py` la importa de aquí como `muestrear`."""
    if n >= len(rejilla):
        return rejilla
    rng = np.random.default_rng(semilla)
    indices = rng.choice(len(rejilla), size=n, replace=False)
    seleccion = [rejilla[i] for i in sorted(indices)]
    if forzar is not None and forzar not in seleccion:
        seleccion[0] = forzar
    return seleccion


def _crear(tipo: str, hiperparametros: dict, n_arboles: int | None = None, semilla: int = 0, device: str = "cpu"):
    if tipo == "rf":
        if n_arboles is None:
            n_arboles = N_ARBOLES_RF
        return RandomForestRegressor(n_estimators=n_arboles, n_jobs=-1, random_state=semilla, **hiperparametros)
    if tipo == "xgb":
        if n_arboles is None:
            n_arboles = 200
        return XGBRegressor(n_estimators=n_arboles, n_jobs=-1, random_state=semilla, tree_method="hist",
                            device=device, eval_metric="mae", **hiperparametros)
    raise ValueError(tipo)


# Tuning por rolling-origin, con los mismos 5 folds que redes y SARIMAX.
def tunear_rolling(tipo: str, datos, columnas_x, columnas_y, folds, configs: list[dict],
                   n_arboles_cv: int, ancla: float) -> tuple[dict, list[dict]]:
    dias_origen = datos.index
    X_todo, Y_todo = datos[columnas_x], datos[columnas_y]
    mascaras = [(dias_origen.isin(f.train), dias_origen.isin(f.val)) for f in folds]
    tabla = []
    mejor = (np.inf, None)
    for j, hiperparametros in enumerate(configs):
        maes = []
        for train, val in mascaras:
            modelo = _crear(tipo, hiperparametros, n_arboles=n_arboles_cv, semilla=0)
            modelo.fit(X_todo[train].to_numpy(), Y_todo[train].to_numpy())
            prediccion = modelo.predict(X_todo[val].to_numpy())
            maes.append(float(np.mean(np.abs(Y_todo[val].to_numpy() - prediccion))))
        maes = np.array(maes)
        mae_cv = float(maes.mean())
        error_estandar = float(maes.std(ddof=1) / np.sqrt(len(maes)))
        tabla.append({"hiperparametros": hiperparametros, "mae_cv": round(mae_cv, 2),
                      "se_cv": round(error_estandar, 2),
                      "mase_cv": round(mae_cv / ancla, 4), "mae_por_fold": [round(x, 1) for x in maes]})
        print(f"    [{tipo} {j + 1:2d}/{len(configs)}] MAE_cv={mae_cv:6.1f} ± {error_estandar:4.1f}  hiperparametros={hiperparametros}")
        if mae_cv < mejor[0]:
            mejor = (mae_cv, hiperparametros)
    tabla.sort(key=lambda fila: fila["mae_cv"])
    return mejor[1], tabla


def _n_arboles_parada_temprana(hiperparametros: dict, X_ajuste, Y_ajuste, frac_val: float = 0.15) -> tuple[int, dict]:
    n_val = max(90, int(frac_val * len(X_ajuste)))
    X_train, Y_train, X_val, Y_val = X_ajuste[:-n_val], Y_ajuste[:-n_val], X_ajuste[-n_val:], Y_ajuste[-n_val:]
    modelo_parada = _crear("xgb", hiperparametros, n_arboles=N_ARBOLES_XGB_MAX, semilla=0)
    modelo_parada.set_params(early_stopping_rounds=30)
    modelo_parada.fit(X_train, Y_train, eval_set=[(X_val, Y_val)], verbose=False)
    n_arboles_mejor = max(50, int(modelo_parada.best_iteration) + 1)
    return n_arboles_mejor, {"best_n_estimators": n_arboles_mejor, "n_val_earlystop": int(n_val)}


def ajustar_final_semillas(tipo: str, hiperparametros: dict, X_ajuste, Y_ajuste):
    """Ajusta el modelo final con las 5 semillas; devuelve (modelos, info_earlystop|None)."""
    info_parada = None
    n_arboles = None
    if tipo == "xgb":
        n_arboles, info_parada = _n_arboles_parada_temprana(hiperparametros, X_ajuste, Y_ajuste)
    modelos = []
    for semilla in SEMILLAS:
        modelo = _crear(tipo, hiperparametros, n_arboles=n_arboles, semilla=semilla)
        modelo.fit(X_ajuste, Y_ajuste)
        modelos.append(modelo)
    return modelos, info_parada


def _metricas(Y_test, pred, ancla) -> dict:
    errores = np.abs(Y_test - pred)
    return {"MAE": float(errores.mean()),
            "RMSE": float(np.sqrt(np.mean((Y_test - pred) ** 2))),
            "MAPE": float(np.mean(errores / np.abs(Y_test)) * 100),
            "MASE": float(errores.mean() / ancla)}


def _metricas_multisemilla(Y_test, predicciones: list[np.ndarray], ancla) -> dict:
    """Métricas por semilla, media ± SD y métricas de la predicción ensemble."""
    por_semilla = [_metricas(Y_test, prediccion, ancla) for prediccion in predicciones]
    resumen = {metrica: {"media": float(np.mean([m[metrica] for m in por_semilla])),
               "sd": float(np.std([m[metrica] for m in por_semilla], ddof=1))}
               for metrica in ("MAE", "RMSE", "MAPE", "MASE")}
    pred_ensemble = np.mean(predicciones, axis=0)
    return {"por_semilla": por_semilla, "resumen": resumen, "ensemble": _metricas(Y_test, pred_ensemble, ancla)}


def importancia_mdi(modelo, columnas_x, top: int = 15):
    importancias = getattr(modelo, "feature_importances_", None)
    if importancias is None:
        estimadores = getattr(modelo, "estimators_", None)
        if estimadores and hasattr(estimadores[0], "feature_importances_"):
            importancias = np.mean([e.feature_importances_ for e in estimadores], axis=0)
    if importancias is None:
        return None
    orden = np.argsort(importancias)[::-1][:top]
    return [(columnas_x[i], round(float(importancias[i]), 4)) for i in orden]


def importancia_perm(modelo, X_val, Y_val, columnas_x, top: int = 15, n_repeats: int = 4):
    """Importancia por permutación fuera de muestra sobre la validación del último fold rolling (2018)."""
    resultado = permutation_importance(modelo, X_val, Y_val, n_repeats=n_repeats, random_state=0, n_jobs=1)
    orden = np.argsort(resultado.importances_mean)[::-1][:top]
    return [(columnas_x[i], round(float(resultado.importances_mean[i]), 4),
             round(float(resultado.importances_std[i]), 4)) for i in orden]


# Curvas de aprendizaje, para leer el sesgo y la varianza.
def curva_aprendizaje(tipo: str, hiperparametros: dict, X_ajuste, Y_ajuste, ancla: float,
                      n_puntos: int = 6, frac_val: float = 0.15) -> list[dict]:
    """MASE de train vs validación según el tamaño del train (val = cola temporal fija)."""
    n_val = max(90, int(frac_val * len(X_ajuste)))
    X_train_completo, Y_train_completo = X_ajuste[:-n_val], Y_ajuste[:-n_val]
    X_val, Y_val = X_ajuste[-n_val:], Y_ajuste[-n_val:]
    n_arboles = 120 if tipo == "rf" else 150
    curva = []
    for fraccion in np.linspace(0.2, 1.0, n_puntos):
        n_dias = max(90, int(fraccion * len(X_train_completo)))
        modelo = _crear(tipo, hiperparametros, n_arboles=n_arboles, semilla=0)
        modelo.fit(X_train_completo[-n_dias:], Y_train_completo[-n_dias:])
        mae_train = float(np.mean(np.abs(Y_train_completo[-n_dias:] - modelo.predict(X_train_completo[-n_dias:]))))
        mae_val = float(np.mean(np.abs(Y_val - modelo.predict(X_val))))
        curva.append({"n_train": int(n_dias), "mase_train": round(mae_train / ancla, 3),
                      "mase_val": round(mae_val / ancla, 3)})
    return curva


def _plot_curvas(curvas: dict):
    """Figura de curvas de aprendizaje (train vs val) por modelo en figuras/arboles/."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    DIR_FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(curvas), figsize=(6 * len(curvas), 4), squeeze=False)
    for ax, (nombre, curva) in zip(axes[0], curvas.items()):
        n_train = [punto["n_train"] for punto in curva]
        ax.plot(n_train, [punto["mase_train"] for punto in curva], "o-", label="train")
        ax.plot(n_train, [punto["mase_val"] for punto in curva], "s-", label="validación")
        ax.set_title(f"Curva de aprendizaje — {nombre} (D+1)")
        ax.set_xlabel("nº días de train"); ax.set_ylabel("MASE"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout()
    ruta = DIR_FIG / "curva_aprendizaje_dayahead.png"
    fig.savefig(ruta, dpi=110); plt.close(fig)
    return str(ruta.relative_to(RAIZ))


# Se persiste todo para que la comparativa lo cargue sin tener que reentrenar.
def _persistir(tipo: str, modelos, predicciones: list[np.ndarray], Y_test, fechas_test, columnas_y):
    """Guarda modelos (joblib), predicciones de test por (fecha, hora) e Y real; devuelve rutas relativas."""
    carpeta = DIR_ART / tipo
    carpeta.mkdir(parents=True, exist_ok=True)
    rutas = {"modelos_joblib": [], "predicciones_parquet": None}
    for semilla, modelo in zip(SEMILLAS, modelos):
        ruta = carpeta / f"modelo_{tipo}_semilla{semilla}.joblib"
        joblib.dump(modelo, ruta, compress=3)
        rutas["modelos_joblib"].append(str(ruta.relative_to(RAIZ)))
    # Predicciones en formato largo (fecha_origen, hora, y_real, pred_ensemble, pred_s{semilla})
    pred_ensemble = np.mean(predicciones, axis=0)
    filas = []
    for i, fecha_origen in enumerate(fechas_test):
        for j, columna in enumerate(columnas_y):
            hora = int(columna[-2:])
            fila = {"fecha_origen": fecha_origen, "hora": hora, "y_real": float(Y_test[i, j]),
                    "pred_ensemble": float(pred_ensemble[i, j])}
            for semilla, prediccion in zip(SEMILLAS, predicciones):
                fila[f"pred_s{semilla}"] = float(prediccion[i, j])
            filas.append(fila)
    df_pred = pd.DataFrame(filas)
    ruta_parquet = carpeta / f"predicciones_base_{tipo}.parquet"
    df_pred.to_parquet(ruta_parquet)
    rutas["predicciones_parquet"] = str(ruta_parquet.relative_to(RAIZ))
    return rutas


def ejecutar(n_configs_xgb: int = N_CONFIGS_XGB) -> dict:
    inicio = time.time()
    ancla = _ancla()
    datos, columnas_x, columnas_y = construir_tabla()
    n_features = len(columnas_x)

    split = split_canonico_precovid(gap_dias=1, guardar=False)
    folds = rolling_origin_folds()
    dias_origen = datos.index
    ajuste = dias_origen.isin(split.train) | dias_origen.isin(split.val)
    test = dias_origen.isin(split.test)

    train_perm = dias_origen.isin(folds[-1].train)
    val_perm = dias_origen.isin(folds[-1].val)
    X_ajuste, Y_ajuste = datos.loc[ajuste, columnas_x].to_numpy(), datos.loc[ajuste, columnas_y].to_numpy()
    X_test, Y_test = datos.loc[test, columnas_x].to_numpy(), datos.loc[test, columnas_y].to_numpy()
    X_train_perm = datos.loc[train_perm, columnas_x].to_numpy()
    Y_train_perm = datos.loc[train_perm, columnas_y].to_numpy()
    X_val_perm, Y_val_perm = datos.loc[val_perm, columnas_x].to_numpy(), datos.loc[val_perm, columnas_y].to_numpy()
    fechas_test = datos.index[test]

    print(f"\n[D+1] features={n_features} · salidas={len(columnas_y)} · fit={int(ajuste.sum())} días · "
          f"test={int(test.sum())} días · {len(folds)} folds (val {[f.val[0].year for f in folds]})")

    salida = {"horizonte": "dayahead", "n_features": n_features, "n_salidas": len(columnas_y),
              "features": columnas_x,
              "fit_dias": int(ajuste.sum()), "test_dias": int(test.sum()),
              "test_rango": f"{fechas_test.min().date()} → {fechas_test.max().date()}",
              "ancla_mase_mw": ancla, "protocolo_seleccion": "rolling-origin 5 folds",
              "semillas": list(SEMILLAS), "modelos": {}}

    curvas = {}
    for tipo, nombre in (("rf", "RandomForest"), ("xgb", "XGBoost")):
        print(f"\n  --- {nombre} ---")
        rejilla = _rejilla(tipo, n_features)
        configs = rejilla if tipo == "rf" else _muestrear(rejilla, n_configs_xgb, semilla=0)
        print(f"  {len(rejilla)} combos → se evalúan {len(configs)} x {len(folds)} folds")
        n_arboles_cv = N_ARBOLES_CV
        hiperparametros, tabla_cv = tunear_rolling(tipo, datos, columnas_x, columnas_y, folds, configs,
                                                   n_arboles_cv, ancla)
        print(f"  hiperparámetros elegidos: MAE_cv={tabla_cv[0]['mae_cv']:.0f} MW · MASE_cv={tabla_cv[0]['mase_cv']} · {hiperparametros}")

        modelos, info_parada = ajustar_final_semillas(tipo, hiperparametros, X_ajuste, Y_ajuste)
        if info_parada:
            print(f"  Early stopping XGB: {info_parada['best_n_estimators']} árboles "
                  f"(val {info_parada['n_val_earlystop']} días)")
        predicciones = [modelo.predict(X_test) for modelo in modelos]
        metricas = _metricas_multisemilla(Y_test, predicciones, ancla)
        resumen_mase = metricas["resumen"]["MASE"]
        print(f"  TEST (5 semillas): MASE={resumen_mase['media']:.3f} ± {resumen_mase['sd']:.3f} · "
              f"ensemble MASE={metricas['ensemble']['MASE']:.3f} MAPE={metricas['ensemble']['MAPE']:.2f}%")

        curva = curva_aprendizaje(tipo, hiperparametros, X_ajuste, Y_ajuste, ancla)
        curvas[nombre] = curva
        rutas = _persistir(tipo, modelos, predicciones, Y_test, fechas_test, columnas_y)

        # Modelo auxiliar fuera de muestra para la permutación: solo sobre <2018, con los hiperparámetros elegidos.
        n_arboles_aux = info_parada["best_n_estimators"] if info_parada else None
        modelo_aux = _crear(tipo, hiperparametros, n_arboles=n_arboles_aux, semilla=0)
        modelo_aux.fit(X_train_perm, Y_train_perm)

        bloque = {"hiperparametros_elegidos": hiperparametros, "mae_cv_mejor": tabla_cv[0]["mae_cv"],
                  "mase_cv_mejor": tabla_cv[0]["mase_cv"], "cv_completo": tabla_cv,
                  "metricas_test": metricas, "early_stopping": info_parada, "curva_aprendizaje": curva,
                  "importancia_mdi_top15": importancia_mdi(modelos[0], columnas_x),
                  "importancia_perm_val_top15": importancia_perm(modelo_aux, X_val_perm, Y_val_perm, columnas_x),
                  "artefactos": rutas}
        salida["modelos"][tipo] = bloque

    try:
        salida["curva_aprendizaje_figura"] = _plot_curvas(curvas)
    except Exception as error:
        salida["curva_aprendizaje_figura"] = {"error": str(error)}
    salida["segundos"] = round(time.time() - inicio, 1)
    print(f"\n[D+1] completado en {salida['segundos']:.0f}s")
    return salida


def main(n_configs_xgb: int = N_CONFIGS_XGB) -> dict:
    DIR_MET.mkdir(parents=True, exist_ok=True)
    resumen = {"horizonte": "day-ahead D+1", "semillas": list(SEMILLAS)}
    resumen["resultado"] = ejecutar(n_configs_xgb)
    resumen["caveat"] = "temperatura de previsión perfecta: MAPE/MASE son cota optimista"
    (DIR_MET / "arboles.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False))
    print(f"\nGuardado {DIR_MET / 'arboles.json'}")
    print("\nTEST (D+1):")
    for tipo, bloque in resumen["resultado"]["modelos"].items():
        resumen_metricas = bloque["metricas_test"]["resumen"]; ensemble = bloque["metricas_test"]["ensemble"]
        print(f"  {tipo:3}: MASE={resumen_metricas['MASE']['media']:.3f}±{resumen_metricas['MASE']['sd']:.3f} "
              f"(ens {ensemble['MASE']:.3f}) MAPE={ensemble['MAPE']:.2f}% MAE={ensemble['MAE']:.0f} RMSE={ensemble['RMSE']:.0f}")
    return resumen


if __name__ == "__main__":
    main()
