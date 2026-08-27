"""Comparativa final + Diebold-Mariano/Holm/MCS sobre las predicciones de test persistidas (D+1)."""
from __future__ import annotations

import json
import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.metricas import ancla_mase
from src.datos_precovid import serie_precovid
from src.redes.datos import construir_xy
from src.features.validacion import split_canonico_precovid
from src.comparativa.dm import (diebold_mariano_diff, holm, model_confidence_set, bootstrap_ci)

DIR_MET = RAIZ / "resultados" / "metricas"
DIR_ARB = RAIZ / "resultados" / "arboles"
DIR_RED = RAIZ / "resultados" / "redes"
DIR_SAR = RAIZ / "resultados" / "sarimax" / "final"
DIR_FIG = RAIZ / "figuras" / "comparativa"


def _de_origen(df: pd.DataFrame, columna_pred: str) -> pd.Series:
    """Parquet con (fecha_origen, hora) [origen D, objetivo D+1] → Series por datetime-objetivo."""
    dt = pd.to_datetime(df["fecha_origen"]) + pd.Timedelta(days=1) + pd.to_timedelta(df["hora"], unit="h")
    return pd.Series(df[columna_pred].to_numpy(), index=pd.DatetimeIndex(dt)).sort_index()


def _de_objetivo(df: pd.DataFrame, columna_pred: str) -> pd.Series:
    """Parquet SARIMAX de los 24 modelos por hora con (datetime, hora) [datetime = día objetivo a las 00:00]."""
    dt = pd.to_datetime(df["datetime"]) + pd.to_timedelta(df["hora"], unit="h")
    return pd.Series(df[columna_pred].to_numpy(), index=pd.DatetimeIndex(dt)).sort_index()


def _de_datetime(df: pd.DataFrame, columna_pred: str) -> pd.Series:
    """Series indexada por `datetime`, que en la SARIMAX continua ya incluye la hora (no hay que sumarla)."""
    return pd.Series(df[columna_pred].to_numpy(),
                     index=pd.DatetimeIndex(pd.to_datetime(df["datetime"]))).sort_index()


def cargar_persistidas() -> dict:
    """{nombre: Series(datetime-objetivo → predicción MW)} de árboles, redes y SARIMAX."""
    # Solo entran los modelos del grid horario D+1 limpio, que es lo que necesita el DM pareado.
    salida, faltan = {}, []
    fuentes = [
        ("XGBoost", DIR_ARB / "xgb" / "predicciones_test_xgb.parquet", _de_origen, "pred_ensemble"),
        ("RandomForest", DIR_ARB / "rf" / "predicciones_test_rf.parquet", _de_origen, "pred_ensemble"),
        ("MLP (red)", DIR_RED / "predicciones_test_mlp.parquet", _de_origen, "pred_ensemble"),
        ("LSTM (red)", DIR_RED / "predicciones_test_lstm.parquet", _de_origen, "pred_ensemble"),
        ("SARIMAX por hora", DIR_SAR / "por_hora_pred.parquet", _de_objetivo, "pred"),
        ("SARIMAX continua", DIR_SAR / "continua_pred.parquet", _de_datetime, "pred"),
    ]
    for nombre, ruta, lector, columna_pred in fuentes:
        if ruta.exists():
            salida[nombre] = lector(pd.read_parquet(ruta), columna_pred)
        else:
            faltan.append(nombre)
    if faltan:
        print(f"  [aviso] sin predicciones persistidas de: {faltan} (se omiten del DM).")
    return salida


def reconstruir_lineales_baselines() -> dict:
    """Comparadores lineales escalados + baselines naive sobre la misma tabla MIMO, con ajuste solo-train."""
    datos, columnas_x, columnas_y = construir_xy(n_dias=1)
    split = split_canonico_precovid(gap_dias=1, guardar=False)
    dias_origen = datos.index
    ajuste = dias_origen.isin(split.train) | dias_origen.isin(split.val)
    test = dias_origen.isin(split.test)
    X_ajuste = datos.loc[ajuste, columnas_x].to_numpy()
    Y_ajuste = datos.loc[ajuste, columnas_y].to_numpy()
    X_test = datos.loc[test, columnas_x].to_numpy()
    origenes = datos.index[test]
    objetivos = pd.DatetimeIndex([dia + pd.Timedelta(days=1) + pd.Timedelta(hours=hora)
                                  for dia in origenes for hora in range(24)])

    def _a_serie(pred2d):
        return pd.Series(np.asarray(pred2d).reshape(-1), index=objetivos).sort_index()

    salida = {}
    ols = make_pipeline(StandardScaler(), LinearRegression()).fit(X_ajuste, Y_ajuste)
    salida["Regresión OLS"] = _a_serie(ols.predict(X_test))
    ridge = make_pipeline(StandardScaler(),
                          RidgeCV(alphas=np.logspace(-2, 4, 13),
                                  cv=TimeSeriesSplit(n_splits=5))).fit(X_ajuste, Y_ajuste)
    salida["Regresión Ridge"] = _a_serie(ridge.predict(X_test))

    serie = serie_precovid()
    salida["Naive semanal (t-168)"] = pd.Series(
        serie.reindex(objetivos - pd.Timedelta(hours=168)).to_numpy(), index=objetivos).sort_index()
    salida["Naive día (t-24)"] = pd.Series(
        serie.reindex(objetivos - pd.Timedelta(hours=24)).to_numpy(), index=objetivos).sort_index()
    return salida


def _metricas(real: np.ndarray, pred: np.ndarray, ancla: float) -> dict:
    errores = real - pred
    return {"MAE": float(np.mean(np.abs(errores))), "RMSE": float(np.sqrt(np.mean(errores ** 2))),
            "MAPE": float(np.mean(np.abs(errores / real)) * 100), "MASE": float(np.mean(np.abs(errores)) / ancla)}


def _metricas_por_hora(real: pd.Series, pred: pd.Series, ancla: float) -> dict:
    errores = (real - pred)
    horas = real.index.hour
    salida = {}
    for hora in range(24):
        mascara = horas == hora
        salida[f"h{hora:02d}"] = {"MAE": round(float(np.abs(errores[mascara]).mean()), 1),
                                  "MASE": round(float(np.abs(errores[mascara]).mean() / ancla), 4)}
    return salida


def _figuras(matriz_dm: pd.DataFrame):
    """Genera el heatmap de p-valores del Diebold-Mariano y devuelve las rutas relativas a la raíz."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src import plots
    plots.usar_estilo("academico")
    DIR_FIG.mkdir(parents=True, exist_ok=True)
    rutas = []

    # Heatmap p-valores DM
    fig, ax = plt.subplots(figsize=(9, 7))
    matriz = matriz_dm.astype(float)
    im = ax.imshow(matriz.to_numpy(), cmap="RdYlGn_r", vmin=0, vmax=0.10)
    ax.set_xticks(range(len(matriz))); ax.set_xticklabels(matriz.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(matriz))); ax.set_yticklabels(matriz.index, fontsize=7)
    for i in range(len(matriz)):
        for j in range(len(matriz)):
            valor = matriz.iloc[i, j]
            if not np.isnan(valor):
                ax.text(j, i, f"{valor:.2f}", ha="center", va="center", fontsize=6)
    ax.set_title("Diebold-Mariano (p-valor; fila vs columna)")
    fig.colorbar(im, ax=ax, shrink=.7, label="p-valor"); fig.tight_layout()
    ruta = DIR_FIG / "dm_heatmap.png"; fig.savefig(ruta, dpi=120); plt.close(fig)
    rutas.append(str(ruta.relative_to(RAIZ)))
    return rutas


def main() -> dict:
    warnings.simplefilter("ignore")
    ancla = ancla_mase()
    print("\nComparativa de modelos (day-ahead D+1)")

    predicciones = {**cargar_persistidas(), **reconstruir_lineales_baselines()}

    # Grid común (intersección de datetimes-objetivo) + real desde la serie (fuente única de verdad)
    serie = serie_precovid()
    comun = None
    for serie_modelo in predicciones.values():
        indices = serie_modelo.dropna().index
        comun = indices if comun is None else comun.intersection(indices)
    comun = comun.sort_values()
    real = pd.Series(serie.reindex(comun).to_numpy(), index=comun)
    print(f"Modelos: {len(predicciones)} | grid común: {len(comun):,} pares — {comun.min()} → {comun.max()}")

    # Métricas + IC bootstrap del MASE + métricas por hora + pérdida |e| diaria
    filas, mae_hora, err_abs, perdida_diaria = {}, {}, {}, {}
    dia = pd.Series(comun.normalize(), index=comun)
    for nombre, serie_modelo in predicciones.items():
        prediccion = serie_modelo.reindex(comun)
        real_np, pred_np = real.to_numpy(), prediccion.to_numpy()
        filas[nombre] = _metricas(real_np, pred_np, ancla)
        mae_hora[nombre] = _metricas_por_hora(real, prediccion, ancla)
        error_abs = np.abs(real_np - pred_np)
        err_abs[nombre] = pd.Series(error_abs, index=comun)
        perdida_diaria[nombre] = err_abs[nombre].groupby(dia).mean()  # pérdida MAE por día-objetivo
        ic = bootstrap_ci((perdida_diaria[nombre] / ancla).to_numpy(), np.mean, B=2000, block=14)
        filas[nombre]["MASE_ic95"] = [round(ic["lo"], 4), round(ic["hi"], 4)]
    tabla = pd.DataFrame({k: {kk: v[kk] for kk in ("MAE", "RMSE", "MAPE", "MASE")} for k, v in filas.items()}).T
    tabla = tabla.sort_values("MASE")
    print("\nTabla comparativa (test, grid común):")
    print(tabla.round({"MAE": 0, "RMSE": 0, "MAPE": 3, "MASE": 4}).to_string())

    modelos = list(tabla.index)
    perdidas_df = pd.DataFrame({m: perdida_diaria[m] for m in modelos}).dropna()

    # DM pareado sobre todos los pares, HLN y HAC. La matriz es de p-valores, fila contra columna,
    # y un loss_diff negativo quiere decir que gana la fila.
    matriz_dm = pd.DataFrame(index=modelos, columns=modelos, dtype=float)
    dm_detalle, p_holm_in = {}, {}
    for modelo_a in modelos:
        for modelo_b in modelos:
            if modelo_a == modelo_b:
                continue
            diferencias = (perdida_diaria[modelo_a] - perdida_diaria[modelo_b]).dropna().to_numpy()
            res_hln = diebold_mariano_diff(diferencias, h=1, hac=False)
            res_hac = diebold_mariano_diff(diferencias, h=1, hac=True)
            matriz_dm.loc[modelo_a, modelo_b] = round(res_hln["p_value"], 4)
            dm_detalle[f"{modelo_a} vs {modelo_b}"] = {"HLN": res_hln, "HAC": res_hac}
    # Holm se alimenta de la variante HAC (Newey-West): la pérdida diaria está autocorrelada
    # y suponer ruido blanco daría p-valores anticonservadores.
    for modelo_a, modelo_b in combinations(modelos, 2):  # Holm sobre pares únicos
        p_holm_in[f"{modelo_a} vs {modelo_b}"] = dm_detalle[f"{modelo_a} vs {modelo_b}"]["HAC"]["p_value"]
    holm_res = holm(p_holm_in, alpha=0.05)

    mcs = model_confidence_set(perdidas_df, alpha=0.10, B=2000, block=14)
    print(f"\nModel Confidence Set (α=0.10): {mcs['conjunto_mcs']}")

    # DM por hora de los pares clave (con cautela de autocorrelación: DM diario es el de referencia)
    pares_clave = [(modelos[0], modelos[1])]
    for candidato in [("MLP (red)", "XGBoost"), ("XGBoost", "Regresión Ridge"), ("MLP (red)", "SARIMAX por hora")]:
        if all(m in predicciones for m in candidato) and candidato not in pares_clave:
            pares_clave.append(candidato)
    dm_hora = {}
    horas = pd.Series(comun.hour, index=comun)
    for modelo_a, modelo_b in pares_clave:
        p_por_hora = {}
        for hora in range(24):
            mascara = (horas == hora).to_numpy()
            p_por_hora[hora] = round(diebold_mariano_diff(
                (err_abs[modelo_a] - err_abs[modelo_b]).to_numpy()[mascara], h=1, hac=True)["p_value"], 4)
        dm_hora[f"{modelo_a} vs {modelo_b}"] = p_por_hora

    try:
        figuras = _figuras(matriz_dm)
    except Exception as error:
        figuras = {"error": str(error)}

    DIR_MET.mkdir(parents=True, exist_ok=True)
    tabla.round(4).to_csv(DIR_MET / "comparativa.csv")
    matriz_dm.to_csv(DIR_MET / "dm_pvalores.csv")
    salida = {
        "horizonte": "day-ahead D+1", "n_modelos": len(predicciones), "modelos": modelos,
        "n_pares_comun": int(len(comun)), "n_dias_comun": int(len(perdidas_df)),
        "periodo": f"{comun.min()} → {comun.max()}", "ancla_mase_mw": ancla,
        "metricas_por_modelo": {k: {kk: round(vv, 4) if isinstance(vv, float) else vv
                                    for kk, vv in v.items()} for k, v in filas.items()},
        "metricas_por_hora": mae_hora,
        "dm_pareado": {k: {"HLN": {kk: (round(vv, 5) if isinstance(vv, float) else vv) for kk, vv in v["HLN"].items()},
                           "HAC": {kk: (round(vv, 5) if isinstance(vv, float) else vv) for kk, vv in v["HAC"].items()}}
                       for k, v in dm_detalle.items()},
        "holm": holm_res, "model_confidence_set": mcs, "dm_por_hora": dm_hora,
        "figuras": figuras,
        "nota": "Temperatura de 'previsión perfecta': cota optimista común a todos los modelos.",
    }
    (DIR_MET / "comparativa.json").write_text(json.dumps(salida, indent=2, ensure_ascii=False))
    print("\nGuardado comparativa.{json,csv} + dm_pvalores.csv + figuras/comparativa/")
    return salida


if __name__ == "__main__":
    main()
