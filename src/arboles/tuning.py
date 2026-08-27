"""Tuning de hiperparámetros de los árboles (Random Forest y XGBoost).
## Uso

    .venv/bin/python -m src.arboles.tuning rf # local, 108 exhaustivas
    .venv/bin/python -m src.arboles.tuning xgb 200 # random search en CPU
    .venv/bin/python -m src.arboles.tuning xgb 200 --device=cuda # en GPU (equivale a --cuda)
    .venv/bin/python -m src.arboles.tuning finales # reentrena el ganador de XGBoost y persiste sus predicciones
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.arboles.modelo import DIR_MET, N_ARBOLES_CV, _ancla, _crear, _muestrear as muestrear
from src.redes.datos import construir_xy
from src.features.validacion import rolling_origin_folds, split_canonico_precovid

XGB_TOPE_ARBOLES = 2000  # cota del early stopping dentro de cada fold
XGB_PACIENCIA = 30
XGB_FRAC_ES = 0.15  # cola del train de cada fold reservada para el early stopping

# RF: 3 x 3 x 4 x 3 = 108 combinaciones, exhaustiva. `max_features` va en nº absoluto sobre p=152.
RF_ESPACIO = {
    "max_depth": [20, 30, None],  # None = sin límite de profundidad
    "min_samples_leaf": [1, 2, 5],  # con max_depth=None este eje pasa a ser el que limita
    "max_features": [50, 76, 114, 152],  # p/3, p/2, 3p/4 y p (bagging puro)
    "max_samples": [None, 0.8, 0.6],  # fracción del bootstrap
}
# XGB: 4 x 2 x 4 x 3 x 2 x 3 x 2 = 1.152 combinaciones, de las que se muestrean N.
XGB_ESPACIO = {
    "max_depth": [6, 8, 10, 12],
    "learning_rate": [0.05, 0.1],  # tasas más bajas exigirían muchos más árboles
    "min_child_weight": [1, 5, 15, 40],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.4, 0.8],
    "reg_lambda": [1, 10, 50],  # el defecto de la librería es 1
    "reg_alpha": [0, 1],  # el defecto de la librería es 0
}

CONFIG_REFERENCIA = {
    "rf":  {"max_depth": 20, "min_samples_leaf": 1, "max_features": 50, "max_samples": None},
    "xgb": {"max_depth": 8, "learning_rate": 0.05, "min_child_weight": 5, "subsample": 0.8,
            "colsample_bytree": 0.8, "reg_lambda": 1, "reg_alpha": 0},
}
MAE_CV_REFERENCIA = {"rf": 578.9, "xgb": 440.8}  # MAE_cv esperado de esas configuraciones


def combinaciones(espacio: dict) -> list[dict]:
    claves = list(espacio)
    return [dict(zip(claves, valores)) for valores in itertools.product(*(espacio[k] for k in claves))]


def _preparar(device: str = "cpu"):
    """Prepara datos y máscaras de folds; con `device='cuda'` sube los datos a memoria de GPU (cupy)."""
    datos, columnas_x, columnas_y = construir_xy(n_dias=1)
    folds = rolling_origin_folds()
    dias_origen = datos.index
    mascaras_por_fold = [(dias_origen.isin(fold.train), dias_origen.isin(fold.val)) for fold in folds]
    X, Y = datos[columnas_x].to_numpy(), datos[columnas_y].to_numpy()
    if device == "cuda":
        try:
            import cupy
            X, Y = cupy.asarray(X, dtype=cupy.float32), cupy.asarray(Y, dtype=cupy.float32)
            mascaras_por_fold = [(cupy.asarray(train), cupy.asarray(val)) for train, val in mascaras_por_fold]
            print("    datos en memoria de GPU (cupy): sin caída al camino lento de predicción", flush=True)
        except Exception as error:
            print(f"    AVISO: no se pudo usar cupy ({type(error).__name__}: {error}); se sigue con numpy "
                  "(mismo resultado, más lento)", flush=True)
            X, Y = datos[columnas_x].to_numpy(), datos[columnas_y].to_numpy()
            mascaras_por_fold = [(dias_origen.isin(fold.train), dias_origen.isin(fold.val)) for fold in folds]
    return X, Y, mascaras_por_fold


def estimar_n_arboles(hiperparametros: dict, X, Y, mascara_train, device: str = "cpu") -> int:
    """Nº de árboles de una configuración de XGBoost por parada temprana sobre la máscara de ajuste dada."""
    X_train, Y_train = X[mascara_train], Y[mascara_train]
    n_val_parada = max(60, int(XGB_FRAC_ES * len(X_train)))
    modelo_parada = _crear("xgb", hiperparametros, n_arboles=XGB_TOPE_ARBOLES, semilla=0, device=device)
    modelo_parada.set_params(early_stopping_rounds=XGB_PACIENCIA)
    modelo_parada.fit(X_train[:-n_val_parada], Y_train[:-n_val_parada],
                      eval_set=[(X_train[-n_val_parada:], Y_train[-n_val_parada:])], verbose=False)
    return max(50, int(modelo_parada.best_iteration) + 1)


def evaluar(tipo: str, hiperparametros: dict, X, Y, mascaras_por_fold,
            device: str = "cpu") -> tuple[list[float], list[int]]:
    """MAE fuera de muestra en cada fold; en XGBoost reajusta cada fold con el nº de árboles estimado."""
    # Los folds son expansivos, así que el último es el de más historia: sobre él se estima el nº de árboles.
    n_fijo = estimar_n_arboles(hiperparametros, X, Y, mascaras_por_fold[-1][0], device) if tipo == "xgb" else None
    maes, arboles = [], []
    for train, val in mascaras_por_fold:
        X_train, Y_train, X_val, Y_val = X[train], Y[train], X[val], Y[val]
        if tipo == "rf":
            modelo = _crear("rf", hiperparametros, n_arboles=N_ARBOLES_CV, semilla=0)
            modelo.fit(X_train, Y_train)
            arboles.append(N_ARBOLES_CV)
        else:
            arboles.append(n_fijo)
            modelo = _crear("xgb", hiperparametros, n_arboles=n_fijo, semilla=0, device=device)
            modelo.fit(X_train, Y_train, verbose=False)
        # `predict` devuelve numpy aunque la entrada sea cupy: se bajan los dos al host antes de restar
        # (`.get()` solo existe en cupy, por eso el `hasattr`).
        _host = lambda arr: arr.get() if hasattr(arr, "get") else arr
        maes.append(float(np.mean(np.abs(_host(Y_val) - _host(modelo.predict(X_val))))))
    return maes, arboles


def _ruta(tipo: str, trozo: int = 0, trozos: int = 1) -> Path:
    """Ruta del JSON de resultados; con `trozos>1` cada proceso escribe el suyo (`..._g0.json`)."""
    sufijo = "" if trozos <= 1 else f"_g{trozo}"
    return DIR_MET / f"tuning_{tipo}{sufijo}.json"


def _cargar(tipo: str, trozo: int = 0, trozos: int = 1) -> dict:
    ruta = _ruta(tipo, trozo, trozos)
    if ruta.exists():
        estado = json.loads(ruta.read_text())
        
        espacio = RF_ESPACIO if tipo == "rf" else XGB_ESPACIO
        if estado.get("combinaciones_totales") != len(combinaciones(espacio)):
            raise SystemExit(f"{ruta.name} se generó con otra rejilla "
                             f"({estado.get('combinaciones_totales')} combinaciones, ahora hay "
                             f"{len(combinaciones(espacio))}). Use un sufijo nuevo en vez de reanudar.")
        return estado
    espacio = RF_ESPACIO if tipo == "rf" else XGB_ESPACIO
    return {"familia": tipo, "espacio": {k: [str(v) for v in vs] for k, vs in espacio.items()},
            "combinaciones_totales": len(combinaciones(espacio)), "ancla_mase": _ancla(),
            "protocolo": "rolling-origin 5 folds, MAE fuera de muestra",
            "config_referencia": CONFIG_REFERENCIA[tipo], "ranking": []}


def _guardar(tipo: str, estado: dict, trozo: int = 0, trozos: int = 1) -> None:
    estado["ranking"].sort(key=lambda registro: registro["mae_cv"])
    estado["n_evaluadas"] = len(estado["ranking"])
    if estado["ranking"]:
        estado["mejor"] = estado["ranking"][0]
    _ruta(tipo, trozo, trozos).write_text(json.dumps(estado, indent=2, ensure_ascii=False))


def fusionar(tipo: str = "xgb", trozos: int = 2) -> dict:
    """Une los JSON de los procesos en el fichero canónico `tuning_<tipo>.json`."""
    estado, vistas = None, set()
    for trozo in range(trozos):
        ruta = _ruta(tipo, trozo, trozos)
        if not ruta.exists():
            print(f"  aviso: falta {ruta.name}", flush=True)
            continue
        parcial = json.loads(ruta.read_text())
        if estado is None:
            estado = {k: v for k, v in parcial.items() if k != "ranking"} | {"ranking": []}
        for registro in parcial["ranking"]:
            clave = json.dumps(registro["hiperparametros"], sort_keys=True)
            if clave not in vistas:
                vistas.add(clave)
                estado["ranking"].append(registro)
    if estado is None:
        raise SystemExit("no hay ningún trozo que fusionar")
    estado["trozos_fusionados"] = trozos
    _guardar(tipo, estado)
    print(f"  fusionadas {len(estado['ranking'])} configuraciones de {trozos} procesos "
          f"→ {_ruta(tipo).name}", flush=True)
    return estado


def buscar(tipo: str, n: int | None = None, device: str = "cpu", presupuesto_s: float | None = None,
           trozo: int = 0, trozos: int = 1) -> dict:
    """Evalúa las configuraciones; con `trozos>1` toma solo los índices ≡ `trozo` (mod `trozos`)."""
    ancla = _ancla()
    espacio = RF_ESPACIO if tipo == "rf" else XGB_ESPACIO
    rejilla = combinaciones(espacio)
    configs = rejilla if (n is None or n >= len(rejilla)) else muestrear(rejilla, n, forzar=CONFIG_REFERENCIA[tipo])
    total_pedidas = len(configs)
    if trozos > 1:
        configs = configs[trozo::trozos]
    estado = _cargar(tipo, trozo, trozos)
    hechas = {json.dumps(registro["hiperparametros"], sort_keys=True) for registro in estado["ranking"]}
    X, Y, mascaras_por_fold = _preparar(device)
    etiqueta = "" if trozos <= 1 else f" · trozo {trozo + 1}/{trozos}"
    print(f"=== {tipo.upper()}{etiqueta} · {len(configs)} configuraciones (de {total_pedidas} pedidas, "
          f"espacio de {len(rejilla)}) · device={device}", flush=True)
    if hechas:
        print(f"    reanudando: {len(hechas)} ya evaluadas, se saltan", flush=True)
    inicio = time.time()
    for j, hiperparametros in enumerate(configs, 1):
        clave = json.dumps(hiperparametros, sort_keys=True)
        if clave in hechas:
            continue
        if presupuesto_s and time.time() - inicio > presupuesto_s:
            print(f"    presupuesto agotado en la config {j}; lo evaluado queda guardado", flush=True)
            break
        maes, arboles = evaluar(tipo, hiperparametros, X, Y, mascaras_por_fold, device=device)
        mae = float(np.mean(maes))
        estado["ranking"].append({
            "hiperparametros": hiperparametros, "mae_cv": round(mae, 2), "mase_cv": round(mae / ancla, 4),
            "se_cv": round(float(np.std(maes, ddof=1) / np.sqrt(len(maes))), 2),
            "mae_por_fold": [round(x, 1) for x in maes], "n_arboles_por_fold": arboles,
            "es_referencia": hiperparametros == CONFIG_REFERENCIA[tipo]})
        _guardar(tipo, estado, trozo, trozos)  # incremental: sobrevive a un corte
        marca = "  <<< REFERENCIA" if hiperparametros == CONFIG_REFERENCIA[tipo] else ""
        print(f"    [{j:3d}/{len(configs)}] MAE_cv={mae:6.1f}  MASE={mae / ancla:.4f}  "
              f"({time.time() - inicio:.0f}s){marca}  {hiperparametros}", flush=True)
        # ETA para dimensionar cuántas configuraciones caben en el presupuesto de la sesión.
        hechas_ahora = len(estado["ranking"]) - len(hechas)
        if hechas_ahora:
            ritmo = (time.time() - inicio) / hechas_ahora
            restan = len(configs) - j
            cabe = int((presupuesto_s - (time.time() - inicio)) / ritmo) if presupuesto_s else restan
            print(f"         ritmo {ritmo / 60:.1f} min/config · quedan {restan} · "
                  f"caben {min(cabe, restan)} en el presupuesto · fin estimado en "
                  f"{min(cabe, restan) * ritmo / 3600:.1f} h", flush=True)
    _guardar(tipo, estado, trozo, trozos)
    if estado["ranking"]:
        mejor = estado["ranking"][0]
        referencia = next((registro for registro in estado["ranking"] if registro["es_referencia"]), None)
        print(f"\n  MEJOR: MAE_cv={mejor['mae_cv']} (MASE {mejor['mase_cv']}) · {mejor['hiperparametros']}", flush=True)
        if referencia:
            print(f"  referencia: MAE_cv={referencia['mae_cv']} (esperado {MAE_CV_REFERENCIA[tipo]}) · "
                  f"mejora del ganador: {mejor['mae_cv'] / referencia['mae_cv'] - 1:+.2%}", flush=True)
    return estado


def finales(hiperparametros: dict, device: str = "cpu", semillas=(0, 1, 2, 3, 4)) -> dict:
    """Reentrena el ganador con 5 semillas sobre el split canónico y persiste TEST y VALIDACIÓN."""
    ancla = _ancla()
    datos, columnas_x, columnas_y = construir_xy(n_dias=1)
    split = split_canonico_precovid()
    dias_origen = datos.index
    mascara_train = dias_origen.isin(split.train)
    mascara_val = dias_origen.isin(split.val)
    mascara_test = dias_origen.isin(split.test)
    X, Y = datos[columnas_x].to_numpy(), datos[columnas_y].to_numpy()
    n_arboles = estimar_n_arboles(hiperparametros, X, Y, mascara_train, device)
    print(f"  finales · {n_arboles} árboles · train={mascara_train.sum()} val={mascara_val.sum()} "
          f"test={mascara_test.sum()} días", flush=True)

    predicciones = {"val": [], "test": []}
    for semilla in semillas:
        modelo = _crear("xgb", hiperparametros, n_arboles=n_arboles, semilla=semilla, device=device)
        modelo.fit(X[mascara_train], Y[mascara_train], verbose=False)
        predicciones["val"].append(modelo.predict(X[mascara_val]))
        predicciones["test"].append(modelo.predict(X[mascara_test]))
        mae = float(np.mean(np.abs(Y[mascara_test] - predicciones["test"][-1])))
        print(f"    semilla {semilla}: MAE test {mae:.1f} MW · MASE {mae / ancla:.4f}", flush=True)

    salida = {"hiperparametros": hiperparametros, "n_arboles": n_arboles, "semillas": list(semillas),
              "ajuste": "solo train"}
    for etiqueta, mascara in (("val", mascara_val), ("test", mascara_test)):
        ensemble = np.mean(predicciones[etiqueta], axis=0)
        real = Y[mascara]
        filas = [{"fecha_origen": pd.Timestamp(fecha), "hora": hora, "y_real": float(real[i, hora]),
                  "pred_ensemble": float(ensemble[i, hora]),
                  **{f"pred_s{semilla}": float(predicciones[etiqueta][k][i, hora])
                     for k, semilla in enumerate(semillas)}}
                 for i, fecha in enumerate(dias_origen[mascara]) for hora in range(real.shape[1])]
        ruta = DIR_MET.parent / "arboles" / "xgb" / f"predicciones_{etiqueta}_xgb.parquet"
        ruta.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(filas).to_parquet(ruta)
        mae = float(np.mean(np.abs(real - ensemble)))
        salida[etiqueta] = {"MAE": round(mae, 2), "MASE": round(mae / ancla, 4),
                            "parquet": str(ruta.relative_to(RAIZ))}
        print(f"  {etiqueta}: MASE ensemble {mae / ancla:.4f} → {ruta.name}", flush=True)
    (DIR_MET / "finales_xgb.json").write_text(json.dumps(salida, indent=2, ensure_ascii=False))
    return salida


def main(argv: list[str] | None = None) -> dict:
    args = list(argv if argv is not None else sys.argv[1:])
    if not args or args[0] not in ("rf", "xgb", "fusionar", "finales"):
        raise SystemExit(__doc__.split("## Uso")[1])
    if args[0] == "fusionar":
        return fusionar("xgb", int(args[1]) if len(args) > 1 else 2)
    if args[0] == "finales":
        ranking = json.loads(_ruta("xgb").read_text())
        print(f"ganador de la búsqueda ({ranking['n_evaluadas']} configuraciones, "
              f"MAE_cv {ranking['mejor']['mae_cv']} MW)", flush=True)
        return finales(ranking["mejor"]["hiperparametros"],
                       device="cuda" if "--cuda" in args else "cpu")

    def opc(nombre, defecto, tipo_=int):
        for arg in args:
            if arg.startswith(f"--{nombre}="):
                return tipo_(arg.split("=", 1)[1])
        return defecto

    tipo = args[0]
    n = next((int(arg) for arg in args[1:] if arg.isdigit()), None)
    device = opc("device", "cuda" if "--cuda" in args else "cpu", str)
    trozos = opc("trozos", 1)
    trozo = opc("trozo", 0)
    presupuesto = opc("presupuesto", None, float)
    return buscar(tipo, n=n, device=device, presupuesto_s=presupuesto, trozo=trozo, trozos=trozos)


if __name__ == "__main__":
    main()
