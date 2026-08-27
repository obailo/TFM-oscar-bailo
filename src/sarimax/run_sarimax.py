"""Runner SARIMAX por fold, resumible y persistente. Horizonte D+1.

Uso:
  python -m src.sarimax.run_sarimax <N>       # rolling fold N in  {1..5} (val = año 2013+N)
  python -m src.sarimax.run_sarimax final     # ajusta 2000-2018 y evalúa en el TEST reservado
  python -m src.sarimax.run_sarimax agregar   # agrega folds + final → resultados/metricas/sarimax_resumen.json

Flags: `--force` (reejecuta aunque exista) · `--solo continua,por_hora` · `--jobs N` (procesos
24 modelos por hora) · `--smoke` (ventana/maxiter mínimos, NO usar en real).
"""
from __future__ import annotations

# BLAS a 1 hilo por worker (lo heredan los procesos hijos del ajuste por hora).
import os
for _var_entorno in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                     "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var_entorno, "1")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.sarimax.datos_sarimax import (fechas_origen_precovid, rolling_origin_folds,
                                       TEST_INI, PURGA_DIAS, ROLLING_VAL_YEARS)
from src.metricas import ancla_mase
from src.sarimax import modelo

DIR_OUT = RAIZ / "resultados" / "sarimax"
DIR_METRICAS = RAIZ / "resultados" / "metricas"
# La continua va la última porque es la cara: así las baratas quedan persistidas antes por si falla.
ESTRUCTURAS = ("por_hora", "continua")


def spec_fold(fold: str):
    """Devuelve (train_dias, eval_dias, etiqueta) del fold. `fold` in  {'1'..'5','final'}."""
    fechas = fechas_origen_precovid()
    if fold == "final":
        test_ini = pd.Timestamp(TEST_INI)
        train = fechas[fechas < test_ini]
        train = train[: max(0, len(train) - PURGA_DIAS)]  # purga D+1
        eval_ = fechas[fechas >= test_ini]
        return train, eval_, "test_2019_2020"
    n = int(fold)
    corte = rolling_origin_folds()[n - 1]
    return corte.train, corte.val, f"val_{ROLLING_VAL_YEARS[n - 1]}"


def _dir_fold(fold: str) -> Path:
    directorio = DIR_OUT / (f"fold_{fold}" if fold != "final" else "final")
    directorio.mkdir(parents=True, exist_ok=True)
    return directorio


def _persistir(dir_fold: Path, nombre: str, resultado: dict):
    """Escribe el parquet de predicciones y el JSON (sin el DataFrame) de una estructura."""
    pred = resultado.pop("pred")
    pred.to_parquet(dir_fold / f"{nombre}_pred.parquet", index=False)
    (dir_fold / f"{nombre}.json").write_text(json.dumps(resultado, indent=2, ensure_ascii=False))


def ejecutar_estructura(nombre: str, fold: str, train, eval_, etiqueta: str,
                        force: bool, jobs: int, smoke: bool):
    dir_fold = _dir_fold(fold)
    ruta_json = dir_fold / f"{nombre}.json"
    if ruta_json.exists() and not force:
        print(f"[skip] {fold}/{nombre} ya existe ({ruta_json})", flush=True)
        return
    inicio = time.time()
    print(f"[run ] {fold}/{nombre} (train {len(train)} días, eval {len(eval_)} días)…", flush=True)
    if nombre == "continua":
        extra = {}
        if smoke:
            extra = {"ventana_anos": 0.17, "maxiter": 15}  # ~2 meses y pocas iteraciones
        resultado = modelo.evaluar_continua(train, eval_, **extra)
    elif nombre == "por_hora":
        resultado = _por_hora(train, eval_, jobs, smoke)
    else:
        raise ValueError(nombre)
    resultado["fold"] = fold
    resultado["eval_periodo"] = etiqueta
    _persistir(dir_fold, nombre, resultado)
    metricas = resultado["metricas"]
    print(f"[ok  ] {fold}/{nombre}: MASE={metricas['MASE']:.3f} MAE={metricas['MAE']:.0f} "
          f"conv={resultado.get('converged')} ({time.time() - inicio:.0f}s)", flush=True)


def _por_hora(train, eval_, jobs: int, smoke: bool) -> dict:
    """24 modelos por hora en paralelo (procesos, BLAS 1 hilo/worker)."""
    ancla = ancla_mase()
    horas = range(0, 24, 6) if smoke else range(24)  # en modo smoke, solo 4 horas
    inicio = time.time()
    resultados = Parallel(n_jobs=jobs, backend="loky", inner_max_num_threads=1)(
        delayed(modelo.evaluar_hora_b)(hora, train, eval_, ancla) for hora in horas)
    return modelo.combinar_por_hora(resultados, ancla, time.time() - inicio)


def agregar() -> dict:
    """Lee fold_1..5 + final → MAE_oos medio ± SE (folds) + métricas del test final por estructura."""
    resumen = {"ancla_mase_horaria_mw": ancla_mase(),
               "folds_val_years": list(ROLLING_VAL_YEARS), "estructuras": {}}
    for nombre in ESTRUCTURAS:
        maes, mases, folds_info = [], [], {}
        for n in range(1, 6):
            ruta_json = DIR_OUT / f"fold_{n}" / f"{nombre}.json"
            if not ruta_json.exists():
                continue
            contenido = json.loads(ruta_json.read_text())
            maes.append(contenido["metricas"]["MAE"]); mases.append(contenido["metricas"]["MASE"])
            folds_info[f"fold_{n}"] = {"val": contenido.get("eval_periodo"), **contenido["metricas"]}
        entrada = {"n_folds": len(maes), "folds": folds_info}
        if maes:
            entrada["MAE_oos_medio"] = float(np.mean(maes))
            entrada["MAE_oos_SE"] = (float(np.std(maes, ddof=1) / np.sqrt(len(maes)))
                                     if len(maes) > 1 else 0.0)
            entrada["MASE_oos_medio"] = float(np.mean(mases))
        ruta_final = DIR_OUT / "final" / f"{nombre}.json"
        if ruta_final.exists():
            contenido = json.loads(ruta_final.read_text())
            test = {"eval_periodo": contenido.get("eval_periodo")}
            # Los órdenes solo viven en la raíz de la estructura continua; en `por_hora` hay uno
            # por hora (dentro de la lista `por_hora`), así que ahí se omiten en lugar de
            # escribir null y aparentar que no hay órdenes.
            for clave in ("order", "seasonal_order"):
                if contenido.get(clave) is not None:
                    test[clave] = contenido[clave]
            test.update({"converged": contenido.get("converged"),
                         **contenido["metricas"],
                         "cobertura_95": contenido.get("cobertura_95")})
            entrada["test"] = test
        resumen["estructuras"][nombre] = entrada
    DIR_METRICAS.mkdir(parents=True, exist_ok=True)
    (DIR_METRICAS / "sarimax_resumen.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False))
    print(f"Guardado {DIR_METRICAS / 'sarimax_resumen.json'}", flush=True)
    return resumen


def main(argv=None):
    parser = argparse.ArgumentParser(description="Ajuste del SARIMAX por fold (D+1).")
    parser.add_argument("comando", help="1..5 | final | agregar")
    parser.add_argument("--force", action="store_true", help="reejecuta aunque exista el fichero")
    parser.add_argument("--solo", default=None,
                        help="subconjunto de estructuras (coma): continua,por_hora")
    parser.add_argument("--jobs", type=int, default=os.cpu_count(), help="procesos en paralelo para los 24 modelos por hora")
    parser.add_argument("--smoke", action="store_true",
                        help="ventana/maxiter mínimos (NO usar en real)")
    args = parser.parse_args(argv)

    if args.comando == "agregar":
        agregar()
        return
    if args.comando not in ("1", "2", "3", "4", "5", "final"):
        parser.error("comando debe ser 1..5, final o agregar")

    estructuras = (tuple(nombre.strip() for nombre in args.solo.split(","))
                   if args.solo else ESTRUCTURAS)
    train, eval_, etiqueta = spec_fold(args.comando)
    print(f"SARIMAX fold={args.comando} ({etiqueta}) · estructuras={estructuras} · "
          f"jobs={args.jobs} · smoke={args.smoke}", flush=True)
    for nombre in estructuras:
        ejecutar_estructura(nombre, args.comando, train, eval_, etiqueta,
                            args.force, args.jobs, args.smoke)


if __name__ == "__main__":
    main()
