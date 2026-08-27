"""Ajuste final del random forest

Uso:
    .venv/bin/python -m src.arboles.finales_rf
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.arboles.modelo import DIR_ART, DIR_MET, N_ARBOLES_CV, _ancla, _crear
from src.features.validacion import split_canonico_precovid
from src.redes.datos import construir_xy

SEMILLAS = (0, 1, 2, 3, 4)


def main() -> dict:
    ancla = _ancla()
    ranking = json.loads((DIR_MET / "tuning_rf.json").read_text())
    mejor = ranking["mejor"]
    hiperparametros = mejor["hiperparametros"]
    print(f"ganador de la búsqueda ({ranking['n_evaluadas']} configuraciones, "
          f"MAE_cv {mejor['mae_cv']} MW): {hiperparametros}", flush=True)

    datos, columnas_x, columnas_y = construir_xy(n_dias=1)
    split = split_canonico_precovid()
    dias_origen = datos.index
    ajuste = dias_origen.isin(split.train) | dias_origen.isin(split.val)
    test = dias_origen.isin(split.test)
    X, Y = datos[columnas_x].to_numpy(), datos[columnas_y].to_numpy()
    print(f"ajuste {ajuste.sum()} días (train+val) · test {test.sum()} días · {N_ARBOLES_CV} árboles",
          flush=True)

    predicciones = []
    for semilla in SEMILLAS:
        inicio = time.time()
        modelo = _crear("rf", hiperparametros, n_arboles=N_ARBOLES_CV, semilla=semilla)
        modelo.fit(X[ajuste], Y[ajuste])
        predicciones.append(modelo.predict(X[test]))
        mae = float(np.mean(np.abs(Y[test] - predicciones[-1])))
        print(f"  semilla {semilla}: MAE test {mae:.1f} MW · MASE {mae / ancla:.4f}  "
              f"({time.time() - inicio:.0f}s)", flush=True)

    ensemble, real = np.mean(predicciones, axis=0), Y[test]
    mae = float(np.mean(np.abs(real - ensemble)))
    filas = [{"fecha_origen": pd.Timestamp(fecha), "hora": hora, "y_real": float(real[i, hora]),
              "pred_ensemble": float(ensemble[i, hora]),
              **{f"pred_s{semilla}": float(predicciones[j][i, hora])
                 for j, semilla in enumerate(SEMILLAS)}}
             for i, fecha in enumerate(dias_origen[test]) for hora in range(real.shape[1])]
    ruta = DIR_ART / "rf" / "predicciones_test_rf.parquet"
    ruta.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(filas).to_parquet(ruta)

    salida = {"hiperparametros": hiperparametros, "n_arboles": N_ARBOLES_CV,
              "semillas": list(SEMILLAS), "ajuste": "train+val",
              "MAE": round(mae, 2), "MASE": round(mae / ancla, 4),
              "parquet": str(ruta.relative_to(RAIZ))}
    (DIR_MET / "finales_rf.json").write_text(json.dumps(salida, indent=2, ensure_ascii=False))
    print(f"\nENSEMBLE: MAE {mae:.1f} MW · MASE {mae / ancla:.4f}", flush=True)
    print(f"guardado {ruta.relative_to(RAIZ)}", flush=True)
    return salida


if __name__ == "__main__":
    main()
