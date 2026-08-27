"""Ajuste final de las redes: MLP y LSTM ganadores con 5 semillas; persiste las predicciones de val y test.

Ejecutar:  python -m src.redes.finales mlp | lstm | ambos
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

from src.metricas import ancla_mase
from src.redes import tuning as T
from src.redes.datos import construir_xy, split_y_escalar
from src.redes.entrenamiento import entrenar_mlp

DIR_MET = RAIZ / "resultados" / "metricas"
DIR_ART = RAIZ / "resultados" / "redes"
SEMILLAS = (0, 1, 2, 3, 4)


def ganador(familia: str) -> dict:
    """Configuración ganadora de la búsqueda, leída del ranking."""
    ruta = DIR_MET / f"tuning_{familia}.json"
    ranking = sorted(json.loads(ruta.read_text()), key=lambda registro: registro["mae_oos_medio"])
    print(f"[{familia}] ganador: MAE_oos {ranking[0]['mae_oos_medio']:.1f} MW, {ranking[0]['config']}",
          flush=True)
    return ranking[0]["config"]


def _persistir(familia: str, etiqueta: str, fechas, real, predicciones: list[np.ndarray],
               ancla: float) -> dict:
    ensemble = np.mean(predicciones, axis=0)
    filas = [{"fecha_origen": pd.Timestamp(fecha), "hora": hora, "y_real": float(real[i, hora]),
              "pred_ensemble": float(ensemble[i, hora]),
              **{f"pred_s{semilla}": float(predicciones[j][i, hora])
                 for j, semilla in enumerate(SEMILLAS)}}
             for i, fecha in enumerate(fechas) for hora in range(real.shape[1])]
    DIR_ART.mkdir(parents=True, exist_ok=True)
    ruta = DIR_ART / f"predicciones_{etiqueta}_{familia}.parquet"
    pd.DataFrame(filas).to_parquet(ruta)
    mae = float(np.mean(np.abs(real - ensemble)))
    print(f"  {etiqueta}: MASE ensemble {mae / ancla:.4f}, guardado en {ruta.name}", flush=True)
    return {"MAE": round(mae, 2), "MASE": round(mae / ancla, 4),
            "parquet": str(ruta.relative_to(RAIZ))}


def finales_mlp(config: dict, ancla: float) -> dict:
    """Ajuste final del MLP con 5 semillas, con parada temprana sobre la validación."""
    import torch
    datos, columnas_x, columnas_y = construir_xy(n_dias=1)
    escalado = split_y_escalar(datos, columnas_x, columnas_y, n_dias=1)
    n_salidas = len(columnas_y)

    dias_origen = datos.index
    mascara_val, mascara_test = escalado["mascaras"]["va"], escalado["mascaras"]["te"]
    salida = {"familia": "mlp", "config": config, "semillas": list(SEMILLAS)}
    X = {clave: torch.tensor(escalado["X"][mascara].to_numpy(), dtype=torch.float32)
         for clave, mascara in (("va", mascara_val), ("te", mascara_test))}
    real = {clave: escalado["Y"][mascara].to_numpy() * escalado["y_desv"] + escalado["y_media"]
            for clave, mascara in (("va", mascara_val), ("te", mascara_test))}
    predicciones = {"va": [], "te": []}
    curvas = []
    for semilla in SEMILLAS:
        inicio = time.time()
        red, historia = entrenar_mlp(escalado, n_salidas=n_salidas, semilla=semilla,
                                     hidden=tuple(config["hidden"]), dropout=config["dropout"],
                                     lr=config["lr"], weight_decay=config["weight_decay"],
                                     batch=config["batch"],
                                     activacion=config["activacion"])
        curvas.append(historia)
        dispositivo = next(red.parameters()).device
        red.eval()
        with torch.no_grad():
            for clave in ("va", "te"):
                predicciones[clave].append(
                    red(X[clave].to(dispositivo)).cpu().numpy() * escalado["y_desv"] + escalado["y_media"])
        mae = float(np.mean(np.abs(real["te"] - predicciones["te"][-1])))
        print(f"    semilla {semilla}: MAE test {mae:.1f} MW, MASE {mae / ancla:.4f} "
              f"({time.time() - inicio:.0f} s)", flush=True)
    salida["curvas"] = curvas  # historia época a época, para las curvas de entrenamiento
    for clave, etiqueta, mascara in (("va", "val", mascara_val), ("te", "test", mascara_test)):
        salida[etiqueta] = _persistir("mlp", etiqueta, dias_origen[mascara], real[clave],
                                      predicciones[clave], ancla)
    return salida


def finales_lstm(config: dict, ancla: float) -> dict:
    """Ajuste final de la LSTM con 5 semillas, con parada temprana sobre la validación."""
    import torch
    Xseq, Xexo, Y, fechas, mascaras = T._prep_lstm(config["L"])
    train, val, test = mascaras["train"], mascaras["val"], mascaras["test"]
    X_esc, exo_esc, Y_esc, y_media, y_desv = T._escalar_fold_lstm(Xseq, Xexo, Y, train)
    n_salidas = Y.shape[1]

    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    salida = {"familia": "lstm", "config": config, "semillas": list(SEMILLAS)}
    real = {"va": Y[val], "te": Y[test]}
    predicciones = {"va": [], "te": []}
    curvas = []
    for semilla in SEMILLAS:
        inicio = time.time()
        modelo, historia = T._entrenar_lstm_exo(X_esc, exo_esc, Y_esc, train, val, config,
                                                n_salidas, semilla=semilla)
        curvas.append(historia)
        modelo.eval()
        with torch.no_grad():
            for clave, mascara in (("va", val), ("te", test)):
                predicciones[clave].append(
                    modelo(torch.tensor(X_esc[mascara], dtype=torch.float32).to(dispositivo),
                           torch.tensor(exo_esc[mascara], dtype=torch.float32).to(dispositivo)
                           ).cpu().numpy() * y_desv + y_media)
        mae = float(np.mean(np.abs(real["te"] - predicciones["te"][-1])))
        print(f"    semilla {semilla}: MAE test {mae:.1f} MW, MASE {mae / ancla:.4f} "
              f"({time.time() - inicio:.0f} s)", flush=True)
    salida["curvas"] = curvas  # historia época a época, para las curvas de entrenamiento
    for clave, etiqueta, mascara in (("va", "val", val), ("te", "test", test)):
        salida[etiqueta] = _persistir("lstm", etiqueta, fechas[mascara], real[clave],
                                      predicciones[clave], ancla)
    return salida


def main(argv: list[str] | None = None):
    args = list(argv if argv is not None else sys.argv[1:])
    familias = ("mlp", "lstm") if not args or args[0] == "ambos" else (args[0],)
    ancla = ancla_mase()
    salida = {}
    for familia in familias:
        print(f"finales {familia.upper()}", flush=True)
        config = ganador(familia)
        salida[familia] = (finales_mlp(config, ancla) if familia == "mlp"
                           else finales_lstm(config, ancla))
        ruta = DIR_MET / f"finales_{familia}.json"
        ruta.write_text(json.dumps(salida[familia], indent=2, ensure_ascii=False))
        print(f"  guardado en {ruta}", flush=True)
    return salida


if __name__ == "__main__":
    main()
