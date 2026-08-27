from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src import metricas as M


def denominador_mase(demanda: pd.Series, train_idx: pd.DatetimeIndex,
                     periodos=(24, 168)) -> dict:
    y = demanda.reindex(train_idx).to_numpy()
    escalas = {m: M.escala_naive(y, m) for m in periodos}
    m_elegido = min(escalas, key=escalas.get)
    return {"escalas": escalas, "m_elegido": m_elegido, "escala": escalas[m_elegido]}


def _origenes_validos(test_idx: pd.DatetimeIndex, demanda: pd.Series, h_max: int):
    """Orígenes t del test para los que y(t+h_max) existe en la serie (mismos t para todo h)."""
    limite = demanda.index[-1] - pd.Timedelta(hours=h_max)
    return test_idx[test_idx <= limite]


def metricas_por_horizonte(preds: pd.DataFrame, demanda: pd.Series,
                           escala_mase: float, horizontes) -> pd.DataFrame:
    """MAE/RMSE/MAPE/MASE por horizonte a partir de las predicciones ŷ(t+h), con el denominador de MASE fijo."""
    origenes = preds.index
    filas = []
    for h in horizontes:
        y_true = demanda.reindex(origenes + pd.Timedelta(hours=h)).to_numpy()
        y_pred = preds[h].to_numpy()
        validos = ~(np.isnan(y_true) | np.isnan(y_pred))
        reales, predichos = y_true[validos], y_pred[validos]
        filas.append({
            "h": h, "n": int(validos.sum()),
            "mae": M.mae(reales, predichos), "rmse": M.rmse(reales, predichos),
            "mape": M.mape(reales, predichos), "mase": M.mase_con_escala(reales, predichos, escala_mase),
        })
    return pd.DataFrame(filas).set_index("h")


def evaluar(modelo, demanda: pd.Series, split, escala_mase: float, horizontes) -> dict:
    modelo.fit(demanda, split.train)
    h_max = max(horizontes)
    origenes = _origenes_validos(split.test, demanda, h_max)
    preds = modelo.predict(origenes, list(horizontes), demanda)
    tabla_metricas = metricas_por_horizonte(preds, demanda, escala_mase, horizontes)
    resumen = {
        "mae_medio": float(tabla_metricas["mae"].mean()),
        "mape_medio": float(tabla_metricas["mape"].mean()),
        "mase_medio": float(tabla_metricas["mase"].mean()),
        "n_origenes": int(len(origenes)),
    }
    if 24 in tabla_metricas.index:
        resumen["mae_h24"] = float(tabla_metricas.loc[24, "mae"])
        resumen["mape_h24"] = float(tabla_metricas.loc[24, "mape"])
    return {"modelo": modelo.nombre, "metricas": tabla_metricas,
            "predicciones": preds, "resumen": resumen}


if __name__ == "__main__":
    from src.datos_precovid import serie_precovid
    from src.features.validacion import split_por_fechas, VAL_INI, TEST_INI
    from src.baselines.modelos import NaiveEstacional

    serie = serie_precovid()
    split = split_por_fechas(serie.index, VAL_INI, TEST_INI, gap=24)  # canónico, horario, gap de 24 h (day-ahead)
    denominador = denominador_mase(serie, split.train)
    print("Escalas MASE (train):", {m: round(v, 1) for m, v in denominador["escalas"].items()})
    print("Naive elegido como ancla:", f"m={denominador['m_elegido']} (escala={denominador['escala']:.1f} MW)")
    resultado = evaluar(NaiveEstacional(168), serie, split, denominador["escala"], horizontes=range(1, 25))
    print("\nNaive semanal, resumen:", {k: round(v, 3) for k, v in resultado["resumen"].items()})
