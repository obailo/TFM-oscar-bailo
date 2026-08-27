"""Carga y limpieza de la serie del EDA: imputa los dos artefactos del cambio de hora (DST)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.datos import cargar_demanda


def picos_dst_otono(demanda: pd.Series) -> pd.Series:
    """Máscara booleana de los picos artificiales del cambio de hora de otoño."""
    indice = demanda.index
    candidatas = ((indice.month == 10) & (indice.dayofweek == 6) & (indice.day >= 25)
                  & (indice.hour.isin([1, 2])))
    vecinos = (demanda.shift(1) + demanda.shift(-1)) / 2
    return pd.Series(candidatas, index=indice) & (demanda > 1.3 * vecinos)


def serie_limpia() -> pd.Series:
    serie = cargar_demanda()["Demanda"].astype(float)

    # 1) Primavera: ceros artificiales -> NaN
    serie = serie.replace(0, np.nan)

    # 2) Otoño: pico (~doble) en el último domingo de octubre
    serie[picos_dst_otono(serie)] = np.nan

    return serie.interpolate(method="time")


if __name__ == "__main__":
    serie = serie_limpia()
    print(f"Serie limpia: {len(serie)} h, {serie.index.min()} -> {serie.index.max()}")
    print(f"Mínimo tras limpieza: {serie.min():,.0f} MW")
