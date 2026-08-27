"""Festivos nacionales de España para el EDA y las features (con complemento manual para 2000-2007)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from dateutil.easter import easter

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import holidays as holidays_lib

# Festivos nacionales fijos (no movibles)
_FIJOS = [(1, 1), (1, 6), (5, 1), (8, 15), (10, 12), (11, 1), (12, 6), (12, 8), (12, 25)]


def fechas_festivas(anio_min: int = 2000, anio_max: int = 2024) -> pd.DatetimeIndex:
    """Fechas de festivos nacionales en [anio_min, anio_max], con los fijos 2000-2007 y el Viernes Santo."""
    festivos_es = holidays_lib.Spain(years=range(anio_min, anio_max + 1))
    fechas = set(pd.to_datetime(sorted(festivos_es.keys())))
    for anio in range(max(anio_min, 2000), min(anio_max, 2007) + 1):
        for mes, dia in _FIJOS:
            fechas.add(pd.Timestamp(anio, mes, dia))
    # Viernes Santo (nacional) para todos los años; Jueves Santo va aparte (regional)
    fechas.update(fechas_semana_santa(anio_min, anio_max))
    return pd.DatetimeIndex(sorted(fechas))


def es_festivo(index: pd.DatetimeIndex) -> pd.Series:
    """Booleana por día; solo nacionales (los regionales se tratan aparte)."""
    fechas = fechas_festivas(index.year.min(), index.year.max())
    return pd.Series(index.normalize().isin(fechas), index=index)


def fechas_semana_santa(anio_min: int = 2000, anio_max: int = 2024) -> pd.DatetimeIndex:
    """Fechas de Semana Santa por año: el Viernes Santo, que es el único festivo nacional
    (el Jueves Santo es autonómico y entra por la vía de los festivos regionales)."""
    fechas = set()
    for anio in range(anio_min, anio_max + 1):
        domingo = pd.Timestamp(easter(anio))
        fechas.add(domingo - pd.Timedelta(days=2))  # Viernes Santo (nacional)
    return pd.DatetimeIndex(sorted(fechas))


if __name__ == "__main__":
    fechas_nacionales = fechas_festivas()
    print(f"Total días festivos 2000-2024: {len(fechas_nacionales)}")
