"""Construye y persiste la serie horaria de demanda peninsular limpia y recortada a < 2020-03-01.

Ejecutar:  python -m src.datos_precovid
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.eda.carga import serie_limpia

CORTE_PRECOVID = "2020-03-01"
RUTA_PRECOVID = RAIZ / "Peninsula_precovid.csv"


def construir_precovid(guardar: bool = True) -> pd.Series:
    serie = serie_limpia()
    serie = serie[serie.index < CORTE_PRECOVID].copy()
    serie.name = "Demanda"
    #verificación de continuidad horaria, huecos, duplicados y NaN
    indice = serie.index
    saltos = (indice.to_series().diff().dropna() != pd.Timedelta(hours=1)).sum()
    duplicados = int(indice.duplicated().sum())
    valores_nan = int(serie.isna().sum())
    assert saltos == 0, f"La serie pre-COVID tiene {saltos} saltos != 1h"
    assert duplicados == 0, f"La serie pre-COVID tiene {duplicados} duplicados"
    assert valores_nan == 0, f"La serie pre-COVID tiene {valores_nan} NaN"
    if guardar:
        serie.to_frame().to_csv(RUTA_PRECOVID, index_label="datetime")
    return serie


def serie_precovid() -> pd.Series:
    """Carga la serie pre-COVID canónica (la materializa la primera vez si no existe)."""
    if RUTA_PRECOVID.exists():
        df = pd.read_csv(RUTA_PRECOVID, parse_dates=["datetime"]).set_index("datetime")
        return df["Demanda"].astype(float)
    return construir_precovid()


if __name__ == "__main__":
    serie = construir_precovid()
    print(f"Datos pre-COVID en {RUTA_PRECOVID.name}")
    print(f"  {len(serie):,} horas, de {serie.index.min()} a {serie.index.max()}")
    print(f"  días completos: {serie.index.normalize().nunique():,} | NaN: {int(serie.isna().sum())} | "
          f"saltos!=1h: 0 | duplicados: 0  (verificado)")
    print(f"  demanda: min {serie.min():,.0f} MW, media {serie.mean():,.0f} MW, max {serie.max():,.0f} MW")
