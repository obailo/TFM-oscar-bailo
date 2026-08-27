"""Carga la serie horaria cruda de demanda peninsular (Peninsula.csv)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# Raíz del proyecto para resolver rutas con independencia del directorio de trabajo desde el que se ejecute.
RAIZ = Path(__file__).resolve().parent.parent
RUTA_CSV = RAIZ / "Peninsula.csv"


def cargar_demanda() -> pd.DataFrame:
    df = pd.read_csv(RUTA_CSV, parse_dates=["datetime"])
    return df.set_index("datetime").sort_index()
