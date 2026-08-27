"""Reestructura por hora (24 series diarias, una por hora del día) y objetivo day-ahead.

Ejecutar:  python -m src.features.por_hora
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.datos_precovid import serie_precovid


def series_por_hora(serie: pd.Series) -> pd.DataFrame:
    """24 series diarias: filas = día (fecha local), columnas = `h00…h23`, valor = demanda."""
    largo = pd.DataFrame({"fecha": serie.index.normalize(), "hora": serie.index.hour,
                          "y": serie.to_numpy()})
    tabla = largo.pivot(index="fecha", columns="hora", values="y")
    tabla.columns = [f"h{hora:02d}" for hora in tabla.columns]
    tabla.index.name = "fecha"
    return tabla


def objetivo_dayahead(serie: pd.Series, n_dias: int = 1) -> pd.DataFrame:
    """Objetivo day-ahead por hora: para cada día-origen D, las 24·n_dias horas de D+1…D+n_dias."""
    if n_dias < 1:
        raise ValueError("n_dias debe ser ≥ 1")
    por_hora = series_por_hora(serie)
    bloques = []
    for dia in range(1, n_dias + 1):
        futuro = por_hora.shift(-dia)  # la fila D toma los valores del día D+d
        futuro.columns = [f"y_d{dia}_{columna}" for columna in por_hora.columns]
        bloques.append(futuro)
    return pd.concat(bloques, axis=1)


def _origenes_validos(objetivo: pd.DataFrame) -> pd.DatetimeIndex:
    """Días-origen D con todo el objetivo disponible (sin NaN de futuro al final)."""
    return objetivo.index[objetivo.notna().all(axis=1)]


if __name__ == "__main__":
    serie = serie_precovid()  # la única fuente pre-COVID, ya recortada
    por_hora = series_por_hora(serie)
    print(f"series_por_hora: {por_hora.shape[0]} días x {por_hora.shape[1]} horas | "
          f"{por_hora.index.min().date()} -> {por_hora.index.max().date()} | "
          f"NaN en días completos={int(por_hora.iloc[:-1].isna().sum().sum())}")

    for n in (1, 2):
        objetivo = objetivo_dayahead(serie, n_dias=n)
        validos = _origenes_validos(objetivo)
        etiqueta = "day-ahead D+1" if n == 1 else "24-48 h (D+1 y D+2)"
        print(f"\nobjetivo_dayahead(n_dias={n}) [{etiqueta}]: {objetivo.shape[1]} salidas | "
              f"orígenes válidos={len(validos)} (de {len(objetivo)})")

    # Verificación anti-leakage y de correctitud
    objetivo1 = objetivo_dayahead(serie, n_dias=1)
    dia_origen = pd.Timestamp("2015-06-10")
    # y_d1_h13 en el origen D debe ser la demanda real de D+1 a las 13:00
    real = float(serie.loc[pd.Timestamp("2015-06-11 13:00:00")])
    objetivo_esperado = float(objetivo1.loc[dia_origen, "y_d1_h13"])
    print("\nChequeo objetivo: y_d1_h13(D=2015-06-10) == demanda(2015-06-11 13:00)?",
          abs(objetivo_esperado - real) < 1e-9, f"({objetivo_esperado:.1f} MW)")
    # Con n_dias=2, y_d2_h00(D) debe ser la demanda de D+2 a las 00:00
    objetivo2 = objetivo_dayahead(serie, n_dias=2)
    real2 = float(serie.loc[pd.Timestamp("2015-06-12 00:00:00")])
    print("Chequeo n_dias=2: y_d2_h00(D=2015-06-10) == demanda(2015-06-12 00:00)?",
          abs(float(objetivo2.loc[dia_origen, "y_d2_h00"]) - real2) < 1e-9)
    # El objetivo es futuro puro: reconstruir la serie desde series_por_hora conserva el total
    print("series_por_hora reconstruye la serie (suma):",
          abs(por_hora.to_numpy().sum() - float(serie.reindex(
              [dia + pd.Timedelta(hours=hora) for dia in por_hora.index for hora in range(24)]).sum())) < 1e-3)
