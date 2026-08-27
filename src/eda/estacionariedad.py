"""Estacionariedad (ADF/KPSS) y volatilidad móvil"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src import plots
from src.datos_precovid import serie_precovid


def _adf_kpss(serie: pd.Series, maxlag: int | None = None) -> dict:
    """ADF (H0: raíz unitaria) y KPSS (H0: estacionaria) sobre una serie ya preparada."""
    serie = serie.dropna()
    resultado_adf = adfuller(serie, maxlag=maxlag, autolag=None if maxlag else "AIC")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # el KPSS satura el p-valor en los extremos y avisa con InterpolationWarning
        resultado_kpss = kpss(serie, regression="c", nlags="auto")
    return {"adf_stat": resultado_adf[0], "adf_p": resultado_adf[1],
            "kpss_stat": resultado_kpss[0], "kpss_p": resultado_kpss[1]}


def tests_estacionariedad(serie: pd.Series) -> dict:
    return _adf_kpss(serie.resample("D").mean())


def tests_estacionariedad_horaria(serie: pd.Series) -> dict:
    """ADF/KPSS sobre la serie horaria en nivel, Δ₂₄, Δ₁₆₈ y Δ₂₄∘Δ₁₆₈."""
    serie_horaria = serie.astype(float)
    transformaciones = {
        "nivel": serie_horaria,
        "d24": serie_horaria.diff(24),
        "d168": serie_horaria.diff(168),
        "d24_d168": serie_horaria.diff(24).diff(168),
    }
    return {nombre: _adf_kpss(transformada, maxlag=48) for nombre, transformada in transformaciones.items()}


def volatilidad(serie: pd.Series, guardar: bool = True):
    media_diaria = serie.resample("D").mean()
    media_movil = media_diaria.rolling(30).mean()
    desv_movil = media_diaria.rolling(30).std()
    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    axes[0].plot(media_diaria.index, media_diaria.values, lw=0.4, alpha=0.4)
    axes[0].plot(media_movil.index, media_movil.values, lw=1.2)
    axes[0].set_ylabel("Media móvil 30d (MW)")
    axes[0].set_title("Media y desviación típica móviles (ventana de 30 días)")
    axes[1].plot(desv_movil.index, desv_movil.values, lw=1.0, color="#b5651d")
    axes[1].set_ylabel("Desv. típica móvil 30d (MW)"); axes[1].set_xlabel("Año")
    if guardar:
        plots.guardar_figura(fig, "eda/estacionariedad/volatilidad", formatos=("png",))
    return fig


if __name__ == "__main__":
    plots.usar_estilo("academico")
    serie = serie_precovid()
    print("Tests (media diaria):", {k: round(v, 4) for k, v in tests_estacionariedad(serie).items()})
    print("Tests (serie horaria):")
    for nombre, dic in tests_estacionariedad_horaria(serie).items():
        print(f"  {nombre:10}", {k: round(v, 4) for k, v in dic.items()})
    volatilidad(serie)
