"""Autocorrelación de la demanda: ACF/PACF y correlaciones con los retardos candidatos."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src import plots
from src.datos_precovid import serie_precovid

LAGS_INTERES = (1, 24, 48, 168, 336, 672)


def acf_pacf(serie: pd.Series, guardar: bool = True):
    fig, axes = plt.subplots(2, 1, figsize=(13, 7))
    plot_acf(serie, lags=180, ax=axes[0])
    axes[0].set_title("ACF de la demanda horaria (180 retardos ≈ 7,5 días; marcas en 24 h y 168 h)")
    axes[0].axvline(24, color="red", ls="--", lw=0.8); axes[0].axvline(168, color="green", ls="--", lw=0.8)
    plot_pacf(serie, lags=72, ax=axes[1], method="ywm")
    axes[1].set_title("PACF de la demanda horaria (72 retardos)")
    axes[1].axvline(24, color="red", ls="--", lw=0.8)
    if guardar:
        plots.guardar_figura(fig, "eda/autocorrelacion/acf_pacf", formatos=("png",))
    return fig


def correlaciones_lag(serie: pd.Series) -> dict:
    return {lag: float(serie.corr(serie.shift(lag))) for lag in LAGS_INTERES}


if __name__ == "__main__":
    plots.usar_estilo("academico")
    serie = serie_precovid()
    acf_pacf(serie)
    for lag, correlacion in correlaciones_lag(serie).items():
        print(f"Correlación lag {lag:>4}h: {correlacion:.4f}")
