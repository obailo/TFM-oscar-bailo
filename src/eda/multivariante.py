"""Detección multivariante de días atípicos por distancia de Mahalanobis (umbral χ²) e Isolation Forest."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.ensemble import IsolationForest

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src import plots
from src.datos_precovid import serie_precovid


def features_diarias(serie: pd.Series) -> pd.DataFrame:
    """Resumen por día: media, desviación, mínimo y máximo de la demanda."""
    por_dia = serie.groupby(serie.index.normalize())
    return pd.DataFrame({"media": por_dia.mean(), "std": por_dia.std(),
                         "min": por_dia.min(), "max": por_dia.max()}).dropna()


def deteccion_atipicos(serie: pd.Series, guardar: bool = True):
    resumen_diario = features_diarias(serie)
    X = resumen_diario.values
    media = X.mean(axis=0)
    inv_cov = np.linalg.inv(np.cov(X.T))
    desviaciones = X - media
    dist_mahalanobis2 = np.einsum("ij,jk,ik->i", desviaciones, inv_cov, desviaciones)
    resumen_diario["mahalanobis2"] = dist_mahalanobis2
    umbral = chi2.ppf(0.999, df=X.shape[1])

    iso = IsolationForest(contamination=0.01, random_state=42).fit(X)
    resumen_diario["iso_anomalia"] = iso.predict(X) == -1

    atipicos = resumen_diario[resumen_diario["mahalanobis2"] > umbral]
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(resumen_diario.index, resumen_diario["mahalanobis2"], lw=0.5)
    ax.axhline(umbral, color="red", ls="--", lw=1.0, label="Umbral χ²(0,999)")
    ax.scatter(atipicos.index, atipicos["mahalanobis2"], color="red", s=12, zorder=3, label="Día atípico")
    ax.set_title("Distancia de Mahalanobis de cada día (resumen [media, std, mín, máx])")
    ax.set_xlabel("Año"); ax.set_ylabel("Mahalanobis²"); ax.legend()
    if guardar:
        plots.guardar_figura(fig, "eda/atipicos/mahalanobis", formatos=("png",))

    top = resumen_diario.sort_values("mahalanobis2", ascending=False).head(15)
    return fig, top, {"umbral_chi2": float(umbral), "n_atipicos_maha": int(len(atipicos)),
                      "n_atipicos_iso": int(resumen_diario["iso_anomalia"].sum())}


if __name__ == "__main__":
    plots.usar_estilo("academico")
    serie = serie_precovid()
    _, top, info = deteccion_atipicos(serie)
    print(info)
    print("\nTop 15 días atípicos (Mahalanobis):")
    print(top[["media", "std", "min", "max", "mahalanobis2"]].round(0))
