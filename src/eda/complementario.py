"""Análisis complementario: ruido, covarianza con calendario y PCA (EDA, sección 12)."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src import plots
from src.datos_precovid import serie_precovid
from src.eda.estacionalidad import stl_diaria
from src.eda.festivos import es_festivo


def ruido(serie: pd.Series, guardar: bool = True, stl_result=None):
    """Analiza el residuo de la STL diaria: histograma, Q-Q, ACF y test de Ljung-Box."""
    if stl_result is None:
        stl_result = stl_diaria(serie)
    residuo = stl_result.resid.dropna()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].hist(residuo.values, bins=60)
    axes[0].set_title("Histograma del residuo (STL diario)")
    axes[0].set_xlabel("Residuo (MW)"); axes[0].set_ylabel("Frecuencia")
    qqplot(residuo.values, line="s", ax=axes[1])
    axes[1].set_title("Gráfico Q-Q del residuo (vs normal)")
    plot_acf(residuo, lags=40, ax=axes[2])
    axes[2].set_title("ACF del residuo")
    if guardar:
        plots.guardar_figura(fig, "eda/ruido/residuo", formatos=("png",))
    ljung_box = acorr_ljungbox(residuo, lags=[7, 14, 30], return_df=True)
    return fig, ljung_box


def covarianza_calendario(serie: pd.Series, guardar: bool = True):
    """Matriz de correlación (Spearman) entre demanda y variables de calendario."""
    festivos = es_festivo(serie.index)
    caracteristicas = pd.DataFrame(index=serie.index)
    caracteristicas["demanda"] = serie.values
    caracteristicas["hora"] = serie.index.hour
    caracteristicas["dia_semana"] = serie.index.dayofweek
    caracteristicas["mes"] = serie.index.month
    caracteristicas["dia_anio"] = serie.index.dayofyear
    caracteristicas["finde"] = (serie.index.dayofweek >= 5).astype(int)
    caracteristicas["festivo"] = festivos.values.astype(int)
    caracteristicas["sin_hora"] = np.sin(2 * np.pi * serie.index.hour / 24)
    caracteristicas["cos_hora"] = np.cos(2 * np.pi * serie.index.hour / 24)
    caracteristicas["sin_anio"] = np.sin(2 * np.pi * serie.index.dayofyear / 365.25)
    caracteristicas["cos_anio"] = np.cos(2 * np.pi * serie.index.dayofyear / 365.25)
    corr = caracteristicas.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title("Correlación de Spearman: demanda y variables de calendario")
    fig.colorbar(im, ax=ax, label="ρ")
    if guardar:
        plots.guardar_figura(fig, "eda/covarianza/correlacion_calendario", formatos=("png",))
    return fig, corr["demanda"].drop("demanda").sort_values(key=np.abs, ascending=False)


def pca_perfiles(serie: pd.Series, guardar: bool = True):
    """PCA de los perfiles diarios (matriz días x 24 h normalizada por su media)."""
    pivote = serie.groupby([serie.index.normalize(), serie.index.hour]).mean().unstack().dropna()
    perfiles = pivote.div(pivote.mean(axis=1), axis=0)
    pca = PCA(n_components=5).fit(perfiles.values)
    componentes = pca.transform(perfiles.values)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].plot(range(1, 6), pca.explained_variance_ratio_ * 100, marker="o")
    axes[0].set_title("Varianza explicada (scree)"); axes[0].set_xlabel("Componente"); axes[0].set_ylabel("%")
    for i in range(3):
        axes[1].plot(range(24), pca.components_[i], marker="o", ms=3, label=f"PC{i+1}")
    axes[1].set_title("Forma de las 3 primeras componentes"); axes[1].set_xlabel("Hora del día"); axes[1].legend()
    mes_dia = pd.DatetimeIndex(perfiles.index).month
    color_estacion = np.select([np.isin(mes_dia, [12, 1, 2]), np.isin(mes_dia, [3, 4, 5]),
                                np.isin(mes_dia, [6, 7, 8])], [0, 1, 2], 3)
    axes[2].scatter(componentes[:, 0], componentes[:, 1], c=color_estacion, cmap="viridis", s=5, alpha=0.4)
    axes[2].set_title("Días en el plano PC1–PC2 (color = estación)")
    axes[2].set_xlabel("PC1"); axes[2].set_ylabel("PC2")
    if guardar:
        plots.guardar_figura(fig, "eda/pca/pca_perfiles", formatos=("png",))
    return fig, pca.explained_variance_ratio_


if __name__ == "__main__":
    plots.usar_estilo("academico")
    serie = serie_precovid()
    _, ljung_box = ruido(serie)
    print("Ljung-Box:\n", ljung_box)
    _, corr = covarianza_calendario(serie)
    print("Correlación con demanda:\n", corr.round(3))
    _, varianza = pca_perfiles(serie)
    print("PCA varianza explicada (%):", (varianza * 100).round(1))
