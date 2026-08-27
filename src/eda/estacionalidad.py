"""Tendencia y estacionalidad múltiple + descomposición STL."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import MSTL, STL

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src import plots
from src.datos_precovid import serie_precovid

MESES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
DIAS = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]


def serie_global(serie: pd.Series, guardar: bool = True):
    """Medias diaria y mensual sobre toda la serie."""
    media_diaria = serie.resample("D").mean()
    media_mensual = serie.resample("MS").mean()
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(media_diaria.index, media_diaria.values, lw=0.5, alpha=0.4, label="Media diaria")
    ax.plot(media_mensual.index, media_mensual.values, lw=1.8, label="Media mensual")
    ax.set_title("Demanda eléctrica peninsular — medias diaria y mensual (2000–2020, pre-COVID)")
    ax.set_xlabel("Año"); ax.set_ylabel("Demanda (MW)"); ax.legend()
    if guardar:
        plots.guardar_figura(fig, "eda/global/serie_global", formatos=("png", "pdf"))
    return fig


def media_anual(serie: pd.Series, guardar: bool = True):
    serie = serie[serie.index.year != 2020]
    medias_anuales = serie.resample("YE").mean()
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(medias_anuales.index.year, medias_anuales.values, marker="o")
    ax.set_title("Demanda media anual"); ax.set_xlabel("Año"); ax.set_ylabel("Demanda media (MW)")
    if guardar:
        plots.guardar_figura(fig, "eda/global/media_anual", formatos=("png", "pdf"))
    return fig


def perfil_mensual(serie: pd.Series, guardar: bool = True):
    datos = [serie[serie.index.month == mes].values for mes in range(1, 13)]
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.boxplot(datos, labels=MESES, showfliers=False)
    ax.set_title("Distribución de la demanda por mes")
    ax.set_xlabel("Mes"); ax.set_ylabel("Demanda (MW)")
    if guardar:
        plots.guardar_figura(fig, "eda/estacionalidad/perfil_mensual", formatos=("png",))
    return fig


def _estacion(index):
    estaciones = pd.Series(index=index, dtype=object)
    estaciones[index.month.isin([12, 1, 2])] = "Invierno"
    estaciones[index.month.isin([3, 4, 5])] = "Primavera"
    estaciones[index.month.isin([6, 7, 8])] = "Verano"
    estaciones[index.month.isin([9, 10, 11])] = "Otoño"
    return estaciones


def perfil_horario(serie: pd.Series, guardar: bool = True):
    estaciones = _estacion(serie.index)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for etiqueta, mascara in [("Laborable", serie.index.dayofweek < 5), ("Fin de semana", serie.index.dayofweek >= 5)]:
        perfil = serie[mascara].groupby(serie[mascara].index.hour).mean()
        axes[0].plot(perfil.index, perfil.values, marker="o", ms=3, label=etiqueta)
    axes[0].set_title("Perfil horario medio: laborable vs fin de semana")
    axes[0].set_xlabel("Hora del día"); axes[0].set_ylabel("Demanda (MW)"); axes[0].legend()
    for estacion in ["Invierno", "Primavera", "Verano", "Otoño"]:
        mascara = estaciones == estacion
        perfil = serie[mascara].groupby(serie[mascara].index.hour).mean()
        axes[1].plot(perfil.index, perfil.values, marker="o", ms=3, label=estacion)
    axes[1].set_title("Perfil horario medio por estación")
    axes[1].set_xlabel("Hora del día"); axes[1].set_ylabel("Demanda (MW)"); axes[1].legend()
    if guardar:
        plots.guardar_figura(fig, "eda/estacionalidad/perfil_horario", formatos=("png",))
    return fig


def heatmaps(serie: pd.Series, guardar: bool = True):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    pivote_hora_mes = serie.groupby([serie.index.hour, serie.index.month]).mean().unstack()
    im0 = axes[0].imshow(pivote_hora_mes.values, aspect="auto", origin="lower", cmap="viridis")
    axes[0].set_title("Demanda media: hora x mes")
    axes[0].set_xlabel("Mes"); axes[0].set_ylabel("Hora del día")
    axes[0].set_xticks(range(12)); axes[0].set_xticklabels(MESES)
    fig.colorbar(im0, ax=axes[0], label="MW")
    pivote_hora_dia = serie.groupby([serie.index.hour, serie.index.dayofweek]).mean().unstack()
    im1 = axes[1].imshow(pivote_hora_dia.values, aspect="auto", origin="lower", cmap="viridis")
    axes[1].set_title("Demanda media: hora x día de la semana")
    axes[1].set_xlabel("Día"); axes[1].set_ylabel("Hora del día")
    axes[1].set_xticks(range(7)); axes[1].set_xticklabels(DIAS)
    fig.colorbar(im1, ax=axes[1], label="MW")
    if guardar:
        plots.guardar_figura(fig, "eda/estacionalidad/heatmaps", formatos=("png",))
    return fig


def stl_diaria(serie: pd.Series):
    """Descomposición STL de la media diaria: periodo semanal y ajuste robusto (los atípicos
    no arrastran la tendencia). Devuelve el resultado STL."""
    return STL(serie.resample("D").mean(), period=7, robust=True).fit()


def descomposicion_mstl(serie: pd.Series, guardar: bool = True):
    """Descomposición MSTL de la serie horaria con doble estacionalidad (24 h y 168 h)."""
    mstl = MSTL(serie.astype(float), periods=(24, 168)).fit()
    estacionales = mstl.seasonal  # una columna por periodo: "seasonal_24" y "seasonal_168"
    fig, axes = plt.subplots(5, 1, figsize=(13, 11), sharex=True)
    axes[0].plot(serie.index, serie.values, lw=0.2); axes[0].set_ylabel("Observada")
    axes[1].plot(mstl.trend.index, mstl.trend.values, lw=0.8); axes[1].set_ylabel("Tendencia")
    axes[2].plot(estacionales.index, estacionales.iloc[:, 0].values, lw=0.2); axes[2].set_ylabel("Estacional (24 h)")
    axes[3].plot(estacionales.index, estacionales.iloc[:, 1].values, lw=0.2); axes[3].set_ylabel("Estacional (168 h)")
    axes[4].plot(mstl.resid.index, mstl.resid.values, lw=0.2); axes[4].set_ylabel("Residuo")
    axes[0].set_title("Descomposición MSTL de la serie horaria (estacionalidad 24 h + 168 h)")
    axes[4].set_xlabel("Año")
    if guardar:
        plots.guardar_figura(fig, "eda/descomposicion/mstl", formatos=("png",))
    return mstl, fig


def reparto_varianza(descomposicion) -> dict:
    """Fracción de varianza que se lleva el residuo frente a la serie observada."""
    residuo = descomposicion.resid.dropna()
    observada = descomposicion.observed.reindex(residuo.index)
    resid_pct = 100 * float(residuo.var() / observada.var())
    return {"residuo_pct": round(resid_pct, 1), "explicado_pct": round(100 - resid_pct, 1)}


if __name__ == "__main__":
    plots.usar_estilo("academico")
    serie = serie_precovid()
    serie_global(serie); media_anual(serie)
    perfil_mensual(serie); perfil_horario(serie); heatmaps(serie)
    stl_diaria(serie)
    descomposicion_mstl(serie)
    print("Estacionalidad: figuras generadas en figuras/eda/")
