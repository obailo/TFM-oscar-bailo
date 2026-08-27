"""Efecto calendario/festivos y estacionalidad de verano/Navidad (EDA)."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src import plots
from src.datos_precovid import serie_precovid
from src.eda.festivos import es_festivo


def _mascaras(serie: pd.Series):
    festivos = es_festivo(serie.index)
    dia_semana = serie.index.dayofweek
    return {
        "Laborable\nnormal": (dia_semana < 5) & (~festivos.values),
        "Festivo\n(entre semana)": (dia_semana < 5) & (festivos.values),
        "Sábado": (dia_semana == 5),
        "Domingo": (dia_semana == 6),
    }


def festivos_total(serie: pd.Series, guardar: bool = True):
    categorias = _mascaras(serie)
    series = [serie[mascara] for mascara in categorias.values()]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.boxplot([grupo.values for grupo in series], labels=list(categorias.keys()), showfliers=False)
    ax.set_title("Demanda: laborable vs festivo nacional vs sábado vs domingo (2000–2020, pre-COVID)")
    ax.set_ylabel("Demanda (MW)")
    if guardar:
        plots.guardar_figura(fig, "eda/festivos/festivos_total", formatos=("png",))
    medias = {k: float(grupo.mean()) for k, grupo in zip(categorias, series)}
    return fig, medias


def verano_navidad(serie: pd.Series, guardar: bool = True):
    """Ciclo anual medio (día del año) resaltando el parón de agosto y la campaña de Navidad."""
    media_diaria = serie.resample("D").mean()
    clave = media_diaria.index.month * 100 + media_diaria.index.day  # MMDD
    ciclo = media_diaria.groupby(clave).mean()
    fechas_eje = pd.to_datetime([f"2000-{k // 100:02d}-{k % 100:02d}" for k in ciclo.index])

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(fechas_eje, ciclo.values, lw=1.4, color="#1f4e79", label="Ciclo anual medio (pre-COVID)")
    ax.axvspan(pd.Timestamp("2000-08-01"), pd.Timestamp("2000-08-31"),
               color="#e69f00", alpha=0.18, label="Agosto (parón estival)")
    ax.axvspan(pd.Timestamp("2000-12-24"), pd.Timestamp("2000-12-31"),
               color="#009e73", alpha=0.18, label="Navidad (24 dic - 6 ene)")
    ax.axvspan(pd.Timestamp("2000-01-01"), pd.Timestamp("2000-01-06"),
               color="#009e73", alpha=0.18)
    ax.set_title("Ciclo anual medio de la demanda: parón estival y Navidad (2000–2020, pre-COVID)")
    ax.set_xlabel("Mes"); ax.set_ylabel("Demanda media diaria (MW)"); ax.legend()
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    if guardar:
        plots.guardar_figura(fig, "eda/calendario/verano_navidad", formatos=("png", "pdf"))

    mes = media_diaria.index.month
    dia = media_diaria.index.day
    base = float(media_diaria.mean())
    media_agosto = float(media_diaria[mes == 8].mean())
    es_navidad = ((mes == 12) & (dia >= 24)) | ((mes == 1) & (dia <= 6))
    media_navidad = float(media_diaria[es_navidad].mean())
    return fig, {
        "media_global": base,
        "media_agosto": media_agosto, "desc_agosto_pct": 100 * (base - media_agosto) / base,
        "media_navidad": media_navidad, "desc_navidad_pct": 100 * (base - media_navidad) / base,
    }


if __name__ == "__main__":
    plots.usar_estilo("academico")
    serie = serie_precovid()
    _, medias = festivos_total(serie)
    _, info_verano_navidad = verano_navidad(serie)
    print("Medias festivos:", {k.replace(chr(10), ' '): round(v) for k, v in medias.items()})
    print("Verano/Navidad:", {k: round(v, 1) for k, v in info_verano_navidad.items()})
