"""Estilos de gráficas y utilidades de figuras.

Uso:
    plots.usar_estilo("academico")
    plots.guardar_figura(fig, "demanda_serie_completa")
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

RAIZ = Path(__file__).resolve().parent.parent
DIR_FIGURAS = RAIZ / "figuras"

_PALETA_ACADEMICO = ["#1f3b57", "#b5651d", "#3a7d44", "#7a3b69", "#555555"]

_BASE = {
    "figure.figsize": (11, 4.5),
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.frameon": False,
    "lines.linewidth": 1.4,
}

_TEMAS = {
    "academico": {
        **_BASE,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "serif",
        "axes.prop_cycle": mpl.cycler(color=_PALETA_ACADEMICO),
        "text.color": "#1a1a1a",
        "axes.labelcolor": "#1a1a1a",
        "xtick.color": "#1a1a1a",
        "ytick.color": "#1a1a1a",
    }
}


def usar_estilo(tema: str = "academico") -> None:
    if tema not in _TEMAS:
        raise ValueError(f"Tema desconocido: {tema!r}. Opciones: {list(_TEMAS)}")
    mpl.rcParams.update(_TEMAS[tema])


def guardar_figura(fig, nombre: str, formatos=("pdf",)) -> list[Path]:
    """Guarda una figura en figuras/ en uno o varios formatos; nombre puede incluir subcarpetas."""
    rutas = []
    for formato in formatos:
        ruta = DIR_FIGURAS / f"{nombre}.{formato}"
        ruta.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(ruta, bbox_inches="tight")
        rutas.append(ruta)
    return rutas
