"""
Uso:  python -m src.features.plot_split
"""
from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src import plots
from src.datos_precovid import serie_precovid
from src.features.validacion import split_canonico_precovid


def figura_split(guardar: bool = True):
    import matplotlib.pyplot as plt

    demanda_diaria = serie_precovid().resample("D").mean()
    split = split_canonico_precovid(guardar=False)

    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.plot(demanda_diaria.index, demanda_diaria.values, lw=0.4, color="0.55", alpha=0.8)
    for nombre, fechas, color in [("Train", split.train, "#1f4e79"),
                                  ("Validación", split.val, "#e69f00"),
                                  ("Test", split.test, "#009e73")]:
        ax.axvspan(fechas[0], fechas[-1], color=color, alpha=0.16,
                   label=f"{nombre} ({len(fechas):,} días-origen)")
    ax.set_ylabel("Demanda media diaria (MW)")
    ax.set_title("Reparto temporal de la serie pre-COVID")
    ax.legend(loc="upper right", ncol=3, fontsize=9)
    if guardar:
        plots.guardar_figura(fig, "features/split_temporal", formatos=("png",))
    return fig


if __name__ == "__main__":
    plots.usar_estilo("academico")
    figura_split()
    print("figura en figuras/features/split_temporal.png")
