"""Figura de las curvas de entrenamiento del MLP y la LSTM (media y rango de las 5 semillas).

Se dibuja desde `finales_{mlp,lstm}.json`, sin reentrenar.
Uso:  python -m src.redes.curvas
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src import plots

DIR_MET = RAIZ / "resultados" / "metricas"


def curvas_entrenamiento(guardar: bool = True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots.usar_estilo("academico")
    fig, ejes = plt.subplots(1, 2, figsize=(12, 4.5))
    for eje, (familia, titulo) in zip(ejes, [("mlp", "MLP multi-salida"), ("lstm", "LSTM")]):
        ruta = DIR_MET / f"finales_{familia}.json"
        if not ruta.exists():
            raise SystemExit(f"falta {ruta.name}: ejecuta antes `python -m src.redes.finales {familia}`")
        curvas = json.loads(ruta.read_text()).get("curvas")
        if not curvas:
            raise SystemExit(f"{ruta.name} no trae 'curvas': hay que reentrenar guardando la historia")
        for tramo in ("train", "val"):
            # La parada temprana deja longitudes distintas por semilla: se promedia sobre las que
            # siguen vivas en cada época, rellenando con NaN, en vez de truncar a la más corta.
            n_epocas = max(len(curva[tramo]) for curva in curvas)
            matriz = np.full((len(curvas), n_epocas), np.nan)
            for i, curva in enumerate(curvas):
                matriz[i, :len(curva[tramo])] = curva[tramo]
            epocas = np.arange(1, n_epocas + 1)
            eje.plot(epocas, np.nanmean(matriz, axis=0), label=f"{tramo} (media)")
            eje.fill_between(epocas, np.nanmin(matriz, axis=0), np.nanmax(matriz, axis=0), alpha=0.15)
        eje.set_title(f"Curva de entrenamiento — {titulo}")
        eje.set_xlabel("época"); eje.set_ylabel("pérdida (MSE escalada)"); eje.legend()
    fig.tight_layout()
    if guardar:
        [ruta_fig] = plots.guardar_figura(fig, "redes/curvas_entrenamiento", formatos=("png",))
        plt.close(fig)
        print(f"figura en {ruta_fig.relative_to(RAIZ)}")
    return fig


if __name__ == "__main__":
    curvas_entrenamiento()
