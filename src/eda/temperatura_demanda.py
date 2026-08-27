from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src import plots
from src.datos_precovid import serie_precovid
from src.features.temperatura import descargar_rep20_horaria


def _datos():
    """Demanda media diaria y temperatura media diaria de cinco capitales, alineadas por fecha."""
    demanda_horaria = serie_precovid()
    demanda = demanda_horaria.resample("D").mean()
    ciudades = ("Madrid", "Barcelona", "València", "Sevilla", "Zaragoza")
    temp_horaria = descargar_rep20_horaria()  # lee el parquet cacheado, no descarga
    columnas = [columna for columna in temp_horaria.columns
                if columna.endswith("__T") and columna.split("__")[0] in ciudades]
    if not columnas:
        raise KeyError(f"no encuentro las columnas de temperatura de {ciudades} en "
                       f"{list(temp_horaria.columns)[:6]}…")
    temperatura = temp_horaria[columnas].mean(axis=1).resample("D").mean()
    fechas_comunes = demanda.index.intersection(temperatura.index)
    return demanda.loc[fechas_comunes], temperatura.loc[fechas_comunes].rename("temperatura")


def relacion_temp_demanda(demanda, temperatura, guardar=True):
    """Nube demanda–temperatura + curva de media por tramo de 1 °C (forma en U)."""
    fig, ax = plt.subplots(figsize=(11, 5))
    n = min(8000, len(demanda))
    indices_muestra = demanda.sample(n, random_state=42).index
    ax.scatter(temperatura.loc[indices_muestra], demanda.loc[indices_muestra], s=6, alpha=0.25, label="días (muestra)")
    tramos = np.arange(np.floor(temperatura.min()), np.ceil(temperatura.max()) + 1, 1.0)
    categorias = pd.cut(temperatura, tramos)
    media = demanda.groupby(categorias, observed=True).mean()
    centros = [iv.mid for iv in media.index]
    ax.plot(centros, media.values, color="#b5651d", lw=2.2, marker="o", ms=3, label="media por °C")
    ax.set_title("Demanda diaria vs temperatura (forma en U: calefacción + refrigeración)")
    ax.set_xlabel("Temperatura media de cinco capitales (°C)"); ax.set_ylabel("Demanda media diaria (MW)"); ax.legend()
    if guardar:
        plots.guardar_figura(fig, "eda/temperatura/relacion_temp_demanda", formatos=("png",))
    correlacion_lineal = float(demanda.corr(temperatura))
    return fig, {"corr_lineal": correlacion_lineal, "temp_min_demanda": float(centros[int(np.argmin(media.values))])}


def main():
    """Regenera las figuras de la relación temperatura-demanda (figuras/eda/temperatura/)."""
    plots.usar_estilo("academico")
    demanda, temperatura = _datos()
    print(f"Temperatura media de cinco capitales (media de las 24 h) | {len(demanda)} días "
          f"(de {demanda.index.min().date()} a {demanda.index.max().date()})")
    _, resultado = relacion_temp_demanda(demanda, temperatura)
    print("Relación temp-demanda:", {k: round(v, 2) for k, v in resultado.items()})


if __name__ == "__main__":
    main()
