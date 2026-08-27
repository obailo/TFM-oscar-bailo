"""Control de calidad de la serie: integridad (NaN, duplicados, huecos, artefactos DST) y distribución."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src import plots
from src.datos import cargar_demanda
from src.datos_precovid import serie_precovid
from src.eda.carga import picos_dst_otono


def integridad(df: pd.DataFrame) -> dict:
    """Comprueba continuidad, duplicados, NaN y los dos artefactos DST."""
    rango = pd.date_range(df.index.min(), df.index.max(), freq="h")
    ceros = df[df["Demanda"] == 0]
    # Pico DST de otoño, con la MISMA regla que aplica `carga.serie_limpia` al limpiarlo
    picos_otono = df[picos_dst_otono(df["Demanda"])]
    info = {
        "n_nan": int(df["Demanda"].isna().sum()),
        "n_duplicados": int(df.index.duplicated().sum()),
        "faltan": len(rango) - len(df),
        "n_ceros_dst_primavera": len(ceros),
        "n_picos_dst_otono": len(picos_otono),
    }
    return info


def distribucion(serie: pd.Series, guardar: bool = True):
    """Histograma + diagrama de caja de la demanda; devuelve estadísticos."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].hist(serie.values, bins=80)
    axes[0].set_title("Histograma de la demanda horaria")
    axes[0].set_xlabel("Demanda (MW)"); axes[0].set_ylabel("Frecuencia")
    axes[1].boxplot(serie.values, vert=True, widths=0.5)
    axes[1].set_title("Diagrama de caja")
    axes[1].set_ylabel("Demanda (MW)"); axes[1].set_xticks([])
    if guardar:
        plots.guardar_figura(fig, "eda/distribucion/distribucion", formatos=("png",))
    return fig, {"media": serie.mean(), "mediana": serie.median(), "std": serie.std(),
                 "skew": serie.skew(), "kurt": serie.kurt()}


if __name__ == "__main__":
    plots.usar_estilo("academico")
    serie = serie_precovid()
    print("Integridad (serie cruda):", integridad(cargar_demanda()))
    print("Integridad (serie pre-COVID limpia):", integridad(serie.to_frame()))
    _, estadisticos = distribucion(serie)
    print("Distribución:", {k: round(v, 2) for k, v in estadisticos.items()})
