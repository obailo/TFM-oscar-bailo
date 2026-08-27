"""Cuantifica la caída de demanda en festivos y fin de semana frente al laborable.

Ejecutar:  python -m src.eda.efecto_festivos
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src import plots
from src.datos_precovid import serie_precovid
from src.features.calendario import festivo_regional_ponderado

# Meses de cada estación meteorológica, para el desglose estacional.
_ESTACIONES = {"Invierno": [12, 1, 2], "Primavera": [3, 4, 5],
               "Verano": [6, 7, 8], "Otoño": [9, 10, 11]}


def demanda_diaria(serie: pd.Series) -> pd.Series:
    """Demanda media diaria (1 valor/día), para casar con las exógenas de resolución diaria."""
    demanda = serie.groupby(serie.index.normalize()).mean()
    demanda.index.name = "fecha"
    demanda.name = "demanda"
    return demanda


def _r2_ols(X: np.ndarray, y: np.ndarray) -> float:
    """R² de OLS multivariante (con intercepto) vía mínimos cuadrados, sin sklearn."""
    X = np.asarray(X, float); y = np.asarray(y, float)
    mascara = np.isfinite(y) & np.isfinite(X).all(axis=1)
    X, y = X[mascara], y[mascara]
    matriz = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(matriz, y, rcond=None)
    residuo = y - matriz @ beta
    sse = float((residuo ** 2).sum())
    sst = float(((y - y.mean()) ** 2).sum())
    return 1.0 - sse / sst if sst > 0 else float("nan")


def efecto_festivos(demanda_diaria: pd.Series, guardar: bool = True):
    """Caída de demanda en festivo (nacional y regional parcial) y en fin de semana, global y por estación."""
    dias = demanda_diaria.index
    festivo_regional = festivo_regional_ponderado(dias)
    finde = pd.Series(dias.dayofweek >= 5, index=dias)
    nacional = festivo_regional >= 0.99
    regional_parcial = (festivo_regional > 0.01) & (festivo_regional < 0.99)
    laborable_normal = (~finde) & (festivo_regional <= 0.01)

    media_laborable = demanda_diaria[laborable_normal].mean()
    grupos = {
        "Laborable normal": laborable_normal,
        "Finde": finde & (festivo_regional <= 0.01),
        "Festivo nacional": nacional,
        "Festivo regional parcial": regional_parcial,
    }
    filas = []
    for nombre, mascara in grupos.items():
        valores = demanda_diaria[mascara]
        filas.append({
            "grupo": nombre,
            "n_dias": int(mascara.sum()),
            "media_MW": float(valores.mean()),
            "caida_pct_vs_laborable": float(100 * (valores.mean() - media_laborable) / media_laborable),
        })
    tabla = pd.DataFrame(filas)

    # Caída del festivo nacional por estación (vs laborable de la misma estación)
    por_estacion = {}
    for estacion, meses in _ESTACIONES.items():
        es_estacion = dias.month.isin(meses)
        laborables_estacion = demanda_diaria[laborable_normal & es_estacion]
        festivos_estacion = demanda_diaria[nacional & es_estacion]
        if len(laborables_estacion) and len(festivos_estacion):
            por_estacion[estacion] = float(100 * (festivos_estacion.mean() - laborables_estacion.mean())
                                           / laborables_estacion.mean())

    # R² incremental del festivo regional ponderado (continuo) sobre finde solo
    y = demanda_diaria.to_numpy()
    finde_float = finde.astype(float).to_numpy()
    r2_finde = _r2_ols(finde_float.reshape(-1, 1), y)
    r2_finde_fr = _r2_ols(np.column_stack([finde_float, festivo_regional.to_numpy()]), y)

    resultado = {
        "tabla": tabla.to_dict("records"),
        "caida_nacional_por_estacion_pct": por_estacion,
        "r2_finde": float(r2_finde),
        "r2_finde_mas_festivo_regional": float(r2_finde_fr),
        "r2_incremental_festivo_regional": float(r2_finde_fr - r2_finde),
    }
    if guardar:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.bar(tabla["grupo"], tabla["caida_pct_vs_laborable"],
               color=["#555555", "#b5651d", "#1f3b57", "#3a7d44"])
        ax.axhline(0, color="black", lw=0.8)
        ax.set_ylabel("Δ demanda vs laborable (%)")
        ax.set_title("Efecto de festivos y fin de semana sobre la demanda diaria")
        ax.tick_params(axis="x", labelrotation=15)
        plots.guardar_figura(fig, "eda/festivos/festivos_efecto", formatos=("png",))
        plt.close(fig)
    return resultado


def main(guardar: bool = True) -> dict:
    plots.usar_estilo("academico")
    resultado = efecto_festivos(demanda_diaria(serie_precovid()), guardar)
    for fila in resultado["tabla"]:
        print(f"  {fila['grupo']:26s} n={fila['n_dias']:5d}  {fila['media_MW']:8.0f} MW  "
              f"{fila['caida_pct_vs_laborable']:+6.1f}%")
    print("  caída del festivo nacional por estación (%):",
          {k: round(valor, 1) for k, valor in resultado["caida_nacional_por_estacion_pct"].items()})
    print(f"  R² incremental del festivo regional sobre finde: {resultado['r2_incremental_festivo_regional']:.4f}")
    return resultado


if __name__ == "__main__":
    main()
