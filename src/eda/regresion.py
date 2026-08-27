from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src import plots
from src.datos_precovid import serie_precovid
from src.eda.autocorrelacion import LAGS_INTERES
from src.eda.festivos import es_festivo


def _features_calendario(serie: pd.Series) -> pd.DataFrame:
    festivos = es_festivo(serie.index)
    return pd.DataFrame({
        "hora": serie.index.hour, "dia_semana": serie.index.dayofweek, "mes": serie.index.month,
        "festivo": festivos.values.astype(int),
    }, index=serie.index)


def baseline_regresion_lineal(serie: pd.Series) -> float:
    """R² in-sample (descriptivo, no predictivo) de la regresión lineal demanda ~ calendario."""

    calendario = _features_calendario(serie)
    X = pd.get_dummies(calendario[["hora", "dia_semana", "mes"]].astype("category"), drop_first=True)
    X["festivo"] = calendario["festivo"].values
    modelo = LinearRegression().fit(X.values, serie.values)
    return float(modelo.score(X.values, serie.values))


def vif_lags(serie: pd.Series) -> pd.Series:
    tabla_lags = pd.DataFrame({f"lag_{lag}": serie.shift(lag) for lag in LAGS_INTERES}).dropna()
    X_const = np.column_stack([np.ones(len(tabla_lags)), tabla_lags.values])  # con constante
    vifs = {col: variance_inflation_factor(X_const, i + 1) for i, col in enumerate(tabla_lags.columns)}
    return pd.Series(vifs)


if __name__ == "__main__":
    plots.usar_estilo("academico")
    serie = serie_precovid()
    print(f"R² regresión lineal (solo calendario): {baseline_regresion_lineal(serie):.3f}")
    print("\nVIF entre lags:\n", vif_lags(serie).round(1))
