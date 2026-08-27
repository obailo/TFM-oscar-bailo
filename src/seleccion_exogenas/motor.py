"""Motor de competición de regresiones OLS: decide por MAE fuera de muestra (rolling-origin,
regla 1-SE) y usa R²_adj y AIC como criterios confirmatorios."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))


def _alinear(X: pd.DataFrame, y) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Alinea X e y por índice común y devuelve además la máscara de filas sin NaN."""
    X = pd.DataFrame(X)
    y = pd.Series(y) if not isinstance(y, pd.Series) else y
    indice = X.index.intersection(y.index)
    X = X.loc[indice]
    y = y.loc[indice].astype(float)
    validas = X.notna().all(axis=1) & y.notna()
    return X.reset_index(drop=True), y.reset_index(drop=True), validas.reset_index(drop=True)


def ajustar_ols(X: pd.DataFrame, y) -> dict:
    """Ajusta OLS (con constante) y devuelve los criterios in-sample. Ignora filas con NaN."""
    X, y, validas = _alinear(X, y)
    X_const = sm.add_constant(X[validas], has_constant="add")
    modelo = sm.OLS(y[validas].to_numpy(), X_const.to_numpy()).fit()
    return {"n": int(modelo.nobs), "k": int(X_const.shape[1] - 1),
            "r2_adj": float(modelo.rsquared_adj), "aic": float(modelo.aic)}


def ajustar_predecir(X_tr, y_tr, X_va, y_va) -> dict:
    """Ajusta en train y evalúa OUT-OF-SAMPLE en val: MAE (en MW) sobre `n` observaciones."""
    X_train, y_train, validas_train = _alinear(X_tr, y_tr)
    X_val, y_val, validas_val = _alinear(X_va, y_va)
    modelo = sm.OLS(y_train[validas_train].to_numpy(),
                    sm.add_constant(X_train[validas_train], has_constant="add").to_numpy()).fit()
    pred = modelo.predict(sm.add_constant(X_val[validas_val], has_constant="add").to_numpy())
    real = y_val[validas_val].to_numpy()
    errores = real - pred
    return {"mae": float(np.mean(np.abs(errores))), "n": int(validas_val.sum())}


def _avisar_n_heterogeneo(tabla: pd.DataFrame, tolerancia_relativa: float = 0.01) -> None:
    """Avisa por stdout si el `n` in-sample difiere entre candidatas más allá de la tolerancia."""
    if "n" not in tabla.columns or len(tabla) < 2:
        return
    n_min, n_max = int(tabla["n"].min()), int(tabla["n"].max())
    if n_max > 0 and (n_max - n_min) / n_max > tolerancia_relativa:
        print(f"  [aviso] n heterogéneo entre candidatas: min={n_min} max={n_max} "
              f"(diferencia {100 * (n_max - n_min) / n_max:.1f}% > tolerancia "
              f"{100 * tolerancia_relativa:.0f}%). Conviene revisarlo antes de comparar "
              f"R²_adj/AIC.")


# Criterio decisor: MAE_oos por rolling-origin más la regla 1-SE. El OLS se ajusta dentro de cada
# fold, solo con su train; R²_adj y AIC quedan como confirmatorios.

def _mae_fold(X_full: pd.DataFrame, y_full: pd.Series, tr_idx, va_idx) -> float | None:
    """MAE OOS de un fold: ajusta OLS en las fechas de train y evalúa en las de val (día-origen)."""
    X_train = X_full.loc[X_full.index.intersection(tr_idx)]
    X_val = X_full.loc[X_full.index.intersection(va_idx)]
    resultado = ajustar_predecir(X_train, y_full.reindex(X_train.index),
                                 X_val, y_full.reindex(X_val.index))
    return resultado["mae"] if resultado["n"] > 0 else None


def competir_rolling(candidatos: dict[str, pd.DataFrame], y: pd.Series,
                     folds: list) -> pd.DataFrame:
    """Competición por rolling-origin: MAE_oos medio ± SE sobre folds más criterios in-sample."""
    filas = []
    for nombre, X in candidatos.items():
        maes = [mae for fold in folds
                if (mae := _mae_fold(X, y, fold.train, fold.val)) is not None]
        if not maes:
            continue
        valores = np.asarray(maes, dtype=float)
        mae_medio = float(valores.mean())
        se = (float(valores.std(ddof=1) / np.sqrt(len(valores)))
              if len(valores) > 1 else float("nan"))
        # confirmatorios in-sample sobre el train del último fold, el más grande, casi el train completo
        train = folds[-1].train
        in_sample = ajustar_ols(X.loc[X.index.intersection(train)],
                                y.reindex(X.index.intersection(train)))
        filas.append({"candidato": nombre, "mae_oos_medio": mae_medio, "mae_oos_se": se,
                      "n_folds": len(valores), "r2_adj": in_sample["r2_adj"],
                      "aic": in_sample["aic"], "k": in_sample["k"], "n": in_sample["n"]})
    columnas = ["candidato", "mae_oos_medio", "mae_oos_se", "n_folds", "r2_adj", "aic", "k", "n"]
    tabla = pd.DataFrame(filas)[columnas]
    if len(tabla):
        tabla = tabla.sort_values("mae_oos_medio").reset_index(drop=True)
    _avisar_n_heterogeneo(tabla)
    return tabla


def elegir_1se(tabla: pd.DataFrame) -> str:
    """Aplica la regla 1-SE + parsimonia a una tabla de `competir_rolling` y devuelve el candidato elegido."""
    if not len(tabla):
        raise ValueError("tabla de competición vacía")
    mejor = tabla.iloc[0]
    se = mejor["mae_oos_se"]
    umbral = mejor["mae_oos_medio"] + (se if pd.notna(se) else 0.0)
    dentro = tabla[tabla["mae_oos_medio"] <= umbral]
    # más parsimonioso (menor k) y, a igualdad, menor MAE
    return dentro.sort_values(["k", "mae_oos_medio"]).iloc[0]["candidato"]
