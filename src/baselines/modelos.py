"""Modelos baseline (naive estacional m=24/m=168 y climatología) con la interfaz común de `evaluacion.py`."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))


class Baseline:
    nombre = "base"

    def fit(self, demanda: pd.Series, train_idx: pd.DatetimeIndex) -> "Baseline":
        """Ajusta el modelo con la información disponible en train (no-op por defecto)."""
        return self

    def predict(self, origenes: pd.DatetimeIndex, horizontes,
                demanda: pd.Series) -> pd.DataFrame:
        """Predice ŷ(t+h) para cada origen t (filas) y cada h en `horizontes` (columnas)."""
        raise NotImplementedError


class NaiveEstacional(Baseline):
    """Naive estacional de periodo m: ŷ(t+h) = y(t + h − m·⌈h/m⌉) (último ciclo completo)."""

    def __init__(self, m: int):
        self.m = int(m)
        self.nombre = f"naive_estacional_m{self.m}"

    def predict(self, origenes, horizontes, demanda):
        origenes = pd.DatetimeIndex(origenes)
        predicciones = {}
        for h in horizontes:
            desfase = h - self.m * math.ceil(h / self.m)
            objetivo = origenes + pd.Timedelta(hours=desfase)
            predicciones[h] = demanda.reindex(objetivo).to_numpy()
        return pd.DataFrame(predicciones, index=origenes)


class Climatologia(Baseline):
    """Media de demanda por (hora x día de semana x mes), estimada solo con train."""

    nombre = "climatologia"

    def fit(self, demanda, train_idx):
        datos = demanda.reindex(train_idx).dropna()
        indice = datos.index
        clave = pd.MultiIndex.from_arrays(
            [indice.hour, indice.dayofweek, indice.month], names=["hora", "dia_semana", "mes"])
        self.medias_ = datos.groupby(clave).mean()
        self.media_global_ = float(datos.mean())
        return self

    def predict(self, origenes, horizontes, demanda):
        origenes = pd.DatetimeIndex(origenes)
        predicciones = {}
        for h in horizontes:
            objetivo = origenes + pd.Timedelta(hours=h)
            clave = pd.MultiIndex.from_arrays([objetivo.hour, objetivo.dayofweek, objetivo.month])
            predicciones[h] = self.medias_.reindex(clave).fillna(self.media_global_).to_numpy()
        return pd.DataFrame(predicciones, index=origenes)


if __name__ == "__main__":
    from src.datos_precovid import serie_precovid
    from src.features.validacion import split_por_fechas, VAL_INI, TEST_INI

    serie = serie_precovid()
    split = split_por_fechas(serie.index, VAL_INI, TEST_INI, gap=24)
    origenes = split.test[:5]
    horizontes_prueba = [1, 24, 25, 48]
    print("Origen de prueba:", origenes[0], "| horizontes:", horizontes_prueba)
    for m in (24, 168):
        naive = NaiveEstacional(m)
        predicciones = naive.predict(origenes, horizontes_prueba, serie)
        print(f"\n{naive.nombre}:\n", predicciones.round(0).to_string())
    origen = origenes[0]
    val_h25 = serie.reindex([origen + pd.Timedelta(hours=25 - 48)]).iloc[0]
    print("\nChequeo naive m24, h=25 usa y(t-23):",
          NaiveEstacional(24).predict([origen], [25], serie).iloc[0, 0] == val_h25)
