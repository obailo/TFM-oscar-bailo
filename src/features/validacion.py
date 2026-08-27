"""Validación temporal: split cronológico con purga y origen rodante de 5 cortes; el test queda reservado."""
from __future__ import annotations

import json
import sys
from collections import namedtuple
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.datos_precovid import serie_precovid

DIR_SPLITS = RAIZ / "resultados" / "splits"

VAL_INI = "2018-01-01"  # validación = último año de train (early-stopping y último fold del rolling)
TEST_INI = "2019-01-01"  # test reservado: de 2019 a febrero de 2020

PURGA_DIAS = 1
# Rolling-origin: ventana expansiva, 5 folds de un año de validación; train = 2000 hasta (año_val - 1).
ROLLING_VAL_YEARS = (2014, 2015, 2016, 2017, 2018)

SplitTemporal = namedtuple("SplitTemporal", ["train", "val", "test"])


def split_por_fechas(index: pd.DatetimeIndex, val_ini, test_ini,
                     gap: int = PURGA_DIAS) -> SplitTemporal:
    """Split con fronteras de fecha fijas y purga de `gap` muestras al final de train y de val."""
    val_ini, test_ini = pd.Timestamp(val_ini), pd.Timestamp(test_ini)
    train = index[index < val_ini]
    val = index[(index >= val_ini) & (index < test_ini)]
    test = index[index >= test_ini]
    # Purga incondicional: si el tramo mide gap o menos se queda vacío, y así no queda un hueco residual
    train = train[: max(0, len(train) - gap)]
    val = val[: max(0, len(val) - gap)]
    return SplitTemporal(train, val, test)


def fechas_origen_precovid() -> pd.DatetimeIndex:
    serie = serie_precovid()
    return pd.DatetimeIndex(serie.index.normalize().unique()).sort_values()


def _resumen_split(split: SplitTemporal, gap: int) -> dict:
    def tramo(fechas):
        return ({"inicio": str(fechas[0]), "fin": str(fechas[-1]), "n": len(fechas)}
                if len(fechas) else {"n": 0})
    return {"gap": gap, "train": tramo(split.train), "val": tramo(split.val),
            "test": tramo(split.test)}


def split_canonico_precovid(guardar: bool = True,
                            gap_dias: int = PURGA_DIAS,
                            fechas_origen: pd.DatetimeIndex | None = None) -> SplitTemporal:

    if fechas_origen is None:
        fechas_origen = fechas_origen_precovid()
    split = split_por_fechas(fechas_origen, VAL_INI, TEST_INI, gap=gap_dias)
    if guardar:
        DIR_SPLITS.mkdir(parents=True, exist_ok=True)
        resumen = {"granularidad": "día-origen", "val_ini": VAL_INI, "test_ini": TEST_INI,
                   "gap_dias": gap_dias, **_resumen_split(split, gap_dias)}
        (DIR_SPLITS / "precovid_canonico.json").write_text(
            json.dumps(resumen, indent=2, ensure_ascii=False))
    return split


def rolling_origin_folds(fechas_origen: pd.DatetimeIndex | None = None,
                         val_years=ROLLING_VAL_YEARS, gap_dias: int = PURGA_DIAS,
                         guardar: bool = False) -> list[SplitTemporal]:
    """Folds de rolling-origin (walk-forward) con ventana expansiva, por día-origen; el test no participa."""
    if fechas_origen is None:
        fechas_origen = fechas_origen_precovid()
    folds = []
    for anio in val_years:
        val_ini = pd.Timestamp(f"{anio}-01-01")
        val_fin = pd.Timestamp(f"{anio + 1}-01-01")
        train = fechas_origen[fechas_origen < val_ini]
        train = train[: max(0, len(train) - gap_dias)]  # purga day-ahead
        val = fechas_origen[(fechas_origen >= val_ini) & (fechas_origen < val_fin)]
        folds.append(SplitTemporal(train, val, fechas_origen[:0]))
    if guardar:
        DIR_SPLITS.mkdir(parents=True, exist_ok=True)
        resumen = {"val_years": list(val_years), "gap_dias": gap_dias, "n_folds": len(folds),
                   "folds": [{"val_year": anio, **_resumen_split(fold, gap_dias)}
                             for anio, fold in zip(val_years, folds)]}
        (DIR_SPLITS / "precovid_rolling.json").write_text(
            json.dumps(resumen, indent=2, ensure_ascii=False))
    return folds


if __name__ == "__main__":
    split = split_canonico_precovid(guardar=True)
    print(f"[canónico] gap_dias={PURGA_DIAS}")
    for nombre_tramo, fechas in zip(("train", "val", "test"), split):
        if len(fechas):
            print(f"  {nombre_tramo:5}: {fechas[0].date()} -> {fechas[-1].date()}  (n={len(fechas):,})")
    print("Purga OK (train<val<test, sin solape):",
          split.train[-1] < split.val[0] and split.val[-1] < split.test[0])

    print(f"\n[rolling-origin] val_years={ROLLING_VAL_YEARS}")
    for anio, fold in zip(ROLLING_VAL_YEARS, rolling_origin_folds(guardar=True)):
        print(f"  fold val={anio}: train {fold.train[0].date()}->{fold.train[-1].date()} "
              f"(n={len(fold.train):,}) | val {fold.val[0].date()}->{fold.val[-1].date()} (n={len(fold.val)})")
