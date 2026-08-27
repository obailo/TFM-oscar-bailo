"""Series objetivo, exógenas y lags de las dos estructuras del SARIMAX (horizonte D+1):
24 modelos por hora y serie horaria continua."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.datos_precovid import serie_precovid
from src.features.por_hora import series_por_hora
from src.features.calendario import (features_calendario, festivo_regional_ponderado,
                                     es_verano, es_navidad)
from src.features.catalogo_calendario import FRANJAS_VERANO, FRANJAS_NAVIDAD
from src.features import catalogo_loader
from src.features.validacion import (split_canonico_precovid, rolling_origin_folds,
                                     fechas_origen_precovid, TEST_INI,
                                     ROLLING_VAL_YEARS, PURGA_DIAS)

# Calendario diario seleccionado (orden fijo); sin festivo_nacional ni posiblePuente.
COLS_CAL = ["dia_semana_sin", "dia_semana_cos", "mes_sin", "mes_cos", "finde",
            "festivo_reg", "esVerano", "esNavidad"]


def serie_horaria() -> pd.Series:
    return serie_precovid()


def panel_por_hora() -> pd.DataFrame:
    """24 series diarias (filas=día, columnas h00..h23) sobre el periodo pre-COVID."""
    return series_por_hora(serie_horaria())


_TEMP_H: pd.DataFrame | None = None


def temp_horaria() -> pd.DataFrame:
    """T HORARIA ganadora de la selección de exógenas: columnas T, T² (índice datetime horario)."""
    global _TEMP_H
    if _TEMP_H is None:
        _TEMP_H = catalogo_loader.temperatura_ganadora_horaria()
    return _TEMP_H


def temp_por_hora(hora: int) -> pd.DataFrame:
    """La T de la hora `hora` como serie DIARIA (índice = día): columnas T, T² de esa hora."""
    temperatura = temp_horaria()
    sub = temperatura[temperatura.index.hour == hora].copy()
    sub.index = sub.index.normalize()
    sub.index.name = "fecha"
    return sub


_FRANJAS: tuple | None = None


def _franjas_elegidas():
    """(ini,fin) de las franjas esVerano/esNavidad elegidas (a partir de su nombre de catálogo)."""
    global _FRANJAS
    if _FRANJAS is None:
        seleccionadas = catalogo_loader.exogenas_seleccionadas()
        verano = FRANJAS_VERANO[seleccionadas["esVerano"].split("_", 1)[1]]
        navidad = FRANJAS_NAVIDAD[seleccionadas["esNavidad"].split("_", 1)[1]]
        _FRANJAS = (verano, navidad)
    return _FRANJAS


def _calendario_diario(fechas: pd.DatetimeIndex) -> pd.DataFrame:
    fechas = pd.DatetimeIndex(fechas)
    (ver_ini, ver_fin), (nav_ini, nav_fin) = _franjas_elegidas()
    calendario = features_calendario(fechas, incluir_ciclicas=True)
    X = pd.DataFrame(index=fechas)
    for columna in ("dia_semana_sin", "dia_semana_cos", "mes_sin", "mes_cos"):
        X[columna] = calendario[columna].to_numpy()
    X["finde"] = calendario["finde"].to_numpy()
    X["festivo_reg"] = festivo_regional_ponderado(fechas).to_numpy()
    X["esVerano"] = es_verano(fechas, ini=ver_ini, fin=ver_fin).to_numpy()
    X["esNavidad"] = es_navidad(fechas, ini=nav_ini, fin=nav_fin).to_numpy()
    return X[COLS_CAL]


def exog_por_hora(hora: int, fechas) -> pd.DataFrame:
    """24 modelos por hora: calendario day-level + la T de la hora `hora` (T, T²)."""
    fechas = pd.DatetimeIndex(fechas)
    X = _calendario_diario(fechas)
    temp_hora = temp_por_hora(hora).reindex(fechas)
    X = X.copy()
    X["T"] = temp_hora["T"].to_numpy()
    X["T2"] = temp_hora["T2"].to_numpy()
    return X


def exog_horaria_continua(idx_horario) -> pd.DataFrame:
    """Continua: calendario diario difundido a las 24 h + T horaria real (sin broadcast)."""
    indice = pd.DatetimeIndex(idx_horario)
    dias = pd.DatetimeIndex(indice.normalize().unique()).sort_values()
    calendario = _calendario_diario(dias)
    salida = calendario.reindex(indice.normalize())  # difunde el calendario del día a sus 24 h
    salida.index = indice
    temperatura = temp_horaria().reindex(indice)  # aquí la T horaria real, sin difundir
    salida["T"] = temperatura["T"].to_numpy()
    salida["T2"] = temperatura["T2"].to_numpy()
    return salida


def lags_horarios(serie_h: pd.Series, lags_horas=(168,)) -> pd.DataFrame:
    """Lags de la serie horaria como exógenas (por defecto t-168, la 2ª estacionalidad)."""
    salida = pd.DataFrame(index=serie_h.index)
    for k in lags_horas:
        salida[f"lag_{k}h"] = serie_h.shift(k)
    return salida


def lags_diarios_hora(porh: pd.DataFrame, hora: int, lags_dias=(1, 2, 7, 14)) -> pd.DataFrame:
    """Lags de la propia hora como exógenas (1/2/7/14 días atrás), legales day-ahead."""
    columna = f"h{hora:02d}"
    salida = pd.DataFrame(index=porh.index)
    for k in lags_dias:
        salida[f"lag_{k}d"] = porh[columna].shift(k)
    return salida


if __name__ == "__main__":
    split = split_canonico_precovid(guardar=False)
    print(f"split canónico: train {len(split.train)} "
          f"({split.train.min().date()}→{split.train.max().date()}) | "
          f"val {len(split.val)} | test {len(split.test)} "
          f"({split.test.min().date()}→{split.test.max().date()})")
    folds = rolling_origin_folds()
    for anio, fold in zip(ROLLING_VAL_YEARS, folds):
        print(f"  fold val={anio}: train {fold.train.min().date()}→{fold.train.max().date()} "
              f"(n={len(fold.train)}) "
              f"| val {fold.val.min().date()}→{fold.val.max().date()} (n={len(fold.val)})")
    panel = panel_por_hora()
    print(f"panel horario: {panel.shape}")
    print(f"T horaria (variante ganadora): {temp_horaria().shape} {list(temp_horaria().columns)}")
    exog_continua = exog_horaria_continua(serie_horaria().index[:48])
    exog_hora = exog_por_hora(13, panel.index)
    print(f"exóg horaria continua (48 h): {exog_continua.shape} {list(exog_continua.columns)}")
    print(f"exóg hora 13: {exog_hora.shape} {list(exog_hora.columns)}")
    print(f"franjas elegidas (verano, navidad): {_franjas_elegidas()}")
