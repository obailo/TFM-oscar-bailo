from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.features import catalogo_loader
from src.datos_precovid import serie_precovid

# Control mínimo de calendario para aislar la aportación de cada bloque: así una variante de
# temperatura/temporada no se premia por capturar estacionalidad que ya capta el calendario.
CAL_CONTROL = ["ciclicas_diasemana_mes", "finde", "festivo_nacional"]

VARIANTES_VERANO = [
    "esVerano_1jun_30sep", "esVerano_15jun_15sep", "esVerano_1jul_15sep",
    "esVerano_1jul_31ago", "esVerano_1jul_31jul", "esVerano_15jul_15sep",
    "esVerano_15jul_31ago", "esVerano_15jul_15ago", "esVerano_1ago_31ago",
    "esVerano_1ago_15ago", "esVerano_1ago_15sep", "esVerano_15ago_31ago",
]
VARIANTES_NAVIDAD = [
    "esNavidad_1dic_7ene", "esNavidad_15dic_7ene", "esNavidad_15dic_6ene",
    "esNavidad_20dic_6ene", "esNavidad_20dic_8ene", "esNavidad_22dic_6ene",
    "esNavidad_22dic_8ene", "esNavidad_22dic_1ene", "esNavidad_23dic_7ene",
    "esNavidad_24dic_6ene", "esNavidad_24dic_1ene", "esNavidad_24dic_7ene",
]

# Catálogo de temperatura (horario): 17 subconjuntos, 5 tratamientos temporales y 5 formas dan las
# 425 variantes. Se compite el catálogo completo, listándolo del registro con `catalogo_loader.listar`.


def bloque_temperatura_hora(nombre: str, hora: int) -> pd.DataFrame:
    """Columnas de una variante de temperatura horaria en la hora `hora`, reindexadas por día."""
    df = catalogo_loader.cargar("temperatura", nombre)
    seleccion = df[df.index.hour == hora]
    salida = pd.DataFrame(seleccion.to_numpy(), columns=list(seleccion.columns),
                          index=seleccion.index.normalize())
    salida.index.name = "fecha"
    return salida


# Objetivos (demanda) por estructura.

def demanda_diaria() -> pd.Series:
    """Demanda MEDIA diaria pre-COVID (objetivo de la estructura serie_unica), indexada por día."""
    serie = serie_precovid()
    demanda = serie.groupby(serie.index.normalize()).mean()
    demanda.index.name = "fecha"
    demanda.name = "y"
    return demanda


def demanda_hora(hora: int) -> pd.Series:
    """Demanda de una HORA concreta de cada día pre-COVID (objetivo de la estructura horaria)."""
    serie = serie_precovid()
    seleccion = serie[serie.index.hour == hora]
    demanda = pd.Series(seleccion.to_numpy(), index=seleccion.index.normalize())
    demanda.index.name = "fecha"
    demanda.name = "y"
    return demanda


# Bloques de regresores, indexados por 'fecha'.

def bloque_calendario_control() -> pd.DataFrame:
    return catalogo_loader.cargar_tabla("calendario", CAL_CONTROL)


def bloque_calendario(nombres: list[str]) -> pd.DataFrame:
    return catalogo_loader.cargar_tabla("calendario", nombres)
