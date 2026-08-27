"""Loader del catálogo de features competibles: carga variantes por nombre, con filtrado de legalidad."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

DIR_CAT = RAIZ / "resultados" / "catalogo"
FAMILIAS = ("temperatura", "calendario", "lags")

# Caché de ficheros ya leídos; varias variantes de lags comparten el mismo parquet.
_cache_ficheros: dict[str, pd.DataFrame] = {}


def _registro(familia: str) -> list[dict]:
    """Lista normalizada de entradas de una familia (cada una con al menos nombre/fichero/columnas)."""
    if familia not in FAMILIAS:
        raise ValueError(f"Familia {familia!r} no válida. Opciones: {FAMILIAS}")
    registro = json.loads(
        (DIR_CAT / familia / f"registro_{familia}.json").read_text(encoding="utf-8"))
    if isinstance(registro, list):
        return registro
    return registro.get("variantes") or registro.get("features") or []


def entradas(familia: str, estructura: str | None = None,
             solo_legales_dayahead: bool = False, solo_legales_h2448: bool = False) -> list[dict]:
    """Entradas de la familia, filtradas por estructura (solo lags) y por legalidad del horizonte."""
    lista = _registro(familia)
    if estructura is not None:
        lista = [entrada for entrada in lista if entrada.get("estructura") == estructura]
    if solo_legales_dayahead:
        lista = [entrada for entrada in lista if entrada.get("legal_dayahead", True)]
    if solo_legales_h2448:
        lista = [entrada for entrada in lista if entrada.get("legal_h2448", True)]
    return lista


def listar(familia: str, **filtros) -> list[str]:
    return [entrada["nombre"] for entrada in entradas(familia, **filtros)]


def _leer_fichero(familia: str, fichero: str) -> pd.DataFrame:
    ruta = DIR_CAT / familia / fichero
    clave = str(ruta)
    if clave not in _cache_ficheros:
        if ruta.suffix == ".parquet":
            df = pd.read_parquet(ruta)
        else:
            df = pd.read_csv(ruta)
            # índice temporal: 'fecha' (diario) o 'datetime' (horario)
            for columna in ("datetime", "fecha"):
                if columna in df.columns:
                    df[columna] = pd.to_datetime(df[columna])
                    df = df.set_index(columna)
                    break
        _cache_ficheros[clave] = df
    return _cache_ficheros[clave]


def limpiar_cache(familia: str, fichero: str) -> None:
    """Saca del caché un fichero ya consumido (libera RAM en barridos de cientos de variantes)."""
    _cache_ficheros.pop(str(DIR_CAT / familia / fichero), None)


def cargar(familia: str, nombre: str) -> pd.DataFrame:
    """Carga una variante por nombre → DataFrame con su(s) columna(s), indexado temporalmente."""
    por_nombre = {entrada["nombre"]: entrada for entrada in _registro(familia)}
    if nombre not in por_nombre:
        raise KeyError(f"Variante {nombre!r} no está en la familia {familia!r}.")
    entrada = por_nombre[nombre]
    df = _leer_fichero(familia, entrada["fichero"])
    columnas = entrada.get("columnas") or [nombre]
    columnas = ([columna for columna in columnas if columna in df.columns]
                or ([nombre] if nombre in df.columns else list(df.columns)))
    return df[columnas].copy()


def cargar_tabla(familia: str, nombres: list[str]) -> pd.DataFrame:
    """Tabla con varias variantes concatenadas por columnas (alineadas por índice)."""
    piezas = [cargar(familia, nombre) for nombre in nombres]
    return pd.concat(piezas, axis=1)


RUTA_EXOGENAS = RAIZ / "resultados" / "metricas" / "exogenas_elegidas.json"


def exogenas_seleccionadas() -> dict:
    if not RUTA_EXOGENAS.exists():
        raise FileNotFoundError(
            f"No existe {RUTA_EXOGENAS}. Ejecuta antes la selección de exógenas: "
            "`.venv/bin/python -m src.seleccion_exogenas.run_seleccion`.")
    return json.loads(RUTA_EXOGENAS.read_text(encoding="utf-8"))["conjunto_seleccionado"]


def temperatura_ganadora_horaria() -> pd.DataFrame:
    """Temperatura HORARIA ganadora de la selección: parquet del catálogo, índice datetime."""
    seleccion = exogenas_seleccionadas()
    return cargar("temperatura", seleccion["temperatura"])


if __name__ == "__main__":
    for familia in FAMILIAS:
        ejemplo = listar(familia)[:3]
        print(f"  {familia}: {len(listar(familia))} variantes; ejemplo {ejemplo}")
    temperatura = cargar("temperatura", listar("temperatura")[0])
    print("  ejemplo temperatura:", temperatura.shape, list(temperatura.columns))
    legales = listar("lags", estructura="horaria", solo_legales_dayahead=True)
    print(f"  lags horarios legales day-ahead: {len(legales)}")
