"""Temperatura horaria (T/HR/TD) de 20 ciudades desde Open-Meteo (ERA5), con caché parquet y subconjuntos."""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))


def _cache_cubre(df: pd.DataFrame, ini, fin) -> bool:
    """True si el índice cacheado cubre todo el rango [ini, fin] pedido (a resolución de día)."""
    if df is None or df.empty:
        return False
    ini_cache, fin_cache = df.index.min().normalize(), df.index.max().normalize()
    return ini_cache <= pd.Timestamp(ini).normalize() and fin_cache >= pd.Timestamp(fin).normalize()


# Las 20 ciudades de rep20 con su (lat, lon): grandes y repartidas por las regiones climáticas
# peninsulares.
CIUDADES_REP20 = {
    "Madrid": (40.417, -3.704), "Barcelona": (41.385, 2.173),
    "València": (39.470, -0.376), "Sevilla": (37.389, -5.984),
    "Bilbao": (43.263, -2.935), "Zaragoza": (41.649, -0.888),
    "Málaga": (36.721, -4.420), "Coruña, A": (43.362, -8.412),
    "Valladolid": (41.652, -4.724), "Murcia": (37.992, -1.131),
    "Oviedo": (43.362, -5.849), "Santander": (43.463, -3.805),
    "Donostia": (43.318, -1.981), "Pamplona": (42.812, -1.645),
    "Alacant/Alicante": (38.345, -0.483), "Granada": (37.177, -3.598),
    "Córdoba": (37.889, -4.779), "Badajoz": (38.879, -6.970),
    "León": (42.598, -5.567), "Albacete": (38.994, -1.858),
}

# 17 subconjuntos por cobertura climática, que barren tanto el tamaño (de 3 a 20) como la
# composición. Las claves son abreviaturas de ciudad; ojo con `Cor` (A Coruña) y `Cór` (Córdoba).
_AB = {'Mad': 'Madrid', 'Bcn': 'Barcelona', 'Val': 'València', 'Sev': 'Sevilla', 'Bil': 'Bilbao',
       'Zar': 'Zaragoza', 'Mal': 'Málaga', 'Cor': 'Coruña, A', 'Vll': 'Valladolid', 'Mur': 'Murcia',
       'Ovi': 'Oviedo', 'San': 'Santander', 'Don': 'Donostia', 'Pam': 'Pamplona',
       'Ala': 'Alacant/Alicante', 'Gra': 'Granada', 'Cór': 'Córdoba', 'Bad': 'Badajoz',
       'León': 'León', 'Alb': 'Albacete'}
_rep10 = ['Mad', 'Bcn', 'Val', 'Sev', 'Mal', 'Bil', 'Zar', 'Mur', 'Cor', 'Vll']
_SUBS_AB = {
    'sub3_med': ['Mad', 'Bcn', 'Val'], 'sub3_sur': ['Mad', 'Bcn', 'Sev'], 'sub3_norte': ['Mad', 'Bcn', 'Bil'],
    'sub5_diverso': ['Mad', 'Bcn', 'Sev', 'Bil', 'Zar'], 'sub5_costa': ['Mad', 'Bcn', 'Val', 'Mal', 'Bil'],
    'sub5_interior': ['Mad', 'Bcn', 'Vll', 'Zar', 'Sev'], 'sub5_grandes': ['Mad', 'Bcn', 'Val', 'Sev', 'Zar'],
    'sub7_diverso': ['Mad', 'Bcn', 'Sev', 'Bil', 'Zar', 'Cor', 'Mal'],
    'sub7_penin': ['Mad', 'Bcn', 'Sev', 'Bil', 'Zar', 'Val', 'Vll'],
    'sub7_sur_med': ['Mad', 'Bcn', 'Val', 'Ala', 'Mal', 'Sev', 'Mur'],
    'sub7_capitales': ['Mad', 'Bcn', 'Val', 'Sev', 'Zar', 'Mal', 'Mur'],
    'rep10': _rep10, 'sub10_alt': ['Mad', 'Bcn', 'Val', 'Sev', 'Gra', 'Bil', 'Zar', 'Ovi', 'Cor', 'Vll'],
    'sub10_norte_interior': ['Mad', 'Bcn', 'Val', 'Sev', 'Bil', 'Zar', 'Vll', 'Cor', 'León', 'Ovi'],
    'sub15_a': _rep10 + ['Ovi', 'Ala', 'Gra', 'León', 'Pam'], 'sub15_b': _rep10 + ['San', 'Cór', 'Bad', 'Alb', 'Don'],
    'rep20': list(_AB.keys()),
}
SUBCONJUNTOS = {nombre: [_AB[a] for a in abbrs] for nombre, abbrs in _SUBS_AB.items()}


# Descarga horaria: es la fuente única, y es pesada, así que va resumible y con pausas.
RUTA_TEMP_REP20_H = RAIZ / "resultados" / "temperatura_rep20_horaria.parquet"


_HVARS = {"temperature_2m": "T", "relative_humidity_2m": "HR", "dew_point_2m": "TD"}
# El rango arranca en 1997 para tener tres años previos con los que calcular la normal estacional a partir de 2000.
INI_HORARIA = "1997-01-01"
FIN_HORARIA = "2020-02-29"


def _descargar_ciudad_horaria(lat, lon, ini, fin, reintentos=6):
    """Descarga horaria de T + HR + TD de una ciudad (hora local Europe/Madrid), con backoff ante HTTP 429."""
    variables = ",".join(_HVARS)
    url = (f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
           f"&start_date={ini}&end_date={fin}&hourly={variables}&timezone=Europe%2FMadrid")
    for i in range(reintentos):
        try:
            with urllib.request.urlopen(url, timeout=180) as respuesta:
                datos = json.load(respuesta)
            return datos["hourly"]
        except urllib.error.HTTPError as error:
            if i == reintentos - 1:
                raise
            espera = 60 * (2 ** i) if error.code == 429 else 5  # si es rate limit, esperar mucho más
            print(f"    HTTP {error.code}; reintento {i+1}/{reintentos} tras {espera}s", flush=True)
            time.sleep(espera)
        except Exception:
            if i == reintentos - 1:
                raise
            time.sleep(5)


def descargar_rep20_horaria(ini=INI_HORARIA, fin=FIN_HORARIA) -> pd.DataFrame:
    """Descarga (resumible, throttled) T + HR + TD horarios de las 20 ciudades rep20; usa la caché si existe."""
    if RUTA_TEMP_REP20_H.exists():
        df = pd.read_parquet(RUTA_TEMP_REP20_H)
        if not _cache_cubre(df, ini, fin):
            print(f"  caché {RUTA_TEMP_REP20_H.name} no cubre [{ini}, {fin}]: re-descarga completa", flush=True)
            df = pd.DataFrame()  # el rango cacheado no basta (p. ej. falta 1997-1999)
    else:
        df = pd.DataFrame()
    RUTA_TEMP_REP20_H.parent.mkdir(parents=True, exist_ok=True)
    for i, (ciudad, (lat, lon)) in enumerate(CIUDADES_REP20.items(), 1):
        if all(f"{ciudad}__{sufijo}" in df.columns for sufijo in _HVARS.values()):
            continue  # la ciudad ya está con sus tres variables, se puede reanudar
        horario = _descargar_ciudad_horaria(lat, lon, ini, fin)
        indice = pd.to_datetime(horario["time"])
        for variable, sufijo in _HVARS.items():
            df[f"{ciudad}__{sufijo}"] = pd.Series(horario[variable], index=indice)
        df = df.sort_index()
        df.to_parquet(RUTA_TEMP_REP20_H)
        print(f"  [{i:2}/{len(CIUDADES_REP20)}] {ciudad}: {len(indice)} h", flush=True)
        time.sleep(30)  # pausa amplia: la petición pesa mucho en el límite horario de Open-Meteo
    return df


def serie_horaria_var(df_h: pd.DataFrame, subconjunto, var: str = "T") -> pd.Series:
    """Serie horaria uniforme (media entre ciudades) de la variable var in  {T,HR,TD} de un subconjunto."""
    ciudades = SUBCONJUNTOS[subconjunto] if isinstance(subconjunto, str) else subconjunto
    columnas = [f"{ciudad}__{var}" for ciudad in ciudades]
    return pd.Series(df_h[columnas].to_numpy().mean(axis=1), index=df_h.index,
                     name=f"{var}_{subconjunto}")


if __name__ == "__main__":
    print("Descargando Rep20 horaria (20 ciudades, T/HR/TD, 1997→2020)...", flush=True)
    df = descargar_rep20_horaria()
    n_ciudades = len({columna.split("__")[0] for columna in df.columns})
    print(f"\nRep20 horaria: {n_ciudades} ciudades x {len(_HVARS)} variables | "
          f"{df.index.min()} -> {df.index.max()} | {len(df):,} h | "
          f"nulos={int(df.isna().sum().sum())}", flush=True)
    print(f"\nSubconjuntos definidos: {len(SUBCONJUNTOS)}")
    for nombre, ciudades in SUBCONJUNTOS.items():
        print(f"  {nombre:22} ({len(ciudades):2} ciudades)")
