"""Catálogo factorial horario de variantes de TEMPERATURA competibles (subconjunto x temporal x forma).

Ejecutar:  python -m src.features.catalogo_temperatura
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.features.temperatura import (
    RUTA_TEMP_REP20_H,
    SUBCONJUNTOS,
    descargar_rep20_horaria,
    serie_horaria_var,
)

DIR_CATALOGO = RAIZ / "resultados" / "catalogo" / "temperatura"
RUTA_REGISTRO = DIR_CATALOGO / "registro_temperatura.json"

FORMAS = ["cruda", "cuadratica", "hddcdd", "indice_calor", "humidex"]
TEMPORALES = ["crudo", "mm2d", "mm3d", "mm7d", "anomalia"]
_MM_DIAS = {"mm2d": 2, "mm3d": 3, "mm7d": 7}

BASE_HDDCDD = 18.0  # base estándar de grados-día en °C, para calefacción y refrigeración
K_ANOMALIA = 3  # años previos con los que se calcula la Normal estacional
VENTANA_DOY = 7  # días arriba y abajo del mismo día del año que entran en la Normal

INI_SALIDA = "2000-01-01"
FIN_SALIDA = "2020-02-29"


# Formas funcionales, a partir de los T/HR/TD crudos ya agregados del subconjunto
def _forma(T: pd.Series, HR: pd.Series, TD: pd.Series, forma: str) -> pd.DataFrame:
    """Aplica la forma física y devuelve un DataFrame con 1-2 columnas (índice = datetime completo)."""
    indice = T.index
    valores_t = T.to_numpy()
    if forma == "cruda":
        columnas = {"T": valores_t}
    elif forma == "cuadratica":
        columnas = {"T": valores_t, "T2": valores_t ** 2}
    elif forma == "hddcdd":
        columnas = {"HDD": np.maximum(BASE_HDDCDD - valores_t, 0.0),
                    "CDD": np.maximum(valores_t - BASE_HDDCDD, 0.0)}
    elif forma == "indice_calor":
        columnas = {"indice_calor": _indice_calor(valores_t, HR.to_numpy())}
    elif forma == "humidex":
        columnas = {"humidex": _humidex(valores_t, TD.to_numpy())}
    else:
        raise ValueError(f"forma desconocida: {forma}")
    return pd.DataFrame(columnas, index=indice)


def _indice_calor(T: np.ndarray, HR: np.ndarray) -> np.ndarray:
    """Índice de calor NWS (Rothfusz) en °C. Para Tf<80 °F devuelve la propia T (sin corrección)."""
    Tf = T * 9.0 / 5.0 + 32.0
    HI = (-42.379 + 2.04901523 * Tf + 10.14333127 * HR - 0.22475541 * Tf * HR
          - 6.83783e-3 * Tf ** 2 - 5.481717e-2 * HR ** 2 + 1.22874e-3 * Tf ** 2 * HR
          + 8.5282e-4 * Tf * HR ** 2 - 1.99e-6 * Tf ** 2 * HR ** 2)
    HI = np.where(Tf < 80.0, Tf, HI)
    return (HI - 32.0) * 5.0 / 9.0


def _humidex(T: np.ndarray, TD: np.ndarray) -> np.ndarray:
    """Humidex canadiense en °C, a partir de temperatura y punto de rocío (TD, °C)."""
    presion_vapor = 6.11 * np.exp(5417.7530 * (1.0 / 273.16 - 1.0 / (273.15 + TD)))
    return T + 0.5555 * (presion_vapor - 10.0)


# Transformaciones temporales, que se aplican a cada columna de la forma por separado
def _temporal(serie: pd.Series, temporal: str) -> pd.Series:
    if temporal == "crudo":
        return serie
    if temporal in _MM_DIAS:
        dias = _MM_DIAS[temporal]
        # media móvil causal de la MISMA hora en los últimos k días (no aplana el ciclo diario)
        return serie.groupby(serie.index.hour).transform(
            lambda serie_hora: serie_hora.rolling(dias, min_periods=1).mean())
    if temporal == "anomalia":
        return _anomalia(serie)
    raise ValueError(f"temporal desconocido: {temporal}")


def _anomalia(serie: pd.Series) -> pd.Series:
    """Anomalía estacional horaria = valor − Normal(hora, día-del-año) de los K años anteriores."""
    valores = serie.to_numpy(dtype=float)
    anios = serie.index.year.to_numpy()
    horas = serie.index.hour.to_numpy()
    dia_anio = serie.index.dayofyear.to_numpy() - 1  # 0..365
    anios_unicos = np.unique(anios)
    indice_anio = {int(anio): i for i, anio in enumerate(anios_unicos)}
    n_anios = len(anios_unicos)
    pos_anio = np.array([indice_anio[int(anio)] for anio in anios])

    cubo = np.full((n_anios, 24, 366), np.nan)
    cubo[pos_anio, horas, dia_anio] = valores
    valido = ~np.isnan(cubo)
    cubo_relleno = np.where(valido, cubo, 0.0)

    # El eje doy se recorre en circular para que la ventana de la Normal no se corte en enero
    # ni en diciembre.
    suma = np.zeros_like(cubo_relleno)
    conteo = np.zeros_like(cubo_relleno)
    for desplazamiento in range(-VENTANA_DOY, VENTANA_DOY + 1):
        suma += np.roll(cubo_relleno, desplazamiento, axis=2)
        conteo += np.roll(valido.astype(float), desplazamiento, axis=2)

    normal = np.full((n_anios, 24, 366), np.nan)
    for i, anio in enumerate(anios_unicos):
        previos = [indice_anio.get(int(anio) - k) for k in range(1, K_ANOMALIA + 1)]
        previos = [j for j in previos if j is not None]
        if not previos:
            continue
        suma_previos = suma[previos].sum(axis=0)
        conteo_previos = conteo[previos].sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            normal[i] = np.where(conteo_previos > 0, suma_previos / conteo_previos, np.nan)

    normal_valores = normal[pos_anio, horas, dia_anio]
    return pd.Series(valores - normal_valores, index=serie.index, name=serie.name)


def construir_variante(T, HR, TD, temporal: str, forma: str) -> pd.DataFrame:
    """DataFrame horario de UNA variante: forma física → temporal por columna → recorte pre-COVID."""
    fisica = _forma(T, HR, TD, forma)
    salida = pd.DataFrame(index=fisica.index)
    for columna in fisica.columns:
        salida[columna] = _temporal(fisica[columna], temporal)

    salida = salida.loc[INI_SALIDA:FIN_SALIDA].astype("float32")
    salida.index.name = "datetime"
    return salida


def nombre_variante(subconjunto: str, temporal: str, forma: str) -> str:
    """Nombre explicativo y estable: temp_<sub>_<temporal>_<forma>."""
    return f"temp_{subconjunto}_{temporal}_{forma}"


def _descripcion(subconjunto, temporal, forma) -> str:
    forma_txt = {
        "cruda": "cruda",
        "cuadratica": "cuadrática (T y T²)",
        "hddcdd": f"grados-día HDD/CDD base {BASE_HDDCDD:g} °C",
        "indice_calor": "índice de calor NWS (T+HR)",
        "humidex": "humidex (T+TD)",
    }[forma]
    temporal_txt = {
        "crudo": "sin suavizado",
        "mm2d": "media móvil misma-hora 2 días",
        "mm3d": "media móvil misma-hora 3 días",
        "mm7d": "media móvil misma-hora 7 días",
        "anomalia": f"anomalía estacional (±{VENTANA_DOY} días, {K_ANOMALIA} años previos)",
    }[temporal]
    return f"Temperatura horaria {subconjunto}, {forma_txt}, {temporal_txt}."


def generar_catalogo(df_h: pd.DataFrame | None = None, guardar: bool = True) -> list[dict]:
    """Genera todas las variantes (subconjunto x temporal x forma) + el registro JSON, desde la caché."""
    if df_h is None:
        if not RUTA_TEMP_REP20_H.exists():
            raise FileNotFoundError(
                f"No existe {RUTA_TEMP_REP20_H}. El catálogo NO re-descarga: genera primero el "
                "parquet horario con `python -m src.features.temperatura`.")
        df_h = descargar_rep20_horaria()  # lee el parquet cacheado, no descarga si ya existe

    if guardar:
        DIR_CATALOGO.mkdir(parents=True, exist_ok=True)
        # Limpia los ficheros huérfanos de catálogos anteriores, para que en la carpeta queden
        # exactamente las variantes vigentes.
        vigentes = {f"{nombre_variante(sub, temporal, forma)}.parquet"
                    for sub in SUBCONJUNTOS for temporal in TEMPORALES for forma in FORMAS}
        for ruta_csv in DIR_CATALOGO.glob("temp_*.csv"):
            ruta_csv.unlink()  # el catálogo pasó a parquet, ningún CSV sigue vigente
        for ruta_parquet in DIR_CATALOGO.glob("temp_*.parquet"):
            if ruta_parquet.name not in vigentes:
                ruta_parquet.unlink()

    registro: list[dict] = []
    for sub in SUBCONJUNTOS:
        # T/HR/TD agregados con media uniforme; se calculan una sola vez por subconjunto.
        T = serie_horaria_var(df_h, sub, "T")
        HR = serie_horaria_var(df_h, sub, "HR")
        TD = serie_horaria_var(df_h, sub, "TD")
        for temporal in TEMPORALES:
            for forma in FORMAS:
                variante = construir_variante(T, HR, TD, temporal, forma)
                nombre = nombre_variante(sub, temporal, forma)
                fichero = f"{nombre}.parquet"
                if guardar:
                    variante.to_parquet(DIR_CATALOGO / fichero)
                registro.append({
                    "nombre": nombre,
                    "fichero": fichero,
                    "subconjunto": sub,
                    "n_ciudades": len(SUBCONJUNTOS[sub]),
                    "temporal": temporal,
                    "forma": forma,
                    "columnas": list(variante.columns),
                    "descripcion": _descripcion(sub, temporal, forma),
                })
    if guardar:
        meta = {
            "familia": "temperatura",
            "agregacion_ciudades": "media simple entre ciudades",
            "periodo": [INI_SALIDA, FIN_SALIDA],
            "indice": "datetime (horario)",
            "base_hddcdd_celsius": BASE_HDDCDD,
            "anomalia": {"anios_previos": K_ANOMALIA, "ventana_doy": VENTANA_DOY},
            "ejes": {
                "subconjunto": list(SUBCONJUNTOS.keys()),
                "temporal": TEMPORALES,
                "forma": FORMAS,
            },
            "n_variantes": len(registro),
            "variantes": registro,
        }
        RUTA_REGISTRO.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return registro


if __name__ == "__main__":
    import time
    inicio = time.time()
    registro = generar_catalogo()
    segundos = time.time() - inicio
    print(f"Catálogo de temperatura: {len(registro)} variantes en {DIR_CATALOGO}  ({segundos:.1f}s)")
    ejes = {eje: sorted({entrada[eje] for entrada in registro})
            for eje in ("subconjunto", "temporal", "forma")}
    for eje, valores in ejes.items():
        print(f"  {eje:12} ({len(valores):2}): {valores}")
    ejemplo = registro[0]
    print(f"Ejemplo: {ejemplo['fichero']} -> columnas {ejemplo['columnas']}")
