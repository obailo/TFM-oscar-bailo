"""
Ejecutar:  .venv/bin/python -m src.features.catalogo_lags
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

from src.datos_precovid import serie_precovid
from src.features.por_hora import series_por_hora

DIR_CAT = RAIZ / "resultados" / "catalogo" / "lags"
DIR_UNICA = DIR_CAT / "serie_unica"
DIR_HORARIA = DIR_CAT / "horaria"


LAGS_HORA = (
    1, 2, 3, 4, 5, 6,
    12, 18, 24, 25, 26,
    48, 49, 72, 96, 120, 
    144, 168, 169, 167, 192, 216,
    336, 337, 504, 672, 720,
)
VENTANAS_HORA = (24, 48, 168, 336, 720)

LAGS_DIA = (1, 2, 3, 4, 5, 6, 7, 8, 14, 21, 28)
VENTANAS_DIA = (7, 14, 28)

FUNCS = {"media": "mean", "std": "std", "min": "min", "max": "max"}


# 1) Serie única (resolución horaria)
def lags_serie_unica(serie: pd.Series, lags=LAGS_HORA) -> pd.DataFrame:
    """lag_h{k} = y(t-k), k en horas (pasado estricto, shift(k))."""
    return pd.DataFrame({f"lag_h{k}": serie.shift(k) for k in lags}, index=serie.index)


def ventanas_serie_unica(serie: pd.Series, ventanas=VENTANAS_HORA) -> pd.DataFrame:
    """Estadísticos móviles (media/std/min/max) en ventanas que terminan en t-1 (fin_t1) o t-24 (fin_t24)."""
    salida = {}
    base_t1 = serie.shift(1)
    base_t24 = serie.shift(24)  # termina en t-24, que es lo legal para day-ahead
    for ventana in ventanas:
        rolling_t1, rolling_t24 = base_t1.rolling(ventana), base_t24.rolling(ventana)
        for nombre, metodo in FUNCS.items():
            salida[f"roll{nombre}_{ventana}h_fin_t1"] = getattr(rolling_t1, metodo)()
            salida[f"roll{nombre}_{ventana}h_fin_t24"] = getattr(rolling_t24, metodo)()
    return pd.DataFrame(salida, index=serie.index)


# 2) Estructura horaria: 24 series diarias, con lags y ventanas en días
def tabla_horaria(serie: pd.Series, lags_dia=LAGS_DIA, ventanas_dia=VENTANAS_DIA) -> pd.DataFrame:
    """Tabla larga por (fecha_origen, hora) con lags y ventanas en DÍAS, sin objetivo (se añade después)."""
    por_hora = series_por_hora(serie)  # filas = día, columnas h00..h23
    piezas = []
    for hora in range(24):
        columna = f"h{hora:02d}"
        serie_hora = por_hora[columna]
        bloque = pd.DataFrame({"hora": hora}, index=por_hora.index)
        # Los lags en días se cuentan antes del objetivo, que es D+1: con k=1 el shift es 0, o sea
        # el propio día-origen D, que en la serie única sería t-24.
        for k in lags_dia:
            bloque[f"lag_{k}d"] = serie_hora.shift(k - 1)
        # Las ventanas en días terminan en D, nunca más allá del origen, así que son legales para
        # day-ahead: la ventana de w días [D-w+1 … D] no incluye el objetivo D+1.
        for ventana in ventanas_dia:
            rolling = serie_hora.rolling(ventana)
            for nombre, metodo in FUNCS.items():
                bloque[f"roll{nombre}_{ventana}d"] = getattr(rolling, metodo)()
        piezas.append(bloque)
    df = pd.concat(piezas)
    df.index.name = "fecha_origen"
    return df.reset_index().sort_values(["fecha_origen", "hora"]).reset_index(drop=True)


def _registro_serie_unica() -> list[dict]:
    registro = []
    for k in LAGS_HORA:
        registro.append({
            "nombre": f"lag_h{k}", "fichero": "serie_unica/lags.parquet",
            "estructura": "serie_unica", "definicion": f"y(t-{k})",
            "lag_horas": k, "ventana": None,
            # con lag de 24 o más, cualquier hora de D+1 mira como mucho al último dato de D
            "legal_dayahead": bool(k >= 24),
            "legal_h2448": True, 
        })
    for ventana in VENTANAS_HORA:
        for estadistico in FUNCS:
            registro.append({
                "nombre": f"roll{estadistico}_{ventana}h_fin_t1",
                "fichero": "serie_unica/ventanas.parquet",
                "estructura": "serie_unica",
                "definicion": f"{estadistico} móvil de {ventana} h terminando en t-1",
                "lag_horas": 1, "ventana": ventana,
                "legal_dayahead": False,  # termina en t-1, o sea horas de D posteriores al origen
                "legal_h2448": True,
            })
            registro.append({
                "nombre": f"roll{estadistico}_{ventana}h_fin_t24",
                "fichero": "serie_unica/ventanas.parquet",
                "estructura": "serie_unica",
                "definicion": f"{estadistico} móvil de {ventana} h terminando en t-24",
                "lag_horas": 24, "ventana": ventana,
                "legal_dayahead": True,  # termina en t-24, legal para cualquier hora de D+1
                "legal_h2448": True,
            })
    return registro


def _registro_horaria() -> list[dict]:
    registro = []
    for k in LAGS_DIA:
        registro.append({
            "nombre": f"lag_{k}d", "fichero": "horaria/tabla.parquet", "estructura": "horaria",
            "definicion": f"demanda de la misma hora {k} días antes del objetivo (k=1 ⇒ día-origen D)",
            "lag_dias": k, "lag_horas_equiv": k * 24, "ventana": None,
            "legal_dayahead": True, "legal_h2448": True,
        })
    for ventana in VENTANAS_DIA:
        for estadistico in FUNCS:
            registro.append({
                "nombre": f"roll{estadistico}_{ventana}d", "fichero": "horaria/tabla.parquet",
                "estructura": "horaria",
                "definicion": (f"{estadistico} móvil de la misma hora sobre los {ventana} días "
                               f"que terminan en D"),
                "lag_dias": 1, "lag_horas_equiv": 24, "ventana": ventana,
                "legal_dayahead": True, "legal_h2448": True,
            })
    return registro


# Verificación anti-leakage explícita
def verificar(serie: pd.Series, tabla_unica: pd.DataFrame, tabla_por_hora: pd.DataFrame) -> bool:
    ok = True
    print("\nAnti-leakage:")

    # (1) Serie única: lag_h{k}(t) == s(t-k) para varios k y varios t
    for k in (1, 24, 168, 336):
        for instante in (pd.Timestamp("2010-06-10 13:00"), pd.Timestamp("2015-12-25 03:00")):
            esperado = serie.get(instante - pd.Timedelta(hours=k), np.nan)
            obtenido = tabla_unica.loc[instante, f"lag_h{k}"]
            correcto = ((np.isnan(esperado) and np.isnan(obtenido))
                        or abs(esperado - obtenido) < 1e-6)
            ok &= correcto
            print(f"  [u] lag_h{k}({instante}) == s(t-{k})? {correcto}")

    # (2) Serie única: ventana fin_t1 NO usa y(t) ni el futuro (media de w=24 == mean(s[t-24:t-1]))
    instante = pd.Timestamp("2012-03-15 10:00")
    esperado = serie.loc[instante - pd.Timedelta(hours=24): instante - pd.Timedelta(hours=1)].mean()
    obtenido = tabla_unica.loc[instante, "rollmedia_24h_fin_t1"]
    correcto = abs(esperado - obtenido) < 1e-6
    ok &= correcto
    print(f"  [u] rollmedia_24h_fin_t1({instante}) == mean(s[t-24..t-1])? {correcto}")
    # robustez: perturbar y(t) NO cambia la ventana fin_t1 (no incluye t)
    serie_perturbada = serie.copy(); serie_perturbada.loc[instante] = serie_perturbada.loc[instante] + 1e6
    valor_perturbado = ventanas_serie_unica(serie_perturbada)["rollmedia_24h_fin_t1"].loc[instante]
    correcto = abs(valor_perturbado - obtenido) < 1e-6
    ok &= correcto
    print(f"  [u] perturbar y(t) no altera rollmedia_24h_fin_t1(t)? {correcto}")

    # (3) Serie única: ventana fin_t24 termina en t-24 (legal day-ahead): == mean(s[t-47:t-24])
    esperado = serie.loc[instante - pd.Timedelta(hours=47): instante - pd.Timedelta(hours=24)].mean()
    obtenido = tabla_unica.loc[instante, "rollmedia_24h_fin_t24"]
    correcto = abs(esperado - obtenido) < 1e-6
    ok &= correcto
    print(f"  [u] rollmedia_24h_fin_t24({instante}) == mean(s[t-47..t-24])? {correcto}")

    # (4) Horaria: lag_1d (k=1) de (origen D, hora h) == demanda(D, h) == t-24 serie única
    dia_origen = pd.Timestamp("2015-06-10")
    fila = tabla_por_hora[(tabla_por_hora.fecha_origen == dia_origen)
                          & (tabla_por_hora.hora == 13)].iloc[0]
    correcto = abs(fila["lag_1d"] - float(serie.loc[dia_origen + pd.Timedelta(hours=13)])) < 1e-6
    ok &= correcto
    print(f"  [h] lag_1d(D=2015-06-10,h13) == demanda(D 13:00)? {correcto}")
    # lag_7d == demanda(D-6, h13) == t-168
    correcto = abs(fila["lag_7d"] - float(
        serie.loc[dia_origen - pd.Timedelta(days=6) + pd.Timedelta(hours=13)])) < 1e-6
    ok &= correcto
    print(f"  [h] lag_7d(D,h13) == demanda(D-6 13:00)? {correcto}")

    # (5) Horaria: ventana rollmedia_7d termina en D (no usa el objetivo D+1).
    serie_h13 = series_por_hora(serie)["h13"]
    esperado = serie_h13.loc[dia_origen - pd.Timedelta(days=6): dia_origen].mean()
    correcto = abs(fila["rollmedia_7d"] - esperado) < 1e-6
    ok &= correcto
    print(f"  [h] rollmedia_7d(D,h13) == mean(demanda h13 de [D-6..D])? {correcto}")

    # (6) Ninguna feature del catálogo llega hasta el objetivo ni lo pasa: en la serie única todos
    # los lags son k>=1 y las ventanas llevan shift>=1, así que como mucho ven t-1; en la horaria
    # todo se queda en D y el objetivo es D+1.
    print(f"  min(k) serie_unica={min(LAGS_HORA)} h | min(k) horaria={min(LAGS_DIA)} d")
    print(f"  {'OK' if ok else 'FALLO: revisar'}")
    return ok


def generar() -> dict:
    serie = serie_precovid()
    DIR_UNICA.mkdir(parents=True, exist_ok=True)
    DIR_HORARIA.mkdir(parents=True, exist_ok=True)

    # Lags y ventanas van a ficheros separados porque cada entrada del registro apunta a uno
    # de los dos.
    lags_unica = lags_serie_unica(serie)
    ventanas_unica = ventanas_serie_unica(serie)
    lags_unica.to_parquet(DIR_UNICA / "lags.parquet")
    ventanas_unica.to_parquet(DIR_UNICA / "ventanas.parquet")
    tabla_unica = pd.concat([lags_unica, ventanas_unica], axis=1)

    tabla_por_hora = tabla_horaria(serie)
    tabla_por_hora.to_parquet(DIR_HORARIA / "tabla.parquet")

    registro = {
        "split": "src.features.validacion.split_canonico_precovid",
        "horizontes": {"day_ahead": "D+1 (24 h)", "24_48h": "D+2"},
        "anti_leakage": "lags shift(k≥1); horaria todo ≤ día-origen D. Verificado.",
        "estructuras": {
            "serie_unica": {
                "resolucion": "horaria", "indice": "datetime",
                "ficheros": ["serie_unica/lags.parquet", "serie_unica/ventanas.parquet"],
                "n_lags": len(LAGS_HORA), "lags_horas": list(LAGS_HORA),
                "ventanas_horas": list(VENTANAS_HORA),
                "nota_dayahead": "lag_h<24 y ventanas fin_t1: solo D+2.",
            },
            "horaria": {
                "resolucion": "24 series diarias (1 por hora)", "indice": "(fecha_origen, hora)",
                "ficheros": ["horaria/tabla.parquet"],
                "n_lags": len(LAGS_DIA), "lags_dias": list(LAGS_DIA),
                "ventanas_dias": list(VENTANAS_DIA),
                "equivalencia": "lag_kd ≡ t-(24·k) de la misma hora.",
            },
        },
        "features": _registro_serie_unica() + _registro_horaria(),
    }
    (DIR_CAT / "registro_lags.json").write_text(
        json.dumps(registro, indent=2, ensure_ascii=False))

    ok = verificar(serie, tabla_unica, tabla_por_hora)
    if not ok:
        # Si algún chequeo anti-leakage falla, el catálogo no debe usarse: contaminaría el
        # modelado con información futura. Mejor abortar aquí.
        raise RuntimeError(
            "Anti-leakage FALLIDO en catalogo_lags.verificar(): alguna feature usaría "
            "información ≥ objetivo (fuga de datos). Se aborta la generación del catálogo.")

    print("\nFicheros generados:")
    print(f"  serie_unica/lags.parquet     {lags_unica.shape[0]:,} x {lags_unica.shape[1]}")
    print(f"  serie_unica/ventanas.parquet {ventanas_unica.shape[0]:,} x {ventanas_unica.shape[1]}")
    print(f"  horaria/tabla.parquet        {tabla_por_hora.shape[0]:,} x {tabla_por_hora.shape[1]} "
          f"(= {tabla_por_hora.fecha_origen.nunique():,} días x 24 h)")
    print(f"  registro_lags.json           {len(registro['features'])} features")
    n_dayahead = sum(feature["legal_dayahead"] for feature in registro["features"])
    print(f"  legal_dayahead: {n_dayahead}/{len(registro['features'])} features")
    return {"ok": ok, "tabla_unica": tabla_unica, "tabla_por_hora": tabla_por_hora, "registro": registro}


if __name__ == "__main__":
    generar()
