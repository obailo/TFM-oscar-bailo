"""Descarga de ENTSO-E la previsión oficial de demanda a un día vista (REE) y calcula su error en test.

Ejecutar:  python -m src.externo.prevision_ree            # descarga (o usa la caché) y calcula el error
           python -m src.externo.prevision_ree --forzar   # vuelve a descargar aunque exista la caché
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.metricas import ancla_mase

DIR_MET = RAIZ / "resultados" / "metricas"
CACHE = DIR_MET / "entsoe_prevision_ree_test.parquet"

URL = "https://transparency.entsoe.eu/load/total/dayAhead/load"
ZONA = "BZN|10YES-REE------0"
HEADERS = {
    "accept": "application/json",
    "content-type": "application/json; charset=utf-8",
    "origin": "https://transparency.entsoe.eu",
    "referer": "https://transparency.entsoe.eu/load/total/dayAhead",
    "user-agent": ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/149.0.0.0 Safari/537.36"),
    "cookie": "uu.app.bpl=0; cookieConsent=true",
}

TEST_INI = "2018-12-31 23:00"
TEST_FIN = "2020-02-29 23:00"
DESFASE_H = 1


def _tramo(desde: str, hasta: str) -> dict:
    body = json.dumps({"dateTimeRange": {"from": desde, "to": hasta}, "areaList": [ZONA],
                         "timeZone": "CET", "sorterList": [], "filterMap": {}}).encode()
    peticion = urllib.request.Request(URL, data=body, method="POST")
    for clave, valor in HEADERS.items():
        peticion.add_header(clave, valor)
    return json.loads(urllib.request.urlopen(peticion, timeout=90).read())


def descargar(ini: str = TEST_INI, fin: str = TEST_FIN, dias: int = 7,
              cache: Path | None = None) -> pd.DataFrame:
    """Previsión a un día vista y demanda real, descargadas en tramos de `dias` y cacheadas en parquet."""
    cache = cache or CACHE
    filas, fallos = [], []
    actual, tope = pd.Timestamp(ini, tz="UTC"), pd.Timestamp(fin, tz="UTC")
    while actual < tope:
        hasta = min(actual + pd.Timedelta(days=dias), tope)
        desde_str, hasta_str = (t.strftime("%Y-%m-%dT%H:%M:%S.000Z") for t in (actual, hasta))
        try:
            datos = _tramo(desde_str, hasta_str)
            for instancia in datos.get("instanceList", []):
                for periodo in instancia["curveData"]["periodList"]:
                    inicio = pd.Timestamp(periodo["timeInterval"]["from"])
                    paso = pd.Timedelta(minutes=60 if periodo["resolution"] == "PT60M" else 15)
                    for clave, valores in periodo["pointMap"].items():
                        if valores and len(valores) > 1 and valores[0] and valores[1]:
                            filas.append({"dt": inicio + int(clave) * paso,
                                          "prevision": float(valores[0]), "real": float(valores[1])})
        except Exception as error:
            fallos.append((desde_str[:10], f"{type(error).__name__} {getattr(error, 'code', '')}"))
        actual = hasta
        time.sleep(0.7)
    if fallos:
        print(f"  {len(fallos)} tramos fallidos: {fallos[:5]}", flush=True)
    df = pd.DataFrame(filas).drop_duplicates("dt").sort_values("dt").reset_index(drop=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    print(f"  {len(df)} puntos, de {df.dt.min()} a {df.dt.max()}, guardados en {cache.name}", flush=True)
    return df


def cargar(forzar: bool = False) -> pd.DataFrame:
    if CACHE.exists() and not forzar:
        return pd.read_parquet(CACHE)
    return descargar()


def comparar_con_serie_propia(df: pd.DataFrame) -> dict:
    """Comprueba que la demanda real de ENTSO-E y la del CSV del trabajo son la misma serie."""
    propia = pd.read_csv(RAIZ / "Peninsula_precovid.csv", parse_dates=["datetime"])
    # +1 h: ENTSO-E etiqueta el intervalo con otro criterio que el CSV de REE. Medido probando desfases de
    # -2 a +2 h, la correlación pasa de 0,910 sin corregir a 0,980 con +1 h (y baja a 0,967 con +2 h).
    entsoe = df.assign(dt=pd.to_datetime(df["dt"]).dt.tz_convert("UTC").dt.tz_localize(None)
                       + pd.Timedelta(hours=DESFASE_H))
    cruce = entsoe.merge(propia.rename(columns={"datetime": "dt", "Demanda": "propia"}), on="dt", how="inner")
    if cruce.empty:
        return {"solapamiento": 0, "aviso": "sin instantes comunes: revisar husos horarios"}
    return {"solapamiento": len(cruce),
            "correlacion": round(float(np.corrcoef(cruce["real"], cruce["propia"])[0, 1]), 4),
            "dif_media_mw": round(float((cruce["real"] - cruce["propia"]).mean()), 1),
            "dif_abs_media_mw": round(float((cruce["real"] - cruce["propia"]).abs().mean()), 1)}


def error_operador(df: pd.DataFrame) -> dict:
    """MAE, MAPE y MASE de la previsión del operador, con el mismo ancla que el resto del trabajo."""
    errores = (df["real"] - df["prevision"]).abs()
    ancla = ancla_mase()
    return {"n": int(len(df)),
            "MAE_mw": round(float(errores.mean()), 1),
            "MAPE_pct": round(float((errores / df["real"].abs()).mean() * 100), 3),
            "MASE": round(float(errores.mean() / ancla), 4),
            "sesgo_mw": round(float((df["real"] - df["prevision"]).mean()), 1)}


def main(argv: list[str] | None = None) -> dict:
    argumentos = list(argv if argv is not None else sys.argv[1:])
    print("Previsión oficial de REE (vía ENTSO-E)", flush=True)
    df = cargar(forzar="--forzar" in argumentos)
    coherencia = comparar_con_serie_propia(df)
    print(f"\ncoherencia con la serie propia: {coherencia}", flush=True)
    error = error_operador(df)
    print(f"error del operador en test: {error}", flush=True)
    salida = {"fuente": "ENTSO-E Transparency Platform, Total Load Day-Ahead / Actual",
              "zona": ZONA, "periodo": [str(df["dt"].min()), str(df["dt"].max())],
              "coherencia_con_serie_propia": coherencia, "error_operador": error,
              "caveat": (
                  "La carga de la zona de oferta de ENTSO-E NO es la demanda peninsular de REE "
                  "(correlación 0,980, diferencia absoluta media ~630 MW). El MASE de esta tabla no es "
                  "comparable con el de los modelos: sirve solo como orden de magnitud.")}
    ruta = DIR_MET / "referencia_operador_ree.json"
    ruta.write_text(json.dumps(salida, indent=2, ensure_ascii=False))
    print(f"\nGuardado en {ruta}", flush=True)
    return salida


if __name__ == "__main__":
    main()
