from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.datos_precovid import serie_precovid
from src.features.por_hora import series_por_hora, objetivo_dayahead
from src.features.calendario import features_calendario, festivo_regional_ponderado, es_verano, es_navidad
from src.features.catalogo_calendario import FRANJAS_VERANO, FRANJAS_NAVIDAD
from src.features.validacion import split_canonico_precovid
from src.features import catalogo_loader

LAGS_ESTRUCTURA = "horaria"
LAGS_DIAS = (1, 2, 7, 14)

_SEL = None
_TEMP_DIA = None


def _seleccionadas() -> dict:
    global _SEL
    if _SEL is None:
        _SEL = catalogo_loader.exogenas_seleccionadas()
    return _SEL


def _franja(nombre_catalogo: str, franjas: dict):
    """(ini, fin) de la franja seleccionada, a partir de su nombre de catálogo (p.ej. esVerano_15jun_15sep)."""
    clave = nombre_catalogo.split("_", 1)[1]
    return franjas[clave]


def _lags_legales(n_dias: int) -> list[int]:
    """Días de lag legales para el horizonte: intersección de LAGS_DIAS con el catálogo filtrado."""
    filtro = {"solo_legales_dayahead": True} if n_dias == 1 else {"solo_legales_h2448": True}
    legales = set(catalogo_loader.listar("lags", estructura=LAGS_ESTRUCTURA, **filtro))
    return [k for k in LAGS_DIAS if f"lag_{k}d" in legales]


def _temp_horaria_por_dia() -> pd.DataFrame:
    """Temperatura ganadora (horaria) pivotada a día x (temp_<col>_hHH), con cache."""
    global _TEMP_DIA
    if _TEMP_DIA is None:
        df = catalogo_loader.temperatura_ganadora_horaria()
        piezas = []
        for columna in df.columns:
            pivotada = df[columna].groupby([df.index.normalize(), df.index.hour]).first().unstack()
            pivotada.columns = [f"temp_{columna}_h{hora:02d}" for hora in pivotada.columns]
            piezas.append(pivotada)
        _TEMP_DIA = pd.concat(piezas, axis=1)
        _TEMP_DIA.index.name = "fecha"
    return _TEMP_DIA


def _exogenas_dia(fechas_obj: pd.DatetimeIndex, sufijo: str = "") -> pd.DataFrame:
    """Exógenas de los días objetivo (D+d): calendario, esVerano/esNavidad y las 24 T horarias."""
    seleccionadas = _seleccionadas()
    ver_ini, ver_fin = _franja(seleccionadas["esVerano"], FRANJAS_VERANO)
    nav_ini, nav_fin = _franja(seleccionadas["esNavidad"], FRANJAS_NAVIDAD)
    fechas_obj = pd.DatetimeIndex(fechas_obj)
    calendario = features_calendario(fechas_obj, incluir_ciclicas=True)
    X = pd.DataFrame(index=range(len(fechas_obj)))
    for columna in ("dia_semana_sin", "dia_semana_cos", "mes_sin", "mes_cos"):
        X[columna] = calendario[columna].to_numpy()
    X["finde"] = calendario["finde"].to_numpy()
    X["festivo_reg"] = festivo_regional_ponderado(fechas_obj).to_numpy()  # subsume el nacional
    X["esVerano"] = es_verano(fechas_obj, ini=ver_ini, fin=ver_fin).to_numpy()
    X["esNavidad"] = es_navidad(fechas_obj, ini=nav_ini, fin=nav_fin).to_numpy()
    temperatura = _temp_horaria_por_dia().reindex(fechas_obj.normalize())  # 24 h de la variante ganadora
    for columna in temperatura.columns:
        X[columna] = temperatura[columna].to_numpy()
    if sufijo:
        X.columns = [f"{columna}{sufijo}" for columna in X.columns]
    return X


def construir_xy(n_dias: int = 1):
    """Tabla por día-origen D con features + 24·n_dias objetivos, sin NaN: devuelve (datos, columnas_x, columnas_y)."""
    serie = serie_precovid()  # fuente ÚNICA pre-COVID (< 2020-03-01)
    panel_horario = series_por_hora(serie)
    y = objetivo_dayahead(serie, n_dias=n_dias)  # 24*n_dias columnas, indexadas por el día-origen D
    y.index = panel_horario.index


    bloques = []
    for k in _lags_legales(n_dias):
        perfil = panel_horario.shift(k - 1)
        perfil.columns = [f"lag{k}d_h{hora:02d}" for hora in range(24)]
        bloques.append(perfil)
    lags = pd.concat(bloques, axis=1)

    # Exógenas seleccionadas de cada día objetivo (D+1 hasta D+n_dias).
    exogenas = []
    for dia in range(1, n_dias + 1):
        fechas_objetivo = pd.DatetimeIndex(panel_horario.index + pd.Timedelta(days=dia))
        exogenas_dia = _exogenas_dia(fechas_objetivo, sufijo=f"_d{dia}")
        exogenas_dia.index = panel_horario.index
        exogenas.append(exogenas_dia)
    X_exogenas = pd.concat(exogenas, axis=1)

    X = pd.concat([lags, X_exogenas], axis=1)
    columnas_x, columnas_y = list(X.columns), list(y.columns)
    datos = pd.concat([X, y], axis=1).dropna()
    return datos, columnas_x, columnas_y


def escalar_tabla_mlp(datos, columnas_x, columnas_y, fechas_train, fechas_val,
                      fechas_test=None) -> dict:
    """Escalado z-score anti-fuga de la tabla del MLP: los estadísticos salen SOLO del train indicado.

    `fechas_train`/`fechas_val`/`fechas_test` son los días-origen de cada tramo (del split canónico o
    de un corte del rolling-origin); sin `fechas_test` la máscara de test queda vacía.
    """
    dias_origen = datos.index
    mascaras = {"tr": dias_origen.isin(fechas_train), "va": dias_origen.isin(fechas_val),
                "te": (dias_origen.isin(fechas_test) if fechas_test is not None
                       else np.zeros(len(dias_origen), bool))}
    # Continuas a escalar = lags (perfiles de demanda) + temperatura horaria (temp_*_d{d}).
    # Cíclicas (sin/cos in [-1,1]) y binarias (finde, festivo_reg, esVerano, esNavidad) NO se escalan.
    continuas = [columna for columna in columnas_x
                 if columna.startswith("lag") or columna.startswith("temp_")]
    medias = datos.loc[mascaras["tr"], continuas].mean()
    desviaciones = datos.loc[mascaras["tr"], continuas].std().replace(0, 1.0)
    y_train = datos.loc[mascaras["tr"], columnas_y].to_numpy()
    y_media, y_desv = float(y_train.mean()), float(y_train.std())
    X_escalada = datos[columnas_x].copy()
    X_escalada[continuas] = (X_escalada[continuas] - medias) / desviaciones
    Y_escalada = (datos[columnas_y] - y_media) / y_desv
    return {"X": X_escalada, "Y": Y_escalada, "mascaras": mascaras,
            "y_media": y_media, "y_desv": y_desv}


def split_y_escalar(datos, columnas_x, columnas_y, n_dias: int = 1) -> dict:
    """Split temporal canónico + escalado solo-train (`escalar_tabla_mlp`), con purga de `n_dias`."""
    split = split_canonico_precovid(gap_dias=n_dias, guardar=False)
    return escalar_tabla_mlp(datos, columnas_x, columnas_y, split.train, split.val, split.test)


def secuencias_lstm(n_dias: int = 1, L: int = 14):
    """Secuencias LSTM: últimos L días por origen D → (X (n,L,24), Y (n,24·n_dias), fechas_origen)."""
    serie = serie_precovid()
    panel_horario = series_por_hora(serie)
    y = objetivo_dayahead(serie, n_dias=n_dias)
    y.index = panel_horario.index
    perfiles = panel_horario.to_numpy()
    objetivos = y.to_numpy()
    fechas = panel_horario.index
    secuencias, objetivos_validos, fechas_origen = [], [], []
    for i in range(L - 1, len(fechas)):
        ventana = perfiles[i - L + 1: i + 1]  # los L días que terminan en D, de forma (L, 24)
        objetivo = objetivos[i]
        if np.isnan(ventana).any() or np.isnan(objetivo).any():
            continue
        secuencias.append(ventana); objetivos_validos.append(objetivo); fechas_origen.append(fechas[i])
    return np.stack(secuencias), np.stack(objetivos_validos), pd.DatetimeIndex(fechas_origen)


def secuencias_lstm_exo(n_dias: int = 1, L: int = 14):
    """Como `secuencias_lstm` más las exógenas del día objetivo: (Xseq, Xexo, Y, fechas)."""
    Xseq, Y, fechas = secuencias_lstm(n_dias=n_dias, L=L)
    bloques = []
    for dia in range(1, n_dias + 1):
        fechas_objetivo = pd.DatetimeIndex(fechas + pd.Timedelta(days=dia))
        bloques.append(_exogenas_dia(fechas_objetivo, sufijo=f"_d{dia}").to_numpy())
    Xexo = np.concatenate(bloques, axis=1).astype("float32")
    mascara = ~np.isnan(Xexo).any(axis=1)  # descarta orígenes con exógenas fuera de rango
    return Xseq[mascara], Xexo[mascara], Y[mascara], fechas[mascara]


def verificar_pipeline() -> dict:
    """Verificación SIN torch de ambos horizontes: shapes, 0 NaN y anti-leakage. Devuelve un dict-resumen."""
    salida = {}
    for n_dias, etiqueta in ((1, "day-ahead D+1"), (2, "24-48h / D+2")):
        datos, columnas_x, columnas_y = construir_xy(n_dias=n_dias)
        escalado = split_y_escalar(datos, columnas_x, columnas_y, n_dias=n_dias)
        Xseq, Yseq, _ = secuencias_lstm(n_dias=n_dias)
        nan_x = int(escalado["X"].isna().sum().sum())
        nan_y = int(escalado["Y"].isna().sum().sum())
        # Anti-leakage: el objetivo del último origen de TRAIN no cae en VAL (purga = n_dias respetada).
        fin_train = datos.index[escalado["mascaras"]["tr"]].max()
        ini_val = datos.index[escalado["mascaras"]["va"]].min()
        gap_ok = (ini_val - fin_train).days > n_dias
        salida[etiqueta] = {
            "n_dias": n_dias,
            "MLP": {"filas": int(len(datos)), "n_features": len(columnas_x),
                    "n_salidas": len(columnas_y)},
            "LSTM": {"X": tuple(Xseq.shape), "Y": tuple(Yseq.shape)},
            "split": {"train": int(escalado["mascaras"]["tr"].sum()),
                      "val": int(escalado["mascaras"]["va"].sum()),
                      "test": int(escalado["mascaras"]["te"].sum())},
            "NaN": {"X": nan_x, "Y": nan_y},
            "anti_leakage": {"fin_train": str(fin_train.date()), "ini_val": str(ini_val.date()),
                             "gap_dias>horizonte": bool(gap_ok)},
            "lags_dias": _lags_legales(n_dias),
            "exogenas_dia": [columna for columna in columnas_x if not columna.startswith("lag")],
        }
    return salida


if __name__ == "__main__":
    resultado = verificar_pipeline()
    for etiqueta, resumen in resultado.items():
        print(f"\n=== Horizonte: {etiqueta} (n_dias={resumen['n_dias']}) ===")
        print(f"  MLP: {resumen['MLP']['filas']:,} filas, {resumen['MLP']['n_features']} features, "
              f"{resumen['MLP']['n_salidas']} salidas")
        print(f"  LSTM: X{resumen['LSTM']['X']}  Y{resumen['LSTM']['Y']}")
        print(f"  split: train={resumen['split']['train']:,}  val={resumen['split']['val']:,}  "
              f"test={resumen['split']['test']:,}")
        print(f"  NaN: X={resumen['NaN']['X']}  Y={resumen['NaN']['Y']}  (deben ser 0)")
        fuga = resumen["anti_leakage"]
        print(f"  anti-leakage: fin_train={fuga['fin_train']} < ini_val={fuga['ini_val']} | "
              f"gap>{resumen['n_dias']}d: {fuga['gap_dias>horizonte']}")
        print(f"  lags (días): {resumen['lags_dias']} | exógenas/día: {len(resumen['exogenas_dia'])} cols")
    print("\nVerificación SIN torch OK para ambos horizontes (shapes correctas, NaN=0, anti-leakage).")
