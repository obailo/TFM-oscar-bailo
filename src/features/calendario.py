"""Variables de calendario crudas para árboles y cíclicas para redes/lineales."""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import holidays as holidays_lib
from dateutil.easter import easter

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.eda.festivos import es_festivo, fechas_festivas


_PERIODOS = {"hora": 24, "dia_semana": 7, "mes": 12}

RUTA_PESOS_CCAA = Path(__file__).resolve().parent / "pesos_ccaa.csv"

_PUENTE_PAD_DIAS = 15


def posible_puente(index: pd.DatetimeIndex) -> pd.Series:
    """Ordinal {0, 0.5, 1} del grado en que un día laborable es posible puente (detección isla/hueco)."""
    ini = index.normalize().min() - pd.Timedelta(days=_PUENTE_PAD_DIAS)
    fin = index.normalize().max() + pd.Timedelta(days=_PUENTE_PAD_DIAS)
    dias = pd.date_range(ini, fin, freq="D")
    festivos = set(pd.DatetimeIndex(fechas_festivas(int(dias.year.min()), int(dias.year.max()))))
    laborable = np.array([(dia.dayofweek < 5) and (dia not in festivos) for dia in dias])

    valores = np.zeros(len(dias))
    i = 0
    while i < len(dias):
        if not laborable[i]:
            i += 1
            continue
        j = i
        while j < len(dias) and laborable[j]:
            j += 1
        racha = j - i
        flanqueada = (i > 0 and not laborable[i - 1]) and (j < len(dias) and not laborable[j])
        if flanqueada:
            if racha == 1:
                valores[i] = 1.0
            elif racha == 2:
                valores[i] = valores[i + 1] = 0.5
        i = j

    serie_dia = pd.Series(valores, index=dias)
    return pd.Series(serie_dia.reindex(index.normalize()).to_numpy(),
                     index=index, name="posiblePuente")


def features_calendario(index: pd.DatetimeIndex, incluir_ciclicas: bool = True) -> pd.DataFrame:
    df = pd.DataFrame(index=index)
    df["hora"] = index.hour
    df["dia_semana"] = index.dayofweek  # 0 = lunes ... 6 = domingo
    df["mes"] = index.month
    df["dia_anio"] = index.dayofyear
    df["finde"] = (index.dayofweek >= 5).astype(int)
    df["festivo"] = es_festivo(index).astype(int).to_numpy()
    df["posiblePuente"] = posible_puente(index).to_numpy()

    if incluir_ciclicas:
        for columna, periodo in _PERIODOS.items():
            angulo = 2 * np.pi * df[columna] / periodo
            df[f"{columna}_sin"] = np.sin(angulo)
            df[f"{columna}_cos"] = np.cos(angulo)
    return df



# CCAA peninsulares, con el código de subdivisión de `holidays` 
_CCAA_PENINSULARES = ["AN", "AR", "AS", "CB", "CL", "CM", "CT", "EX",
                      "GA", "MC", "MD", "NC", "PV", "RI", "VC"]


def cargar_pesos_ccaa() -> dict:
    """Peso de demanda de cada CCAA peninsular (código `holidays` -> peso), de `pesos_ccaa.csv`."""
    if not RUTA_PESOS_CCAA.exists():
        raise FileNotFoundError(
            f"No existe {RUTA_PESOS_CCAA}. Contiene el reparto de demanda por CCAA.")
    df = pd.read_csv(RUTA_PESOS_CCAA)
    codigos = df["codigo_iso"].str.replace("ES-", "", regex=False)
    # El filtro por código deja fuera Canarias, Baleares, Ceuta y Melilla con independencia
    # de lo que traiga el CSV.
    pesos = {codigo: float(peso) for codigo, peso in zip(codigos, df["peso_normalizado"])
             if codigo in _CCAA_PENINSULARES}
    suma = sum(pesos.values())
    if abs(suma - 1.0) > 1e-3:  # los pesos de las 15 peninsulares deben sumar 1
        raise ValueError(
            f"Los pesos de CCAA peninsulares deben sumar ≈1 (suman {suma:.4f}). "
            f"Revisa {RUTA_PESOS_CCAA}.")
    return pesos


_FEST_AUTON = {
    "AN": [("fija", 2, 28), ("pascua", -3)],
    "AR": [("fija", 4, 23), ("pascua", -3)],
    "AS": [("fija", 9, 8), ("pascua", -3)],
    "CB": [("fija", 7, 28), ("fija", 9, 15), ("pascua", -3)],  # sin Santiago
    "CL": [("fija", 4, 23), ("fija", 3, 19), ("fija", 7, 25), ("pascua", -3)],
    "CM": [("fija", 5, 31), ("fija", 3, 19), ("pascua", 60), ("pascua", -3)],
    "CT": [("fija", 6, 24), ("fija", 9, 11), ("fija", 12, 26), ("pascua", 1)],  # sin Jueves Santo
    "EX": [("fija", 9, 8), ("fija", 3, 19), ("pascua", -3)],
    "GA": [("fija", 7, 25), ("fija", 5, 17), ("pascua", -3)],  # sin San José ni San Xoán
    "MC": [("fija", 6, 9), ("fija", 3, 19), ("pascua", -3)],
    "MD": [("fija", 5, 2), ("fija", 3, 19), ("fija", 7, 25), ("pascua", 60), ("pascua", -3)],
    "NC": [("fija", 12, 3), ("fija", 3, 19), ("fija", 7, 25), ("pascua", 1), ("pascua", -3)],
    "PV": [("fija", 3, 19), ("fija", 7, 25), ("pascua", 1), ("pascua", -3)],
    "RI": [("fija", 6, 9), ("fija", 3, 19), ("fija", 7, 25), ("pascua", 1), ("pascua", -3)],
    "VC": [("fija", 10, 9), ("fija", 3, 19), ("pascua", 1)],  # sin Jueves Santo
}


def _fechas_autonomicas(codigo: str, anios) -> list:
    fechas = []
    for anio in anios:
        pascua = pd.Timestamp(easter(anio))
        for regla in _FEST_AUTON.get(codigo, []):
            if regla[0] == "fija":
                fechas.append(pd.Timestamp(year=anio, month=regla[1], day=regla[2]))
            else:
                fechas.append(pascua + pd.Timedelta(days=regla[1]))
    return fechas


def _peso_festivo_por_dia(anio_min: int, anio_max: int) -> dict:
    """{día -> peso de festivo regional} = sum_ccaa 1{ccaa de fiesta}·peso(ccaa)."""
    pesos = cargar_pesos_ccaa()
    anios = range(anio_min, anio_max + 1)
    acumulado = defaultdict(float)
    for codigo, peso in pesos.items():
        for fecha in holidays_lib.Spain(subdiv=codigo, years=anios):
            fecha = pd.Timestamp(fecha)
            if fecha.year >= 2008:
                acumulado[fecha] += peso
    if anio_min <= 2007:
        anios_pre = range(anio_min, min(anio_max, 2007) + 1)
        nacionales_pre = [fecha for fecha in fechas_festivas(anio_min, min(anio_max, 2007))
                          if fecha.year <= 2007]
       
        for codigo, peso in pesos.items():
            for fecha in nacionales_pre:
                acumulado[fecha] += peso
            for fecha in _fechas_autonomicas(codigo, anios_pre):
                acumulado[fecha] += peso
    return acumulado


def festivo_regional_ponderado(index: pd.DatetimeIndex) -> pd.Series:
    acumulado = _peso_festivo_por_dia(int(index.year.min()), int(index.year.max()))
    dias = index.normalize()
    return pd.Series([min(1.0, acumulado.get(dia, 0.0)) for dia in dias], index=index,
                     name="festivo_regional")


def _en_franja(index: pd.DatetimeIndex, ini, fin, envuelve: bool):
    mes_dia = index.month * 100 + index.day
    ini_mes_dia, fin_mes_dia = ini[0] * 100 + ini[1], fin[0] * 100 + fin[1]
    if envuelve:  # la franja cruza el cambio de año, como diciembre a enero
        return (mes_dia >= ini_mes_dia) | (mes_dia <= fin_mes_dia)
    return (mes_dia >= ini_mes_dia) & (mes_dia <= fin_mes_dia)


def es_verano(index: pd.DatetimeIndex, ini, fin) -> pd.Series:
    """Binaria de temporada estival en la franja (ini, fin), cada una como par (mes, día)."""
    mascara = np.asarray(_en_franja(index, ini, fin, envuelve=False), dtype=int)
    return pd.Series(mascara, index=index, name="esVerano")


def es_navidad(index: pd.DatetimeIndex, ini, fin) -> pd.Series:
    """Binaria de temporada navideña (cruza el cambio de año) en la franja (ini, fin)."""
    mascara = np.asarray(_en_franja(index, ini, fin, envuelve=True), dtype=int)
    return pd.Series(mascara, index=index, name="esNavidad")


if __name__ == "__main__":
    horas = pd.date_range("2023-01-01", periods=24, freq="h")
    calendario = features_calendario(horas)
    print("Columnas:", list(calendario.columns))
    print("\nCodificación cíclica de la hora (0h, 6h, 12h, 18h):")
    print(calendario.loc[horas[[0, 6, 12, 18]], ["hora", "hora_sin", "hora_cos"]].round(3))

    print("\nPesos CCAA (sum):", round(sum(cargar_pesos_ccaa().values()), 5))
    dias = pd.date_range("2019-01-01", "2019-12-31", freq="D")
    festivo_reg = festivo_regional_ponderado(dias)
    print("Días con festivo regional>0 en 2019:", int((festivo_reg > 0).sum()),
          "| nacionales (≈1):", int((festivo_reg > 0.99).sum()),
          "| solo regionales (0<p<0.99):", int(((festivo_reg > 0) & (festivo_reg < 0.99)).sum()))
    print("  11-sep-2019 (Diada, solo Cataluña) ->", round(float(festivo_reg.loc["2019-09-11"]), 3),
          "| 25-dic-2019 (nacional) ->", round(float(festivo_reg.loc["2019-12-25"]), 3))

    print("esVerano 2019:", int(es_verano(dias, ini=(6, 15), fin=(9, 15)).sum()), "días |",
          "esNavidad 2019:", int(es_navidad(dias, ini=(12, 24), fin=(1, 6)).sum()), "días")
