"""Competición de exógenas por OLS que congela el conjunto elegido en `exogenas_elegidas.json` + CSV.

Ejecutar:  .venv/bin/python -m src.seleccion_exogenas.run_seleccion
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.features import catalogo_loader
from src.features.validacion import rolling_origin_folds
from src.seleccion_exogenas import candidatas
from src.seleccion_exogenas.motor import competir_rolling, elegir_1se, ajustar_predecir

DIR_OUT = RAIZ / "resultados" / "metricas"

# Horas para el reporte de estabilidad del ganador de temperatura (valle/punta + mediodía/tarde).
HORAS_ESTABILIDAD = [4, 9, 13, 20]


def _folds():
    """Folds rolling-origin (día-origen) para la selección (test reservado, no se usa aquí)."""
    return rolling_origin_folds(guardar=False)


# 1) Temperatura: competición del catálogo completo, hora a hora.

def competir_temperatura(folds, guardar_csv=True) -> tuple[pd.DataFrame, str]:
    """Compite las 425 variantes de temperatura por hora, por rango medio del MAE_oos entre las 24 h;
    entre las que quedan a Δrango ≤ 1 del mejor gana la de menos regresores (parsimonia)."""
    calendario = candidatas.bloque_calendario_control()
    demanda_por_hora = {hora: candidatas.demanda_hora(hora) for hora in range(24)}
    nombres = catalogo_loader.listar("temperatura")
    # mae[variante][hora] guarda el MAE_oos medio sobre los folds; n_regresores, el nº de regresores
    mae = {nombre: {} for nombre in nombres}
    n_regresores = {}
    for i, nombre in enumerate(nombres, 1):
        for hora in range(24):
            temp_hora = candidatas.bloque_temperatura_hora(nombre, hora)
            X = calendario.join(temp_hora, how="inner")
            y = demanda_por_hora[hora]
            maes = []
            for fold in folds:
                train = X.index.intersection(fold.train)
                val = X.index.intersection(fold.val)
                resultado = ajustar_predecir(X.loc[train], y.reindex(train),
                                             X.loc[val], y.reindex(val))
                if resultado["n"] > 0:
                    maes.append(resultado["mae"])
            if maes:
                mae[nombre][hora] = float(np.mean(maes))
        n_regresores[nombre] = candidatas.bloque_temperatura_hora(nombre, 12).shape[1]
        catalogo_loader.limpiar_cache("temperatura", f"{nombre}.parquet")  # libera RAM
        if i % 50 == 0:
            print(f"  temperatura: {i}/{len(nombres)} variantes", flush=True)

    mat_mae = pd.DataFrame(mae).T.reindex(nombres)  # filas = variante, columnas = hora
    rangos = mat_mae.rank(axis=0, method="min")  # rango por hora, 1 = mejor MAE
    resumen = pd.DataFrame({
        "candidato": mat_mae.index,
        "rango_medio": rangos.mean(axis=1).to_numpy(),
        "mae_medio_horas": mat_mae.mean(axis=1).to_numpy(),  # informativo: mezcla escalas de hora
        "k": [n_regresores[nombre] for nombre in mat_mae.index],
        "horas_top3": (rangos <= 3).sum(axis=1).to_numpy(),
    }).sort_values("rango_medio").reset_index(drop=True)
    # Ganador: mejor rango medio; entre casi-empatados (Δrango≤1) el más parsimonioso.
    umbral = resumen.iloc[0]["rango_medio"] + 1.0
    cerca = resumen[resumen["rango_medio"] <= umbral]
    ganador = cerca.sort_values(["k", "rango_medio"]).iloc[0]["candidato"]

    print(f"\nTEMPERATURA: top 8 de {len(resumen)} (rango medio 24 h):")
    print(resumen.head(8).to_string(index=False))
    # Estabilidad: ¿gana la misma variante en las horas representativas?
    ganadores_hora = {hora: mat_mae[hora].idxmin()
                      for hora in HORAS_ESTABILIDAD if hora in mat_mae.columns}
    print(f"Ganador global: {ganador}")
    print(f"Ganadores por hora {HORAS_ESTABILIDAD}: {ganadores_hora}")
    if guardar_csv:
        DIR_OUT.mkdir(parents=True, exist_ok=True)
        resumen.to_csv(DIR_OUT / "seleccion_temperatura.csv", index=False)
    return resumen, ganador


# 2) Franjas de temporada (verano y navidad): rolling + 1-SE sobre la demanda diaria.

def competir_temporada(y_dia, folds, variantes: list[str], etiqueta: str,
                       guardar_csv=True) -> tuple[pd.DataFrame, str]:
    """Compite franjas (esVerano o esNavidad) sobre el control de calendario, criterio MAE_oos rolling."""
    calendario = candidatas.bloque_calendario_control()
    candidatos = {}
    for variante in variantes:
        candidatos[variante] = calendario.join(candidatas.bloque_calendario([variante]), how="left")
    tabla = competir_rolling(candidatos, y_dia, folds)
    ganador = elegir_1se(tabla)
    print(f"\n{etiqueta}: franjas, top 5 de {len(tabla)}:")
    print(tabla.head(5).to_string(index=False))
    print(f"  ganador (1-SE): {ganador}")
    if guardar_csv:
        tabla.to_csv(DIR_OUT / f"seleccion_{etiqueta}.csv", index=False)
    return tabla, ganador


# 3) Festivos: best-subset 2⁴ decisorio (rolling + 1-SE).

_FESTIVOS_VARS = ["finde", "festivo_nacional", "festivo_regional_ponderado", "posiblePuente"]


def competir_festivos(y_dia, folds, guardar_csv=True) -> tuple[pd.DataFrame, list[str]]:
    """Best-subset de las 4 variables de festivos (2⁴=16, 15 no vacías) sobre el control de cíclicas."""
    ciclicas = candidatas.bloque_calendario(["ciclicas_diasemana_mes"])
    columnas = {variable: candidatas.bloque_calendario([variable]) for variable in _FESTIVOS_VARS}
    candidatos = {}
    for tamano in range(1, len(_FESTIVOS_VARS) + 1):
        for combo in itertools.combinations(_FESTIVOS_VARS, tamano):
            X = ciclicas.copy()
            for variable in combo:
                X = X.join(columnas[variable], how="left")
            candidatos["+".join(combo)] = X
    candidatos["solo_ciclicas"] = ciclicas
    tabla = competir_rolling(candidatos, y_dia, folds)
    ganador = elegir_1se(tabla)
    elegidas = [] if ganador == "solo_ciclicas" else ganador.split("+")
    print(f"\nFESTIVOS: best-subset 2⁴, top 6 de {len(tabla)}:")
    print(tabla.head(6).to_string(index=False))
    print(f"  subconjunto ganador: {elegidas}")
    if guardar_csv:
        tabla.to_csv(DIR_OUT / "seleccion_festivos.csv", index=False)
    return tabla, elegidas


# Orquestación.

def main(guardar: bool = True) -> dict:
    folds = _folds()
    print(f"Rolling-origin: {len(folds)} folds | train último fold = {len(folds[-1].train)} días")
    y_dia = candidatas.demanda_diaria()

    _, temp_ganadora = competir_temperatura(folds, guardar_csv=guardar)
    _, verano_ganadora = competir_temporada(
        y_dia, folds, candidatas.VARIANTES_VERANO, "verano", guardar_csv=guardar)
    _, navidad_ganadora = competir_temporada(
        y_dia, folds, candidatas.VARIANTES_NAVIDAD, "navidad", guardar_csv=guardar)
    _, festivos_ganadores = competir_festivos(y_dia, folds, guardar_csv=guardar)

    temp_columnas = list(catalogo_loader.cargar("temperatura", temp_ganadora).columns)
    elegido = {
        "temperatura": temp_ganadora,
        "esVerano": verano_ganadora,
        "esNavidad": navidad_ganadora,
        # Informativo: los modelos fijan el conjunto ganador (finde + festivo regional
        # ponderado) en su propio preparado de datos, no lo releen de aquí.
        "festivos_elegidos": festivos_ganadores,
    }
    salida = {
        "criterio": "MAE_oos rolling-origin (5 folds); temperatura por rango medio + "
                    "parsimonia (Δrango≤1), temporada y festivos con regla 1-SE",
        "split": "rolling_origin_folds (val 2014-2018); test reservado",
        "conjunto_seleccionado": elegido,
    }
    if guardar:
        DIR_OUT.mkdir(parents=True, exist_ok=True)
        (DIR_OUT / "exogenas_elegidas.json").write_text(
            json.dumps(salida, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nExógenas elegidas:\n  temperatura: {temp_ganadora} {temp_columnas}")
        print(f"  esVerano: {verano_ganadora} | esNavidad: {navidad_ganadora} | "
              f"festivos: {festivos_ganadores}")
        print(f"  guardado en {DIR_OUT}/exogenas_elegidas.json")
    return salida


if __name__ == "__main__":
    main()
