"""
Ejecutar:  python -m src.features.catalogo_calendario
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.datos_precovid import serie_precovid
from src.features.calendario import (
    es_navidad,
    es_verano,
    features_calendario,
    festivo_regional_ponderado,
)

SALIDA = RAIZ / "resultados" / "catalogo" / "calendario"

FRANJAS_VERANO = {
    "1jun_30sep": ((6, 1), (9, 30)),
    "15jun_15sep": ((6, 15), (9, 15)),
    "1jul_15sep": ((7, 1), (9, 15)),
    "1jul_31ago": ((7, 1), (8, 31)),
    "1jul_31jul": ((7, 1), (7, 31)),
    "15jul_15sep": ((7, 15), (9, 15)),
    "15jul_31ago": ((7, 15), (8, 31)),
    "15jul_15ago": ((7, 15), (8, 15)),
    "1ago_31ago": ((8, 1), (8, 31)),
    "1ago_15ago": ((8, 1), (8, 15)),
    "1ago_15sep": ((8, 1), (9, 15)),
    "15ago_31ago": ((8, 15), (8, 31)),
}

FRANJAS_NAVIDAD = {
    "1dic_7ene": ((12, 1), (1, 7)),
    "15dic_7ene": ((12, 15), (1, 7)),
    "15dic_6ene": ((12, 15), (1, 6)),
    "20dic_6ene": ((12, 20), (1, 6)),
    "20dic_8ene": ((12, 20), (1, 8)),
    "22dic_6ene": ((12, 22), (1, 6)),
    "22dic_8ene": ((12, 22), (1, 8)),
    "22dic_1ene": ((12, 22), (1, 1)),
    "23dic_7ene": ((12, 23), (1, 7)),
    "24dic_6ene": ((12, 24), (1, 6)),
    "24dic_1ene": ((12, 24), (1, 1)),
    "24dic_7ene": ((12, 24), (1, 7)),
}


def _indice_dias() -> pd.DatetimeIndex:
    serie = serie_precovid()
    dias = pd.DatetimeIndex(serie.index.normalize().unique()).sort_values()
    dias.name = "fecha"
    return dias


def _guardar(df: pd.DataFrame, fichero: str) -> None:
    df.index.name = "fecha"
    df.to_csv(SALIDA / fichero, index_label="fecha")


def generar_catalogo() -> list[dict]:
    """Genera todos los CSV del catálogo y devuelve la lista de metadatos (registro)."""
    SALIDA.mkdir(parents=True, exist_ok=True)
    dias = _indice_dias()
    registro: list[dict] = []

    for nombre, (ini, fin) in FRANJAS_VERANO.items():
        columna = f"esVerano_{nombre}"
        serie = es_verano(dias, ini=ini, fin=fin).astype(int)
        serie.name = columna
        fichero = f"esVerano_{nombre}.csv"
        _guardar(serie.to_frame(), fichero)
        registro.append({
            "nombre": columna,
            "fichero": fichero,
            "tipo": "binaria",
            "parametros": {"ini": list(ini), "fin": list(fin)},
            "columnas": [columna],
            "descripcion": f"Temporada estival, franja {nombre.replace('_', '–')}.",
        })

    for nombre, (ini, fin) in FRANJAS_NAVIDAD.items():
        columna = f"esNavidad_{nombre}"
        serie = es_navidad(dias, ini=ini, fin=fin).astype(int)
        serie.name = columna
        fichero = f"esNavidad_{nombre}.csv"
        _guardar(serie.to_frame(), fichero)
        registro.append({
            "nombre": columna,
            "fichero": fichero,
            "tipo": "binaria",
            "parametros": {"ini": list(ini), "fin": list(fin)},
            "columnas": [columna],
            "descripcion": f"Temporada navideña, franja {nombre.replace('_', '–')}.",
        })

    calendario = features_calendario(dias, incluir_ciclicas=False)

    festivo_nac = calendario["festivo"].astype(int).rename("festivo_nacional")
    _guardar(festivo_nac.to_frame(), "festivo_nacional.csv")
    registro.append({
        "nombre": "festivo_nacional",
        "fichero": "festivo_nacional.csv",
        "tipo": "binaria",
        "parametros": {},
        "columnas": ["festivo_nacional"],
        "descripcion": "Festivo nacional (1/0), incluye Viernes Santo.",
    })

    # posiblePuente es ordinal {0, 0,5, 1}: no castear a int, se perdería el 0,5
    puente = calendario["posiblePuente"].astype(float).rename("posiblePuente")
    _guardar(puente.to_frame(), "posiblePuente.csv")
    registro.append({
        "nombre": "posiblePuente",
        "fichero": "posiblePuente.csv",
        "tipo": "ordinal",
        "parametros": {"regla": "racha de 1 día laborable entre no-laborables→1,0; 2 días→0,5; ≥3→0"},
        "columnas": ["posiblePuente"],
        "descripcion": "Grado de posible puente (0/0,5/1) alrededor de festivos nacionales.",
    })

    finde = calendario["finde"].astype(int).rename("finde")
    _guardar(finde.to_frame(), "finde.csv")
    registro.append({
        "nombre": "finde",
        "fichero": "finde.csv",
        "tipo": "binaria",
        "parametros": {},
        "columnas": ["finde"],
        "descripcion": "Fin de semana (sáb/dom = 1).",
    })

    fest_regional = festivo_regional_ponderado(dias).astype(float).rename("festivo_regional_ponderado")
    _guardar(fest_regional.to_frame(), "festivo_regional_ponderado.csv")
    registro.append({
        "nombre": "festivo_regional_ponderado",
        "fichero": "festivo_regional_ponderado.csv",
        "tipo": "continua",
        "parametros": {"rango": [0.0, 1.0], "fuente_pesos": "REE ISE 2019"},
        "columnas": ["festivo_regional_ponderado"],
        "descripcion": "Festivo regional ponderado in [0,1] = sum_ccaa 1{fiesta}·peso(ccaa).",
    })

    calendario_ciclico = features_calendario(dias, incluir_ciclicas=True)
    columnas_ciclicas = ["dia_semana_sin", "dia_semana_cos", "mes_sin", "mes_cos"]
    ciclicas = calendario_ciclico[columnas_ciclicas].astype(float)
    _guardar(ciclicas, "ciclicas_diasemana_mes.csv")
    registro.append({
        "nombre": "ciclicas_diasemana_mes",
        "fichero": "ciclicas_diasemana_mes.csv",
        "tipo": "continua",
        "parametros": {"periodos": {"dia_semana": 7, "mes": 12}, "orden_fourier": 1},
        "columnas": columnas_ciclicas,
        "descripcion": "Codificación cíclica (Fourier 1.er orden) de día-de-semana y mes.",
    })

    return registro


def main() -> None:
    registro = generar_catalogo()
    ruta_registro = SALIDA / "registro_calendario.json"
    ruta_registro.write_text(json.dumps(registro, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Catálogo de calendario en {SALIDA}")
    print(f"  {len(registro)} variantes | registro: {ruta_registro.name}")
    for entrada in registro:
        print(f"    [{entrada['tipo']:>8}] {entrada['fichero']:<34} cols={entrada['columnas']}")


if __name__ == "__main__":
    main()
