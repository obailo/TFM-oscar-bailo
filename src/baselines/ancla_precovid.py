"""
Ejecutar:  .venv/bin/python -m src.baselines.ancla_precovid
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.datos_precovid import serie_precovid, CORTE_PRECOVID
from src.features.validacion import split_por_fechas, VAL_INI, TEST_INI
from src.baselines.modelos import NaiveEstacional, Climatologia
from src.baselines.evaluacion import denominador_mase, evaluar

GAP_DAYAHEAD = 24
HORIZONTES = range(1, 25)


def main() -> dict:
    serie = serie_precovid()
    print(f"Serie pre-COVID: {len(serie):,} h | {serie.index.min()} -> {serie.index.max()}")
    split = split_por_fechas(serie.index, VAL_INI, TEST_INI, gap=GAP_DAYAHEAD)
    for nombre, indice in zip(("train", "val", "test"), split):
        if len(indice):
            print(f"  {nombre:5}: {indice[0]} -> {indice[-1]}  (n={len(indice):,})")

    ancla_train = serie.index[serie.index < pd.Timestamp(TEST_INI)]
    ancla_train = ancla_train[: max(0, len(ancla_train) - GAP_DAYAHEAD)]
    denominador = denominador_mase(serie, ancla_train)
    print(f"\n== ANCLA DEL MASE (train COMPLETO 2000-2018: {ancla_train[0]} -> {ancla_train[-1]}, "
          f"n={len(ancla_train):,}) ==")
    print("  escalas naive (MAE in-sample):",
          {periodo: round(valor, 1) for periodo, valor in denominador["escalas"].items()})
    print(f"  ANCLA = naive m={denominador['m_elegido']}: {denominador['escala']:.1f} MW (denominador único)")

    print("\n== Baselines sobre el test pre-COVID (h=1…24) ==")
    filas = []
    for modelo in (NaiveEstacional(24), NaiveEstacional(168), Climatologia()):
        resultado = evaluar(modelo, serie, split, denominador["escala"], horizontes=HORIZONTES)
        resumen = resultado["resumen"]
        filas.append({
            "modelo": resultado["modelo"], "MAE_MW": round(resumen["mae_medio"], 1),
            "MAPE_pct": round(resumen["mape_medio"], 2), "MASE": round(resumen["mase_medio"], 3),
            "MAPE_h24": round(float(resultado["metricas"].loc[24, "mape"]), 2),
            "n_origenes": resumen["n_origenes"],
        })
        print(f"  {resultado['modelo']:22} MAE={resumen['mae_medio']:.0f} MW  "
              f"MAPE={resumen['mape_medio']:.2f}%  MASE={resumen['mase_medio']:.3f}")
    tabla = pd.DataFrame(filas)

    dir_salida = RAIZ / "resultados" / "metricas"
    dir_salida.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(dir_salida / "baselines_precovid.csv", index=False)
    (dir_salida / "ancla_mase.json").write_text(json.dumps({
        "corte_precovid": CORTE_PRECOVID, "val_ini": VAL_INI, "test_ini": TEST_INI,
        "gap": GAP_DAYAHEAD, "horizontes": "1-24",
        "ancla_train": f"2000-01-01 -> {ancla_train[-1].date()} (2000-2018, train completo, val incluido)",
        "ancla_mase_mw": round(denominador["escala"], 2), "m_elegido": denominador["m_elegido"],
        "escalas_mae": {periodo: round(valor, 2) for periodo, valor in denominador["escalas"].items()},
    }, indent=2, ensure_ascii=False))
    print(f"\nGuardado: {dir_salida / 'baselines_precovid.csv'}  y  ancla_mase.json")
    return {"ancla": denominador, "tabla": tabla}


if __name__ == "__main__":
    main()
