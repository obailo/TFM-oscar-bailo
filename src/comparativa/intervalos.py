"""Intervalos de predicción al 95 % por cuantiles empíricos de los residuos de 2018, por hora del día.

Ejecutar:  python -m src.comparativa.intervalos
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.arboles.modelo import N_ARBOLES_CV, _crear
from src.redes.datos import construir_xy
from src.features.validacion import split_canonico_precovid

DIR_MET = RAIZ / "resultados" / "metricas"
DIR_RED = RAIZ / "resultados" / "redes"
DIR_ARB = RAIZ / "resultados" / "arboles"
DIR_SAR = RAIZ / "resultados" / "sarimax"
RUTA = DIR_MET / "intervalos.json"


def _tabla_por_origen(df: pd.DataFrame, columna_pred="pred_ensemble") -> pd.DataFrame:
    """Parquet (fecha_origen, hora) → DataFrame con datetime-objetivo, hora, real y predicción."""
    dt = pd.to_datetime(df["fecha_origen"]) + pd.Timedelta(days=1) + pd.to_timedelta(df["hora"], unit="h")
    return pd.DataFrame({"dt": dt, "hora": df["hora"].to_numpy(),
                         "real": df["y_real"].to_numpy(), "pred": df[columna_pred].to_numpy()})


def _tabla_por_objetivo(df: pd.DataFrame) -> pd.DataFrame:
    dt = pd.to_datetime(df["datetime"]) + pd.to_timedelta(df["hora"], unit="h")
    return pd.DataFrame({"dt": dt, "hora": df["hora"].to_numpy(),
                         "real": df["real"].to_numpy(), "pred": df["pred"].to_numpy()})


def reajustar_rf() -> tuple:
    """Reajusta el RF solo sobre `train` (sin 2018) y predice 2018 para obtener residuos de calibración."""
    bloques_arboles = json.loads((DIR_MET / "arboles.json").read_text())["resultado"]["modelos"]
    datos, columnas_x, columnas_y = construir_xy(n_dias=1)
    split = split_canonico_precovid(gap_dias=1, guardar=False)
    dias_origen = datos.index
    train, val = dias_origen.isin(split.train), dias_origen.isin(split.val)
    X_train, Y_train = datos.loc[train, columnas_x].to_numpy(), datos.loc[train, columnas_y].to_numpy()
    X_val, Y_val = datos.loc[val, columnas_x].to_numpy(), datos.loc[val, columnas_y].to_numpy()

    hiper = bloques_arboles["rf"]["hiperparametros_elegidos"]
    # Los mismos árboles que en la validación cruzada y en el ajuste final del RF.
    print(f"  reajustando RF solo-train (hiperparametros {hiper}, {N_ARBOLES_CV} árboles)…", flush=True)
    modelo_rf = _crear("rf", hiper, n_arboles=N_ARBOLES_CV, semilla=0).fit(X_train, Y_train)
    return dias_origen[val], Y_val, modelo_rf.predict(X_val)


def _a_largo(origenes, real2d, pred2d) -> pd.DataFrame:
    """(n_dias, 24) → formato largo con datetime-objetivo y hora."""
    filas = []
    for i, fecha_origen in enumerate(origenes):
        base = pd.Timestamp(fecha_origen) + pd.Timedelta(days=1)
        for hora in range(real2d.shape[1]):
            filas.append({"dt": base + pd.Timedelta(hours=hora), "hora": hora,
                          "real": float(real2d[i, hora]), "pred": float(pred2d[i, hora])})
    return pd.DataFrame(filas)


def cargar() -> dict:
    """{modelo: (calibración 2018, test)} en formato largo. Solo los modelos que tienen las dos cosas."""
    fuentes, faltan = {}, []
    for nombre, familia in (("MLP (red)", "mlp"), ("LSTM (red)", "lstm")):
        ruta_val = DIR_RED / f"predicciones_val_{familia}.parquet"
        ruta_test = DIR_RED / f"predicciones_test_{familia}.parquet"
        if ruta_val.exists() and ruta_test.exists():
            fuentes[nombre] = (_tabla_por_origen(pd.read_parquet(ruta_val)), _tabla_por_origen(pd.read_parquet(ruta_test)))
        else:
            faltan.append(nombre)

    ruta_val, ruta_test = DIR_SAR / "fold_5" / "por_hora_pred.parquet", DIR_SAR / "final" / "por_hora_pred.parquet"
    if ruta_val.exists() and ruta_test.exists():
        calibracion = _tabla_por_objetivo(pd.read_parquet(ruta_val))
        calibracion = calibracion[calibracion["dt"].dt.year == 2018]  # el fold 5 evalúa 2018
        fuentes["SARIMAX por hora"] = (calibracion, _tabla_por_objetivo(pd.read_parquet(ruta_test)))
    else:
        faltan.append("SARIMAX por hora")

    # El XGBoost tiene persistidas sus predicciones de validación y su ajuste final es solo-train,
    # así que no hay que reajustarlo: el modelo que calibra y el que predice son el mismo.
    ruta_val_xgb = DIR_ARB / "xgb" / "predicciones_val_xgb.parquet"
    ruta_test_xgb = DIR_ARB / "xgb" / "predicciones_test_xgb.parquet"
    if ruta_val_xgb.exists() and ruta_test_xgb.exists():
        fuentes["XGBoost"] = (_tabla_por_origen(pd.read_parquet(ruta_val_xgb)), _tabla_por_origen(pd.read_parquet(ruta_test_xgb)))
        print("  XGBoost: val y test persistidos con ajuste solo-train → sin reajuste", flush=True)
    else:
        faltan.append("XGBoost")

    try:
        # El Random Forest no tiene predicciones de validación persistidas: hay que reajustarlo.
        ruta_test_rf = DIR_ARB / "rf" / "predicciones_test_rf.parquet"
        if ruta_test_rf.exists():
            origenes, real2d, pred2d = reajustar_rf()
            fuentes["RandomForest"] = (_a_largo(origenes, real2d, pred2d),
                                       _tabla_por_origen(pd.read_parquet(ruta_test_rf)))
        else:
            faltan.append("RandomForest")
    except Exception as error:
        print(f"  [aviso] Random Forest no reajustado: {error}", flush=True)
        faltan.append("RandomForest")

    if faltan:
        print(f"  [aviso] sin calibración+test para: {faltan}", flush=True)
    return fuentes


def intervalos_por_hora(calibracion: pd.DataFrame, test: pd.DataFrame,
                        alpha: float = 0.05) -> dict:
    """Cuantiles empíricos del residuo por hora sobre la calibración; cobertura y amplitud."""
    calibracion = calibracion.assign(res=calibracion["real"] - calibracion["pred"])
    # Cuantil CONFORME con estadístico de orden
    n_cal = calibracion.groupby("hora")["res"].size().min()
    ajuste = 1.0 / max(n_cal, 1)  # corrección de muestra finita (split conformal)
    inferior_hora = calibracion.groupby("hora")["res"].quantile(max(alpha / 2 - ajuste, 0.0), interpolation="lower")
    superior_hora = calibracion.groupby("hora")["res"].quantile(min(1 - alpha / 2 + ajuste, 1.0), interpolation="higher")
    inferior = test["pred"] + test["hora"].map(inferior_hora).to_numpy()
    superior = test["pred"] + test["hora"].map(superior_hora).to_numpy()
    dentro = (test["real"] >= inferior) & (test["real"] <= superior)
    # De comparación, una banda única sin desagregar por hora, para justificar el desagregado.
    inferior_global, superior_global = calibracion["res"].quantile(alpha / 2), calibracion["res"].quantile(1 - alpha / 2)
    dentro_global = ((test["real"] >= test["pred"] + inferior_global)
                     & (test["real"] <= test["pred"] + superior_global))
    return {"nivel_nominal": 1 - alpha,
            "cobertura_por_hora": float(dentro.mean()),
            "amplitud_media_mw": float((superior - inferior).mean()),
            "cobertura_banda_unica": float(dentro_global.mean()),
            "amplitud_banda_unica_mw": float(superior_global - inferior_global),
            "n_calibracion": int(len(calibracion)), "n_test": int(len(test)),
            "amplitud_por_hora_mw": {int(hora): float(superior_hora[hora] - inferior_hora[hora])
                                     for hora in sorted(inferior_hora.index)}}


def main() -> dict:
    print("\nIntervalos de predicción (nominal 95 %)")
    fuentes = cargar()
    resultados = {}
    for nombre, (calibracion, test) in fuentes.items():
        resultados[nombre] = intervalos_por_hora(calibracion, test)
        registro = resultados[nombre]
        print(f"\n{nombre}:")
        print(f"  calibración {registro['n_calibracion']:,} residuos (2018) · test {registro['n_test']:,}")
        print(f"  por hora    : cobertura {registro['cobertura_por_hora']:.3f} · "
              f"amplitud {registro['amplitud_media_mw']:.0f} MW")
        print(f"  banda única : cobertura {registro['cobertura_banda_unica']:.3f} · "
              f"amplitud {registro['amplitud_banda_unica_mw']:.0f} MW")

    print("\nBanda por hora (mismo método para todos):")
    print(f"{'modelo':16s} {'cobertura':>10s} {'amplitud':>10s}")
    for nombre in sorted(resultados, key=lambda k: resultados[k]["amplitud_media_mw"]):
        registro = resultados[nombre]
        print(f"{nombre:16s} {registro['cobertura_por_hora']:10.3f} {registro['amplitud_media_mw']:9.0f} MW")

    salida = {"metodo": "cuantiles empíricos de residuos por hora del día",
              "nivel_nominal": 1 - 0.05, "calibracion": "2018",
              "caveats": ["en las redes, 2018 guió también la parada temprana → garantía aproximada",
                          "calibrar con un año y aplicar a 2019-2020 supone estabilidad entre periodos",
                          "el Random Forest se reajusta solo-train para calibrar; su modelo de test usó train+val",
                          "el RF de calibración usa los hiperparámetros de arboles.json (rejilla corta de "
                          "modelo._rejilla) y el RF de test los de tuning_rf.json (búsqueda amplia): hoy "
                          "coinciden, pero si dejaran de hacerlo la banda se calibraría con otro modelo"],
              "resultados": resultados}
    RUTA.parent.mkdir(parents=True, exist_ok=True)
    RUTA.write_text(json.dumps(salida, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nGuardado {RUTA.name}")
    return salida


if __name__ == "__main__":
    main()
