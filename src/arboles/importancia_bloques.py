"""
Ejecutar:  .venv/bin/python -m src.arboles.importancia_bloques
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.arboles.modelo import N_ARBOLES_CV, _ancla, _crear
from src.redes.datos import construir_xy
from src.features.validacion import rolling_origin_folds

DIR_MET = RAIZ / "resultados" / "metricas"
DIR_FIG = RAIZ / "figuras" / "arboles"
N_REPS = 5  # repeticiones de la permutación por bloque
SEMILLA = 0


def _leer(nombre: str, comando: str) -> dict:
    """Carga un JSON de resultados; si falta, aborta diciendo qué hay que ejecutar antes."""
    ruta = DIR_MET / nombre
    if not ruta.exists():
        raise SystemExit(f"falta resultados/metricas/{nombre}: ejecute antes `{comando}`.")
    return json.loads(ruta.read_text())


def ganadores() -> tuple[dict[str, dict], dict[str, int]]:
    """
    Hiperparámetros y nº de árboles de los ganadores, leídos de los JSON de la búsqueda.
    """
    hiperparametros = {
        "xgb": _leer("tuning_xgb.json", "python -m src.arboles.tuning xgb 200")["mejor"]["hiperparametros"],
        "rf": _leer("tuning_rf.json", "python -m src.arboles.tuning rf")["mejor"]["hiperparametros"],
    }
    n_arboles = {
        "xgb": int(_leer("finales_xgb.json", "python -m src.arboles.tuning finales")["n_arboles"]),
        "rf": N_ARBOLES_CV,
    }
    return hiperparametros, n_arboles


def definir_bloques(columnas_x: list[str]) -> dict[str, list[str]]:
    """Define los bloques de columnas en dos niveles: familias de información y desglose por día y franja."""
    lags = [c for c in columnas_x if c.startswith("lag")]
    temp = [c for c in columnas_x if c.startswith("temp")]
    calendario = [c for c in columnas_x if not c.startswith(("lag", "temp"))]

    def horas(columnas, lo, hi):
        return [c for c in columnas if lo <= int(c.rsplit("_h", 1)[1][:2]) <= hi]

    bloques = {
        "N1 · retardos de demanda": lags,
        "N1 · temperatura": temp,
        "N1 · calendario": calendario,
        "N2 · retardo d-1": [c for c in lags if c.startswith("lag1d")],
        "N2 · retardo d-2": [c for c in lags if c.startswith("lag2d")],
        "N2 · retardo d-7": [c for c in lags if c.startswith("lag7d")],
        "N2 · retardo d-14": [c for c in lags if c.startswith("lag14d")],
        "N2 · temperatura madrugada (0-7 h)": horas(temp, 0, 7),
        "N2 · temperatura jornada (8-15 h)": horas(temp, 8, 15),
        "N2 · temperatura tarde-noche (16-23 h)": horas(temp, 16, 23),
        "CONTROL · todas las variables": list(columnas_x),
    }
    n1 = (len(bloques["N1 · retardos de demanda"]) + len(bloques["N1 · temperatura"])
          + len(bloques["N1 · calendario"]))
    assert n1 == len(columnas_x), f"el nivel 1 no particiona las columnas: {n1} de {len(columnas_x)}"
    return bloques


def permutar_bloque(X: np.ndarray, cols: list[int], rng) -> np.ndarray:
    """Baraja las filas de `cols` con una única reordenación compartida por todas las columnas del bloque."""
    X_permutado = X.copy()
    X_permutado[:, cols] = X[rng.permutation(len(X))][:, cols]
    return X_permutado


def medir(tipo: str, datos, columnas_x, columnas_y, ancla: float,
          hiperparametros: dict, n_arboles: int) -> dict:
    folds = rolling_origin_folds()
    dias_origen = datos.index
    train, val = dias_origen.isin(folds[-1].train), dias_origen.isin(folds[-1].val)
    X_train, Y_train = datos[columnas_x][train].to_numpy(), datos[columnas_y][train].to_numpy()
    X_val, Y_val = datos[columnas_x][val].to_numpy(), datos[columnas_y][val].to_numpy()
    print(f"[{tipo}] ajuste sobre {train.sum()} días (< 2018) · permutación sobre {val.sum()} días (2018)",
          flush=True)

    inicio = time.time()
    modelo = _crear(tipo, hiperparametros, n_arboles=n_arboles, semilla=SEMILLA)
    modelo.fit(X_train, Y_train)
    base = float(np.mean(np.abs(Y_val - modelo.predict(X_val))))
    print(f"[{tipo}] ajustado en {time.time() - inicio:.0f} s · MAE sin permutar {base:.1f} MW "
          f"(MASE {base / ancla:.4f})", flush=True)

    posicion = {columna: i for i, columna in enumerate(columnas_x)}
    salida = []
    for nombre, columnas in definir_bloques(columnas_x).items():
        indices = [posicion[columna] for columna in columnas]
        rng = np.random.default_rng(SEMILLA)
        maes = [float(np.mean(np.abs(Y_val - modelo.predict(permutar_bloque(X_val, indices, rng)))))
                for _ in range(N_REPS)]
        media = float(np.mean(maes))
        salida.append({"bloque": nombre, "n_columnas": len(columnas),
                       "mae_permutado": round(media, 1), "sd": round(float(np.std(maes, ddof=1)), 1),
                       "aumento_mae_mw": round(media - base, 1),
                       "aumento_mase": round((media - base) / ancla, 4)})
        print(f"    {nombre:42s} n={len(columnas):3d}  +{media - base:7.1f} MW  "
              f"(+{(media - base) / ancla:.4f} MASE)", flush=True)

    techo = next(registro["aumento_mae_mw"] for registro in salida if registro["bloque"].startswith("CONTROL"))
    for registro in salida:
        registro["fraccion_del_techo"] = round(registro["aumento_mae_mw"] / techo, 3) if techo else None
    return {"hiperparametros": hiperparametros, "n_arboles": n_arboles, "mae_base": round(base, 1),
            "mase_base": round(base / ancla, 4), "n_repeticiones": N_REPS, "bloques": salida}


def figura(resultados: dict):
    """Barras horizontales del aumento de MASE por bloque, un panel por modelo."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    DIR_FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True)
    for ax, tipo in zip(axes, ("xgb", "rf")):
        bloques = [r for r in resultados[tipo]["bloques"] if not r["bloque"].startswith("CONTROL")][::-1]
        colores = ["#c0392b" if r["bloque"].startswith("N1") else "#5d8aa8" for r in bloques]
        ax.barh([r["bloque"].split(" · ")[1] for r in bloques],
                [r["aumento_mase"] for r in bloques], color=colores)
        ax.set_title({"xgb": "XGBoost", "rf": "Random Forest"}[tipo])
        ax.set_xlabel("aumento del MASE al permutar el bloque")
        ax.grid(axis="x", alpha=.3)
    fig.suptitle("Importancia por bloques (permutación conjunta, validación 2018)")
    fig.tight_layout()
    ruta = DIR_FIG / "importancia_bloques.png"
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    print(f"figura → {ruta}", flush=True)
    return ruta


def main():
    ancla = _ancla()
    hiperparametros, n_arboles = ganadores()
    datos, columnas_x, columnas_y = construir_xy(n_dias=1)
    print(f"{len(columnas_x)} columnas · ancla MASE {ancla} MW", flush=True)
    for tipo in ("xgb", "rf"):
        print(f"  ganador {tipo}: {n_arboles[tipo]} árboles · {hiperparametros[tipo]}", flush=True)
    resultados = {tipo: medir(tipo, datos, columnas_x, columnas_y, ancla,
                              hiperparametros[tipo], n_arboles[tipo]) for tipo in ("xgb", "rf")}
    resultados["ancla_mase_mw"] = ancla
    resultados["protocolo"] = "ajuste < 2018, permutación por bloques sobre 2018"
    ruta = DIR_MET / "importancia_bloques.json"
    ruta.write_text(json.dumps(resultados, indent=2, ensure_ascii=False))
    print(f"métricas → {ruta}", flush=True)
    figura(resultados)
    return resultados


if __name__ == "__main__":
    main()
