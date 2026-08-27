"""Punto de entrada de las redes (MLP y LSTM, day-ahead D+1): búsqueda de hiperparámetros y ajuste final.

## Uso

    python -m src.redes.run_redes mlp           # búsqueda del MLP  (200 de 360)
    python -m src.redes.run_redes lstm          # búsqueda del LSTM (100 de 192)
    python -m src.redes.run_redes mlp --epocas-por-corte     # estima las épocas en cada corte
    python -m src.redes.run_redes control mlp   # SOLO el control interno, sin gastar el resto
    python -m src.redes.run_redes finales mlp   # reentrena el ganador con 5 semillas y persiste test y val
"""
from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.redes import tuning as T
from src.features.validacion import rolling_origin_folds
from src.redes.datos import construir_xy

MLP_SPACE = {
    "hidden": [(640, 320), (896, 448), (1152, 576),
               (896, 448, 224), (1152, 576, 288)],
    "dropout": [0.0, 0.05, 0.1, 0.2],
    "lr": [1e-4, 2.5e-4, 5e-4],
    "weight_decay": [1e-7, 1e-6, 1e-5],
    "batch": [128, 256],
    "activacion": ["gelu"],
}
LSTM_SPACE = {
    "hidden": [128, 192, 256, 384],
    "dropout": [0.0, 0.05, 0.1, 0.2],
    "weight_decay": [1e-7, 1e-6, 1e-5],
    "lr": [5e-4, 1e-3],
    "L": [14],
    "batch": [64, 128],
    "activacion": ["gelu"],
}
N_CONFIGS = {"mlp": 200, "lstm": 100}
PRESUPUESTO_S = 10.5 * 3600  # tope de la búsqueda: al agotarse para en limpio y es reanudable

# Cifras de referencia del control interno: permiten comprobar que una reejecución reproduce.
REFERENCIA = {
    "mlp": {"mae_oos": 317.2,
            "config": {"hidden": [1152, 576], "dropout": 0.05, "lr": 5e-4,
                       "weight_decay": 1e-7, "batch": 256, "activacion": "gelu"}},
    "lstm": {"mae_oos": 319.6,
             "config": {"hidden": 384, "dropout": 0.05, "weight_decay": 1e-5, "lr": 5e-4,
                        "L": 14, "batch": 64, "activacion": "gelu"}},
}

# Margen que se le concede al control antes de declararlo fallido: el nivel puede moverse un poco entre
# entornos y versiones de librería, pero no dispararse.
UMBRAL_CONTROL = 0.15


def combinaciones(espacio: dict) -> int:
    """Tamaño del espacio, para reportar cobertura sin depender de lo que guarde el JSON."""
    return len(list(itertools.product(*espacio.values())))


def control(familia: str, epocas_por_corte: bool = False, abortar: bool = True,
            datos=None) -> dict:
    """Control interno: evalúa la configuración de referencia y aborta si el MAE_oos se dispara."""
    espacio = MLP_SPACE if familia == "mlp" else LSTM_SPACE
    config = T._normalizar_config(REFERENCIA[familia]["config"], espacio)
    config_id = T._id_config(familia, config)
    referencia = REFERENCIA[familia]["mae_oos"]

    print(f"\nControl {familia.upper()}: {config_id}", flush=True)
    print(f"  referencia: MAE_oos {referencia} MW", flush=True)

    # El control se reutiliza si ya está en el parcial (para no repetirlo al reanudar): no arrastrar
    # parciales de pruebas, porque un tope de épocas distinto daría por bueno un número que no lo es.
    registros = T._cargar_registros(familia)
    previo = next((registro for registro in registros if registro["id"] == config_id), None)
    if previo is not None:
        print("  (ya evaluado; se reutiliza)", flush=True)
        maes, epocas = previo["mae_oos_folds"], previo.get("epocas_por_fold")
    else:
        folds = rolling_origin_folds()
        if familia == "mlp":
            tabla, columnas_x, columnas_y = datos if datos is not None else construir_xy(n_dias=1)
            maes, epocas = T._maes_mlp_parada_separada(config, folds, tabla, columnas_x, columnas_y,
                                                       epocas_por_corte)
        else:
            Xseq, Xexo, Y, fechas = (datos if datos is not None
                                     else T.secuencias_lstm_exo(n_dias=1, L=config["L"]))
            maes, epocas = T._maes_lstm_parada_separada(config, folds, Xseq, Xexo, Y, fechas,
                                                        epocas_por_corte)
        registros.append({"id": config_id, "config": config, "mae_oos_folds": maes,
                          "mae_oos_medio": float(np.mean(maes)),
                          "mae_oos_se": float(np.std(maes, ddof=1) / np.sqrt(len(maes))),
                          "epocas_por_fold": epocas, "es_control": True})
        T._guardar_registros(familia, registros)

    mae = float(sum(maes) / len(maes))
    desviacion = mae / referencia - 1
    print(f"  MAE_oos {mae:.1f} MW ({desviacion:+.1%} vs {referencia} MW)", flush=True)
    print(f"      por corte: {[round(valor, 1) for valor in maes]}", flush=True)
    if epocas:
        print(f"      épocas por corte: {epocas}", flush=True)
    if desviacion > UMBRAL_CONTROL:
        mensaje = f"  Control FALLIDO: {desviacion:+.1%} > {UMBRAL_CONTROL:.0%}; se aborta."
        print(mensaje, flush=True)
        if abortar:
            raise SystemExit(mensaje)
    elif desviacion < 0:
        print("  Aviso: el nivel baja respecto a la referencia.", flush=True)
    else:
        print("  Control OK.", flush=True)
    print(flush=True)
    return {"familia": familia, "id": config_id, "mae_oos": mae, "referencia": referencia,
            "desviacion": desviacion, "mae_oos_folds": maes, "epocas_por_fold": epocas,
            "ok": desviacion <= UMBRAL_CONTROL}


def buscar(familia: str, n: int | None = None, presupuesto: float = PRESUPUESTO_S,
           epocas_por_corte: bool = False) -> list[dict]:
    """Búsqueda de una familia sobre las rejillas del módulo, reanudable y precedida del control."""
    espacio = MLP_SPACE if familia == "mlp" else LSTM_SPACE
    n = n or N_CONFIGS[familia]
    total = combinaciones(espacio)
    print(f"{familia.upper()}: {n} de {total} combinaciones ({n / total:.1%}) · estado en {familia}",
          flush=True)
    print(f"    referencia: MAE_oos {REFERENCIA[familia]['mae_oos']} MW", flush=True)
    print(f"    épocas estimadas {'en cada corte' if epocas_por_corte else 'una vez, en el corte mayor'}",
          flush=True)
    datos = None
    if familia == "mlp":
        datos = construir_xy(n_dias=1)
    forzar = T._normalizar_config(REFERENCIA[familia]["config"], espacio)
    control(familia, epocas_por_corte=epocas_por_corte, datos=datos)
    folds = rolling_origin_folds()
    inicio = time.time()
    if familia == "mlp":
        tabla, columnas_x, columnas_y = datos
        return T.buscar_mlp(folds, tabla, columnas_x, columnas_y, espacio=espacio, n_configs=n,
                            presupuesto=presupuesto, t0=inicio, etiqueta=familia,
                            epocas_por_corte=epocas_por_corte, forzar=forzar)
    return T.buscar_lstm(folds, espacio=espacio, n_configs=n, presupuesto=presupuesto, t0=inicio,
                         etiqueta=familia, epocas_por_corte=epocas_por_corte, forzar=forzar)


def finales(familia: str) -> dict:
    """Reentrena el ganador con 5 semillas y persiste test y validación (delega en `finales.py`)."""
    from src.redes import finales as F
    return F.main([familia])


def main(argv: list[str] | None = None):
    args = list(argv if argv is not None else sys.argv[1:])
    if not args or args[0] not in ("mlp", "lstm", "finales", "control"):
        raise SystemExit(__doc__.split("## Uso")[1])
    epocas_por_corte = "--epocas-por-corte" in args
    if args[0] == "finales":
        return finales(args[1] if len(args) > 1 else "mlp")
    if args[0] == "control":
        return control(args[1] if len(args) > 1 else "mlp", epocas_por_corte=epocas_por_corte,
                       abortar=False)
    n = next((int(arg) for arg in args[1:] if arg.isdigit()), None)
    presupuesto = next((float(arg.split("=")[1]) for arg in args if arg.startswith("--presupuesto=")),
                       PRESUPUESTO_S)
    return buscar(args[0], n=n, presupuesto=presupuesto, epocas_por_corte=epocas_por_corte)


if __name__ == "__main__":
    main()
