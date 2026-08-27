from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]


def ancla_mase() -> float:
    """Escala del MASE horario pre-COVID (naive m=168), común a todos los modelos."""
    ruta = RAIZ / "resultados" / "metricas" / "ancla_mase.json"
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe {ruta}. Ejecuta antes los baselines, que son quienes fijan el ancla: "
            "`.venv/bin/python -m src.baselines.run_baselines`.")
    return float(json.loads(ruta.read_text())["ancla_mase_mw"])


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def escala_naive(y_train: np.ndarray, m: int) -> float:
    return float(np.mean(np.abs(y_train[m:] - y_train[:-m])))


def mase_con_escala(y_true: np.ndarray, y_pred: np.ndarray, escala: float) -> float:
    return float(np.mean(np.abs(y_true - y_pred)) / escala)

