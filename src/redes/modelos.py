"""Arquitecturas PyTorch: MLP multi-salida (24) y LSTM con exógenas. Requiere `torch`."""
from __future__ import annotations

import torch
import torch.nn as nn

# Activaciones candidatas, referenciadas por nombre (serializable a JSON y al `id` de cada configuración).
ACTIVACIONES = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}


def _resolver_activacion(activacion) -> type[nn.Module]:
    """Acepta el nombre (clave de `ACTIVACIONES`) o directamente una clase `nn.Module`."""
    if isinstance(activacion, str):
        if activacion not in ACTIVACIONES:
            raise ValueError(f"activación '{activacion}' desconocida; opciones: {sorted(ACTIVACIONES)}")
        return ACTIVACIONES[activacion]
    return activacion


class MLP(nn.Module):
    """Perceptrón multicapa multi-salida: n_entradas → hidden… → n_salidas (24). Estrategia MIMO directa."""

    def __init__(self, n_entradas: int, n_salidas: int = 24, hidden=(256, 128), dropout: float = 0.2,
                 activacion="relu"):
        super().__init__()
        Activacion = _resolver_activacion(activacion)  # la misma en todas las capas ocultas
        capas, entrada = [], n_entradas
        for unidades in hidden:
            capas += [nn.Linear(entrada, unidades), Activacion(), nn.Dropout(dropout)]
            entrada = unidades
        capas.append(nn.Linear(entrada, n_salidas))  # salida lineal, es una regresión
        self.red = nn.Sequential(*capas)

    def forward(self, x):
        return self.red(x)


class LSTMRedExo(nn.Module):
    """LSTM con exógenas: último estado oculto concatenado con las exógenas del día objetivo → n_salidas."""

    def __init__(self, n_features: int, n_exogenas: int, hidden: int = 128, capas: int = 1, n_salidas: int = 24,
                 dropout: float = 0.2, activacion="relu"):
        super().__init__()
        Activacion = _resolver_activacion(activacion)  # solo la cabeza densa; las puertas de la celda son fijas
        self.lstm = nn.LSTM(n_features, hidden, num_layers=capas, batch_first=True,
                            dropout=dropout if capas > 1 else 0.0)
        self.cabeza = nn.Sequential(nn.Linear(hidden + n_exogenas, hidden), Activacion(),
                                    nn.Dropout(dropout), nn.Linear(hidden, n_salidas))

    def forward(self, x, exo):
        salida, _ = self.lstm(x)
        oculto = torch.cat([salida[:, -1, :], exo], dim=1)
        return self.cabeza(oculto)
