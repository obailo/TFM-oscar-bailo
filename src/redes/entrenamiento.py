"""Bucle de entrenamiento del MLP, con parada temprana, dropout, weight decay y semilla fijada."""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))


def fijar_semilla(semilla: int):
    import torch
    random.seed(semilla); np.random.seed(semilla); torch.manual_seed(semilla)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(semilla)
    torch.backends.cudnn.deterministic = True  # reproducibilidad del media±sd entre semillas (cuDNN)
    torch.backends.cudnn.benchmark = False


def entrenar_mlp(datos_escalados: dict, n_salidas: int, hidden=(256, 128), dropout=0.2, lr=1e-3,
                 weight_decay=1e-4, epocas=300, paciencia=20, batch=64, semilla=0, device=None,
                 activacion="relu", epocas_fijas=None):
    """Entrena el MLP MIMO con parada temprana (o `epocas_fijas` sin ella). Devuelve (modelo, historia)."""
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    from src.redes.modelos import MLP

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    fijar_semilla(semilla)
    f32 = lambda df: torch.tensor(df.to_numpy(), dtype=torch.float32)
    mascaras = datos_escalados["mascaras"]
    X_train = f32(datos_escalados["X"][mascaras["tr"]])
    Y_train = f32(datos_escalados["Y"][mascaras["tr"]])
    X_val = f32(datos_escalados["X"][mascaras["va"]]).to(device)
    Y_val = f32(datos_escalados["Y"][mascaras["va"]]).to(device)
    modelo = MLP(X_train.shape[1], n_salidas=n_salidas, hidden=hidden, dropout=dropout,
                 activacion=activacion).to(device)
    optimizador = torch.optim.Adam(modelo.parameters(), lr=lr, weight_decay=weight_decay)
    funcion_perdida = nn.MSELoss()
    cargador = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch, shuffle=True)
    mejor, espera, mejores_pesos, historia = float("inf"), 0, None, {"train": [], "val": []}
    if epocas_fijas is not None:  # nº de épocas ya decidido fuera, sin parada
        for _ in range(int(epocas_fijas)):
            modelo.train()
            for xb, yb in cargador:
                optimizador.zero_grad(); perdida = funcion_perdida(modelo(xb.to(device)), yb.to(device))
                perdida.backward(); optimizador.step()
        return modelo, historia
    for _ in range(epocas):
        modelo.train()
        for xb, yb in cargador:
            optimizador.zero_grad(); perdida = funcion_perdida(modelo(xb.to(device)), yb.to(device))
            perdida.backward(); optimizador.step()
        modelo.eval()
        with torch.no_grad():
            perdida_train = funcion_perdida(modelo(X_train.to(device)), Y_train.to(device)).item()
            perdida_val = funcion_perdida(modelo(X_val), Y_val).item()
        historia["train"].append(perdida_train); historia["val"].append(perdida_val)
        if perdida_val < mejor:
            mejor, espera = perdida_val, 0
            mejores_pesos = {k: v.detach().cpu().clone() for k, v in modelo.state_dict().items()}
        else:
            espera += 1
            if espera >= paciencia:
                break
    if mejores_pesos:
        modelo.load_state_dict(mejores_pesos)
    return modelo, historia
