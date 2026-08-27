"""Tuning de redes (MLP y LSTM-exo) por random search con rolling-origin de 5 folds; reanudable."""
from __future__ import annotations

import itertools
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.redes.datos import escalar_tabla_mlp, secuencias_lstm_exo
from src.redes.entrenamiento import entrenar_mlp, fijar_semilla
from src.features.validacion import split_canonico_precovid

DIR_OUT = RAIZ / "resultados" / "metricas"


def _muestrear(espacio: dict, n: int, semilla: int = 0) -> list[dict]:
    """`n` configuraciones distintas del producto cartesiano del espacio, con barajado determinista."""
    combinaciones = [dict(zip(espacio, valores))
                     for valores in itertools.product(*espacio.values())]
    random.Random(semilla).shuffle(combinaciones)
    return combinaciones[:n]


def _id_config(prefijo: str, config: dict) -> str:
    return prefijo + ":" + ";".join(f"{clave}={config[clave]}" for clave in sorted(config))


def _ruta_parcial(familia: str) -> Path:
    return DIR_OUT / f"tuning_{familia}.json"


def _cargar_registros(familia: str) -> list[dict]:
    """Configuraciones ya evaluadas de esa familia (lista vacía si es la primera corrida)."""
    ruta = _ruta_parcial(familia)
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else []


def _guardar_registros(familia: str, registros: list[dict]):
    DIR_OUT.mkdir(parents=True, exist_ok=True)
    _ruta_parcial(familia).write_text(json.dumps(registros, indent=2, ensure_ascii=False), encoding="utf-8")


def _mae_va_mlp(modelo, datos_escalados) -> float:
    import torch
    dispositivo = next(modelo.parameters()).device
    X_val = torch.tensor(datos_escalados["X"][datos_escalados["mascaras"]["va"]].to_numpy(),
                         dtype=torch.float32).to(dispositivo)
    real = (datos_escalados["Y"][datos_escalados["mascaras"]["va"]].to_numpy()
            * datos_escalados["y_desv"] + datos_escalados["y_media"])
    modelo.eval()
    with torch.no_grad():
        prediccion = modelo(X_val).cpu().numpy() * datos_escalados["y_desv"] + datos_escalados["y_media"]
    return float(np.abs(real - prediccion).mean())


# Protocolo de parada separada: la parada se decide sobre una cola reservada del train del corte y
# luego se reentrena con el train completo ese nº de épocas, para que la validación que puntúa no
# intervenga en nada.
FRAC_PARADA = 0.15  # fracción del train del corte que se reserva para decidir la parada
MIN_PARADA = 90  # suelo en días-origen (~3 meses): con menos, la curva de parada es puro ruido
MIN_EPOCAS = 5  # suelo del nº de épocas estimado, para no reentrenar un modelo sin entrenar


def _particion_parada(mascara_train: np.ndarray, frac: float = FRAC_PARADA,
                      minimo: int = MIN_PARADA) -> tuple[np.ndarray, np.ndarray]:
    """Divide la máscara de train de un corte en (train reducido, cola de parada), sin barajar."""
    posiciones = np.flatnonzero(np.asarray(mascara_train))
    if len(posiciones) < 3 * minimo:
        raise ValueError(
            f"train del corte demasiado corto ({len(posiciones)}) para reservar {minimo} de parada")
    n_parada = max(minimo, int(round(frac * len(posiciones))))
    corte = posiciones[-n_parada:]
    train_reducido = np.asarray(mascara_train).copy()
    train_reducido[corte] = False
    cola_parada = np.zeros_like(train_reducido)
    cola_parada[corte] = True
    return train_reducido, cola_parada


def _comprobar_disjuntos(train_reducido: np.ndarray, cola_parada: np.ndarray, val: np.ndarray):
    assert not (cola_parada & val).any(), "el tramo de parada se solapa con el de puntuación (protocolo roto)"
    assert not (train_reducido & val).any(), "el train se solapa con el de puntuación"
    assert cola_parada.sum() > 0 and train_reducido.sum() > 0 and val.sum() > 0, \
        "algún tramo del corte ha quedado vacío"


def _epocas_parada_mlp(datos_escalados, config, n_salidas, semilla=0) -> int:
    """Nº de épocas de una configuración de MLP, por parada temprana sobre la cola del train del corte."""
    train_reducido, cola_parada = _particion_parada(datos_escalados["mascaras"]["tr"])
    _comprobar_disjuntos(train_reducido, cola_parada, datos_escalados["mascaras"]["va"])
    datos_parada = {**datos_escalados,
                    "mascaras": {"tr": train_reducido, "va": cola_parada,
                                 "te": datos_escalados["mascaras"]["te"]}}
    _, historia = entrenar_mlp(datos_parada, n_salidas=n_salidas, semilla=semilla,
                               hidden=tuple(config["hidden"]),
                               dropout=config["dropout"], lr=config["lr"],
                               weight_decay=config["weight_decay"], batch=config["batch"],
                               activacion=config["activacion"])
    return max(MIN_EPOCAS, int(np.argmin(historia["val"])) + 1)


def _maes_mlp_parada_separada(config, folds, datos, columnas_x, columnas_y,
                              epocas_por_corte=False) -> tuple[list, list]:
    """MAE de validación de cada corte del MLP con parada separada. Devuelve (maes, épocas)."""
    n_salidas = len(columnas_y)
    n_fijo = None
    if not epocas_por_corte:
        datos_ultimo = escalar_tabla_mlp(datos, columnas_x, columnas_y, folds[-1].train, folds[-1].val)
        n_fijo = _epocas_parada_mlp(datos_ultimo, config, n_salidas)
    maes, epocas = [], []
    for fold in folds:
        datos_escalados = escalar_tabla_mlp(datos, columnas_x, columnas_y, fold.train, fold.val)
        n_epocas = n_fijo if n_fijo is not None else _epocas_parada_mlp(datos_escalados, config, n_salidas)
        modelo, _ = entrenar_mlp(datos_escalados, n_salidas=n_salidas, semilla=0,
                                 hidden=tuple(config["hidden"]),
                                 dropout=config["dropout"], lr=config["lr"],
                                 weight_decay=config["weight_decay"], batch=config["batch"],
                                 activacion=config["activacion"], epocas_fijas=n_epocas)
        maes.append(_mae_va_mlp(modelo, datos_escalados))
        epocas.append(n_epocas)
    return maes, epocas


def _normalizar_config(config: dict, espacio: dict) -> dict:
    """Alinea una config escrita a mano con la forma del producto cartesiano del espacio (mismo `_id_config`)."""
    salida = dict(config)
    for clave, valores in espacio.items():
        if clave in salida and any(isinstance(valor, tuple) for valor in valores):
            salida[clave] = tuple(salida[clave])
    return salida


def _orden_configs(espacio, n_configs, semilla, forzar=None) -> list[dict]:
    """Muestra del espacio con la configuración de control (`forzar`) colocada la primera."""
    configs = _muestrear(espacio, n_configs, semilla=semilla)
    if forzar is None:
        return configs
    forzar = _normalizar_config(forzar, espacio)
    resto = [config for config in configs if config != forzar]
    return [forzar] + resto[:max(0, n_configs - 1)]


def buscar_mlp(folds, datos, columnas_x, columnas_y, espacio, n_configs, presupuesto, t0=None,
               etiqueta="mlp", epocas_por_corte=False, forzar=None) -> list[dict]:
    """Búsqueda del MLP sobre `espacio`; `etiqueta` nombra el fichero de estado y `forzar` va la primera."""
    # t0=None significa "ahora". Un `t0=0.0` literal haría que `time.time() - 0` (≈1,8e9 s) superase
    # el presupuesto y la búsqueda se cortase en la primera configuración.
    t0 = t0 if t0 else time.time()
    registros = _cargar_registros(etiqueta)
    hechas = {registro["id"] for registro in registros}
    if hechas:
        print(f"  [MLP] reanudando: {len(hechas)} configuraciones ya evaluadas", flush=True)
    for config in _orden_configs(espacio, n_configs, semilla=0, forzar=forzar):
        config_id = _id_config("mlp", config)
        if config_id in hechas:
            continue
        if time.time() - t0 > presupuesto:
            print("  [MLP] presupuesto agotado, paro limpio (resumible)", flush=True); break
        maes, epocas = _maes_mlp_parada_separada(config, folds, datos, columnas_x, columnas_y,
                                                 epocas_por_corte)
        registro = {"id": config_id, "config": config, "mae_oos_folds": maes,
                    "mae_oos_medio": float(np.mean(maes)),
                    "mae_oos_se": float(np.std(maes, ddof=1) / np.sqrt(len(maes))),
                    "epocas_por_fold": epocas}
        registros.append(registro); _guardar_registros(etiqueta, registros)
        print(f"  [MLP] {len(registros)}/{n_configs} {config_id}: MAE_oos {registro['mae_oos_medio']:.1f} "
              f"± {registro['mae_oos_se']:.1f}", flush=True)
    return sorted(registros, key=lambda registro: registro["mae_oos_medio"])


def _entrenar_lstm_exo(X_secuencias, X_exogenas, Y_escalada, train, val, config, n_salidas, semilla=0,
                       epocas=300, paciencia=20, batch=None, epocas_fijas=None):
    """Entrena el LSTM-exo de un corte, con parada temprana o con `epocas_fijas` (`val` puede ser None)."""
    batch = batch if batch is not None else config["batch"]
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    from src.redes.modelos import LSTMRedExo
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fijar_semilla(semilla)
    f32 = lambda a: torch.tensor(a, dtype=torch.float32)
    modelo = LSTMRedExo(n_features=X_secuencias.shape[2], n_exogenas=X_exogenas.shape[1],
                        hidden=config["hidden"], capas=1, n_salidas=n_salidas,
                        dropout=config["dropout"],
                        activacion=config["activacion"]).to(device)
    optimizador = torch.optim.Adam(modelo.parameters(), lr=config["lr"],
                                   weight_decay=config["weight_decay"])
    funcion_perdida = nn.MSELoss()
    cargador = DataLoader(TensorDataset(f32(X_secuencias[train]), f32(X_exogenas[train]),
                                        f32(Y_escalada[train])), batch_size=batch, shuffle=True)
    historia = {"train": [], "val": []}
    if epocas_fijas is not None:  # nº de épocas ya decidido fuera, sin parada temprana
        for _ in range(int(epocas_fijas)):
            modelo.train()
            for xb, eb, yb in cargador:
                optimizador.zero_grad()
                perdida = funcion_perdida(modelo(xb.to(device), eb.to(device)), yb.to(device))
                perdida.backward(); optimizador.step()
        return modelo, historia
    X_train_t = f32(X_secuencias[train]).to(device)
    exo_train_t = f32(X_exogenas[train]).to(device)
    Y_train_t = f32(Y_escalada[train]).to(device)
    X_val_t = f32(X_secuencias[val]).to(device)
    exo_val_t = f32(X_exogenas[val]).to(device)
    Y_val_t = f32(Y_escalada[val]).to(device)
    mejor, espera, mejores_pesos = float("inf"), 0, None
    for _ in range(epocas):
        modelo.train()
        for xb, eb, yb in cargador:
            optimizador.zero_grad()
            perdida = funcion_perdida(modelo(xb.to(device), eb.to(device)), yb.to(device))
            perdida.backward(); optimizador.step()
        modelo.eval()
        with torch.no_grad():
            perdida_train = funcion_perdida(modelo(X_train_t, exo_train_t), Y_train_t).item()
            perdida_val = funcion_perdida(modelo(X_val_t, exo_val_t), Y_val_t).item()
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


def _prep_lstm(L: int):
    """(Xseq, Xexo, Y, fechas) + máscaras del split canónico para un L dado."""
    Xseq, Xexo, Y, fechas = secuencias_lstm_exo(n_dias=1, L=L)
    split = split_canonico_precovid(gap_dias=1, guardar=False)
    mascaras = {nombre: fechas.isin(getattr(split, nombre)) for nombre in ("train", "val", "test")}
    return Xseq, Xexo, Y, fechas, mascaras


def _escalar_fold_lstm(Xseq, Xexo, Y, train):
    """Escalado anti-fuga de un corte del LSTM, con estadísticos solo del train del corte."""
    media, desv = Xseq[train].mean(), (Xseq[train].std() or 1.0)
    exo_media = Xexo[train].mean(axis=0)
    exo_desv = Xexo[train].std(axis=0); exo_desv[exo_desv == 0] = 1.0
    y_media, y_desv = float(Y[train].mean()), float(Y[train].std())
    return ((Xseq - media) / desv, (Xexo - exo_media) / exo_desv, (Y - y_media) / y_desv,
            y_media, y_desv)


def _epocas_parada_lstm(X_secuencias, X_exogenas, Y_escalada, train, val, config, n_salidas, semilla=0) -> int:
    """Nº de épocas de una configuración de LSTM, por parada temprana sobre la cola del train del corte."""
    train_reducido, cola_parada = _particion_parada(train)
    _comprobar_disjuntos(train_reducido, cola_parada, val)
    _, historia = _entrenar_lstm_exo(X_secuencias, X_exogenas, Y_escalada, train_reducido, cola_parada,
                                     config, n_salidas, semilla=semilla)
    return max(MIN_EPOCAS, int(np.argmin(historia["val"])) + 1)


def _maes_lstm_parada_separada(config, folds, Xseq, Xexo, Y, fechas, epocas_por_corte=False):
    """Como `_maes_mlp_parada_separada`, para el LSTM-exo."""
    import torch
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    n_salidas = Y.shape[1]
    mascaras = [(fechas.isin(fold.train), fechas.isin(fold.val)) for fold in folds]
    n_fijo = None
    if not epocas_por_corte:
        train, val = mascaras[-1]
        X_esc, exo_esc, Y_esc, _, _ = _escalar_fold_lstm(Xseq, Xexo, Y, train)
        n_fijo = _epocas_parada_lstm(X_esc, exo_esc, Y_esc, train, val, config, n_salidas)
    maes, epocas = [], []
    for train, val in mascaras:
        X_esc, exo_esc, Y_esc, y_media, y_desv = _escalar_fold_lstm(Xseq, Xexo, Y, train)
        n_epocas = (n_fijo if n_fijo is not None
                    else _epocas_parada_lstm(X_esc, exo_esc, Y_esc, train, val, config, n_salidas))
        modelo, _ = _entrenar_lstm_exo(X_esc, exo_esc, Y_esc, train, None, config, n_salidas, semilla=0,
                                       epocas_fijas=n_epocas)
        modelo.eval()
        with torch.no_grad():
            prediccion = modelo(torch.tensor(X_esc[val], dtype=torch.float32).to(dispositivo),
                                torch.tensor(exo_esc[val], dtype=torch.float32).to(dispositivo)
                                ).cpu().numpy() * y_desv + y_media
        maes.append(float(np.abs(Y[val] - prediccion).mean()))
        epocas.append(n_epocas)
    return maes, epocas


def buscar_lstm(folds, espacio, n_configs, presupuesto, t0=None, etiqueta="lstm",
                epocas_por_corte=False, forzar=None) -> list[dict]:
    """Búsqueda del LSTM. `espacio`/`etiqueta`/`epocas_por_corte`/`forzar` como en `buscar_mlp`."""
    t0 = t0 if t0 else time.time()  # ver la nota en `buscar_mlp`
    registros = _cargar_registros(etiqueta)
    hechas = {registro["id"] for registro in registros}
    if hechas:
        print(f"  [LSTM] reanudando: {len(hechas)} configuraciones ya evaluadas", flush=True)
    cache = {}  # L -> datos preparados, se reutilizan entre configuraciones con el mismo L
    for config in _orden_configs(espacio, n_configs, semilla=1, forzar=forzar):
        config_id = _id_config("lstm", config)
        if config_id in hechas:
            continue
        if time.time() - t0 > presupuesto:
            print("  [LSTM] presupuesto agotado, paro limpio (resumible)", flush=True); break
        L = config["L"]
        if L not in cache:
            cache[L] = secuencias_lstm_exo(n_dias=1, L=L)
        Xseq, Xexo, Y, fechas = cache[L]
        maes, epocas = _maes_lstm_parada_separada(config, folds, Xseq, Xexo, Y, fechas, epocas_por_corte)
        registro = {"id": config_id, "config": config, "mae_oos_folds": maes,
                    "mae_oos_medio": float(np.mean(maes)),
                    "mae_oos_se": float(np.std(maes, ddof=1) / np.sqrt(len(maes))),
                    "epocas_por_fold": epocas}
        registros.append(registro); _guardar_registros(etiqueta, registros)
        print(f"  [LSTM] {len(registros)}/{n_configs} {config_id}: MAE_oos {registro['mae_oos_medio']:.1f} "
              f"± {registro['mae_oos_se']:.1f}", flush=True)
    return sorted(registros, key=lambda registro: registro["mae_oos_medio"])
