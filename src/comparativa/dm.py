"""Contrastes para comparar modelos: Diebold-Mariano, Holm y Model Confidence Set."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _lrv_nw(diferencias: np.ndarray, dbar: float, lags: int) -> float:
    """Varianza HAC (Newey-West)."""
    gamma0 = float(np.mean((diferencias - dbar) ** 2))
    varianza = gamma0
    for k in range(1, lags + 1):
        peso = 1.0 - k / (lags + 1.0)
        covarianza = float(np.mean((diferencias[k:] - dbar) * (diferencias[:-k] - dbar)))
        varianza += 2.0 * peso * covarianza
    return varianza


def diebold_mariano_diff(diferencias, h: int = 1, hac: bool = False, hac_lags: int | None = None) -> dict:
    """Diebold-Mariano sobre el diferencial de pérdida; d_bar < 0 significa que gana el modelo 1."""
    diferencias = np.asarray(diferencias, dtype=float)
    n = len(diferencias)
    if n < 3:
        return {"dm": float("nan"), "p_value": float("nan"),
                "mean_loss_diff": float(np.mean(diferencias)) if n else float("nan"),
                "n": n, "mejor": "indeterminado"}
    dbar = float(diferencias.mean())
    if hac:
        lags = hac_lags if hac_lags is not None else int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
        varianza = _lrv_nw(diferencias, dbar, max(0, lags))
    else:
        varianza = float(np.mean((diferencias - dbar) ** 2))
        for k in range(1, h):
            covarianza = float(np.mean((diferencias[k:] - dbar) * (diferencias[:-k] - dbar)))
            varianza += 2.0 * covarianza
    var_dbar = varianza / n
    if var_dbar <= 0:
        return {"dm": float("nan"), "p_value": float("nan"), "mean_loss_diff": dbar, "n": n,
                "mejor": "indeterminado"}
    dm = dbar / np.sqrt(var_dbar)
    correccion_hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_hln = dm * correccion_hln
    p_valor = float(2 * stats.t.sf(abs(dm_hln), df=n - 1))
    mejor = "modelo1" if dbar < 0 else "modelo2"
    if p_valor > 0.05:
        mejor = "empate (no significativo)"
    return {"dm": float(dm_hln), "p_value": p_valor, "mean_loss_diff": dbar, "n": n, "mejor": mejor,
            "hac": bool(hac)}


def holm(pvals: dict, alpha: float = 0.05) -> dict:
    """Corrección de Holm para comparaciones múltiples."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    salida, previo = {}, 0.0
    for rango, (clave, p_valor) in enumerate(items):
        p_ajustado = min(1.0, (m - rango) * p_valor)
        p_ajustado = max(p_ajustado, previo)
        previo = p_ajustado
        salida[clave] = {"p": float(p_valor), "p_holm": float(p_ajustado),
                         "rechaza": bool(p_ajustado < alpha)}
    return salida


def _block_indices(T: int, block: int, B: int, seed: int = 0) -> np.ndarray:
    """Índices para el bootstrap de bloques móviles."""
    rng = np.random.default_rng(seed)
    block = max(1, min(block, T))
    n_bloques = int(np.ceil(T / block))
    inicios = rng.integers(0, T - block + 1, size=(B, n_bloques))
    desplazamiento = np.arange(block)
    indices = (inicios[:, :, None] + desplazamiento[None, None, :]).reshape(B, -1)[:, :T]
    return indices


def bootstrap_ci(x, stat=np.mean, B: int = 2000, block: int = 14, seed: int = 0,
                 alpha: float = 0.05) -> dict:
    x = np.asarray(x, dtype=float)
    T = len(x)
    indices = _block_indices(T, block, B, seed)
    muestras = np.array([stat(x[i]) for i in indices])
    return {"estimador": float(stat(x)), "lo": float(np.percentile(muestras, 100 * alpha / 2)),
            "hi": float(np.percentile(muestras, 100 * (1 - alpha / 2))), "se": float(muestras.std(ddof=1))}


def model_confidence_set(perdidas, alpha: float = 0.10, B: int = 2000, block: int = 14,
                         seed: int = 0) -> dict:
    """Model Confidence Set: elimina modelos hasta quedarse con los indistinguibles del mejor."""
    perdidas = pd.DataFrame(perdidas).dropna()
    modelos = list(perdidas.columns)
    T = len(perdidas)
    indices = _block_indices(T, block, B, seed)
    supervivientes = modelos.copy()
    orden_eliminacion, p_mcs, p_acumulado = [], {}, 0.0
    while len(supervivientes) > 1:
        perdidas_vivas = perdidas[supervivientes].to_numpy()  # (T, k)
        desviacion = perdidas_vivas - perdidas_vivas.mean(axis=1, keepdims=True)
        dbar = desviacion.mean(axis=0)
        medias_boot = desviacion[indices].mean(axis=1)  # (B, k) medias bootstrap
        varianzas = np.maximum(((medias_boot - dbar) ** 2).mean(axis=0), 1e-12)
        t_i = dbar / np.sqrt(varianzas)
        t_boot = (medias_boot - dbar) / np.sqrt(varianzas)  # (B, k) bajo H0, centrado
        Tmax, Tmax_boot = float(t_i.max()), t_boot.max(axis=1)
        # (1+k)/(B+1) y no la fracción cruda: el estadístico observado cuenta como una realización
        # bajo H0, así el p-valor bootstrap nunca es 0 ni 1 (Davison y Hinkley 1997).
        p_valor = float((1 + (Tmax_boot >= Tmax).sum()) / (len(Tmax_boot) + 1))
        p_acumulado = max(p_acumulado, p_valor)  # el p-valor del MCS es el máximo acumulado
        peor = supervivientes[int(np.argmax(t_i))]  # el de mayor desviación positiva es el peor
        p_mcs[peor] = p_acumulado
        orden_eliminacion.append({"modelo": peor, "p_mcs": round(p_acumulado, 4), "t_max": round(Tmax, 3)})
        if p_valor >= alpha:
            break  # no se rechaza H0, así que todos los supervivientes están en el MCS
        supervivientes.remove(peor)
    for modelo in supervivientes:
        p_mcs.setdefault(modelo, 1.0)
    return {"alpha": alpha, "B": B, "block": block, "conjunto_mcs": supervivientes,
            "p_mcs_por_modelo": {m: round(float(p_mcs.get(m, 1.0)), 4) for m in modelos},
            "orden_eliminacion": orden_eliminacion}
