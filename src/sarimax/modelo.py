"""SARIMAX en dos estructuras: serie horaria continua y 24 modelos por hora. Horizonte D+1."""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.metricas import ancla_mase
from src.sarimax.datos_sarimax import (
    serie_horaria, panel_por_hora, exog_por_hora,
    exog_horaria_continua, lags_horarios, lags_diarios_hora)

DIR_METRICAS = RAIZ / "resultados" / "metricas"

# Órdenes por estructura, de la identificación Box-Jenkins manual.
# Continua: Δ₂₄ (D=1, s=24) para el ciclo diario, AR(2) intradía y AR estacional s=24; el ciclo
# semanal entra como lag t-168 exógeno. Las MA quedan fuera por coste y estabilidad.
ORDER_CONTINUA = (2, 0, 0)
SEASONAL_CONTINUA = (1, 1, 0, 24)
LAGS_CONTINUA = (168,)  # la semana, como exógena: es la segunda estacionalidad
VENTANA_ANOS_CONTINUA = None  # None = historia completa; un número recorta el train a esos años

# 24 modelos por hora: rejilla pequeña, con el orden elegido por AICc hora a hora. Δ₇ (semanal) en todas.
GRID_ORDER_B = [(1, 0, 0), (2, 0, 0), (2, 0, 1)]
GRID_SEASONAL_B = [(1, 1, 0, 7), (0, 1, 1, 7)]
LAGS_DIAS_B = (1, 2, 7, 14)  # t-24, t-48, t-168 y t-336, estandarizados

MAXITER_DEFECTO = 500


def _metricas(real: np.ndarray, pred: np.ndarray, ancla: float) -> dict:
    real, pred = np.asarray(real, float), np.asarray(pred, float)
    errores = real - pred
    errores_abs = np.abs(errores)
    return {"MAE": float(errores_abs.mean()), "RMSE": float(np.sqrt(np.mean(errores ** 2))),
            "MAPE": float(np.mean(errores_abs / real) * 100),
            "MASE": float(errores_abs.mean() / ancla),
            "n": int(len(real))}


def _fit_sarimax(y: pd.Series, X: pd.DataFrame | None, order, seasonal_order,
                 maxiter: int = MAXITER_DEFECTO):
    # `approximate_diffuse` + `enforce_*=False`: sin ellos la init estacionaria falla y el optimizador no converge.
    exog = None if X is None else X.to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        modelo = SARIMAX(y.to_numpy(), exog=exog, order=order, seasonal_order=seasonal_order,
                         enforce_stationarity=False, enforce_invertibility=False)
        modelo.initialize_approximate_diffuse()
        return modelo.fit(disp=False, maxiter=maxiter)


def _aicc(resultado) -> float:
    k, n = resultado.params.size, resultado.nobs
    denominador = n - k - 1
    return float(resultado.aic) + (2 * k * (k + 1) / denominador if denominador > 0 else np.inf)


def _converged(resultado) -> bool:
    try:
        return bool(resultado.mle_retvals.get("converged"))
    except Exception as error:  # sin `mle_retvals` no se puede saber: se reporta como no convergido
        print(f"  [aviso] convergencia no consultable ({type(error).__name__}: {error}); "
              f"se reporta converged=False.", flush=True)
        return False


def _diagnostico_resid(resid: np.ndarray, periodo_estacional: int) -> dict:
    resid = np.asarray(resid, float)
    resid = resid[np.isfinite(resid)]
    lags_ljung_box = ([periodo_estacional, 2 * periodo_estacional] if periodo_estacional else [10, 20])
    nlags = max(2 * periodo_estacional, 20) if periodo_estacional else 20
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ljung_box = acorr_ljungbox(resid, lags=lags_ljung_box, return_df=True)
        acf_valores = acf(resid, nlags=min(nlags, len(resid) - 1))
        jarque_bera_p = float(stats.jarque_bera(resid)[1])
    acf_abs = np.abs(acf_valores[1:])
    return {"ljungbox_lags": lags_ljung_box,
            "ljungbox_p": [float(valor) for valor in ljung_box["lb_pvalue"].to_numpy()],
            "jarque_bera_p": jarque_bera_p,
            "acf_resid_estacional": (float(acf_valores[periodo_estacional])
                                     if periodo_estacional and periodo_estacional < len(acf_valores)
                                     else None),
            "acf_abs_max": float(acf_abs.max()) if acf_abs.size else None,
            "acf_abs_media": float(acf_abs.mean()) if acf_abs.size else None,
            "resid_std": float(np.std(resid)), "resid_media": float(np.mean(resid))}


def _resumen_resid(resid: np.ndarray) -> dict:
    resid = np.asarray(resid, float)
    resid = resid[np.isfinite(resid)]
    return {"n": int(resid.size), "media": float(resid.mean()), "std": float(resid.std()),
            "min": float(resid.min()), "max": float(resid.max())}


def _pred_1paso(resultado, ypost: pd.Series, Xpost: pd.DataFrame | None, eval_dias):
    """Predicción 1-paso (D+1) + IC 95 % sobre `eval_dias`, extendiendo el filtro con `ypost`."""
    n_ajuste = resultado.nobs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        resultado_ext = resultado.append(ypost.to_numpy(),
                                         exog=(None if Xpost is None else Xpost.to_numpy()),
                                         refit=False)
        prediccion = resultado_ext.get_prediction(start=n_ajuste, dynamic=False)
        media = np.asarray(prediccion.predicted_mean)
        intervalo = np.asarray(prediccion.conf_int(alpha=0.05))
    mascara = ypost.index.isin(eval_dias)
    return (ypost.index[mascara], ypost.to_numpy()[mascara], media[mascara],
            intervalo[mascara, 0], intervalo[mascara, 1])


# 24 modelos, uno por hora del día.
def _estandarizar_lags(X: pd.DataFrame, tr_mask: np.ndarray) -> pd.DataFrame:
    """z-score train-only de las columnas de lag (en MW). Las demás exógenas quedan intactas."""
    columnas = [columna for columna in X.columns if columna.startswith("lag_")]
    if not columnas:
        return X
    medias = X.loc[tr_mask, columnas].mean()
    desviaciones = X.loc[tr_mask, columnas].std().replace(0, 1.0)
    X_escalada = X.copy()
    X_escalada[columnas] = (X[columnas] - medias) / desviaciones
    return X_escalada


def evaluar_hora_b(hora: int, train_dias, eval_dias, ancla: float) -> dict:
    """Un modelo horario (hora `hora`): SARIMAX Δ₇ + lags estandarizados, orden por AICc, eval D+1."""
    panel = panel_por_hora()
    y = panel[f"h{hora:02d}"]
    X = pd.concat([exog_por_hora(hora, panel.index),
                   lags_diarios_hora(panel, hora, lags_dias=LAGS_DIAS_B)], axis=1)
    validos = X.notna().all(axis=1) & y.notna()
    y, X = y[validos], X[validos]
    train = y.index.isin(train_dias)
    X = _estandarizar_lags(X, train)
    yfit, Xfit = y[train], X[train]

    # Selección de orden por AICc (sin usar el eval).
    mejor = None
    for order in GRID_ORDER_B:
        for orden_estacional in GRID_SEASONAL_B:
            try:
                candidato = _fit_sarimax(yfit, Xfit, order, orden_estacional)
                aicc_cand = _aicc(candidato)
            except Exception as error:  # el orden no ajusta: queda fuera de la comparación por AICc
                print(f"  [aviso] hora {hora:02d}: {order}x{orden_estacional} no ajusta "
                      f"({type(error).__name__}: {error}); se descarta.", flush=True)
                continue
            if mejor is None or aicc_cand < mejor[0]:
                mejor = (aicc_cand, order, orden_estacional, candidato)
    if mejor is None:
        raise RuntimeError(f"hora {hora:02d}: ningún orden ajustó.")
    aicc, order, orden_estacional, resultado = mejor

    fin_fit = yfit.index.max()
    eval_max = pd.DatetimeIndex(eval_dias).max()
    post = (y.index > fin_fit) & (y.index <= eval_max)
    fechas, real, pred, lo, hi = _pred_1paso(resultado, y[post], X[post], eval_dias)
    return {"hora": int(hora), "order": list(order), "seasonal_order": list(orden_estacional),
            "aic": round(float(resultado.aic), 1), "aicc": round(aicc, 1),
            "converged": _converged(resultado),
            "idx": fechas, "real": real, "pred": pred, "lo95": lo, "hi95": hi,
            "diagnosticos": _diagnostico_resid(resultado.resid, orden_estacional[3]),
            "metricas": _metricas(real, pred, ancla)}


def combinar_por_hora(por_hora: list[dict], ancla: float, segundos: float) -> dict:
    """Agrega los 24 dicts de `evaluar_hora_b` → métricas globales (ancla HORARIA compartida) + artefactos."""
    fechas = np.concatenate([np.asarray(resultado["idx"]) for resultado in por_hora])
    horas = np.concatenate([np.full(len(resultado["idx"]), resultado["hora"])
                            for resultado in por_hora])
    real = np.concatenate([resultado["real"] for resultado in por_hora])
    pred = np.concatenate([resultado["pred"] for resultado in por_hora])
    lo = np.concatenate([resultado["lo95"] for resultado in por_hora])
    hi = np.concatenate([resultado["hi95"] for resultado in por_hora])
    df_pred = pd.DataFrame({"datetime": pd.DatetimeIndex(fechas), "hora": horas, "real": real,
                            "pred": pred, "lo95": lo, "hi95": hi})
    metricas = _metricas(real, pred, ancla)
    cobertura = float(np.mean((real >= lo) & (real <= hi)))
    resumen_horas = [{"hora": resultado["hora"], "order": resultado["order"],
                      "seasonal_order": resultado["seasonal_order"],
                      "aicc": resultado["aicc"], "converged": resultado["converged"],
                      "MASE": resultado["metricas"]["MASE"], "MAE": resultado["metricas"]["MAE"],
                      "acf_abs_max": resultado["diagnosticos"]["acf_abs_max"]}
                     for resultado in por_hora]
    return {"estructura": "por_hora", "ancla_mw": round(ancla, 2),
            "converged": bool(all(resultado["converged"] for resultado in por_hora)),
            "metricas": metricas, "cobertura_95": cobertura,
            "por_hora": resumen_horas,
            "diagnosticos": {"acf_abs_max_medio":
                             float(np.mean([resultado["diagnosticos"]["acf_abs_max"]
                                            for resultado in por_hora])),
                             "por_hora": [resultado["diagnosticos"] for resultado in por_hora]},
            "config": {"lags_dias": list(LAGS_DIAS_B), "grid_order": [list(o) for o in GRID_ORDER_B],
                       "grid_seasonal": [list(o) for o in GRID_SEASONAL_B],
                       "seleccion_orden": "AICc por hora"},
            "segundos": round(segundos, 1), "pred": df_pred}


# Serie horaria continua, s=24.
def evaluar_continua(train_dias, eval_dias, order=ORDER_CONTINUA, seasonal_order=SEASONAL_CONTINUA,
                     lags=LAGS_CONTINUA, ventana_anos: float = VENTANA_ANOS_CONTINUA,
                     maxiter: int = MAXITER_DEFECTO) -> dict:
    """SARIMAX sobre la serie horaria continua (Δ₂₄, s=24, lag t-168 exógeno) con rolling day-ahead."""
    inicio = time.time()
    serie = serie_horaria()
    X = pd.concat([exog_horaria_continua(serie.index), lags_horarios(serie, lags)], axis=1)
    validos = X.notna().all(axis=1) & serie.notna()
    serie, X = serie[validos], X[validos]

    train_dias = pd.DatetimeIndex(train_dias)
    eval_dias = pd.DatetimeIndex(eval_dias)
    val_ini = eval_dias.min()
    dias_serie = serie.index.normalize()
    mascara_ajuste = dias_serie.isin(train_dias)
    if ventana_anos:  # con None o 0 se usa la historia completa
        desde = val_ini - pd.Timedelta(days=round(ventana_anos * 365))
        mascara_ajuste = mascara_ajuste & (serie.index >= desde)
    yfit = serie[mascara_ajuste]

    # Diferenciación estacional Δ_s MANUAL (con D=1,s=24 el espacio de estados no cabe en memoria):
    # se ajusta w=y−y_{t−s} con D=0 y se re-integra ŷ(t)=ŵ(t)+y(t−s), observado al predecir D+1.
    periodo_estacional = seasonal_order[3]
    orden_estacional_sin_dif = (seasonal_order[0], 0, seasonal_order[2], periodo_estacional)
    serie_dif = serie - serie.shift(periodo_estacional)
    X_dif = X - X.shift(periodo_estacional)
    dias_dif = serie_dif.index.normalize()
    w_ajuste = serie_dif[mascara_ajuste].dropna()
    X_dif_ajuste = X_dif.loc[w_ajuste.index]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        modelo = SARIMAX(w_ajuste.to_numpy(), exog=X_dif_ajuste.to_numpy(), order=order,
                         seasonal_order=orden_estacional_sin_dif, enforce_stationarity=False,
                         enforce_invertibility=False)
        resultado = modelo.fit(disp=False, maxiter=maxiter)
    convergio = _converged(resultado)
    resid_fit = resultado.resid
    # AIC del ajuste, antes del rolling extend, para que sea comparable con el de la otra estructura.
    aic_fit = float(resultado.aic)
    train_desde = str(w_ajuste.index.min().date())  # primer instante realmente ajustado

    # Rolling day-ahead: desde el día siguiente al fin del fit hasta el fin del eval.
    dias_disponibles = pd.DatetimeIndex(dias_serie.unique()).sort_values()
    fin_fit_dia = w_ajuste.index.normalize().max()
    dias_rolling = dias_disponibles[(dias_disponibles > fin_fit_dia)
                                    & (dias_disponibles <= eval_dias.max())]
    reales, predicciones, limites_inf, limites_sup, fechas_obj, horas_obj = [], [], [], [], [], []
    un_dia = pd.Timedelta(days=1)
    for dia_objetivo in dias_rolling:
        mascara_dia = dias_dif == dia_objetivo
        X_dif_dia, w_dia = X_dif[mascara_dia], serie_dif[mascara_dia]
        real_dia = serie[dias_serie == dia_objetivo]
        # y(t−s): la demanda del día anterior, ya observada
        prev_dia = serie[dias_serie == (dia_objetivo - un_dia)]
        if (len(X_dif_dia) == 24 and len(prev_dia) == 24 and dia_objetivo in eval_dias
                and not w_dia.isna().any()):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                forecast = resultado.get_forecast(steps=24, exog=X_dif_dia.to_numpy())
                w_media = np.asarray(forecast.predicted_mean)
                intervalo = np.asarray(forecast.conf_int(alpha=0.05))
            base = prev_dia.to_numpy()  # para re-integrar al nivel
            reales.append(real_dia.to_numpy()); predicciones.append(w_media + base)
            limites_inf.append(intervalo[:, 0] + base); limites_sup.append(intervalo[:, 1] + base)
            fechas_obj.extend(real_dia.index); horas_obj.extend(real_dia.index.hour)
        # Avanza el origen con la diferencia observada del día. `extend` (no `append`) reusa parámetros
        # y estado sin re-almacenar la historia: memoria plana y equivalente numéricamente.
        if len(w_dia) == 24 and not w_dia.isna().any() and len(X_dif_dia) == 24:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                resultado = resultado.extend(w_dia.to_numpy(), exog=X_dif_dia.to_numpy())

    real = np.concatenate(reales); pred = np.concatenate(predicciones)
    lo = np.concatenate(limites_inf); hi = np.concatenate(limites_sup)
    ancla = ancla_mase()
    metricas = _metricas(real, pred, ancla)
    cobertura = float(np.mean((real >= lo) & (real <= hi)))
    df_pred = pd.DataFrame({"datetime": pd.DatetimeIndex(fechas_obj), "hora": horas_obj,
                            "real": real, "pred": pred, "lo95": lo, "hi95": hi})
    df_horas = pd.DataFrame({"hora": horas_obj, "real": real, "pred": pred})
    por_hora = [{"hora": int(hora), **{clave: valor for clave, valor
                                       in _metricas(grupo["real"].to_numpy(),
                                                    grupo["pred"].to_numpy(), ancla).items()
                                       if clave in ("MAE", "MAPE", "MASE")}}
                for hora, grupo in df_horas.groupby("hora")]
    return {"estructura": "continua", "order": list(order),
            "seasonal_order": list(seasonal_order), "converged": convergio,
            "ancla_mw": round(ancla, 2),
            "aic": round(aic_fit, 1),
            "metricas": metricas, "cobertura_95": cobertura, "por_hora": por_hora,
            "diagnosticos": _diagnostico_resid(resid_fit, seasonal_order[3]),
            "residuos_resumen": _resumen_resid(resid_fit),
            "config": {"lags_exog": list(lags), "ventana_anos": ventana_anos,
                       "train_desde": train_desde,
                       "fit_n": int(len(yfit)), "maxiter": maxiter},
            "segundos": round(time.time() - inicio, 1), "pred": df_pred}


if __name__ == "__main__":
    # Smoke barato: una hora del panel horario sobre un fold corto.
    from src.sarimax.datos_sarimax import rolling_origin_folds
    fold = rolling_origin_folds()[0]
    train = fold.train[fold.train >= pd.Timestamp("2012-01-01")]  # train corto para el smoke
    eval_dias = fold.val[:20]
    print("== hora 13 (smoke) ==")
    resultado_b = evaluar_hora_b(13, train, eval_dias, ancla_mase())
    print({clave: resultado_b[clave] for clave in ("order", "aicc", "converged", "metricas")})
