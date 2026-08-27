"""Ejecuta el EDA desde scripts y regenera sus figuras en figuras/eda/ (el efecto de los
festivos se regenera aparte, con `python -m src.eda.efecto_festivos`, y la relación
temperatura-demanda también, con `python -m src.eda.temperatura_demanda`, porque necesita
el parquet de temperatura que genera `src.features.run_features`).

Uso:  python -m src.eda.run_eda
"""
from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src import plots
from src.datos import cargar_demanda
from src.datos_precovid import serie_precovid
from src.eda import (autocorrelacion, calendario, calidad,
                     estacionalidad, estacionariedad, multivariante, regresion)


def main():
    plots.usar_estilo("academico")
    serie = serie_precovid()

    
    print("Integridad (serie cruda):", calidad.integridad(cargar_demanda()))
    print("Integridad (serie pre-COVID limpia):", calidad.integridad(serie.to_frame()))
    calidad.distribucion(serie)

    estacionalidad.serie_global(serie); estacionalidad.media_anual(serie)
    estacionalidad.perfil_mensual(serie)
    estacionalidad.perfil_horario(serie); estacionalidad.heatmaps(serie)
    stl = estacionalidad.stl_diaria(serie)
    mstl, _ = estacionalidad.descomposicion_mstl(serie)
    print("Descomposición solo semanal:", estacionalidad.reparto_varianza(stl))
    print("Descomposición doble (24 h + 168 h):", estacionalidad.reparto_varianza(mstl))

    autocorrelacion.acf_pacf(serie)
    print("Correlaciones lag:", {k: round(v, 3) for k, v in autocorrelacion.correlaciones_lag(serie).items()})

    calendario.festivos_total(serie)
    print("Verano/Navidad:", {k: round(v, 1) for k, v in calendario.verano_navidad(serie)[1].items()})

    print("Estacionariedad (media diaria):",
          {k: round(v, 4) for k, v in estacionariedad.tests_estacionariedad(serie).items()})
    print("Estacionariedad (serie horaria):")
    for nombre, dic in estacionariedad.tests_estacionariedad_horaria(serie).items():
        print(f"   {nombre:10}", {k: round(v, 4) for k, v in dic.items()})
    estacionariedad.volatilidad(serie)

    _, _, info_atip = multivariante.deteccion_atipicos(serie)
    print("Días atípicos:", info_atip)
    print("R² regresión lineal (calendario):", round(regresion.baseline_regresion_lineal(serie), 3))
    print("VIF lags:", regresion.vif_lags(serie).round(1).to_dict())

    print("EDA completo: figuras en figuras/eda/")


if __name__ == "__main__":
    main()
