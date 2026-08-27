# Predicción de la demanda eléctrica peninsular

Predice la demanda eléctrica horaria de la España peninsular a un día vista: 
dado lo observado hasta el final del día D, las 24 horas del día D+1. Se comparan SARIMAX,
redes neuronales (MLP y LSTM), Random Forest y XGBoost, una regresión lineal y
varios modelos ingenuos, todos sobre los mismos datos y el mismo protocolo de validación.

## Instalación

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Todo funciona en CPU; solo las redes se benefician de una GPU. `requirements.txt` instala la build de
PyTorch con CUDA, que es la que PyPI sirve por defecto en Linux y ocupa unos 2 GB. Si no hay GPU, la
build de CPU es mucho más ligera y hay que pedirla explícitamente:

```bash
.venv/bin/pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
```

## Datos

Los dos CSV vienen en el repositorio:

- `Peninsula.csv` — volcado crudo del operador, de 2000 a 2024.
- `Peninsula_precovid.csv` — la serie de trabajo: limpia, recortada a febrero de 2020, 176.760 horas.

El segundo se deriva del primero y se regenera con `.venv/bin/python -m src.datos_precovid`.

La temperatura no viaja en el repositorio: se descarga de Open-Meteo la primera vez que se ejecuta
`src.features.run_features` y queda cacheada en `resultados/`. Son veinte ciudades desde 1997 y las
peticiones van espaciadas, así que ese paso tarda un rato.

## Orden de ejecución

Cada etapa consume lo que dejó la anterior:

```bash
.venv/bin/python -m src.eda.run_eda                       # exploración de la serie
.venv/bin/python -m src.eda.efecto_festivos               # efecto de festivos y fin de semana
.venv/bin/python -m src.features.run_features             # variables y particiones temporales
.venv/bin/python -m src.features.plot_split               # figura del reparto temporal
.venv/bin/python -m src.eda.temperatura_demanda           # relación temperatura-demanda
.venv/bin/python -m src.seleccion_exogenas.run_seleccion  # selección de variables exógenas
.venv/bin/python -m src.baselines.run_baselines           # modelos ingenuos y ancla del MASE

.venv/bin/python -m src.sarimax.run_sarimax 1             # SARIMAX: los cinco cortes rodantes,
.venv/bin/python -m src.sarimax.run_sarimax 2             #   uno a uno
.venv/bin/python -m src.sarimax.run_sarimax 3
.venv/bin/python -m src.sarimax.run_sarimax 4
.venv/bin/python -m src.sarimax.run_sarimax 5
.venv/bin/python -m src.sarimax.run_sarimax final         # ajuste final sobre el test
.venv/bin/python -m src.sarimax.run_sarimax agregar       # resumen de los cinco cortes

.venv/bin/python -m src.redes.run_redes mlp               # búsqueda de hiperparámetros del MLP
.venv/bin/python -m src.redes.run_redes lstm              # búsqueda de hiperparámetros de la LSTM
.venv/bin/python -m src.redes.finales ambos               # ajuste final de las dos
.venv/bin/python -m src.redes.curvas                      # curvas de entrenamiento

.venv/bin/python -m src.arboles.tuning rf                 # búsqueda de hiperparámetros (RF)
.venv/bin/python -m src.arboles.tuning xgb 200            # búsqueda de hiperparámetros (XGBoost)
.venv/bin/python -m src.arboles.run_arboles               # ajuste base, importancias y curvas
.venv/bin/python -m src.arboles.tuning finales            # ajuste final de XGBoost
.venv/bin/python -m src.arboles.finales_rf                # ajuste final del Random Forest
.venv/bin/python -m src.arboles.importancia_bloques       # importancia por bloques

.venv/bin/python -m src.comparativa.run_comparativa       # comparación y contrastes
.venv/bin/python -m src.comparativa.intervalos            # intervalos de predicción
.venv/bin/python -m src.externo.prevision_ree             # referencia del operador (opcional)
```

Cuatro puntos del orden que no son evidentes:

- `run_features` va antes que `src.eda.temperatura_demanda`, porque esa figura necesita el parquet de
  temperatura que deja la primera.
- `run_baselines` va antes que cualquier modelo: es quien fija el ancla del MASE que todos comparten.
- `run_arboles` va antes que `tuning finales`, porque el ajuste final de XGBoost tiene que ser el
  último en escribir sus predicciones.
- Los cinco cortes del SARIMAX se lanzan por separado, para poder repartirlos o reanudarlos; el quinto
  lo necesitan los intervalos de predicción.

Si a un módulo le falta un artefacto de una etapa anterior, aborta con un mensaje que dice qué comando
hay que ejecutar antes.

## Salidas

Cada etapa deja sus métricas en JSON y sus predicciones en parquet dentro de `resultados/`, y sus
gráficas en `figuras/`. Ninguna de las dos carpetas se versiona: se regeneran ejecutando la etapa.

## Estructura

La lógica vive en los módulos de cada paquete; los `run_*.py` solo orquestan.

| Paquete | Contenido |
|---|---|
| `src/eda/` | Calidad de la serie, estacionalidades, estacionariedad, autocorrelación, festivos, temperatura |
| `src/features/` | Catálogos de temperatura, calendario y retardos; tabla final y protocolo de validación temporal |
| `src/seleccion_exogenas/` | Competición de regresiones para elegir las variables exógenas |
| `src/baselines/` | Modelos ingenuos y ancla del MASE |
| `src/sarimax/` | SARIMAX en dos estructuras: 24 modelos horarios y serie horaria continua |
| `src/redes/` | Perceptrón multicapa multi-salida y LSTM, en PyTorch |
| `src/arboles/` | Random Forest, XGBoost e importancia de variables |
| `src/externo/` | Previsión publicada por el operador, como referencia |
| `src/comparativa/` | Comparación final, Diebold-Mariano, Model Confidence Set e intervalos |

## Datos de terceros

Las temperaturas proceden del reanálisis **ERA5** servido por **Open-Meteo**. Se emplean con fines exclusivamente académicos.
