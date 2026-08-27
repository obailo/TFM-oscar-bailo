"""Regenera el feature catálogo de temperatura, calendario y lags y el split canónico pre-COVID.

Uso:  python -m src.features.run_features
"""
from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.datos_precovid import serie_precovid
from src.features import catalogo_calendario, catalogo_lags, catalogo_temperatura
from src.features.temperatura import descargar_rep20_horaria
from src.features.validacion import split_canonico_precovid


def main():
    print("Regenerando el catálogo de variables (pre-COVID)", flush=True)

    serie = serie_precovid()
    print(f"1) Serie pre-COVID: {len(serie):,} h | {serie.index.min().date()} -> "
          f"{serie.index.max().date()}", flush=True)

    print("2) Temperatura Rep20 horaria (Open-Meteo/ERA5; usa caché si existe)…", flush=True)
    descargar_rep20_horaria()

    print("3) Catálogo de temperatura…", flush=True)
    variantes_temp = catalogo_temperatura.generar_catalogo()
    print(f"   {len(variantes_temp)} variantes en resultados/catalogo/temperatura/", flush=True)

    print("4) Catálogo de calendario…", flush=True)
    catalogo_calendario.main()

    print("5) Catálogo de lags…", flush=True)
    resultado_lags = catalogo_lags.generar()
    n_lags = (len(resultado_lags.get("registro", {}).get("features", []))
              if isinstance(resultado_lags, dict) else "?")
    print(f"   {n_lags} features de lags/ventanas en resultados/catalogo/lags/", flush=True)

    print("6) Split canónico pre-COVID…", flush=True)
    split = split_canonico_precovid(guardar=True)
    print(f"   train={len(split.train)}, val={len(split.val)}, test={len(split.test)} "
          f"(días-origen)", flush=True)

    print("\nHecho: resultados/catalogo/ y resultados/splits/precovid_canonico.json",
          flush=True)


if __name__ == "__main__":
    main()
