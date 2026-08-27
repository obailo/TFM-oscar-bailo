"""
Uso:  python -m src.baselines.run_baselines
"""
from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.baselines.ancla_precovid import main

if __name__ == "__main__":
    main()
