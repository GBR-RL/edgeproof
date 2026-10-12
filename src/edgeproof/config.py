"""Paths shared by the stages."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("EDGEPROOF_ROOT", Path(__file__).resolve().parents[2]))
DATA = ROOT / "data"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
SEED = 13
