"""Paths shared by the stages."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("EDGEPROOF_ROOT", Path(__file__).resolve().parents[2]))
DATA = ROOT / "data"
# The YOLO export can live elsewhere, e.g. on a compute node's local disk during training.
YOLO_DATA = Path(os.environ.get("EDGEPROOF_YOLO_DATA", DATA / "yolo"))
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
SEED = 13
