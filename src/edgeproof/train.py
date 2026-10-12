"""Baseline training: the FP32 models every deployment variant is measured against.

Two model families, so the study can say whether an architecture choice matters on the line:
YOLO26 predicts boxes end to end (no non-maximum suppression), YOLO11 needs NMS after the
network, which costs CPU time that a latency table on the network alone would hide.

Compute nodes on the cluster have no internet, so ``prefetch`` downloads everything Ultralytics
fetches at run time (pretrained weights, the AMP check model, the plotting font) beforehand.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from edgeproof.config import SEED
from edgeproof.device import pick_device, ultralytics_device

Family = Literal["yolo26", "yolo11"]
Task = Literal["detect", "segment"]


@dataclass(frozen=True, slots=True)
class Recipe:
    family: Family = "yolo26"
    size: Literal["n", "s", "m"] = "s"
    task: Task = "detect"
    imgsz: int = 1024  # VisA defects are often a few dozen pixels on 1,500-pixel images
    epochs: int = 100
    batch: int = 32
    patience: int = 30
    seed: int = SEED

    @property
    def name(self) -> str:
        return f"{self.family}{self.size}-{'seg' if self.task == 'segment' else 'det'}-{self.imgsz}"

    @property
    def weights(self) -> str:
        return f"{self.family}{self.size}{'-seg' if self.task == 'segment' else ''}.pt"


def train_args(
    recipe: Recipe, data_yaml: Path, project: Path, *, device: str, workers: int, fraction: float
) -> dict[str, Any]:
    """Keyword arguments for ``YOLO.train``; everything not listed keeps the Ultralytics default."""
    return {
        "data": str(data_yaml),
        "imgsz": recipe.imgsz,
        "epochs": recipe.epochs,
        "batch": recipe.batch,
        "patience": recipe.patience,
        "seed": recipe.seed,
        "deterministic": True,
        "device": ultralytics_device(device),
        "workers": workers,
        "fraction": fraction,
        "project": str(project),
        "name": recipe.name,
        "exist_ok": True,
        "single_cls": True,
    }


def metrics_from(results: dict[str, Any]) -> dict[str, float]:
    """Ultralytics keys like ``metrics/mAP50-95(B)`` as ``box_map50_95``; ``(M)`` is the mask."""
    out = {}
    for key, value in results.items():
        if not key.startswith("metrics/"):
            continue
        metric, _, kind = key.removeprefix("metrics/").partition("(")
        prefix = {"B)": "box", "M)": "mask"}.get(kind, "")
        slug = metric.lower().replace("-", "_")
        out[f"{prefix}_{slug}" if prefix else slug] = round(float(value), 5)
    return out


def prefetch(recipes: list[Recipe], pretrained: Path) -> list[Path]:
    """Downloads what a training run would fetch, so it can run without internet."""
    from ultralytics.utils import WEIGHTS_DIR
    from ultralytics.utils.checks import check_font
    from ultralytics.utils.downloads import attempt_download_asset

    pretrained.mkdir(parents=True, exist_ok=True)
    paths = [Path(attempt_download_asset(str(pretrained / r.weights))) for r in recipes]
    attempt_download_asset(str(WEIGHTS_DIR / "yolo26n.pt"))  # Ultralytics' AMP self-check model
    check_font("Arial.ttf")
    return paths


def train(
    recipe: Recipe,
    yolo_dir: Path,
    models: Path,
    *,
    device: str | None = None,
    workers: int = 8,
    fraction: float = 1.0,
) -> dict[str, Any]:
    """Trains one recipe, keeps the best checkpoint as ``models/<name>.pt`` and scores it on
    the held-out test split. Returns the summary also written to ``models/<name>.json``."""
    from ultralytics import YOLO  # type: ignore[attr-defined,unused-ignore]

    device = pick_device(device)
    start = models / "pretrained" / recipe.weights
    model = YOLO(str(start if start.exists() else recipe.weights))
    data_yaml = yolo_dir / recipe.task / "data.yaml"
    runs = models / "runs"
    model.train(
        **train_args(recipe, data_yaml, runs, device=device, workers=workers, fraction=fraction)
    )
    best = runs / recipe.name / "weights" / "best.pt"
    models.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, models / f"{recipe.name}.pt")
    summary = {
        "recipe": asdict(recipe),
        "device": device,
        "fraction": fraction,
        "test": evaluate(models / f"{recipe.name}.pt", data_yaml, recipe.imgsz, device),
    }
    (models / f"{recipe.name}.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def evaluate(
    weights: Path, data_yaml: Path, imgsz: int, device: str | None = None, split: str = "test"
) -> dict[str, float]:
    from ultralytics import YOLO  # type: ignore[attr-defined,unused-ignore]

    model = YOLO(str(weights))
    result = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=imgsz,
        batch=16,
        device=ultralytics_device(pick_device(device)),
        plots=False,
        verbose=False,
    )
    return metrics_from(result.results_dict)
