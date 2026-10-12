"""Turns a trained checkpoint into deployable files: ONNX (for ONNX Runtime and TensorRT) and
OpenVINO IR. Shapes are static (batch 1, square input): a line camera sends one fixed-size image
at a time, and static shapes let every runtime plan its memory once.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Literal

Runtime = Literal["torch", "onnxruntime", "openvino"]
RUNTIMES: tuple[Runtime, ...] = ("torch", "onnxruntime", "openvino")
FORMATS = {"onnxruntime": "onnx", "openvino": "openvino"}


def artifact(models: Path, name: str, runtime: Runtime) -> Path:
    """Where a model lives for a runtime; the OpenVINO entry is the IR folder."""
    folder = models / "exported" / name
    return {
        "torch": folder / f"{name}.pt",
        "onnxruntime": folder / f"{name}.onnx",
        "openvino": folder / f"{name}_openvino_model",
    }[runtime]


def export(models: Path, name: str, runtime: Runtime, imgsz: int) -> Path:
    """Exports ``models/<name>.pt`` for ``runtime`` unless the file already exists."""
    target = artifact(models, name, runtime)
    if target.exists():
        return target
    source = artifact(models, name, "torch")
    source.parent.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        shutil.copy2(models / f"{name}.pt", source)  # Ultralytics writes exports next to the input
    if runtime == "torch":
        return source
    from ultralytics import YOLO  # type: ignore[attr-defined]

    # nms=False keeps an end-to-end head where the model has one (YOLO26); the Ultralytics
    # default would export the NMS head instead and hand NMS back to the CPU.
    YOLO(str(source)).export(
        format=FORMATS[runtime], imgsz=imgsz, batch=1, dynamic=False, nms=False
    )
    if not target.exists():
        raise FileNotFoundError(f"export did not produce {target}")
    return target


def size_mb(path: Path) -> float:
    files = [path] if path.is_file() else [p for p in path.rglob("*") if p.is_file()]
    weights = [p for p in files if p.suffix in {".pt", ".onnx", ".bin", ".xml"}]
    return round(sum(p.stat().st_size for p in weights) / 1e6, 2)
