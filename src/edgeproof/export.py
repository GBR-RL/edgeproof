"""Turns a trained checkpoint into deployable files: ONNX (for ONNX Runtime and TensorRT) and
OpenVINO IR. Shapes are static (batch 1, square input): a line camera sends one fixed-size image
at a time, and static shapes let every runtime plan its memory once.

YOLO26 carries two heads: a one-to-one head that needs no NMS ("e2e") and the one-to-many head
whose boxes go through NMS ("nms"). Ultralytics runs the NMS head unless told otherwise, so the
head is an explicit choice here and part of every result. YOLO11 has only the NMS head.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Literal

Runtime = Literal["torch", "onnxruntime", "openvino"]
RUNTIMES: tuple[Runtime, ...] = ("torch", "onnxruntime", "openvino")
Head = Literal["e2e", "nms"]
FORMATS = {"onnxruntime": "onnx", "openvino": "openvino"}


def artifact(models: Path, name: str, runtime: Runtime, head: Head = "e2e") -> Path:
    """Where a model lives for a runtime; the OpenVINO entry is the IR folder."""
    folder = models / "exported" / name
    if head == "nms":
        folder /= "nms"
    return {
        "torch": folder / f"{name}.pt",
        "onnxruntime": folder / f"{name}.onnx",
        "openvino": folder / f"{name}_openvino_model",
    }[runtime]


def resolve_head(checkpoint: Path, head: Head) -> Head:
    """The requested head if the model has it; models without a one-to-one head run NMS."""
    if head == "nms":
        return head
    from ultralytics import YOLO  # type: ignore[attr-defined,unused-ignore]

    model: Any = YOLO(str(checkpoint)).model
    return "e2e" if getattr(model.model[-1], "one2one_cv2", None) is not None else "nms"


def export(models: Path, name: str, runtime: Runtime, imgsz: int, head: Head = "e2e") -> Path:
    """Exports ``models/<name>.pt`` for ``runtime`` unless the file already exists."""
    target = artifact(models, name, runtime, head)
    if target.exists():
        return target
    source = artifact(models, name, "torch", head)
    source.parent.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        shutil.copy2(models / f"{name}.pt", source)  # Ultralytics writes exports next to the input
    if runtime == "torch":
        return source
    from ultralytics import YOLO  # type: ignore[attr-defined,unused-ignore]

    # nms=False exports the one-to-one head; None (the default) the raw one-to-many head, whose
    # NMS then runs in post-processing on the CPU.
    YOLO(str(source)).export(
        format=FORMATS[runtime], imgsz=imgsz, batch=1, dynamic=False,
        nms=False if head == "e2e" else None,
    )  # fmt: skip
    if not target.exists():
        raise FileNotFoundError(f"export did not produce {target}")
    return target


def size_mb(path: Path) -> float:
    files = [path] if path.is_file() else [p for p in path.rglob("*") if p.is_file()]
    weights = [p for p in files if p.suffix in {".pt", ".onnx", ".bin", ".xml", ".engine"}]
    return round(sum(p.stat().st_size for p in weights) / 1e6, 2)
