"""Post-training INT8 quantization, one variant at a time, so each choice can be priced.

The choices measured, with ONNX Runtime (QDQ format) and OpenVINO (NNCF):

- calibration method: min-max, entropy or percentile ranges for the activations (ONNX Runtime);
  NNCF's performance (symmetric) or mixed (asymmetric activations) preset (OpenVINO);
- weights per output channel or per tensor (ONNX Runtime; NNCF is always per channel);
- the detection head quantized or kept in float: the head turns features into box coordinates
  and confidences, where an INT8 rounding step moves a box edge or a score directly;
- calibration set size.

Calibration images are a seeded sample of the training split, preprocessed exactly as at
inference (letterbox to a square, RGB, 0-1).
"""

from __future__ import annotations

import itertools
import random
import re
import shutil
import tempfile
from collections.abc import Callable, Iterator, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from unittest.mock import patch

import numpy as np

from edgeproof.config import SEED
from edgeproof.export import Head, artifact

PAD = 114  # Ultralytics' letterbox gray


@dataclass(frozen=True, slots=True)
class PTQ:
    runtime: Literal["onnxruntime", "openvino"] = "onnxruntime"
    method: str = "minmax"  # minmax | entropy | percentile (ORT); performance | mixed (OpenVINO)
    per_channel: bool = True
    keep_head: bool = False
    calib: int = 128
    head: Head = "e2e"  # which detection head is exported, not whether it is quantized

    def __post_init__(self) -> None:
        allowed = {"onnxruntime": ("minmax", "entropy", "percentile"),
                   "openvino": ("performance", "mixed")}[self.runtime]  # fmt: skip
        if self.method not in allowed:
            raise ValueError(f"{self.runtime} supports {allowed}, not {self.method!r}")

    @property
    def tag(self) -> str:
        channel = ("-pc" if self.per_channel else "-pt") if self.runtime == "onnxruntime" else ""
        scope = "headfp" if self.keep_head else "full"
        return f"int8-{self.method}{channel}-{scope}-c{self.calib}"


def letterbox(image: np.ndarray, imgsz: int) -> np.ndarray:
    """BGR image to a (1, 3, imgsz, imgsz) float32 RGB tensor in [0, 1], aspect kept, padded."""
    import cv2

    h, w = image.shape[:2]
    r = min(imgsz / h, imgsz / w)
    nw, nh = round(w * r), round(h * r)
    if (nw, nh) != (w, h):
        image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dw, dh = (imgsz - nw) / 2, (imgsz - nh) / 2
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT,
                               value=(PAD, PAD, PAD))  # fmt: skip
    x = image[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return np.ascontiguousarray(x)


def calibration_images(yolo_dir: Path, n: int, seed: int = SEED) -> list[Path]:
    """A fixed sample of training images; the same for every variant of the same size."""
    images = sorted((yolo_dir / "images" / "train").iterdir())
    return sorted(random.Random(f"calib-{seed}").sample(images, min(n, len(images))))


def tensors(paths: Sequence[Path], imgsz: int) -> Iterator[np.ndarray]:
    import cv2

    for p in paths:
        image = cv2.imread(str(p))
        if image is None:
            raise FileNotFoundError(p)
        yield letterbox(image, imgsz)


def head_nodes(ops: Sequence[tuple[str, str]]) -> list[str]:
    """Names of the head's operations: every node of the last module that holds a Sigmoid
    (the confidence output), plus all Sigmoids. ``ops`` are (name, type) pairs in graph order."""
    sigmoids = [name for name, kind in ops if kind == "Sigmoid"]
    if not sigmoids:
        return []
    match = re.search(r"model\.(\d+)", sigmoids[-1])
    if match is None:
        return sigmoids
    scope = re.compile(rf"model\.{match.group(1)}[/.]")
    return [name for name, kind in ops if kind == "Sigmoid" or scope.search(name)]


CHUNK = 8  # calibration images whose intermediate tensors are held in memory at once


def chunks(get_next: Callable[[], Any], n: int) -> Iterator[list[Any]]:
    """Successive lists of up to ``n`` items from a reader that returns None when it is done."""
    items = iter(get_next, None)
    while chunk := list(itertools.islice(items, n)):
        yield chunk


@contextmanager
def _calibrate_in_chunks(reader_type: Any) -> Iterator[None]:
    """ONNX Runtime's calibrators keep every intermediate tensor of every calibration image before
    computing ranges or histograms: well over 100 GB for 128 images at 1024 pixels. Both the
    min-max ranges and the histograms merge across calls, so the images are fed a few at a time.
    (Min-max has an option for this, CalibMaxIntermediateOutputs, but in ONNX Runtime 1.31 it
    drops each full chunk without computing its ranges.)"""
    from onnxruntime.quantization import calibrate

    with ExitStack() as stack:
        for cls in (calibrate.MinMaxCalibrater, calibrate.HistogramCalibrater):
            original = cls.collect_data

            def chunked(self: Any, reader: Any, original: Any = original) -> None:
                for chunk in chunks(reader.get_next, CHUNK):
                    original(self, reader_type(x[next(iter(x))] for x in chunk))

            stack.enter_context(patch.object(cls, "collect_data", chunked))
        yield


def quantize_onnx(source: Path, target: Path, ptq: PTQ, calib: Sequence[Path], imgsz: int) -> Path:
    import onnx
    from onnxruntime.quantization import (
        CalibrationDataReader,
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    model = onnx.load(str(source))
    input_name = model.graph.input[0].name
    exclude = head_nodes([(n.name, n.op_type) for n in model.graph.node]) if ptq.keep_head else []

    class Reader(CalibrationDataReader):  # type: ignore[misc,unused-ignore]
        def __init__(self, items: Iterator[np.ndarray]) -> None:
            self.it = items

        def get_next(self) -> dict[str, np.ndarray] | None:
            x = next(self.it, None)
            return None if x is None else {input_name: x}

    methods = {"minmax": CalibrationMethod.MinMax, "entropy": CalibrationMethod.Entropy,
               "percentile": CalibrationMethod.Percentile}  # fmt: skip
    target.parent.mkdir(parents=True, exist_ok=True)
    # ONNX Runtime writes a shape-inferred copy next to the model it quantizes and deletes it
    # afterwards, so variants quantized in parallel from one export would delete each other's.
    with tempfile.TemporaryDirectory() as tmp, _calibrate_in_chunks(Reader):
        own = shutil.copy2(source, Path(tmp) / source.name)
        quantize_static(
            str(own), str(target), Reader(tensors(calib, imgsz)),
            quant_format=QuantFormat.QDQ,
            per_channel=ptq.per_channel,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            calibrate_method=methods[ptq.method],
            nodes_to_exclude=exclude,
        )  # fmt: skip
    # Ultralytics reads task, stride, image size and the end-to-end flag from the metadata.
    quantized = onnx.load(str(target))
    del quantized.metadata_props[:]
    quantized.metadata_props.extend(model.metadata_props)
    onnx.save(quantized, str(target))
    return target


def quantize_openvino(
    source: Path, target: Path, ptq: PTQ, calib: Sequence[Path], imgsz: int
) -> Path:
    import nncf
    import openvino as ov

    xml = next(source.glob("*.xml"))
    model = ov.Core().read_model(str(xml))
    ignored = None
    if ptq.keep_head:
        names = head_nodes([(op.get_friendly_name(), op.get_type_name())
                            for op in model.get_ordered_ops()])  # fmt: skip
        ignored = nncf.IgnoredScope(names=names, validate=False)
    preset = {"performance": nncf.QuantizationPreset.PERFORMANCE,
              "mixed": nncf.QuantizationPreset.MIXED}[ptq.method]  # fmt: skip
    quantized = nncf.quantize(
        model, nncf.Dataset(list(tensors(calib, imgsz))), preset=preset,
        subset_size=len(calib), ignored_scope=ignored,
    )  # fmt: skip
    target.mkdir(parents=True, exist_ok=True)
    ov.save_model(quantized, str(target / xml.name), compress_to_fp16=False)
    shutil.copy2(source / "metadata.yaml", target / "metadata.yaml")
    return target


def variant_path(models: Path, name: str, ptq: PTQ) -> Path:
    folder = artifact(models, name, "torch", ptq.head).parent / "ptq"
    if ptq.runtime == "onnxruntime":
        return folder / f"{name}-{ptq.tag}.onnx"
    return folder / f"{name}-{ptq.tag}_openvino_model"


def build(models: Path, name: str, ptq: PTQ, yolo_dir: Path, imgsz: int) -> Path:
    """Quantizes the FP32 export of ``name``; reuses an existing variant."""
    from edgeproof.export import export

    target = variant_path(models, name, ptq)
    if target.exists():
        return target
    source = export(models, name, ptq.runtime, imgsz, ptq.head)
    calib = calibration_images(yolo_dir, ptq.calib)
    if ptq.runtime == "onnxruntime":
        return quantize_onnx(source, target, ptq, calib, imgsz)
    return quantize_openvino(source, target, ptq, calib, imgsz)
