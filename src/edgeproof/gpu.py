"""Deployment on an NVIDIA GPU: TensorRT engines in FP32, FP16 and INT8, against PyTorch.

TensorRT 11 builds strongly typed networks only: the precision is part of the ONNX graph, not a
builder flag. Ultralytics bakes it in with NVIDIA ModelOpt before the build: FP16 through
AutoCast (operations whose range does not fit FP16 stay in FP32), INT8 as explicit quantize and
dequantize (Q/DQ) nodes, calibrated on training images for a model trained in float (PTQ) or taken
from training for a quantization-aware trained model (QAT). Both quantize the same layers: every
convolution except the head's output convolutions and the DFL. FP32 engines may use TF32 on the
tensor cores, TensorRT's default.

Latency is the engine alone, batch 1, with the input already in GPU memory: one
``execute_async_v3`` and a stream synchronization per call, timed on the host. Pre- and
post-processing run as on the CPU and are reported from the accuracy pass.
"""

from __future__ import annotations

import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import numpy as np

from edgeproof.export import Head, artifact

Precision = Literal["fp32", "fp16", "int8"]
QUANTIZE: dict[str, int | None] = {"fp32": None, "fp16": 16, "int8": 8}
DEVICE = "0"  # Ultralytics' name for the first CUDA device


def engine_path(models: Path, name: str, head: Head, precision: Precision) -> Path:
    """Each precision in its own folder: the export writes intermediate ONNX files next to the
    checkpoint, which must not replace the CPU exports."""
    return artifact(models, name, "torch", head).parent / "tensorrt" / precision / f"{name}.engine"


def export_engine(
    models: Path, name: str, imgsz: int, head: Head, precision: Precision, data_yaml: Path,
    *, qat: bool = False,
) -> Path:  # fmt: skip
    """Builds the TensorRT engine of ``models/<name>.pt`` unless it exists. PTQ calibrates on the
    training split (shuffled with a fixed seed; ModelOpt takes the first 512 images)."""
    if qat and precision != "int8":
        raise ValueError("a QAT model carries its INT8 ranges and exports to INT8 only")
    target = engine_path(models, name, head, precision)
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    source = target.with_suffix(".pt")
    shutil.copy2(models / f"{name}.pt", source)
    calibration = {"data": str(data_yaml), "split": "train"} if precision == "int8" and not qat else {}
    from ultralytics import YOLO  # type: ignore[attr-defined,unused-ignore]

    YOLO(str(source)).export(
        format="engine", imgsz=imgsz, batch=1, dynamic=False, device=DEVICE,
        nms=False if head == "e2e" else None, quantize=QUANTIZE[precision], **calibration,
    )  # fmt: skip
    if not target.exists():
        raise FileNotFoundError(f"export did not produce {target}")
    return target


def _engine_runner(path: Path, imgsz: int) -> Callable[[], None]:
    import tensorrt as trt
    import torch

    with path.open("rb") as f:
        meta = int.from_bytes(f.read(4), byteorder="little", signed=True)
        f.seek(4 + meta)  # Ultralytics writes its metadata in front of the serialized engine
        engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
    buffers = []  # kept alive as long as the context points at them
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = tuple(engine.get_tensor_shape(name))
        dtype = torch.from_numpy(np.empty(0, trt.nptype(engine.get_tensor_dtype(name)))).dtype
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            if shape != (1, 3, imgsz, imgsz):
                raise ValueError(f"engine input {shape}, expected (1, 3, {imgsz}, {imgsz})")
            gen = torch.Generator(device="cuda").manual_seed(0)
            tensor = torch.rand(shape, generator=gen, device="cuda").to(dtype)
        else:
            tensor = torch.empty(shape, dtype=dtype, device="cuda")
        context.set_tensor_address(name, tensor.data_ptr())
        buffers.append(tensor)
    stream = torch.cuda.Stream()

    def run() -> None:
        context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()

    run.buffers = buffers  # type: ignore[attr-defined]
    return run


def _torch_runner(path: Path, imgsz: int, head: Head) -> Callable[[], None]:
    import torch
    from ultralytics import YOLO  # type: ignore[attr-defined,unused-ignore]

    net: Any = YOLO(str(path)).model
    net.end2end = head == "e2e"
    net = net.fuse().float().eval().cuda()
    x = torch.rand((1, 3, imgsz, imgsz), generator=torch.Generator().manual_seed(0)).cuda()

    def run() -> None:
        with torch.inference_mode():
            net(x)
        torch.cuda.synchronize()

    return run


def latency(
    path: Path, runtime: Literal["torch", "tensorrt"], imgsz: int, *, head: Head = "e2e",
    runs: int = 1000, warmup: int = 100,
) -> dict[str, Any]:  # fmt: skip
    """Milliseconds per image on the GPU, batch 1, the network alone."""
    import torch

    run = _engine_runner(path, imgsz) if runtime == "tensorrt" else _torch_runner(path, imgsz, head)
    for _ in range(warmup):
        run()
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        run()
        times.append((time.perf_counter() - start) * 1000)
    t = np.asarray(times)
    return {
        "gpu": torch.cuda.get_device_name(0),
        "p50_ms": round(float(np.percentile(t, 50)), 3),
        "p99_ms": round(float(np.percentile(t, 99)), 3),
        "mean_ms": round(float(t.mean()), 3),
    }
