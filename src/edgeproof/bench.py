"""Measures one deployment variant: network latency in its own runtime, and accuracy end to end.

Latency is timed on the bare runtime call at batch 1 after a warm-up, so the numbers compare the
runtimes and not the Python around them. Pre- and post-processing (resize, NMS) are reported
separately from Ultralytics' own per-stage timing: NMS is exactly the cost YOLO26 removes.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from edgeproof.export import Runtime
from edgeproof.metrics import recall_at_far

FARS = (0.01, 0.05)


def _runner(path: Path, runtime: Runtime, imgsz: int, threads: int) -> Callable[[], Any]:
    x = np.random.default_rng(0).random((1, 3, imgsz, imgsz), dtype=np.float32)
    if runtime == "onnxruntime":
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        feed = {session.get_inputs()[0].name: x}
        return lambda: session.run(None, feed)
    if runtime == "openvino":
        import openvino as ov

        xml = next(path.glob("*.xml"))
        compiled = ov.Core().compile_model(
            str(xml), "CPU", {"INFERENCE_NUM_THREADS": threads, "PERFORMANCE_HINT": "LATENCY"}
        )
        request = compiled.create_infer_request()
        return lambda: request.infer({0: x})
    import torch
    from ultralytics import YOLO  # type: ignore[attr-defined]

    torch.set_num_threads(threads)
    net: Any = YOLO(str(path)).model
    net = net.fuse().float().eval()  # conv+batch-norm folded, as every export does
    tensor = torch.from_numpy(x)

    def run() -> Any:
        with torch.inference_mode():
            return net(tensor)

    return run


def latency(
    path: Path, runtime: Runtime, imgsz: int, *, threads: int = 4, runs: int = 100, warmup: int = 10
) -> dict[str, float]:
    """Milliseconds per image for the network alone, batch 1."""
    run = _runner(path, runtime, imgsz, threads)
    for _ in range(warmup):
        run()
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        run()
        times.append((time.perf_counter() - start) * 1000)
    t = np.asarray(times)
    return {
        "threads": threads,
        "p50_ms": round(float(np.percentile(t, 50)), 2),
        "p99_ms": round(float(np.percentile(t, 99)), 2),
        "mean_ms": round(float(t.mean()), 2),
    }


def accuracy(
    path: Path, task: str, yolo_dir: Path, imgsz: int, fars: tuple[float, ...] = FARS
) -> dict[str, Any]:
    """Test-split mAP plus image-level recall at fixed false-alarm rates, through the runtime
    itself, and the median pre- and post-processing time per image."""
    from ultralytics import YOLO  # type: ignore[attr-defined]

    from edgeproof.train import metrics_from

    model: Any = YOLO(str(path), task=task)  # Ultralytics' result types are loose unions
    result = model.val(
        data=str(yolo_dir / task / "data.yaml"), split="test", imgsz=imgsz, batch=1,
        device="cpu", plots=False, verbose=False,
    )  # fmt: skip
    out: dict[str, Any] = metrics_from(result.results_dict)
    defects: list[float] = []
    normals: list[float] = []
    pre: list[float] = []
    post: list[float] = []
    images = sorted((yolo_dir / task / "images" / "test").iterdir())
    for r in model.predict([str(p) for p in images], imgsz=imgsz, conf=0.001, device="cpu",
                           stream=True, verbose=False):  # fmt: skip
        score = float(r.boxes.conf.max()) if len(r.boxes) else 0.0
        (defects if "_defect_" in Path(r.path).name else normals).append(score)
        pre.append(r.speed["preprocess"])
        post.append(r.speed["postprocess"])
    for far in fars:
        at = recall_at_far(defects, normals, far)
        out[f"recall_at_far{far:g}"] = at["recall"]
    out["preprocess_ms"] = round(float(np.median(pre)), 2)
    out["postprocess_ms"] = round(float(np.median(post)), 2)
    return out


def cpu_name() -> str:
    import platform

    name = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return f"{name} ({os.cpu_count()} logical cores)"
