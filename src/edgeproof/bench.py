"""Measures one deployment variant: network latency in its own runtime, and accuracy end to end.

Latency is timed on the bare runtime call at batch 1 after a warm-up, so the numbers compare the
runtimes and not the Python around them. Pre- and post-processing (resize, NMS) are reported
separately from Ultralytics' own per-stage timing: NMS is exactly the cost YOLO26 removes.

Every runtime runs in FP32 on a set number of threads. Neither holds by default: OpenVINO computes
in bfloat16 on CPUs with AMX (Xeon Sapphire Rapids and newer), and ONNX Runtime sizes its thread
pool for all cores of the machine, not the cores a Slurm job may use.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np

from edgeproof.export import Head, Runtime
from edgeproof.metrics import recall_at_far

FARS = (0.01, 0.05)


def ov_config(threads: int) -> dict[str, Any]:
    return {"INFERENCE_NUM_THREADS": threads, "INFERENCE_PRECISION_HINT": "f32"}


def usable_cores() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # not on Linux
        return os.cpu_count() or 1


@contextmanager
def pinned(threads: int) -> Iterator[None]:
    """Sessions that Ultralytics opens inside this block run in FP32 on ``threads`` threads."""
    with ExitStack() as stack:
        try:
            import onnxruntime as ort
        except ImportError:
            pass
        else:
            session = ort.InferenceSession

            def ort_session(path: Any, sess_options: Any = None, *args: Any, **kwargs: Any) -> Any:
                if sess_options is None:
                    sess_options = ort.SessionOptions()
                    sess_options.intra_op_num_threads = threads
                    # Idle workers otherwise spin between runs and starve the pre- and
                    # post-processing, which share the same cores.
                    sess_options.add_session_config_entry("session.intra_op.allow_spinning", "0")
                return session(path, sess_options, *args, **kwargs)

            stack.enter_context(patch.object(ort, "InferenceSession", ort_session))
        try:
            import openvino as ov
        except ImportError:
            pass
        else:

            class Core(ov.Core):  # type: ignore[misc,unused-ignore]
                def compile_model(self, *args: Any, config: Any = None, **kwargs: Any) -> Any:
                    config = {**(config or {}), **ov_config(threads)}
                    return super().compile_model(*args, config=config, **kwargs)

            stack.enter_context(patch.object(ov, "Core", Core))
        try:
            import torch
        except ImportError:
            pass
        else:
            before = torch.get_num_threads()
            torch.set_num_threads(threads)
            stack.callback(torch.set_num_threads, before)
        yield


def _runner(
    path: Path, runtime: Runtime, imgsz: int, threads: int, head: Head
) -> Callable[[], Any]:
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
            str(xml), "CPU", {**ov_config(threads), "PERFORMANCE_HINT": "LATENCY"}
        )
        request = compiled.create_infer_request()
        return lambda: request.infer({0: x})
    import torch
    from ultralytics import YOLO  # type: ignore[attr-defined,unused-ignore]

    torch.set_num_threads(threads)
    net: Any = YOLO(str(path)).model
    net.end2end = head == "e2e"  # before fusing, which drops the other head
    net = net.fuse().float().eval()  # conv+batch-norm folded, as every export does
    tensor = torch.from_numpy(x)

    def run() -> Any:
        with torch.inference_mode():
            return net(tensor)

    return run


def latency(
    path: Path, runtime: Runtime, imgsz: int, *, head: Head = "e2e", threads: int = 4,
    runs: int = 100, warmup: int = 10,
) -> dict[str, float]:  # fmt: skip
    """Milliseconds per image for the network alone, batch 1."""
    run = _runner(path, runtime, imgsz, threads, head)
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
    path: Path, task: str, yolo_dir: Path, imgsz: int, *, head: Head = "e2e",
    fars: tuple[float, ...] = FARS, threads: int | None = None,
) -> dict[str, Any]:  # fmt: skip
    """Test-split mAP plus image-level recall at fixed false-alarm rates, through the runtime
    itself, and the median pre- and post-processing time per image. Runs on all usable cores
    unless ``threads`` is given; latency is measured separately."""
    with pinned(threads or usable_cores()):
        return _accuracy(path, task, yolo_dir, imgsz, head=head, fars=fars)


def _accuracy(
    path: Path, task: str, yolo_dir: Path, imgsz: int, *, head: Head, fars: tuple[float, ...]
) -> dict[str, Any]:
    from ultralytics import YOLO  # type: ignore[attr-defined,unused-ignore]

    from edgeproof.train import metrics_from

    model: Any = YOLO(str(path), task=task)  # Ultralytics' result types are loose unions
    nms = False if head == "e2e" else None  # None: Ultralytics' default, the NMS head
    result = model.val(
        data=str(yolo_dir / task / "data.yaml"), split="test", imgsz=imgsz, batch=1,
        device="cpu", nms=nms, plots=False, verbose=False,
    )  # fmt: skip
    out: dict[str, Any] = metrics_from(result.results_dict)
    defects: list[float] = []
    normals: list[float] = []
    pre: list[float] = []
    post: list[float] = []
    # A folder, not a list of paths: Ultralytics runs a list as one batch, all test images at once.
    images = yolo_dir / task / "images" / "test"
    for r in model.predict(str(images), imgsz=imgsz, batch=1, conf=0.001, device="cpu",
                           nms=nms, stream=True, verbose=False):  # fmt: skip
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
