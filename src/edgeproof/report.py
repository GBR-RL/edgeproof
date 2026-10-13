"""Markdown tables for the README, built from the result files: accuracy from the runs on shared
nodes, latency from the one run on a whole node (slurm/latency.sbatch), joined per variant.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

Row = dict[str, Any]
Key = tuple[str, str, str, str]

HEADS = {"e2e": "NMS-free", "nms": "NMS"}
RUNTIMES = {"torch": "PyTorch", "onnxruntime": "ONNX Runtime", "openvino": "OpenVINO"}
TAG = re.compile(  # see quantize.PTQ.tag
    r"int8-(?P<method>[a-z]+)(?:-(?P<channel>pc|pt))?-(?P<scope>full|headfp)-c(?P<n>\d+)"
)


def load(path: Path) -> list[Row]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def key(row: Row) -> Key:
    return row["model"], row["head"], row["runtime"], row["precision"]


def join(accuracy: list[Row], latency: list[Row]) -> list[Row]:
    """Accuracy rows with the latency of the same variant from the whole-node run. The latency an
    accuracy run measured itself is dropped: other jobs shared its node."""
    timed = {key(r): r for r in latency}  # a later measurement of a variant wins
    out = []
    for row in accuracy:
        t = timed.get(key(row), {})
        out.append(row | {"p50_ms": t.get("p50_ms"), "p99_ms": t.get("p99_ms")})
    return out


def _f(value: float | None, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _ms(value: float | None) -> str:
    return "-" if value is None else f"{value:.0f}"


def _delta(value: float, base: float | None) -> str:
    return f"{value:.3f}" if base is None else f"{value:.3f} ({value - base:+.3f})"


def fp32_table(rows: list[Row]) -> str:
    lines = [
        "| Model | Head | Runtime | p50 ms | p99 ms | mAP50 | mAP50-95 | Mask mAP50 "
        "| Recall @ 1% false alarms | Size MB |",
        "|---|---|---|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for r in rows:
        if r["precision"] != "fp32":
            continue
        lines.append(
            f"| {r['model']} | {HEADS[r['head']]} | {RUNTIMES[r['runtime']]} | {_ms(r['p50_ms'])} "
            f"| {_ms(r['p99_ms'])} | {_f(r['box_map50'])} | {_f(r['box_map50_95'])} "
            f"| {_f(r.get('mask_map50'))} | {_f(r['recall_at_far0.01'], 2)} | {r['size_mb']:.1f} |"
        )
    return "\n".join(lines)


def int8_table(rows: list[Row]) -> str:
    """Each INT8 variant against FP32 on the same model, head and runtime."""
    base = {key(r)[:3]: r for r in rows if r["precision"] == "fp32"}
    lines = [
        "| Model | Runtime | Calibration | Head | p50 ms | Speed-up | mAP50 (vs FP32) "
        "| mAP50-95 (vs FP32) | Recall @ 1% false alarms | Size MB |",
        "|---|---|---|---|--:|--:|--:|--:|--:|--:|",
    ]
    for r in rows:
        tag = TAG.fullmatch(r["precision"])
        if tag is None:
            continue
        fp32 = base.get(key(r)[:3], {})
        method = tag["method"] + {"pt": ", per-tensor", "pc": ""}.get(tag["channel"] or "", "")
        head = "float" if tag["scope"] == "headfp" else "INT8"
        speedup = (
            f"{fp32['p50_ms'] / r['p50_ms']:.1f}x" if fp32.get("p50_ms") and r["p50_ms"] else "-"
        )
        lines.append(
            f"| {r['model']} | {RUNTIMES[r['runtime']]} | {method} | {head} | {_ms(r['p50_ms'])} "
            f"| {speedup} | {_delta(r['box_map50'], fp32.get('box_map50'))} "
            f"| {_delta(r['box_map50_95'], fp32.get('box_map50_95'))} "
            f"| {_f(r['recall_at_far0.01'], 2)} | {r['size_mb']:.1f} |"
        )
    return "\n".join(lines)
