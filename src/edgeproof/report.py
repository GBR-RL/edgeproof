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
    r"int8-(?P<method>[a-z]+)(?:-(?P<channel>pc|pt))?-(?P<scope>full|headfp|decodefp)-c(?P<n>\d+)"
)
SCOPES = {"full": "INT8", "headfp": "float", "decodefp": "INT8, decoding in float"}
CALIB = "128"  # calibration images unless a row says otherwise


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
    if base is None:
        return f"{value:.3f}"
    d = value - base
    return f"{value:.3f} ({0.0 if abs(d) < 5e-4 else d:+.3f})"  # no "-0.000"


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


def _speedup(base: float | None, value: float | None) -> str:
    return f"{base / value:.1f}x" if base and value else "-"


def _mask(r: Row, base: Row) -> str:
    if r.get("mask_map50") is None:
        return "-"
    return _delta(r["mask_map50"], base.get("mask_map50"))


def int8_table(rows: list[Row]) -> str:
    """Each INT8 variant against FP32 on the same model, head and runtime."""
    base = {key(r)[:3]: r for r in rows if r["precision"] == "fp32"}
    lines = [
        "| Model | Head | Runtime | Calibration | Head ops | p50 ms | Speed-up | mAP50 (vs FP32) "
        "| mAP50-95 (vs FP32) | Mask mAP50 (vs FP32) | Recall @ 1% false alarms | Size MB |",
        "|---|---|---|---|---|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for r in rows:
        tag = TAG.fullmatch(r["precision"])
        if tag is None:
            continue
        fp32 = base.get(key(r)[:3], {})
        method = tag["method"] + {"pt": ", per-tensor", "pc": ""}.get(tag["channel"] or "", "")
        if tag["n"] != CALIB:
            method += f", {tag['n']} images"
        lines.append(
            f"| {r['model']} | {HEADS[r['head']]} | {RUNTIMES[r['runtime']]} | {method} "
            f"| {SCOPES[tag['scope']]} | {_ms(r['p50_ms'])} "
            f"| {_speedup(fp32.get('p50_ms'), r['p50_ms'])} "
            f"| {_delta(r['box_map50'], fp32.get('box_map50'))} "
            f"| {_delta(r['box_map50_95'], fp32.get('box_map50_95'))} | {_mask(r, fp32)} "
            f"| {_f(r['recall_at_far0.01'], 2)} | {r['size_mb']:.1f} |"
        )
    return "\n".join(lines)


def bf16_table(rows: list[Row]) -> str:
    """OpenVINO computing in bfloat16 against FP32 on the same model and head."""
    base = {key(r)[:3]: r for r in rows if r["precision"] == "fp32"}
    lines = [
        "| Model | Head | FP32 p50 ms | BF16 p50 ms | Speed-up | mAP50 (vs FP32) "
        "| mAP50-95 (vs FP32) | Mask mAP50 (vs FP32) | Recall @ 1% false alarms |",
        "|---|---|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for r in rows:
        if r["precision"] != "bf16":
            continue
        fp32 = base.get(key(r)[:3], {})
        lines.append(
            f"| {r['model']} | {HEADS[r['head']]} | {_ms(fp32.get('p50_ms'))} | {_ms(r['p50_ms'])} "
            f"| {_speedup(fp32.get('p50_ms'), r['p50_ms'])} "
            f"| {_delta(r['box_map50'], fp32.get('box_map50'))} "
            f"| {_delta(r['box_map50_95'], fp32.get('box_map50_95'))} | {_mask(r, fp32)} "
            f"| {_f(r['recall_at_far0.01'], 2)} |"
        )
    return "\n".join(lines)
