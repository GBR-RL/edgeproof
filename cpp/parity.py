"""Compares edgeproof-run with the Python pipeline on the same ONNX model and images: the same
detections, and how long each stage takes in C++ and in Ultralytics on the same threads.

Usage: python cpp/parity.py --model m.onnx --images dir --cpp detections.jsonl --timing t.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from edgeproof.bench import pinned


def iou(a: np.ndarray, b: np.ndarray) -> float:
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def compare(cpp: list[list[float]], py: list[list[float]]) -> dict[str, Any]:
    """Greedy one-to-one matching of two images' detections by IoU, highest score first."""
    a, b = np.asarray(cpp, dtype=float).reshape(-1, 5), np.asarray(py, dtype=float).reshape(-1, 5)
    used: set[int] = set()
    box_diff, score_diff, matched = 0.0, 0.0, 0
    for row in a:
        best, j = 0.5, -1
        for k, other in enumerate(b):
            if k not in used and (v := iou(row, other)) > best:
                best, j = v, k
        if j >= 0:
            used.add(j)
            matched += 1
            box_diff = max(box_diff, float(np.abs(row[:4] - b[j, :4]).max()))
            score_diff = max(score_diff, float(abs(row[4] - b[j, 4])))
    return {"cpp": len(a), "python": len(b), "matched": matched, "box": box_diff,
            "score": score_diff}  # fmt: skip


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--cpp", type=Path, required=True, help="edgeproof-run detections")
    p.add_argument("--timing", type=Path, required=True, help="edgeproof-run summary line")
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--out", type=Path, default=Path("parity.json"))
    args = p.parse_args()

    from ultralytics import YOLO

    cpp = {}
    for line in args.cpp.read_text().splitlines():
        r = json.loads(line)
        cpp[r["image"]] = r["boxes"]
    timing = json.loads(args.timing.read_text().strip().splitlines()[-1])
    model: Any = YOLO(str(args.model), task="detect")
    nms = None if timing["head"] == "nms" else False
    stats, speeds = [], {"preprocess": [], "inference": [], "postprocess": []}
    with pinned(args.threads):
        for r in model.predict(str(args.images), imgsz=args.imgsz, batch=1, conf=args.conf,
                               device="cpu", nms=nms, stream=True, verbose=False):  # fmt: skip
            boxes = r.boxes
            py = np.concatenate([boxes.xyxy.numpy(), boxes.conf.numpy()[:, None]], axis=1)
            stats.append(compare(cpp[Path(r.path).name], py.tolist()))
            for k in speeds:
                speeds[k].append(r.speed[k])
    n = len(stats)
    same = sum(s["cpp"] == s["python"] == s["matched"] for s in stats)
    python = {f"{k}_ms": round(float(np.median(v)), 2) for k, v in speeds.items()}
    python["total_ms"] = round(sum(python.values()), 2)
    report = {
        "model": args.model.name,
        "images": n,
        "identical_detections": same,
        "max_box_diff_px": round(max(s["box"] for s in stats), 3),
        "max_score_diff": round(max(s["score"] for s in stats), 5),
        "python": python,
        "cpp": {k: timing[k] for k in ("preprocess_ms", "inference_ms", "postprocess_ms",
                                       "total_ms")},  # fmt: skip
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    if same < 0.99 * n or report["max_score_diff"] > 0.01:
        raise SystemExit("C++ and Python disagree")


if __name__ == "__main__":
    main()
