"""Performance gate for CI: rebuilds deployment variants from the trained checkpoints, measures
them on the runner and fails when accuracy or latency regress.

Accuracy is compared with the reference measured on the cluster: export and quantization are
deterministic up to floating-point order, so a drop beyond a small tolerance means a change in
the code (export settings, quantization scope, calibration, pre- or post-processing), not in the
machine. Latency is compared with a budget per CPU architecture, set generously above what the
shared GitHub runners measure, so that only a real slowdown fails a pull request.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

Row = dict[str, Any]


@dataclass(frozen=True, slots=True)
class Variant:
    name: str
    args: list[str]  # an edgeproof bench or ptq command
    map50: float | None = None  # reference, None: latency only
    recall: float | None = None
    budget_ms: dict[str, float] = field(default_factory=dict)  # per architecture
    archs: tuple[str, ...] = ()  # where it runs; empty: everywhere

    def runs_on(self, arch: str) -> bool:
        return not self.archs or arch in self.archs


@dataclass(frozen=True, slots=True)
class Gate:
    variants: list[Variant]
    map50_tolerance: float = 0.01
    recall_tolerance: float = 0.03


def load(path: Path) -> Gate:
    import yaml

    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    tolerance = cfg.get("tolerance", {})
    variants = [
        Variant(
            name=v["name"],
            args=shlex.split(v["args"]),
            map50=v.get("map50"),
            recall=v.get("recall"),
            budget_ms=v.get("budget_ms", {}),
            archs=tuple(v.get("archs", ())),
        )
        for v in cfg["variants"]
    ]
    return Gate(variants, tolerance.get("map50", 0.01), tolerance.get("recall", 0.03))


def failures(gate: Gate, variant: Variant, row: Row) -> list[str]:
    """Why ``row`` fails the gate; empty when it passes."""
    out = []
    if variant.map50 is not None and row["box_map50"] < variant.map50 - gate.map50_tolerance:
        out.append(f"mAP50 {row['box_map50']:.3f} < {variant.map50:.3f} - {gate.map50_tolerance}")
    if variant.recall is not None:
        recall = row["recall_at_far0.01"]
        if recall < variant.recall - gate.recall_tolerance:
            out.append(f"recall {recall:.3f} < {variant.recall:.3f} - {gate.recall_tolerance}")
    budget = variant.budget_ms.get(row["arch"])
    if budget is not None and row["p50_ms"] > budget:
        out.append(f"p50 {row['p50_ms']:.0f} ms > budget {budget:.0f} ms")
    return out


def summary(gate: Gate, rows: list[tuple[Variant, Row]]) -> str:
    """A Markdown table of every variant against its reference and budget."""
    lines = [
        "| Variant | Arch | p50 ms | Budget ms | mAP50 | Reference | Recall @ 1% | Reference "
        "| Result |",
        "|---|---|--:|--:|--:|--:|--:|--:|---|",
    ]

    def f(value: float | None, fmt: str) -> str:
        return "-" if value is None else format(value, fmt)

    for variant, row in rows:
        problems = failures(gate, variant, row)
        lines.append(
            f"| {variant.name} | {row['arch']} | {row['p50_ms']:.0f} "
            f"| {f(variant.budget_ms.get(row['arch']), '.0f')} | {f(row.get('box_map50'), '.3f')} "
            f"| {f(variant.map50, '.3f')} | {f(row.get('recall_at_far0.01'), '.2f')} "
            f"| {f(variant.recall, '.2f')} | {'; '.join(problems) or 'pass'} |"
        )
    return "\n".join(lines)
