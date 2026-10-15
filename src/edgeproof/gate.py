"""Performance gate for CI: rebuilds deployment variants from the trained checkpoints, measures
them on the runner and fails when accuracy or latency regress.

Accuracy is compared with the reference measured on the cluster: export and quantization are
deterministic up to floating-point order, so a drop beyond a small tolerance means a change in
the code (export settings, quantization scope, calibration, pre- or post-processing), not in the
machine. Latency cannot be compared across machines: GitHub's runners come with different CPUs
(an x86-64 runner may be a Zen 3, a Zen 4 or a Xeon, and INT8 speed differs by 2x between them).
So a quantized variant has to be a minimum factor faster than its float counterpart measured in
the same run, which holds on any CPU unless quantization stopped working, and absolute budgets
per architecture only catch gross slowdowns.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

Row = dict[str, Any]


@dataclass(frozen=True, slots=True)
class Speedup:
    over: str  # name of the variant to compare with, measured earlier in the same run
    min: float


@dataclass(frozen=True, slots=True)
class Variant:
    name: str
    args: list[str]  # an edgeproof bench or ptq command
    map50: float | None = None  # reference, None: latency only
    recall: float | None = None
    budget_ms: dict[str, float] = field(default_factory=dict)  # per architecture
    speedup: Speedup | None = None
    archs: tuple[str, ...] = ()  # where it runs; empty: everywhere
    cpu_flags: tuple[str, ...] = ()  # needs one of these CPU features; empty: none

    def skip_reason(self, arch: str, flags: set[str]) -> str | None:
        if self.archs and arch not in self.archs:
            return f"not on {arch}"
        if self.cpu_flags and not flags.intersection(self.cpu_flags):
            return f"needs {' or '.join(self.cpu_flags)}, which this CPU lacks"
        return None


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
            speedup=Speedup(**v["speedup"]) if "speedup" in v else None,
            archs=tuple(v.get("archs", ())),
            cpu_flags=tuple(v.get("cpu_flags", ())),
        )
        for v in cfg["variants"]
    ]
    return Gate(variants, tolerance.get("map50", 0.01), tolerance.get("recall", 0.03))


def cpu_flags() -> set[str]:
    """The CPU's feature flags as Linux reports them; empty elsewhere."""
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.exists():
        return set()
    for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(("flags", "Features")):  # x86 / ARM
            return set(line.split(":", 1)[1].split())
    return set()


def speedup(variant: Variant, row: Row, measured: dict[str, Row]) -> float | None:
    """How many times faster ``row`` is than the variant it is compared with, if that ran."""
    base = measured.get(variant.speedup.over) if variant.speedup else None
    return base["p50_ms"] / row["p50_ms"] if base else None


def failures(gate: Gate, variant: Variant, row: Row, measured: dict[str, Row]) -> list[str]:
    """Why ``row`` fails the gate; empty when it passes. ``measured``: rows of this run by name."""
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
    factor = speedup(variant, row, measured)
    if variant.speedup and factor is not None and factor < variant.speedup.min:
        out.append(f"{factor:.2f}x over {variant.speedup.over} < {variant.speedup.min}x")
    return out


def summary(gate: Gate, rows: list[tuple[Variant, Row]]) -> str:
    """A Markdown table of every variant against its reference and budget."""
    measured = {variant.name: row for variant, row in rows}
    lines = [
        "| Variant | Arch | p50 ms | Budget ms | Speed-up (min) | mAP50 | Reference "
        "| Recall @ 1% | Reference | Result |",
        "|---|---|--:|--:|--:|--:|--:|--:|--:|---|",
    ]

    def f(value: float | None, fmt: str) -> str:
        return "-" if value is None else format(value, fmt)

    for variant, row in rows:
        problems = failures(gate, variant, row, measured)
        factor = speedup(variant, row, measured)
        gain = "-"
        if variant.speedup and factor is not None:
            gain = f"{factor:.1f}x ({variant.speedup.min}x)"
        lines.append(
            f"| {variant.name} | {row['arch']} | {row['p50_ms']:.0f} "
            f"| {f(variant.budget_ms.get(row['arch']), '.0f')} | {gain} "
            f"| {f(row.get('box_map50'), '.3f')} | {f(variant.map50, '.3f')} "
            f"| {f(row.get('recall_at_far0.01'), '.2f')} | {f(variant.recall, '.2f')} "
            f"| {'; '.join(problems) or 'pass'} |"
        )
    return "\n".join(lines)
