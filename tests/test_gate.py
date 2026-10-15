from pathlib import Path

from edgeproof.gate import Gate, Variant, failures, load, summary


def _row(**kw: object) -> dict[str, object]:
    return {"arch": "x86_64", "p50_ms": 100.0, "box_map50": 0.41, "recall_at_far0.01": 0.78} | kw


def test_load_splits_the_command(tmp_path: Path) -> None:
    config = tmp_path / "gate.yaml"
    config.write_text(
        "tolerance: {map50: 0.02}\n"
        "variants:\n"
        "  - {name: a, args: bench --name m --runtimes openvino, map50: 0.4,"
        " budget_ms: {aarch64: 300}}\n"
    )
    gate = load(config)
    assert gate.map50_tolerance == 0.02
    assert gate.recall_tolerance == 0.03
    v, x86_only = gate.variants
    assert v.runs_on("aarch64") and v.runs_on("x86_64")
    assert x86_only.runs_on("x86_64") and not x86_only.runs_on("aarch64")
    assert v.args == ["bench", "--name", "m", "--runtimes", "openvino"]
    assert v.budget_ms == {"aarch64": 300}
    assert v.recall is None


def test_failures_within_tolerance_and_budget() -> None:
    gate = Gate([])
    v = Variant("a", [], map50=0.415, recall=0.80, budget_ms={"x86_64": 120.0})
    assert failures(gate, v, _row()) == []
    assert failures(gate, v, _row(box_map50=0.40, p50_ms=150.0, **{"recall_at_far0.01": 0.70})) == [
        "mAP50 0.400 < 0.415 - 0.01",
        "recall 0.700 < 0.800 - 0.03",
        "p50 150 ms > budget 120 ms",
    ]
    # no budget for this architecture: latency is reported, not checked
    assert failures(gate, v, _row(arch="aarch64", p50_ms=999.0)) == []


def test_summary_marks_each_variant() -> None:
    gate = Gate([])
    ok, slow = Variant("ok", []), Variant("slow", [], budget_ms={"x86_64": 50.0})
    lines = summary(gate, [(ok, _row()), (slow, _row())]).splitlines()
    assert lines[2].endswith("| pass |")
    assert lines[3].endswith("| p50 100 ms > budget 50 ms |")
