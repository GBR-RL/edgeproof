from pathlib import Path

from edgeproof.gate import Gate, Speedup, Variant, failures, load, summary


def _row(**kw: object) -> dict[str, object]:
    return {"arch": "x86_64", "p50_ms": 100.0, "box_map50": 0.41, "recall_at_far0.01": 0.78} | kw


def test_load_splits_the_command(tmp_path: Path) -> None:
    config = tmp_path / "gate.yaml"
    config.write_text(
        "tolerance: {map50: 0.02}\n"
        "variants:\n"
        "  - {name: a, args: bench --name m --runtimes openvino, map50: 0.4,"
        " budget_ms: {aarch64: 300}}\n"
        "  - {name: b, args: bench --name m, archs: [x86_64], cpu_flags: [avx512_vnni, avx_vnni],"
        " speedup: {over: a, min: 1.5}}\n"
    )
    gate = load(config)
    assert gate.map50_tolerance == 0.02
    assert gate.recall_tolerance == 0.03
    v, int8 = gate.variants
    assert v.args == ["bench", "--name", "m", "--runtimes", "openvino"]
    assert v.budget_ms == {"aarch64": 300}
    assert v.recall is None
    assert v.speedup is None
    assert int8.speedup == Speedup("a", 1.5)
    assert int8.cpu_flags == ("avx512_vnni", "avx_vnni")


def test_variants_skip_other_architectures_and_missing_cpu_features() -> None:
    anywhere = Variant("a", [])
    vnni = Variant("b", [], archs=("x86_64",), cpu_flags=("avx512_vnni", "avx_vnni"))
    assert anywhere.skip_reason("aarch64", set()) is None
    assert vnni.skip_reason("aarch64", {"asimd"}) == "not on aarch64"
    assert vnni.skip_reason("x86_64", {"avx2"}) == (
        "needs avx512_vnni or avx_vnni, which this CPU lacks"
    )
    assert vnni.skip_reason("x86_64", {"avx2", "avx_vnni"}) is None


def test_failures_within_tolerance_and_budget() -> None:
    gate = Gate([])
    v = Variant("a", [], map50=0.415, recall=0.80, budget_ms={"x86_64": 120.0})
    assert failures(gate, v, _row(), {}) == []
    assert failures(
        gate, v, _row(box_map50=0.40, p50_ms=150.0, **{"recall_at_far0.01": 0.70}), {}
    ) == [
        "mAP50 0.400 < 0.415 - 0.01",
        "recall 0.700 < 0.800 - 0.03",
        "p50 150 ms > budget 120 ms",
    ]
    # no budget for this architecture: latency is reported, not checked
    assert failures(gate, v, _row(arch="aarch64", p50_ms=999.0), {}) == []


def test_speedup_is_measured_against_the_float_variant_of_the_same_run() -> None:
    gate = Gate([])
    int8 = Variant("int8", [], speedup=Speedup("fp32", 1.5))
    fp32 = _row(p50_ms=300.0)
    assert failures(gate, int8, _row(p50_ms=150.0), {"fp32": fp32}) == []
    assert failures(gate, int8, _row(p50_ms=250.0), {"fp32": fp32}) == ["1.20x over fp32 < 1.5x"]
    # the float variant did not run here: nothing to compare with
    assert failures(gate, int8, _row(p50_ms=250.0), {}) == []


def test_summary_marks_each_variant() -> None:
    gate = Gate([])
    ok, slow = Variant("ok", []), Variant("slow", [], budget_ms={"x86_64": 50.0})
    int8 = Variant("int8", [], speedup=Speedup("ok", 1.5))
    lines = summary(gate, [(ok, _row()), (slow, _row()), (int8, _row(p50_ms=40.0))]).splitlines()
    assert lines[2].endswith("| pass |")
    assert lines[3].endswith("| p50 100 ms > budget 50 ms |")
    assert "| 2.5x (1.5x) |" in lines[4]
    assert lines[4].endswith("| pass |")
