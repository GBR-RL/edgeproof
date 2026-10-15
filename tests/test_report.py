from typing import Any

from edgeproof.report import TAG, bf16_table, fp32_table, gpu_table, int8_table, join


def _row(precision: str, runtime: str = "openvino", **kw: Any) -> dict[str, Any]:
    return {"model": "m", "head": "e2e", "runtime": runtime, "precision": precision,
            "p50_ms": 999.0, "p99_ms": 999.0, "box_map50": 0.4, "box_map50_95": 0.2,
            "recall_at_far0.01": 0.8, "size_mb": 38.5} | kw  # fmt: skip


def test_join_takes_latency_from_the_whole_node_run() -> None:
    accuracy = [_row("fp32"), _row("int8-mixed-full-c128")]
    latency = [{"model": "m", "head": "e2e", "runtime": "openvino", "precision": "fp32",
                "p50_ms": 100.0, "p99_ms": 110.0}]  # fmt: skip
    fp32, int8 = join(accuracy, latency)
    assert (fp32["p50_ms"], fp32["p99_ms"]) == (100.0, 110.0)
    assert int8["p50_ms"] is None  # shared-node timing is not reported
    assert fp32["box_map50"] == 0.4


def test_int8_rows_compare_with_fp32_of_the_same_runtime() -> None:
    rows = [
        _row("fp32", p50_ms=100.0),
        _row("fp32", runtime="onnxruntime", p50_ms=120.0, box_map50=0.5),
        _row("int8-performance-headfp-c128", p50_ms=40.0, box_map50=0.39, size_mb=11.9),
    ]
    table = int8_table(rows).splitlines()
    assert len(table) == 3
    assert table[2] == (
        "| m | NMS-free | OpenVINO | performance | float | 40 | 2.5x | 0.390 (-0.010) "
        "| 0.200 (+0.000) | - | 0.80 | 11.9 |"
    )


def test_int8_rows_name_the_scope_calibration_size_and_mask() -> None:
    rows = [
        _row("fp32", p50_ms=100.0, mask_map50=0.41),
        _row("int8-percentile-pc-decodefp-c512", runtime="onnxruntime", p50_ms=50.0),
        _row("int8-performance-full-c32", p50_ms=25.0, mask_map50=0.4),
    ]
    ov, ort = int8_table(rows).splitlines()[2:]  # grouped in the order runtimes first appear
    assert "| percentile, 512 images | INT8, decoding in float | 50 | - |" in ort
    assert "| performance, 32 images | INT8 | 25 | 4.0x |" in ov
    assert "| 0.400 (-0.010) | 0.80 |" in ov


def test_int8_rows_are_grouped_by_variant_whatever_the_file_order() -> None:
    rows = [
        _row("fp32"),
        _row("fp32", runtime="onnxruntime"),
        _row("int8-percentile-pc-headfp-c128", runtime="onnxruntime"),
        _row("int8-performance-full-c512"),
        _row("int8-performance-full-c32"),
        _row("int8-performance-decodefp-c128"),
        _row("int8-performance-full-c128"),
    ]
    order = [line.split(" | ")[3:5] for line in int8_table(rows).splitlines()[2:]]
    assert order == [
        ["performance, 32 images", "INT8"],
        ["performance", "INT8"],
        ["performance", "INT8, decoding in float"],
        ["performance, 512 images", "INT8"],
        ["percentile", "float"],
    ]


def test_bf16_rows_compare_with_fp32_openvino() -> None:
    rows = [_row("fp32", p50_ms=90.0), _row("bf16", p50_ms=30.0, box_map50=0.401)]
    lines = bf16_table(rows).splitlines()
    assert len(lines) == 3
    assert lines[2] == (
        "| m | NMS-free | 90 | 30 | 3.0x | 0.401 (+0.001) | 0.200 (+0.000) | - | 0.80 |"
    )


def test_fp32_table_skips_int8_and_marks_missing_values() -> None:
    rows = [_row("fp32", p50_ms=None, p99_ms=None), _row("int8-minmax-pc-full-c128")]
    lines = fp32_table(rows).splitlines()
    assert len(lines) == 3
    assert lines[2] == "| m | NMS-free | OpenVINO | - | - | 0.400 | 0.200 | - | 0.80 | 38.5 |"


def test_tag_parses_every_ptq_variant() -> None:
    assert TAG.fullmatch("int8-entropy-pt-headfp-c32").groupdict() == {
        "method": "entropy", "channel": "pt", "scope": "headfp", "n": "32",
    }  # fmt: skip
    assert TAG.fullmatch("int8-performance-decodefp-c512")["scope"] == "decodefp"  # type: ignore[index]
    assert TAG.fullmatch("int8-mixed-full-c128")["channel"] is None  # type: ignore[index]


def test_rounding_to_zero_has_no_sign_flip() -> None:
    rows = [_row("fp32", box_map50=0.41509), _row("int8-performance-full-c128", box_map50=0.41491)]
    assert "0.415 (+0.000)" in int8_table(rows)


def test_gpu_rows_compare_with_pytorch_of_the_baseline() -> None:
    rows = [
        _row("fp32", runtime="torch", p50_ms=8.0, p99_ms=8.5),
        _row("int8", runtime="tensorrt", p50_ms=2.0, p99_ms=2.1, box_map50=0.41),
        _row("int8", runtime="tensorrt", model="m-qat", p50_ms=2.0, p99_ms=2.1),
    ]
    lines = gpu_table(rows).splitlines()
    assert lines[3] == (
        "| m | NMS-free | baseline | TensorRT INT8 | 2.00 | 2.10 | 4.0x | 0.410 (+0.010) "
        "| 0.200 (+0.000) | - | 0.80 | 38.5 |"
    )
    assert lines[4].startswith("| m | NMS-free | +10 epochs QAT | TensorRT INT8 | 2.00 |")
