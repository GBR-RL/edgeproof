import sys
import types
from typing import Any

import pytest

from edgeproof.bench import ov_config, pinned


class _Options:
    intra_op_num_threads = 0

    def __init__(self) -> None:
        self.config: dict[str, str] = {}

    def add_session_config_entry(self, key: str, value: str) -> None:
        self.config[key] = value


def _fake_runtimes(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any]:
    """Stand-ins with the call signatures Ultralytics uses, so the test needs neither runtime."""
    ort: Any = types.ModuleType("onnxruntime")
    ort.SessionOptions = _Options
    ort.InferenceSession = lambda path, sess_options=None, **kw: sess_options

    class Core:
        def compile_model(self, model: str, device_name: str = "CPU", config: Any = None) -> Any:
            return config

    ov: Any = types.ModuleType("openvino")
    ov.Core = Core
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    monkeypatch.setitem(sys.modules, "openvino", ov)
    monkeypatch.setitem(sys.modules, "torch", None)  # import fails: torch is skipped
    return ort, ov


def test_pinned_sets_threads_and_fp32(monkeypatch: pytest.MonkeyPatch) -> None:
    ort, ov = _fake_runtimes(monkeypatch)
    with pinned(8):
        options = ort.InferenceSession("m.onnx", providers=["CPUExecutionProvider"])
        config = ov.Core().compile_model("m.xml", "CPU", config={"PERFORMANCE_HINT": "LATENCY"})
    assert options.intra_op_num_threads == 8
    assert options.config == {"session.intra_op.allow_spinning": "0"}
    assert config == {"PERFORMANCE_HINT": "LATENCY", **ov_config(8)}
    assert config["INFERENCE_PRECISION_HINT"] == "f32"


def test_pinned_keeps_explicit_options_and_restores(monkeypatch: pytest.MonkeyPatch) -> None:
    ort, ov = _fake_runtimes(monkeypatch)
    mine = _Options()
    with pinned(8):
        assert ort.InferenceSession("m.onnx", mine) is mine
    assert ort.InferenceSession("m.onnx") is None  # the original again
    assert ov.Core().compile_model("m.xml") is None
