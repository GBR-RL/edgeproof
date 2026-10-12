from pathlib import Path

from edgeproof.export import artifact, size_mb


def test_artifact_paths_per_runtime(tmp_path: Path) -> None:
    folder = tmp_path / "exported" / "m"
    assert artifact(tmp_path, "m", "torch") == folder / "m.pt"
    assert artifact(tmp_path, "m", "onnxruntime") == folder / "m.onnx"
    assert artifact(tmp_path, "m", "openvino") == folder / "m_openvino_model"


def test_size_counts_weights_only(tmp_path: Path) -> None:
    ir = tmp_path / "m_openvino_model"
    ir.mkdir()
    (ir / "m.xml").write_bytes(b"x" * 1_000_000)
    (ir / "m.bin").write_bytes(b"x" * 2_000_000)
    (ir / "metadata.yaml").write_bytes(b"x" * 5_000_000)
    assert size_mb(ir) == 3.0
