from pathlib import Path

import pytest

from edgeproof.gpu import engine_path, export_engine


def test_each_precision_builds_in_its_own_folder(tmp_path: Path) -> None:
    folder = tmp_path / "exported" / "m"
    assert engine_path(tmp_path, "m", "e2e", "fp16") == folder / "tensorrt" / "fp16" / "m.engine"
    assert engine_path(tmp_path, "m", "nms", "int8") == (
        folder / "nms" / "tensorrt" / "int8" / "m.engine"
    )


def test_qat_models_export_int8_only(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="INT8"):
        export_engine(tmp_path, "m-qat", 64, "e2e", "fp16", data_yaml=tmp_path, qat=True)


def test_existing_engines_are_reused(tmp_path: Path) -> None:
    target = engine_path(tmp_path, "m", "e2e", "fp32")
    target.parent.mkdir(parents=True)
    target.touch()
    assert export_engine(tmp_path, "m", 64, "e2e", "fp32", data_yaml=tmp_path) == target
