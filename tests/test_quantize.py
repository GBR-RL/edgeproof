from pathlib import Path

import numpy as np
import pytest

from edgeproof.quantize import PAD, PTQ, calibration_images, head_nodes, letterbox, variant_path


def test_tag_names_every_choice() -> None:
    assert PTQ().tag == "int8-minmax-pc-full-c128"
    assert PTQ(method="entropy", per_channel=False, keep_head=True, calib=32).tag == (
        "int8-entropy-pt-headfp-c32"
    )
    # NNCF is always per channel, so the tag leaves it out.
    assert PTQ(runtime="openvino", method="mixed").tag == "int8-mixed-full-c128"


def test_method_must_match_runtime() -> None:
    with pytest.raises(ValueError, match="openvino"):
        PTQ(runtime="openvino", method="entropy")
    with pytest.raises(ValueError, match="onnxruntime"):
        PTQ(method="performance")


def test_letterbox_keeps_aspect_and_pads() -> None:
    image = np.zeros((100, 200, 3), np.uint8)
    image[..., 0] = 255  # blue in BGR
    x = letterbox(image, 64)
    assert x.shape == (1, 3, 64, 64)
    assert x.dtype == np.float32
    assert x.flags["C_CONTIGUOUS"]
    # 200x100 scales to 64x32: 16 rows of padding above and below.
    assert np.allclose(x[0, :, :16], PAD / 255)
    assert np.allclose(x[0, :, 48:], PAD / 255)
    # RGB order: blue lands in the last channel.
    assert np.allclose(x[0, 2, 16:48], 1.0)
    assert np.allclose(x[0, 0, 16:48], 0.0)


def test_calibration_sample_is_fixed_and_nested_in_the_split(tmp_path: Path) -> None:
    train = tmp_path / "images" / "train"
    train.mkdir(parents=True)
    for i in range(50):
        (train / f"{i:03d}.png").touch()
    a, b = calibration_images(tmp_path, 8), calibration_images(tmp_path, 8)
    assert a == b
    assert len(a) == 8
    assert a == sorted(a)
    assert len(calibration_images(tmp_path, 500)) == 50


def test_head_nodes_take_the_last_module_with_a_sigmoid() -> None:
    ops = [
        ("/model.0/conv/Conv", "Conv"),
        ("/model.22/cv2/Conv", "Conv"),
        ("/model.23/cv2.0/Conv", "Conv"),
        ("/model.23/Sigmoid", "Sigmoid"),
        ("/model.23/Concat", "Concat"),
        ("/model.230/Conv", "Conv"),
    ]
    assert head_nodes(ops) == ["/model.23/cv2.0/Conv", "/model.23/Sigmoid", "/model.23/Concat"]
    assert head_nodes([("/a/Conv", "Conv")]) == []


def test_variant_paths_per_runtime(tmp_path: Path) -> None:
    folder = tmp_path / "exported" / "m" / "ptq"
    assert variant_path(tmp_path, "m", PTQ()) == folder / "m-int8-minmax-pc-full-c128.onnx"
    assert variant_path(tmp_path, "m", PTQ(runtime="openvino", method="performance")) == (
        folder / "m-int8-performance-full-c128_openvino_model"
    )
    assert variant_path(tmp_path, "m", PTQ(head="nms")) == (
        folder.parent / "nms" / "ptq" / "m-int8-minmax-pc-full-c128.onnx"
    )
