from pathlib import Path

import numpy as np
import pytest

from edgeproof.quantize import (
    PAD,
    PTQ,
    calibration_images,
    chunks,
    decode_nodes,
    excluded,
    head_nodes,
    letterbox,
    variant_path,
)


def test_tag_names_every_choice() -> None:
    assert PTQ().tag == "int8-minmax-pc-full-c128"
    assert PTQ(method="entropy", per_channel=False, keep_head=True, calib=32).tag == (
        "int8-entropy-pt-headfp-c32"
    )
    assert PTQ(keep_decode=True).tag == "int8-minmax-pc-decodefp-c128"
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


ONNX_OPS = [
    ("/model.0/conv/Conv", "Conv"),
    ("/model.0/act/Sigmoid", "Sigmoid"),  # SiLU, exported as Sigmoid and Mul
    ("/model.22/cv2/Conv", "Conv"),
    ("/model.23/cv2.0/cv2.0.0/conv/Conv", "Conv"),
    ("/model.23/cv2.0/cv2.0.0/act/Sigmoid", "Sigmoid"),
    ("/model.23/dfl/conv/Conv", "Conv"),
    ("/model.23/Sigmoid", "Sigmoid"),
    ("/model.23/Concat_3", "Concat"),
    ("/model.230/Conv", "Conv"),
]
OPENVINO_OPS = [
    ("__module.model.0.conv/aten::_convolution/Convolution", "Convolution"),
    ("__module.model.23.cv2.0.0.conv/aten::_convolution/Convolution", "Convolution"),
    ("__module.model.23.cv2.0.0.act/aten::silu_/Swish", "Swish"),
    ("__module.model.23.dfl.conv/aten::_convolution/Convolution", "Convolution"),
    ("__module.model.23/aten::sigmoid/Sigmoid", "Sigmoid"),
    ("__module.model.23/aten::cat/Concat_3", "Concat"),
]


def test_head_nodes_take_the_last_module_with_a_sigmoid() -> None:
    assert head_nodes(ONNX_OPS) == [name for name, _ in ONNX_OPS[3:8]]
    assert head_nodes(OPENVINO_OPS) == [name for name, _ in OPENVINO_OPS[1:]]
    assert head_nodes([("/a/Conv", "Conv")]) == []


def test_decode_nodes_leave_the_head_convolutions_out() -> None:
    assert decode_nodes(ONNX_OPS) == [
        "/model.23/dfl/conv/Conv", "/model.23/Sigmoid", "/model.23/Concat_3",
    ]  # fmt: skip
    assert decode_nodes(OPENVINO_OPS) == [name for name, _ in OPENVINO_OPS[3:]]


def test_excluded_nodes_follow_the_scope() -> None:
    assert excluded(ONNX_OPS, PTQ()) == []
    assert excluded(ONNX_OPS, PTQ(keep_head=True)) == head_nodes(ONNX_OPS)
    assert excluded(ONNX_OPS, PTQ(keep_decode=True)) == decode_nodes(ONNX_OPS)
    with pytest.raises(ValueError, match="keep_head"):
        PTQ(keep_head=True, keep_decode=True)


def test_variant_paths_per_runtime(tmp_path: Path) -> None:
    folder = tmp_path / "exported" / "m" / "ptq"
    assert variant_path(tmp_path, "m", PTQ()) == folder / "m-int8-minmax-pc-full-c128.onnx"
    assert variant_path(tmp_path, "m", PTQ(runtime="openvino", method="performance")) == (
        folder / "m-int8-performance-full-c128_openvino_model"
    )
    assert variant_path(tmp_path, "m", PTQ(head="nms")) == (
        folder.parent / "nms" / "ptq" / "m-int8-minmax-pc-full-c128.onnx"
    )


@pytest.mark.parametrize(("n", "sizes"), [(20, [8, 8, 4]), (16, [8, 8]), (3, [3]), (0, [])])
def test_chunks_cover_every_item_once(n: int, sizes: list[int]) -> None:
    items = iter(range(n))
    out = list(chunks(lambda: next(items, None), 8))
    assert [len(c) for c in out] == sizes
    assert [x for c in out for x in c] == list(range(n))
