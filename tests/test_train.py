from pathlib import Path

from edgeproof.train import Recipe, metrics_from, train_args


def test_recipe_names_and_weights() -> None:
    det = Recipe()
    seg = Recipe(family="yolo11", size="n", task="segment", imgsz=640)
    assert (det.name, det.weights) == ("yolo26s-det-1024", "yolo26s.pt")
    assert (seg.name, seg.weights) == ("yolo11n-seg-640", "yolo11n-seg.pt")


def test_train_args_use_ultralytics_device_names() -> None:
    args = train_args(
        Recipe(), Path("d.yaml"), Path("runs"), device="cuda", workers=4, fraction=1.0
    )
    assert args["device"] == "0"
    assert args["single_cls"] is True
    assert args["deterministic"] is True
    assert args["name"] == "yolo26s-det-1024"
    assert (
        train_args(Recipe(), Path("d"), Path("r"), device="cuda:1", workers=4, fraction=1.0)[
            "device"
        ]
        == "1"
    )


def test_metrics_are_renamed_per_head() -> None:
    raw = {"metrics/precision(B)": 0.5, "metrics/mAP50-95(B)": 0.25,
           "metrics/mAP50(M)": 0.4, "fitness": 0.3}  # fmt: skip
    assert metrics_from(raw) == {"box_precision": 0.5, "box_map50_95": 0.25, "mask_map50": 0.4}
