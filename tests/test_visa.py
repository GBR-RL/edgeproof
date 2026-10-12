from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from edgeproof.data import visa


def make_part(root: Path, part: str, defects: int, normals: int) -> None:
    images = root / part / "Data" / "Images"
    masks = root / part / "Data" / "Masks" / "Anomaly"
    for folder in (images / "Anomaly", images / "Normal", masks):
        folder.mkdir(parents=True, exist_ok=True)
    blank = np.zeros((40, 60, 3), np.uint8)
    for i in range(defects):
        cv2.imwrite(str(images / "Anomaly" / f"{i:03d}.JPG"), blank)
        mask = np.zeros((40, 60), np.uint8)
        mask[5:15, 10:30] = 255  # one 20x10 defect
        mask[30:32, 50:51] = 255  # a 2-pixel fragment, below the minimum area
        cv2.imwrite(str(masks / f"{i:03d}.png"), mask)
    for i in range(normals):
        cv2.imwrite(str(images / "Normal" / f"{i:04d}.JPG"), blank)


def test_split_is_deterministic_and_disjoint(tmp_path: Path) -> None:
    for part in ("pcb1", "candle"):
        make_part(tmp_path, part, defects=10, normals=200)
    a = visa.make_split(tmp_path, parts=("pcb1", "candle"))
    b = visa.make_split(tmp_path, parts=("pcb1", "candle"))
    assert a == b
    assert len({s.image for s in a}) == len(a)
    counts = Counter((s.part, s.split, s.mask is not None) for s in a)
    assert counts[("pcb1", "train", True)] == 6
    assert counts[("pcb1", "test", True)] == 2
    assert counts[("pcb1", "test", False)] == visa.NORMALS["test"]
    assert all(s.mask == visa.mask_for(s.image) for s in a if s.mask)


def test_regions_drop_fragments_and_labels_are_normalised() -> None:
    mask = np.zeros((40, 60), np.uint8)
    mask[5:15, 10:30] = 255
    mask[30:32, 50:51] = 255
    polygons = visa.regions(mask)
    assert len(polygons) == 1
    det, seg = visa.label_lines(polygons, 60, 40)
    cls, cx, cy, w, h = det[0].split()
    assert cls == "0"
    assert abs(float(cx) - 19.5 / 60) < 0.02
    assert abs(float(cy) - 9.5 / 40) < 0.03
    assert abs(float(w) - 19 / 60) < 0.02
    assert abs(float(h) - 9 / 40) < 0.03
    coords = [float(v) for v in seg[0].split()[1:]]
    assert len(coords) >= 6
    assert all(0 <= v <= 1 for v in coords)


def test_export_writes_both_tasks(tmp_path: Path) -> None:
    make_part(tmp_path / "visa", "pcb1", defects=10, normals=200)
    samples = visa.make_split(tmp_path / "visa", parts=("pcb1",))
    counts = visa.export_yolo(samples, tmp_path / "visa", tmp_path / "yolo")
    assert counts["train_defects"] == 6
    for task in ("detect", "segment"):
        assert (tmp_path / "yolo" / task / "data.yaml").exists()
        labels = list((tmp_path / "yolo" / task / "labels" / "train").glob("*.txt"))
        images = list((tmp_path / "yolo" / task / "images" / "train").glob("*"))
        assert len(labels) == len(images) == 6 + visa.NORMALS["train"]
