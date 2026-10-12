"""VisA as a supervised defect dataset: every part, every defect, detection and segmentation labels.

VisA (Amazon, CC BY 4.0, https://registry.opendata.aws/visa/) has 12 industrial parts (PCBs,
capsules, candles, cashews, ...): 100 defective images per part with pixel masks of the defects,
and 500 to 1,000 defect-free images per part.
Its official splits target unsupervised anomaly detection, so a supervised split is made here:
per part, the defective images are divided 60/20/20 into train, validation and test with a fixed
seed, and a fixed number of defect-free images joins each split (to learn and to measure false
alarms). A quantization study needs a large test set: two INT8 variants often differ by a point
of mAP, which a few hundred images cannot resolve.

The labels are one class, "defect": a box per connected defect region for detection, and the
region's outline as a polygon for segmentation.
"""

from __future__ import annotations

import json
import random
import shutil
import tarfile
import urllib.request
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

VISA_URL = "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar"
VISA_SIZE_BYTES = 1_929_840_640
PARTS: tuple[str, ...] = ("candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1",
                          "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum")  # fmt: skip
SPLITS: tuple[str, ...] = ("train", "val", "test")
SHARES = (0.6, 0.2, 0.2)
NORMALS = {"train": 60, "val": 20, "test": 60}  # defect-free images per part and split
MIN_AREA = 16  # pixels: smaller mask fragments are noise
Split = Literal["train", "val", "test"]


@dataclass(frozen=True, slots=True)
class Sample:
    part: str
    image: str  # relative to the VisA root, e.g. "pcb1/Data/Images/Anomaly/000.JPG"
    mask: str | None  # relative to the VisA root; None for defect-free images
    split: Split


def download(url: str = VISA_URL, dest: Path | None = None, size: int = VISA_SIZE_BYTES) -> Path:
    """Downloads the archive unless a complete copy exists."""
    dest = dest or Path("data/raw/VisA_20220922.tar")
    if dest.exists() and dest.stat().st_size == size:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(".part")
    with urllib.request.urlopen(url) as response, partial.open("wb") as out:
        shutil.copyfileobj(response, out, length=1 << 20)
    if partial.stat().st_size != size:
        raise OSError(f"incomplete download: {partial.stat().st_size} of {size} bytes")
    partial.replace(dest)
    return dest


def extract(tar_path: Path, out_dir: Path) -> Path:
    """Extracts the archive (with a path-traversal guard) and returns the VisA root."""
    root = find_root(out_dir) if out_dir.exists() else None
    if root is not None:
        return root
    out_dir.mkdir(parents=True, exist_ok=True)
    resolved = out_dir.resolve()
    with tarfile.open(tar_path) as tar:
        members = []
        for member in tar.getmembers():
            if not (out_dir / member.name).resolve().is_relative_to(resolved):
                raise ValueError(f"unsafe path in archive: {member.name}")
            members.append(member)
        tar.extractall(out_dir, members=members)
    found = find_root(out_dir)
    if found is None:
        raise FileNotFoundError(f"no VisA parts under {out_dir}")
    return found


def find_root(base: Path) -> Path | None:
    """The directory that contains ``<part>/Data/Images``."""
    for candidate in (base, *sorted(p for p in base.iterdir() if p.is_dir())):
        if (candidate / PARTS[0] / "Data" / "Images").is_dir():
            return candidate
    return None


def _images(root: Path, part: str, kind: str) -> list[str]:
    folder = root / part / "Data" / "Images" / kind
    return sorted(p.relative_to(root).as_posix() for p in folder.iterdir() if p.is_file())


def mask_for(image: str) -> str:
    """VisA keeps each defect mask under Masks/Anomaly with the image's stem and a .png suffix."""
    path = Path(image)
    return (path.parents[2] / "Masks" / "Anomaly" / f"{path.stem}.png").as_posix()


def make_split(root: Path, parts: Iterable[str] = PARTS, seed: int = 13) -> list[Sample]:
    """Assigns every defective image and a fixed sample of defect-free images to one split."""
    samples: list[Sample] = []
    for part in parts:
        rng = random.Random(f"{seed}-{part}")  # string seeds hash stably across machines
        defects = _images(root, part, "Anomaly")
        rng.shuffle(defects)
        n_train = round(SHARES[0] * len(defects))
        n_val = round(SHARES[1] * len(defects))
        cuts = {"train": defects[:n_train], "val": defects[n_train : n_train + n_val],
                "test": defects[n_train + n_val :]}  # fmt: skip
        normals = _images(root, part, "Normal")
        rng.shuffle(normals)
        start = 0
        for split in SPLITS:
            samples += [Sample(part, img, mask_for(img), split) for img in cuts[split]]  # type: ignore[arg-type]
            take = normals[start : start + NORMALS[split]]
            start += NORMALS[split]
            samples += [Sample(part, img, None, split) for img in take]  # type: ignore[arg-type]
    return samples


def write_split(samples: Sequence[Sample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(asdict(s)) + "\n")


def read_split(path: Path) -> list[Sample]:
    with path.open(encoding="utf-8") as f:
        return [Sample(**json.loads(line)) for line in f if line.strip()]


def regions(mask: np.ndarray) -> list[np.ndarray]:
    """Outer contours of the connected defect regions, each an (n, 2) array of pixel points."""
    import cv2

    binary = (mask > 0).astype(np.uint8)
    if binary.ndim == 3:
        binary = binary.max(axis=2)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        if cv2.contourArea(c) < MIN_AREA:
            continue
        approx = cv2.approxPolyDP(c, 1.0, True)  # within a pixel of the mask outline
        points = approx.reshape(-1, 2)
        if len(points) >= 3:
            out.append(points)
    return out


def label_lines(
    polygons: Sequence[np.ndarray], width: int, height: int
) -> tuple[list[str], list[str]]:
    """YOLO detection lines (class cx cy w h) and segmentation lines (class x1 y1 x2 y2 ...),
    normalised to [0, 1], single class 0 = defect."""
    det, seg = [], []
    for p in polygons:
        x0, y0 = p.min(axis=0)
        x1, y1 = p.max(axis=0)
        det.append(f"0 {(x0 + x1) / 2 / width:.6f} {(y0 + y1) / 2 / height:.6f} "
                   f"{(x1 - x0) / width:.6f} {(y1 - y0) / height:.6f}")  # fmt: skip
        coords = " ".join(f"{x / width:.6f} {y / height:.6f}" for x, y in p)
        seg.append(f"0 {coords}")
    return det, seg


def export_yolo(samples: Sequence[Sample], root: Path, out: Path) -> dict[str, Any]:
    """Writes images once and two label sets (detect/, segment/) with a data YAML each."""
    import cv2

    counts: Counter[str] = Counter()
    for task in ("detect", "segment"):
        for split in SPLITS:
            (out / task / "labels" / split).mkdir(parents=True, exist_ok=True)
    for s in samples:
        image_dir = out / "images" / s.split
        image_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{s.part}_{'defect' if s.mask else 'normal'}_{Path(s.image).stem}"
        src = root / s.image
        dst = image_dir / f"{stem}{src.suffix.lower()}"
        if not dst.exists():
            shutil.copy2(src, dst)
        image = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(src)
        h, w = image.shape[:2]
        polygons = []
        if s.mask is not None:
            mask = cv2.imread(str(root / s.mask), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(root / s.mask)
            polygons = regions(mask)
        det, seg = label_lines(polygons, w, h)
        for task, lines in (("detect", det), ("segment", seg)):
            (out / task / "labels" / s.split / f"{stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else "")
            )
        counts[f"{s.split}_images"] += 1
        counts[f"{s.split}_defects"] += len(polygons)
    for task in ("detect", "segment"):
        # Ultralytics finds labels by replacing "images" with "labels" in the image path, so each
        # task gets a symlink-free mirror: its own images folder pointing at the shared files.
        task_images = out / task / "images"
        if not task_images.exists():
            shutil.copytree(out / "images", task_images, copy_function=_link)
        (out / task / "data.yaml").write_text(
            f"path: {(out / task).resolve().as_posix()}\n"
            "train: images/train\nval: images/val\ntest: images/test\nnames:\n  0: defect\n"
        )
    return dict(counts)


def _link(src: str, dst: str) -> str:
    """Hard links where the file system allows them, copies otherwise."""
    import os

    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
    return dst
