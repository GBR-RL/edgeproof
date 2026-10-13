"""Command line entry point."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from edgeproof import __version__
from edgeproof.config import DATA, MODELS, RESULTS, YOLO_DATA

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.callback()
def main() -> None:
    """What accuracy it costs to deploy an inspection model on edge hardware."""


@app.command()
def data(
    tar: Annotated[Path | None, typer.Option(help="Use an existing VisA archive.")] = None,
    out: Annotated[Path, typer.Option(help="Data directory.")] = DATA,
) -> None:
    """Download VisA, split it per part and export YOLO detection and segmentation labels."""
    from edgeproof.data import visa

    archive = tar or visa.download(dest=out / "raw" / "VisA_20220922.tar")
    root = visa.extract(archive, out / "visa")
    samples = visa.make_split(root)
    visa.write_split(samples, out / "split.jsonl")
    counts = visa.export_yolo(samples, root, out / "yolo")
    (out / "yolo" / "counts.json").write_text(json.dumps(counts, indent=2) + "\n")
    typer.echo(json.dumps(counts))


def _recipe(family: str, size: str, task: str, imgsz: int, epochs: int, batch: int) -> Any:
    from edgeproof.train import Recipe

    return Recipe(family=family, size=size, task=task, imgsz=imgsz, epochs=epochs, batch=batch)  # type: ignore[arg-type]


@app.command()
def prefetch(
    families: Annotated[str, typer.Option(help="Families, comma-separated.")] = "yolo26,yolo11",
    sizes: Annotated[str, typer.Option(help="Comma-separated model sizes.")] = "s",
) -> None:
    """Download pretrained weights and assets so training runs without internet."""
    from edgeproof.train import Recipe
    from edgeproof.train import prefetch as fetch

    recipes = [Recipe(family=f, size=s, task=t)  # type: ignore[arg-type]
               for f in families.split(",") for s in sizes.split(",")
               for t in ("detect", "segment")]  # fmt: skip
    for path in fetch(recipes, MODELS / "pretrained"):
        typer.echo(path)


@app.command()
def train(
    family: Annotated[str, typer.Option(help="yolo26 or yolo11.")] = "yolo26",
    size: Annotated[str, typer.Option(help="n, s or m.")] = "s",
    task: Annotated[str, typer.Option(help="detect or segment.")] = "detect",
    imgsz: Annotated[int, typer.Option(help="Training image size.")] = 1024,
    epochs: Annotated[int, typer.Option(help="Maximum epochs.")] = 100,
    batch: Annotated[int, typer.Option(help="Batch size.")] = 32,
    workers: Annotated[int, typer.Option(help="Data loader processes.")] = 8,
    fraction: Annotated[float, typer.Option(help="Share of the training set (smoke tests).")] = 1.0,
    device: Annotated[str | None, typer.Option(help="auto, cpu, cuda, cuda:1, mps.")] = None,
) -> None:
    """Train one baseline and score it on the test split."""
    from edgeproof.train import train as run

    recipe = _recipe(family, size, task, imgsz, epochs, batch)
    summary = run(recipe, YOLO_DATA, MODELS, device=device, workers=workers, fraction=fraction)
    typer.echo(json.dumps(summary["test"]))


def _measure(
    name: str, runtime: str, precision: str, path: Path, *, head: str, threads: int, runs: int,
    accuracy: bool, out: Path,
) -> dict[str, Any]:  # fmt: skip
    """One result row: latency and size on this machine, accuracy through the runtime."""
    import platform

    from edgeproof import bench as b
    from edgeproof.export import size_mb

    summary = json.loads((MODELS / f"{name}.json").read_text())
    imgsz, task = summary["recipe"]["imgsz"], summary["recipe"]["task"]
    row: dict[str, Any] = {"model": name, "head": head, "runtime": runtime,
                           "precision": precision, "cpu": b.cpu_name(),
                           "arch": platform.machine().lower(),
                           "size_mb": size_mb(path)}  # fmt: skip
    row |= b.latency(path, runtime, imgsz, head=head, threads=threads, runs=runs)  # type: ignore[arg-type]
    if accuracy:
        row |= b.accuracy(path, task, YOLO_DATA, imgsz, head=head)  # type: ignore[arg-type]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    typer.echo(json.dumps(row))
    return row


def _imgsz(name: str) -> int:
    return int(json.loads((MODELS / f"{name}.json").read_text())["recipe"]["imgsz"])


def _head(name: str, head: str) -> Any:
    from edgeproof.export import resolve_head

    return resolve_head(MODELS / f"{name}.pt", head)  # type: ignore[arg-type]


Threads = Annotated[int, typer.Option(help="CPU threads per inference.")]
Runs = Annotated[int, typer.Option(help="Timed runs per variant.")]
SkipAccuracy = Annotated[bool, typer.Option(help="Latency only.")]
HeadOpt = Annotated[str, typer.Option(help="e2e (NMS-free, YOLO26 only) or nms.")]
Out = Annotated[Path | None, typer.Option(help="Result file (JSON lines), appended to.")]


@app.command()
def bench(
    name: Annotated[str, typer.Option(help="Trained model, e.g. yolo26s-det-1024.")],
    runtimes: Annotated[str, typer.Option(help="Runtimes, comma-separated.")] = (
        "torch,onnxruntime,openvino"
    ),
    head: HeadOpt = "e2e",
    threads: Threads = 4,
    runs: Runs = 100,
    skip_accuracy: SkipAccuracy = False,
    out: Out = None,
) -> None:
    """Export a trained model to each runtime and measure latency, size and accuracy on CPU."""
    from edgeproof.export import export

    head = _head(name, head)
    for runtime in runtimes.split(","):
        path = export(MODELS, name, runtime, _imgsz(name), head)  # type: ignore[arg-type]
        _measure(name, runtime, "fp32", path, head=head, threads=threads, runs=runs,
                 accuracy=not skip_accuracy, out=out or RESULTS / "runtimes.jsonl")  # fmt: skip


@app.command()
def ptq(
    name: Annotated[str, typer.Option(help="Trained model, e.g. yolo26s-det-1024.")],
    runtime: Annotated[str, typer.Option(help="onnxruntime or openvino.")] = "onnxruntime",
    method: Annotated[str, typer.Option(help="minmax|entropy|percentile or performance|mixed.")] = (
        "minmax"
    ),
    per_channel: Annotated[bool, typer.Option(help="Per-channel weight scales (ORT).")] = True,
    keep_head: Annotated[bool, typer.Option(help="Keep the detection head in float.")] = False,
    calib: Annotated[int, typer.Option(help="Calibration images.")] = 128,
    head: HeadOpt = "e2e",
    threads: Threads = 4,
    runs: Runs = 100,
    skip_accuracy: SkipAccuracy = False,
    out: Out = None,
) -> None:
    """Quantize a trained model to INT8 after training and measure the variant."""
    from edgeproof.quantize import PTQ, build

    spec = PTQ(runtime=runtime, method=method, per_channel=per_channel,  # type: ignore[arg-type]
               keep_head=keep_head, calib=calib, head=_head(name, head))  # fmt: skip
    path = build(MODELS, name, spec, YOLO_DATA, _imgsz(name))
    _measure(name, runtime, spec.tag, path, head=spec.head, threads=threads, runs=runs,
             accuracy=not skip_accuracy, out=out or RESULTS / "ptq.jsonl")  # fmt: skip


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)
