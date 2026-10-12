"""Command line entry point."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from edgeproof import __version__
from edgeproof.config import DATA, MODELS, YOLO_DATA

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


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)
