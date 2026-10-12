"""Command line entry point."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from edgeproof import __version__
from edgeproof.config import DATA

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


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)
