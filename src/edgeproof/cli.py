"""Command line entry point."""

from __future__ import annotations

import typer

from edgeproof import __version__

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.callback()
def main() -> None:
    """What accuracy it costs to deploy an inspection model on edge hardware."""


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)
