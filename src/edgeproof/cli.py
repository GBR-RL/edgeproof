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


@app.command()
def finetune(
    name: Annotated[str, typer.Option(help="Trained baseline, e.g. yolo26s-det-1024.")],
    qat: Annotated[bool, typer.Option(help="With INT8 fake quantization (QAT).")] = True,
    epochs: Annotated[int, typer.Option(help="Epochs.")] = 10,
    lr0: Annotated[float, typer.Option(help="Initial AdamW learning rate.")] = 2e-4,
    batch: Annotated[int | None, typer.Option(help="Batch size, the baseline's by default.")] = (
        None
    ),
    workers: Annotated[int, typer.Option(help="Data loader processes.")] = 8,
    fraction: Annotated[float, typer.Option(help="Share of the training set (smoke tests).")] = 1.0,
    device: Annotated[str | None, typer.Option(help="auto, cpu, cuda, cuda:1, mps.")] = None,
) -> None:
    """Continue training a baseline with quantization-aware training, or without as a control."""
    from edgeproof.train import finetune as run

    summary = run(name, MODELS, YOLO_DATA, qat=qat, epochs=epochs, lr0=lr0, batch=batch,
                  device=device, workers=workers, fraction=fraction)  # fmt: skip
    typer.echo(json.dumps(summary["test"]))


def _measure(
    name: str, runtime: str, precision: str, path: Path, *, head: str, threads: int, runs: int,
    accuracy: bool, out: Path, compute: str = "fp32",
) -> dict[str, Any]:  # fmt: skip
    """One result row: latency and size on this machine, accuracy through the runtime.
    ``compute`` is the precision OpenVINO runs the float operations in."""
    import platform

    from edgeproof import bench as b
    from edgeproof.export import size_mb

    summary = json.loads((MODELS / f"{name}.json").read_text())
    imgsz, task = summary["recipe"]["imgsz"], summary["recipe"]["task"]
    row: dict[str, Any] = {"model": name, "head": head, "runtime": runtime,
                           "precision": precision, "cpu": b.cpu_name(),
                           "arch": platform.machine().lower(),
                           "size_mb": size_mb(path)}  # fmt: skip
    kw: dict[str, Any] = {"head": head, "compute": compute}
    row |= b.latency(path, runtime, imgsz, threads=threads, runs=runs, **kw)  # type: ignore[arg-type]
    if accuracy:
        row |= b.accuracy(path, task, YOLO_DATA, imgsz, **kw)
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
    precision: Annotated[str, typer.Option(help="fp32, or bf16 (OpenVINO only).")] = "fp32",
    threads: Threads = 4,
    runs: Runs = 100,
    skip_accuracy: SkipAccuracy = False,
    out: Out = None,
) -> None:
    """Export a trained model to each runtime and measure latency, size and accuracy on CPU."""
    from edgeproof.export import export

    if precision not in {"fp32", "bf16"} or (precision == "bf16" and runtimes != "openvino"):
        raise typer.BadParameter("--precision is fp32, or bf16 with --runtimes openvino")
    head = _head(name, head)
    for runtime in runtimes.split(","):
        path = export(MODELS, name, runtime, _imgsz(name), head)  # type: ignore[arg-type]
        _measure(name, runtime, precision, path, head=head, threads=threads, runs=runs,
                 accuracy=not skip_accuracy, out=out or RESULTS / "runtimes.jsonl",
                 compute=precision)  # fmt: skip


@app.command()
def ptq(
    name: Annotated[str, typer.Option(help="Trained model, e.g. yolo26s-det-1024.")],
    runtime: Annotated[str, typer.Option(help="onnxruntime or openvino.")] = "onnxruntime",
    method: Annotated[str, typer.Option(help="minmax|entropy|percentile or performance|mixed.")] = (
        "minmax"
    ),
    per_channel: Annotated[bool, typer.Option(help="Per-channel weight scales (ORT).")] = True,
    keep_head: Annotated[bool, typer.Option(help="Keep the detection head in float.")] = False,
    float_decode: Annotated[bool, typer.Option(help="Keep only the head's decoding in float.")] = (
        False
    ),
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
               keep_head=keep_head, keep_decode=float_decode, calib=calib,
               head=_head(name, head))  # fmt: skip
    path = build(MODELS, name, spec, YOLO_DATA, _imgsz(name))
    _measure(name, runtime, spec.tag, path, head=spec.head, threads=threads, runs=runs,
             accuracy=not skip_accuracy, out=out or RESULTS / "ptq.jsonl")  # fmt: skip


@app.command()
def gpu(
    name: Annotated[str, typer.Option(help="Trained model, e.g. yolo26s-det-1024 or -qat.")],
    runtime: Annotated[str, typer.Option(help="tensorrt or torch.")] = "tensorrt",
    precision: Annotated[str, typer.Option(help="fp32, fp16 or int8 (TensorRT).")] = "fp16",
    head: HeadOpt = "e2e",
    runs: Annotated[int, typer.Option(help="Timed runs.")] = 1000,
    skip_accuracy: SkipAccuracy = False,
    out: Out = None,
) -> None:
    """Measure a model on the GPU: a TensorRT engine, or PyTorch for reference. INT8 engines of a
    QAT model (``-qat``) take its trained ranges; of any other model, PTQ ranges."""
    import platform

    from edgeproof import bench as b
    from edgeproof import gpu as g
    from edgeproof.export import size_mb

    summary = json.loads((MODELS / f"{name}.json").read_text())
    imgsz, task = summary["recipe"]["imgsz"], summary["recipe"]["task"]
    qat = bool(summary.get("finetune", {}).get("qat"))
    head = _head(name, head)
    if runtime == "tensorrt":
        data_yaml = YOLO_DATA / task / "data.yaml"
        path = g.export_engine(MODELS, name, imgsz, head, precision, data_yaml=data_yaml, qat=qat)  # type: ignore[arg-type]
    elif runtime == "torch" and precision == "fp32":
        path, precision = MODELS / f"{name}.pt", "int8-sim" if qat else "fp32"
    else:
        raise typer.BadParameter("--runtime tensorrt, or torch with --precision fp32")
    row: dict[str, Any] = {"model": name, "head": head, "runtime": runtime,
                           "precision": precision, "cpu": b.cpu_name(),
                           "arch": platform.machine().lower(),
                           "size_mb": size_mb(path)}  # fmt: skip
    row |= g.latency(path, runtime, imgsz, head=head, runs=runs)  # type: ignore[arg-type]
    if not skip_accuracy:
        row |= b.accuracy(path, task, YOLO_DATA, imgsz, head=head, device=g.DEVICE)  # type: ignore[arg-type]
    out = out or RESULTS / "gpu.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    typer.echo(json.dumps(row))


@app.command()
def gate(
    config: Annotated[Path, typer.Option(help="Gate definition (YAML).")] = Path("perf/gate.yaml"),
    out: Annotated[Path, typer.Option(help="Result rows (JSON lines).")] = RESULTS / "gate.jsonl",
    summary: Annotated[Path | None, typer.Option(help="Append the Markdown table here.")] = None,
) -> None:
    """Build, measure and check every variant of the gate; exit 1 if any regresses."""
    from edgeproof import gate as g

    spec = g.load(config)
    rows = []
    for variant in spec.variants:
        typer.echo(f"== {variant.name}: edgeproof {' '.join(variant.args)}")
        app([*variant.args, "--out", str(out)], standalone_mode=False)
        last = out.read_text(encoding="utf-8").splitlines()[-1]
        rows.append((variant, json.loads(last)))
    table = g.summary(spec, rows)
    typer.echo(table)
    if summary is not None:
        with summary.open("a", encoding="utf-8") as f:
            f.write(table + "\n")
    if any(g.failures(spec, variant, row) for variant, row in rows):
        raise typer.Exit(1)


@app.command()
def report(
    accuracy: Annotated[Path, typer.Option(help="Accuracy rows (JSON lines).")] = (
        RESULTS / "accuracy.jsonl"
    ),
    latency: Annotated[Path, typer.Option(help="Whole-node latency rows (JSON lines).")] = (
        RESULTS / "latency.jsonl"
    ),
    gpu: Annotated[Path, typer.Option(help="GPU rows (JSON lines).")] = RESULTS / "gpu.jsonl",
) -> None:
    """Print the FP32, BF16, INT8 and GPU result tables as Markdown."""
    from edgeproof import report as r

    rows = r.join(r.load(accuracy), r.load(latency) if latency.exists() else [])
    tables = [r.fp32_table(rows), r.bf16_table(rows), r.int8_table(rows)]
    if gpu.exists():
        tables.append(r.gpu_table(r.load(gpu)))
    typer.echo("\n\n".join(tables))


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)
