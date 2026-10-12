#!/usr/bin/env bash
# One-time environment on a CLAIX login node: Python venv, the package, pretrained weights.
# Installing is light work and allowed on login nodes; everything heavy runs through Slurm.
set -euo pipefail
BASE="${EDGEPROOF_BASE:-$HPCWORK/edgeproof}"
module load Python/3.11.5
cd "$BASE/repo"
[ -d "$BASE/venv" ] || python -m venv "$BASE/venv"
source "$BASE/venv/bin/activate"
pip install -q --upgrade pip
pip install -q -e . "ultralytics>=8.4" "onnx>=1.16" "onnxruntime>=1.18"
export EDGEPROOF_ROOT="$BASE"
cd "$BASE"  # Ultralytics keeps its AMP check model in ./weights
edgeproof prefetch --families yolo26,yolo11 --sizes s
python -c "import torch, ultralytics; print(torch.__version__, torch.version.cuda, ultralytics.__version__)"
