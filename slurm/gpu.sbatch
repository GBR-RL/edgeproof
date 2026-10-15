#!/usr/bin/env bash
# GPU work on one H100: quantization-aware training and the TensorRT measurements, in the GPU
# environment (slurm/setup-gpu.sh). Runs every edgeproof command of a file in order, so one job
# copies the data once. Same node share as train.sbatch; submit through submit.sh.
# Usage: slurm/submit.sh HH:MM:SS slurm/gpu.sbatch <commands file>
#SBATCH --job-name=edgeproof-gpu
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
set -euo pipefail
BASE="${EDGEPROOF_BASE:-$HPCWORK/edgeproof}"
module load Python/3.11.5
source "$BASE/venv-gpu/bin/activate"
export EDGEPROOF_ROOT="$BASE" YOLO_OFFLINE=1
cd "$BASE"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 3)"
LOCAL="${TMPDIR:-/tmp}/edgeproof-data"
mkdir -p "$LOCAL"
cp -a "$BASE/data/yolo" "$LOCAL/"
sed -i "s#^path: .*/yolo/#path: $LOCAL/yolo/#" "$LOCAL"/yolo/*/data.yaml
grep -v '^\s*\(#\|$\)' "$1" | while read -r -a cmd; do
  echo "== edgeproof ${cmd[*]}"
  EDGEPROOF_YOLO_DATA="$LOCAL/yolo" edgeproof "${cmd[@]}" < /dev/null
done
