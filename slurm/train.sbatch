#!/usr/bin/env bash
# One baseline on one H100. Billing is per node share: one GPU blocks 24 cores and 122 GB, so the
# job takes exactly that much and no more. Submit through submit.sh, which checks the GPU quota.
#SBATCH --job-name=edgeproof-train
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=00:45:00
set -euo pipefail
BASE="${EDGEPROOF_BASE:-$HPCWORK/edgeproof}"
module load Python/3.11.5
source "$BASE/venv/bin/activate"
export EDGEPROOF_ROOT="$BASE" YOLO_OFFLINE=1
cd "$BASE"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
# Data on the node-local disk: thousands of small reads are slow on the shared file system.
LOCAL="${TMPDIR:-/tmp}/edgeproof-data"
mkdir -p "$LOCAL"
cp -a "$BASE/data/yolo" "$LOCAL/"  # -a keeps the hard links between the task folders
sed -i "s#^path: .*/yolo/#path: $LOCAL/yolo/#" "$LOCAL"/yolo/*/data.yaml
EDGEPROOF_YOLO_DATA="$LOCAL/yolo" edgeproof train --workers 16 "$@"
