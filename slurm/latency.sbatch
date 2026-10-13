#!/usr/bin/env bash
# Latency of every model, head and runtime on one whole CPU node, so no other job shares its
# memory bandwidth or caches; accuracy comes from bench.sbatch. A whole node bills all 96 cores,
# so this job measures latency only and stays short.
# Usage: sbatch --output=... slurm/latency.sbatch <out file> <model>:<head> [...]
#SBATCH --job-name=edgeproof-latency
#SBATCH --partition=c23ms
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:30:00
set -euo pipefail
BASE="${EDGEPROOF_BASE:-$HPCWORK/edgeproof}"
module load Python/3.11.5
source "$BASE/venv/bin/activate"
export EDGEPROOF_ROOT="$BASE" YOLO_OFFLINE=1 CUDA_VISIBLE_DEVICES=""
cd "$BASE"
lscpu | grep "Model name"
OUT="$1"; shift
for variant in "$@"; do
  edgeproof bench --name "${variant%%:*}" --head "${variant##*:}" --skip-accuracy --out "$OUT"
done
