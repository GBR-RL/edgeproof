#!/usr/bin/env bash
# Latency of every variant on one whole CPU node, so no other job shares its memory bandwidth or
# caches; accuracy comes from cpu.sbatch. A whole node bills all 96 cores, so this job measures
# latency only and stays short. Each line of the variants file is one edgeproof command (bench or
# ptq) that has already run once, so its exports and quantized models exist.
# Usage: sbatch --output=... slurm/latency.sbatch <out file> <variants file>
#SBATCH --job-name=edgeproof-latency
#SBATCH --partition=c23ms
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:50:00
set -euo pipefail
BASE="${EDGEPROOF_BASE:-$HPCWORK/edgeproof}"
module load Python/3.11.5
source "$BASE/venv/bin/activate"
export EDGEPROOF_ROOT="$BASE" YOLO_OFFLINE=1 CUDA_VISIBLE_DEVICES=""
cd "$BASE"
lscpu | grep "Model name"
OUT="$1"
grep -v '^\s*\(#\|$\)' "$2" | while read -r -a cmd; do
  edgeproof "${cmd[@]}" --skip-accuracy --out "$OUT" < /dev/null
done
