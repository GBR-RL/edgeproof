#!/usr/bin/env bash
# One edgeproof command (bench, ptq) on a CLAIX-2023 CPU node (Xeon Platinum 8468), with the data
# on the node-local disk. Billing follows the larger of the core and memory shares, so memory
# stays at the per-core default; for more memory, ask for more cores (--cpus-per-task=16).
# Latency runs on 4 threads; the other cores keep the job's own evaluation from competing with it.
# Usage: sbatch --output=... slurm/cpu.sbatch bench --name <model> --head <e2e|nms> --out <file>
#        sbatch --output=... slurm/cpu.sbatch ptq --name <model> --method entropy --out <file>
#SBATCH --job-name=edgeproof-cpu
#SBATCH --partition=c23ms
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2540M
#SBATCH --time=01:30:00
set -euo pipefail
BASE="${EDGEPROOF_BASE:-$HPCWORK/edgeproof}"
module load Python/3.11.5
source "$BASE/venv/bin/activate"
export EDGEPROOF_ROOT="$BASE" YOLO_OFFLINE=1 CUDA_VISIBLE_DEVICES=""
cd "$BASE"
lscpu | grep "Model name"
LOCAL="${TMPDIR:-/tmp}/edgeproof-data"
mkdir -p "$LOCAL"
cp -a "$BASE/data/yolo" "$LOCAL/"
sed -i "s#^path: .*/yolo/#path: $LOCAL/yolo/#" "$LOCAL"/yolo/*/data.yaml
EDGEPROOF_YOLO_DATA="$LOCAL/yolo" edgeproof "$@"
