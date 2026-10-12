#!/usr/bin/env bash
# Builds the VisA split and YOLO labels on the free CPU partition (download runs on the login
# node beforehand: compute nodes have no internet).
#SBATCH --job-name=edgeproof-data
#SBATCH --partition=devel
#SBATCH --cpus-per-task=8
#SBATCH --time=00:45:00
set -euo pipefail
BASE="${EDGEPROOF_BASE:-$HPCWORK/edgeproof}"
module load Python/3.11.5
source "$BASE/venv/bin/activate"
export EDGEPROOF_ROOT="$BASE"
edgeproof data --tar "$BASE/data/raw/VisA_20220922.tar"
