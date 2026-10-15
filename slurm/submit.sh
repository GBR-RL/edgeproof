#!/usr/bin/env bash
# Submits a GPU job only if the requested time fits the remaining GPU quota with a reserve.
# Usage: slurm/submit.sh HH:MM:SS <sbatch script> [script args...]
set -euo pipefail
LIMIT="$1"; shift
RESERVE_GPU_H="${RESERVE_GPU_H:-1}"
IFS=: read -r h m s <<<"$LIMIT"
want=$(awk -v h="$h" -v m="$m" -v s="$s" 'BEGIN { printf "%.2f", h + m / 60 + s / 3600 }')
usage=$(r_wlm_usage -q)
# Case-insensitive: r_wlm_usage has printed both "GPU-h" and "gpu-h".
consumable=$(awk -F: 'tolower($1) == "consumable gpu-h" { gsub(/ /, "", $2); print $2 }' <<<"$usage")
used=$(awk -F: 'tolower($1) == "consumed gpu-h current month" { gsub(/ /, "", $2); print $2 }' <<<"$usage")
if [ -z "$consumable" ] || [ -z "$used" ]; then
  echo "refusing: could not read the GPU quota" >&2
  exit 1
fi
left=$(awk -v c="$consumable" -v u="$used" 'BEGIN { printf "%.2f", c - u }')
echo "GPU-h requested $want, left $left, reserve $RESERVE_GPU_H"
if awk -v w="$want" -v l="$left" -v r="$RESERVE_GPU_H" 'BEGIN { exit !(w + r > l) }'; then
  echo "refusing: not enough GPU quota" >&2
  exit 1
fi
mkdir -p "$HPCWORK/edgeproof/logs"
sbatch --time="$LIMIT" --output="$HPCWORK/edgeproof/logs/%x-%j.out" "$@"
