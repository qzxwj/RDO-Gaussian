#!/bin/bash
set -uo pipefail

# RDO-Gaussian BungeeNeRF batch on one GPU: rate0 (high), rate2 (mid), rate5 (low), sequentially.
# Usage: bash launch_bungee_gpu7.sh [GPU_ID]
# NOTE: no `set -e` on purpose: launch_bungee_rate.sh tolerates per-scene
# failures (see 2026-08-11 pompidou OOM), so one bad scene must not abort
# the remaining rates.
GPU_ID="${1:-7}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Reduce CUDA memory fragmentation (see launch_bungee_rate.sh).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$ROOT/output/_logs"
MASTER_LOG="$ROOT/output/_logs/bungee_all_rates_gpu${GPU_ID}_$(date +%Y%m%d_%H%M%S).log"
echo "$MASTER_LOG" > "$ROOT/output/_logs/current_bungee_all_rates.log"

{
  echo "[$(date '+%F %T')] RDO BungeeNeRF batch on GPU ${GPU_ID}: rates 0 -> 2 -> 5"
  for RATE_IDX in 0 2 5; do
    echo "[$(date '+%F %T')] ########## RATE_IDX=${RATE_IDX} ##########"
    if ! bash "$ROOT/scripts/launch_bungee_rate.sh" "$RATE_IDX" "$GPU_ID"; then
      echo "[$(date '+%F %T')] RATE_IDX=${RATE_IDX} exited non-zero; continuing"
    fi
  done
  echo "[$(date '+%F %T')] All RDO BungeeNeRF finished: $(date -Is)"
} 2>&1 | tee "$MASTER_LOG"
