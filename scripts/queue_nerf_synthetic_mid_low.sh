#!/bin/bash
set -euo pipefail

# Wait for free GPUs and launch RDO midrate (rate2) then lowrate (rate5).
# Never steal GPUs reserved for live wave-1 jobs (LG=0, SOG=1, NOSH=2, RDO0=3).

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAUNCH="${ROOT}/scripts/launch_nerf_synthetic_rate.sh"
MIN_FREE_MIB=20000
LOG_DIR="${ROOT}/output/_logs"
mkdir -p "$LOG_DIR"
QUEUE_LOG="${LOG_DIR}/queue_nerf_synthetic_mid_low_$(date +%Y%m%d_%H%M%S).log"

reserved_gpus() {
  local out=()
  local lg sog nosh rdo0
  lg=$(cat /home/lsh/Data/xwj/LightGaussian/outputs/_logs/nerf_synthetic_gpu0.pid 2>/dev/null || true)
  # Also honor LG resume pid / any active LG launcher pid files
  local lg_resume
  lg_resume=$(cat /home/lsh/Data/xwj/LightGaussian/outputs/_logs/nerf_synthetic_resume.pid 2>/dev/null || true)
  sog=$(cat /home/lsh/Data/xwj/Self-Organizing-Gaussians/output/_logs/nerf_synthetic_sog_gpu1.pid 2>/dev/null || true)
  nosh=$(cat /home/lsh/Data/xwj/Self-Organizing-Gaussians/output/_logs/nerf_synthetic_nosh_gpu2.pid 2>/dev/null || true)
  rdo0=$(cat /home/lsh/Data/xwj/RDO-Gaussian/output/_logs/nerf_synthetic_rate0.pid 2>/dev/null || true)
  if [[ -n "$lg" ]] && kill -0 "$lg" 2>/dev/null; then out+=(0); fi
  if [[ -n "$sog" ]] && kill -0 "$sog" 2>/dev/null; then out+=(1); fi
  if [[ -n "$nosh" ]] && kill -0 "$nosh" 2>/dev/null; then out+=(2); fi
  if [[ -n "$rdo0" ]] && kill -0 "$rdo0" 2>/dev/null; then out+=(3); fi
  # Explicitly reserved GPUs written by this queue after launching jobs.
  if [[ -f "${LOG_DIR}/reserved_gpus.txt" ]]; then
    out+=($(cat "${LOG_DIR}/reserved_gpus.txt"))
  fi
  echo "${out[*]}"
}

find_free_gpu() {
  local reserved
  reserved=" $(reserved_gpus) "
  while IFS=',' read -r idx free; do
    idx="${idx// /}"
    free="${free// /}"
    if [[ "$free" -ge "$MIN_FREE_MIB" ]] && [[ "$reserved" != *" $idx "* ]]; then
      echo "$idx"
      return 0
    fi
  done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)
  return 1
}

wait_and_launch() {
  local rate="$1"
  local name="$2"
  local marker="${LOG_DIR}/nerf_synthetic_rate${rate}.started"
  if [[ -f "$marker" ]]; then
    echo "[$(date '+%F %T')] ${name} already started (marker exists)" | tee -a "$QUEUE_LOG"
    return 0
  fi
  if [[ -f "${ROOT}/output/nerf_synthetic/ship/rate${rate}/results.json" ]]; then
    echo "[$(date '+%F %T')] ${name} already complete" | tee -a "$QUEUE_LOG"
    touch "$marker"
    return 0
  fi

  echo "[$(date '+%F %T')] waiting for free non-reserved GPU for ${name} (rate${rate}); reserved=[$(reserved_gpus)]" | tee -a "$QUEUE_LOG"
  while true; do
    gpu="$(find_free_gpu || true)"
    if [[ -n "${gpu}" ]]; then
      echo "[$(date '+%F %T')] launching ${name} on GPU ${gpu} (reserved=[$(reserved_gpus)])" | tee -a "$QUEUE_LOG"
      touch "$marker"
      echo "$gpu" >> "${LOG_DIR}/reserved_gpus.txt"
      nohup bash "$LAUNCH" "$rate" "$gpu" \
        > "${LOG_DIR}/launcher_rate${rate}_gpu${gpu}.stdout" 2>&1 &
      echo $! > "${LOG_DIR}/nerf_synthetic_rate${rate}.pid"
      echo "[$(date '+%F %T')] pid=$(cat "${LOG_DIR}/nerf_synthetic_rate${rate}.pid")" | tee -a "$QUEUE_LOG"
      # Keep GPU reserved for the lifetime of this rate job via reserved_gpus.txt
      return 0
    fi
    sleep 60
  done
}

{
  echo "[$(date '+%F %T')] queue start for RDO midrate/lowrate"
  echo "[$(date '+%F %T')] initial reserved=[$(reserved_gpus)]"
  wait_and_launch 2 "RDO-Gaussian (midrate)"
  wait_and_launch 5 "RDO-Gaussian (lowrate)"
  echo "[$(date '+%F %T')] queue finished launching mid+low"
} 2>&1 | tee -a "$QUEUE_LOG"
