#!/bin/bash
set -euo pipefail

# Usage: launch_bungee_rate.sh <RATE_IDX> <GPU_ID>
# RATE_IDX: 0=highrate, 2=midrate, 5=lowrate (synthetic lambda table)
# BungeeNeRF 6-scene subset; outputs to output/bungeenerf/<scene>/rate<IDX>.

RATE_IDX="${1:?RATE_IDX required}"
GPU_ID="${2:?GPU_ID required}"

if ! [[ "$RATE_IDX" =~ ^(0|2|5)$ ]]; then
  echo "RATE_IDX must be 0, 2, or 5, got: ${RATE_IDX}" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

source /home/lsh/Documents/miniconda3/etc/profile.d/conda.sh
conda activate rdo_gaussian
PYTHON="${PYTHON:-/home/lsh/Documents/miniconda3/envs/rdo_gaussian/bin/python}"
DATA_ROOT="/home/lsh/Data/xwj/data/bungeenerf"

declare -a SCENES=(amsterdam bilbao hollywood pompidou quebec rome)

# Indices align with run_synthetic.sh lists; we only use 0/2/5.
declare -a sh_mask_lambda_list=(0.0005 0.001 0.0025 0.005 0.01 0.025)
declare -a gs_mask_lambda_list=(0.0001 0.0002 0.0005 0.001 0.002 0.005)

SH_MASK_LAMBDA="${sh_mask_lambda_list[$RATE_IDX]}"
GS_MASK_LAMBDA="${gs_mask_lambda_list[$RATE_IDX]}"

mkdir -p output/_logs
LOG="output/_logs/bungee_rate${RATE_IDX}_gpu${GPU_ID}_$(date +%Y%m%d_%H%M%S).log"
echo "$LOG" > "output/_logs/current_bungee_rate${RATE_IDX}.log"

{
  echo "[$(date '+%F %T')] start RDO bungee RATE_IDX=${RATE_IDX} GPU=${GPU_ID}"
  echo "sh_mask_lambda=${SH_MASK_LAMBDA} gs_mask_lambda=${GS_MASK_LAMBDA}"

  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
  # Reduce CUDA memory fragmentation: without this, densify peak (~23GB on
  # 1.6K bungeenerf) leaves unusable cached blocks and OOMs. See 2026-08-11
  # hollywood rate0 crash at iter 15000.
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

  for SCENE in "${SCENES[@]}"; do
    path_source="${DATA_ROOT}/${SCENE}"
    path_output="output/bungeenerf/${SCENE}/rate${RATE_IDX}"

    echo "[$(date '+%F %T')] ########## ${SCENE}/rate${RATE_IDX} ##########"
    echo "SOURCE=${path_source} OUT=${path_output}"

    if [[ -f "${path_output}/results.json" ]]; then
      echo "[$(date '+%F %T')] SKIP ${SCENE}/rate${RATE_IDX} (results.json exists)"
      cat "${path_output}/results.json"
      continue
    fi

    if [[ -d "${path_output}" ]]; then
      echo "[$(date '+%F %T')] CLEAN incomplete ${path_output}"
      rm -rf "${path_output}"
    fi
    mkdir -p "${path_output}"

    echo "[$(date '+%F %T')] TRAIN ${SCENE}/rate${RATE_IDX}"
    # densify_grad_threshold raised 0.0002 -> 0.0003 to cap gaussian count:
    # pompidou exploded to ~7.6M gaussians and OOM'd the 24GB card at iter
    # 15000 (2026-08-11). vq_patch_size lowered 16384 -> 8192 to halve the
    # VQ-phase cdist peak. Per-scene failures are tolerated so one OOM no
    # longer aborts the whole rate batch.
    if ! "${PYTHON}" -u train.py \
      -s="${path_source}" \
      -m="${path_output}" \
      -w \
      --iterations 30000 \
      --densify_grad_threshold 0.0003 \
      --vq_cb_lr 0.0002 \
      --vq_logits_lr 0.002 \
      --vq_scale_lmbda 32768 \
      --vq_rot_lmbda 256 \
      --vq_dc_lmbda 256 \
      --vq_sh1_lmbda 256 \
      --vq_sh2_lmbda 256 \
      --vq_sh3_lmbda 256 \
      --vq_scale_cbsize 4096 \
      --vq_rot_cbsize 4096 \
      --vq_dc_cbsize 4096 \
      --vq_sh1_cbsize 2048 \
      --vq_sh2_cbsize 2048 \
      --vq_sh3_cbsize 2048 \
      --vq_patch_size 8192 \
      --sh_mask_lambda "${SH_MASK_LAMBDA}" \
      --sh_mask_lr 0.005 \
      --gs_mask_lambda "${GS_MASK_LAMBDA}" \
      --gs_mask_lr 0.01 \
      --eval; then
      echo "[$(date '+%F %T')] FAILED ${SCENE}/rate${RATE_IDX} (train) — continuing"
      continue
    fi

    echo "[$(date '+%F %T')] RENDER ${SCENE}/rate${RATE_IDX}"
    if ! "${PYTHON}" render.py -m "${path_output}" -s "${path_source}" -w --skip_train --eval; then
      echo "[$(date '+%F %T')] FAILED ${SCENE}/rate${RATE_IDX} (render) — continuing"
      continue
    fi

    echo "[$(date '+%F %T')] METRICS ${SCENE}/rate${RATE_IDX}"
    if ! "${PYTHON}" metrics.py -m "${path_output}"; then
      echo "[$(date '+%F %T')] FAILED ${SCENE}/rate${RATE_IDX} (metrics) — continuing"
      continue
    fi

    if [[ ! -f "${path_output}/results.json" ]]; then
      echo "ERROR: missing ${path_output}/results.json" >&2
      exit 1
    fi
    echo "[$(date '+%F %T')] DONE ${SCENE}/rate${RATE_IDX}"
    cat "${path_output}/results.json"
  done

  echo "[$(date '+%F %T')] all scenes finished for RATE_IDX=${RATE_IDX} on GPU ${GPU_ID}"
} 2>&1 | tee "$LOG"
