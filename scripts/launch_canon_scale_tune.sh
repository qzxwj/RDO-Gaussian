#!/usr/bin/env bash
set -euo pipefail

# One-arm sequential tuner on GPU 4 only.
# Usage: launch_canon_scale_tune.sh <scene> <arm> <vq_scale_lmbda> [vq_scale_cbsize]
# Waits if GPU 4 is busy. Does not launch a second arm.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

source /home/lsh/Documents/miniconda3/etc/profile.d/conda.sh
conda activate rdo_gaussian
PYTHON="${PYTHON:-/home/lsh/Documents/miniconda3/envs/rdo_gaussian/bin/python}"
DATA_ROOT="${DATA_ROOT:-/home/lsh/Data/xwj/data}"

GPU=4
POLL_SEC=60
SH_MASK_LAMBDA=0.05
GS_MASK_LAMBDA=0.005
VQ_PATCH=16384

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: $0 <scene> <arm> <vq_scale_lmbda> [vq_scale_cbsize]" >&2
  exit 2
fi

SCENE="$1"
ARM="$2"
SCALE_LMBDA="$3"
SCALE_CBSIZE="${4:-8192}"

PROTECTED_ARMS=(
  rate2
  rate2_retrain
  rate0
  rate5
  canon_rate2
  stage_rate2
)
for protected in "${PROTECTED_ARMS[@]}"; do
  if [[ "${ARM}" == "${protected}" ]]; then
    echo "Refuse to overwrite protected arm: ${ARM}" >&2
    exit 1
  fi
done

if [[ "${SCENE}" == "garden" ]]; then
  RES_ARG="-r=4"
else
  RES_ARG="-r=2"
fi

PATH_SOURCE="${DATA_ROOT}/mipnerf360/${SCENE}"
PATH_OUTPUT="output/mipnerf360/${SCENE}/${ARM}"
LOG="output/_logs/canon_scale_${ARM}_${SCENE}.log"

gpu_busy() {
  local used
  used="$(nvidia-smi -i "${GPU}" --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)"
  used="$(echo "${used}" | tr -d '[:space:]')"
  [[ -n "${used}" ]]
}

wait_gpu_free() {
  if ! gpu_busy; then
    return 0
  fi
  echo "[$(date '+%F %T')] WAIT GPU ${GPU} busy"
  nvidia-smi -i "${GPU}" --query-compute-apps=pid,process_name,used_memory --format=csv || true
  while gpu_busy; do
    sleep "${POLL_SEC}"
  done
  echo "[$(date '+%F %T')] GPU ${GPU} is free"
}

if [[ ! -d "${PATH_SOURCE}/images" || ! -d "${PATH_SOURCE}/sparse" ]]; then
  echo "[$(date '+%F %T')] ERROR missing data ${PATH_SOURCE}" >&2
  exit 1
fi

mkdir -p output/_logs

if [[ -f "${PATH_OUTPUT}/results.json" ]]; then
  echo "[$(date '+%F %T')] SKIP ${SCENE}/${ARM} (results.json exists)"
  exit 0
fi

if [[ -d "${PATH_OUTPUT}" ]]; then
  echo "[$(date '+%F %T')] CLEAN incomplete ${PATH_OUTPUT}"
  rm -rf "${PATH_OUTPUT}"
fi
mkdir -p "${PATH_OUTPUT}"

wait_gpu_free

echo "[$(date '+%F %T')] TRAIN ${SCENE}/${ARM} GPU=${GPU} ${RES_ARG} lambda_scale=${SCALE_LMBDA} K_scale=${SCALE_CBSIZE}"
(
  export CUDA_VISIBLE_DEVICES="${GPU}"
  "${PYTHON}" -u train.py \
    -s="${PATH_SOURCE}" \
    -m="${PATH_OUTPUT}" \
    ${RES_ARG} \
    --iterations 30000 \
    --vq_cb_lr 0.0002 \
    --vq_logits_lr 0.002 \
    --vq_scale_lmbda "${SCALE_LMBDA}" \
    --vq_rot_lmbda 256 \
    --vq_dc_lmbda 256 \
    --vq_sh1_lmbda 256 \
    --vq_sh2_lmbda 256 \
    --vq_sh3_lmbda 256 \
    --vq_scale_cbsize "${SCALE_CBSIZE}" \
    --vq_rot_cbsize 8192 \
    --vq_dc_cbsize 8192 \
    --vq_sh1_cbsize 4096 \
    --vq_sh2_cbsize 4096 \
    --vq_sh3_cbsize 4096 \
    --vq_patch_size "${VQ_PATCH}" \
    --sh_mask_lambda "${SH_MASK_LAMBDA}" \
    --sh_mask_lr 0.05 \
    --gs_mask_lambda "${GS_MASK_LAMBDA}" \
    --gs_mask_lr 0.01 \
    --eval \
    --geo_canonicalize \
    --no_absgs \
    --no_freeze_masks_at_vq \
    --vq_start_iter 15001 \
    --rate_constrain_iter 20001

  echo "[$(date '+%F %T')] RENDER ${SCENE}/${ARM}"
  "${PYTHON}" render.py -m "${PATH_OUTPUT}" -s "${PATH_SOURCE}" --skip_train

  echo "[$(date '+%F %T')] METRICS ${SCENE}/${ARM}"
  "${PYTHON}" metrics.py -m "${PATH_OUTPUT}"

  if [[ ! -f "${PATH_OUTPUT}/results.json" ]]; then
    echo "ERROR: missing ${PATH_OUTPUT}/results.json" >&2
    exit 1
  fi
  echo "[$(date '+%F %T')] DONE ${SCENE}/${ARM}"
) >"${LOG}" 2>&1
