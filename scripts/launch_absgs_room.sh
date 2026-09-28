#!/usr/bin/env bash
set -euo pipefail

# Isolate AbsGS on the canon+stage λ4096 30k stack. Room on GPU 7 only.
# Share VRAM; do not wait for exclusive access.
# Does not launch garden. Does not kill other jobs. Does not overwrite protected arms.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

source /home/lsh/Documents/miniconda3/etc/profile.d/conda.sh
conda activate rdo_gaussian
PYTHON="${PYTHON:-/home/lsh/Documents/miniconda3/envs/rdo_gaussian/bin/python}"
DATA_ROOT="${DATA_ROOT:-/home/lsh/Data/xwj/data}"

ARM="absgs_canon_stage_l4096_rate2"
SCENE="room"
GPU=7
SCALE_LMBDA=4096
SCALE_CBSIZE=8192
SH_MASK_LAMBDA=0.05
GS_MASK_LAMBDA=0.005
VQ_PATCH=16384
# Share GPU 7 with existing jobs. Do not wait for exclusive access. Do not kill others.

PROTECTED_ARMS=(
  rate2
  rate2_retrain
  rate0
  rate5
  shac_lst_rate2
  canon_rate2
  stage_rate2
  canon_l4096_rate2
  canon_stage_l4096_rate2
)
for protected in "${PROTECTED_ARMS[@]}"; do
  if [[ "${ARM}" == "${protected}" ]]; then
    echo "Refuse to overwrite protected arm: ${ARM}" >&2
    exit 1
  fi
done

gpu_busy() {
  local gpu="$1"
  local used
  used="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)"
  used="$(echo "${used}" | tr -d '[:space:]')"
  [[ -n "${used}" ]]
}

PATH_SOURCE="${DATA_ROOT}/mipnerf360/${SCENE}"
PATH_OUTPUT="output/mipnerf360/${SCENE}/${ARM}"
LOG="output/_logs/absgs_room.log"

mkdir -p output/_logs

if [[ ! -d "${PATH_SOURCE}/images" || ! -d "${PATH_SOURCE}/sparse" ]]; then
  echo "[$(date '+%F %T')] ERROR missing data ${PATH_SOURCE}" >&2
  exit 1
fi

if [[ -f "${PATH_OUTPUT}/results.json" ]]; then
  echo "[$(date '+%F %T')] SKIP ${SCENE}/${ARM} (results.json exists)" | tee -a "${LOG}"
  "${PYTHON}" -u scripts/compare_absgs.py
  exit 0
fi

if [[ -d "${PATH_OUTPUT}" ]]; then
  echo "[$(date '+%F %T')] REFUSE incomplete output already exists at ${PATH_OUTPUT}" >&2
  exit 1
fi

if gpu_busy "${GPU}"; then
  echo "[$(date '+%F %T')] SHARE GPU ${GPU}; existing jobs left running"
  nvidia-smi -i "${GPU}" --query-compute-apps=pid,process_name,used_memory --format=csv || true
fi
mkdir -p "${PATH_OUTPUT}"

echo "[$(date '+%F %T')] TRAIN ${SCENE}/${ARM} GPU=${GPU} -r=2 absgs+canon+stage lambda_scale=${SCALE_LMBDA}"
(
  export CUDA_VISIBLE_DEVICES="${GPU}"
  "${PYTHON}" -u train.py \
    -s="${PATH_SOURCE}" \
    -m="${PATH_OUTPUT}" \
    -r=2 \
    --eval \
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
    --gs_mask_lr 0.01

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

"${PYTHON}" -u scripts/compare_absgs.py
echo "[$(date '+%F %T')] room complete; garden not launched"
