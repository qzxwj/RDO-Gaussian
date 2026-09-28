#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"

RATE_IDX="${1:-0}"
if ! [[ "$RATE_IDX" =~ ^[0-5]$ ]]; then
  echo "RATE_IDX must be an integer in 0..5, got: ${RATE_IDX}" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/home/lsh/Documents/miniconda3/envs/rdo_gaussian/bin/python}"
DATA_ROOT="${DATA_ROOT:-/home/lsh/Data/xwj/data}"

declare -a sh_mask_lambda_list=(
  0.005
  0.02
  0.05
  0.1
  0.2
  0.5
)
declare -a gs_mask_lambda_list=(
  0.0005
  0.002
  0.005
  0.01
  0.02
  0.05
)

SH_MASK_LAMBDA="${sh_mask_lambda_list[$RATE_IDX]}"
GS_MASK_LAMBDA="${gs_mask_lambda_list[$RATE_IDX]}"

# Outdoor / indoor resolution for Mip-NeRF 360 (3DGS convention).
# Smaller vq_patch_size on mip360 avoids ECVQ distance-matrix OOM on large point clouds.
MIP360_OUTDOOR="bicycle flowers garden stump treehill"
MIP360_INDOOR="bonsai counter kitchen room"

# dataset_key:data_subdir:scene
declare -a JOBS=(
  "tandt:tandt:train"
  "tandt:tandt:truck"
  "db:blending:drjohnson"
  "db:blending:playroom"
  "mipnerf360:mipnerf360:bicycle"
  "mipnerf360:mipnerf360:bonsai"
  "mipnerf360:mipnerf360:counter"
  "mipnerf360:mipnerf360:flowers"
  "mipnerf360:mipnerf360:garden"
  "mipnerf360:mipnerf360:kitchen"
  "mipnerf360:mipnerf360:room"
  "mipnerf360:mipnerf360:stump"
  "mipnerf360:mipnerf360:treehill"
)

resolution_args_for() {
  local dataset_key="$1"
  local scene="$2"
  if [[ "${dataset_key}" != "mipnerf360" ]]; then
    echo ""
    return
  fi
  if [[ " ${MIP360_OUTDOOR} " == *" ${scene} "* ]]; then
    echo "-r=4"
  elif [[ " ${MIP360_INDOOR} " == *" ${scene} "* ]]; then
    echo "-r=2"
  else
    echo "-r=4"
  fi
}

vq_patch_size_for() {
  local dataset_key="$1"
  if [[ "${dataset_key}" == "mipnerf360" ]]; then
    echo "16384"
  else
    echo "65536"
  fi
}

echo "[$(date '+%F %T')] start RATE_IDX=${RATE_IDX} sh_mask_lambda=${SH_MASK_LAMBDA} gs_mask_lambda=${GS_MASK_LAMBDA} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[$(date '+%F %T')] PYTHON=${PYTHON}"

for job in "${JOBS[@]}"; do
  IFS=':' read -r dataset_key data_subdir scene <<< "$job"
  path_source="${DATA_ROOT}/${data_subdir}/${scene}"
  path_output="output/${dataset_key}/${scene}/rate${RATE_IDX}"

  if [[ ! -d "${path_source}/images" || ! -d "${path_source}/sparse" ]]; then
    echo "[$(date '+%F %T')] ERROR missing data for ${dataset_key}/${scene}: ${path_source}" >&2
    exit 1
  fi

  if [[ -f "${path_output}/results.json" ]]; then
    echo "[$(date '+%F %T')] SKIP ${dataset_key}/${scene}/rate${RATE_IDX} (results.json exists)"
    continue
  fi

  # Drop incomplete outputs from a previous failed run of this scene/rate.
  if [[ -d "${path_output}" ]]; then
    echo "[$(date '+%F %T')] CLEAN incomplete ${path_output}"
    rm -rf "${path_output}"
  fi

  mkdir -p "${path_output}"
  res_arg="$(resolution_args_for "${dataset_key}" "${scene}")"
  vq_patch="$(vq_patch_size_for "${dataset_key}")"

  echo "[$(date '+%F %T')] TRAIN ${dataset_key}/${scene}/rate${RATE_IDX} ${res_arg} vq_patch_size=${vq_patch}"
  # shellcheck disable=SC2086
  "${PYTHON}" -u train.py \
    -s="${path_source}" \
    -m="${path_output}" \
    ${res_arg} \
    --iterations 30000 \
    --vq_cb_lr 0.0002 \
    --vq_logits_lr 0.002 \
    --vq_scale_lmbda 32768 \
    --vq_rot_lmbda 256 \
    --vq_dc_lmbda 256 \
    --vq_sh1_lmbda 256 \
    --vq_sh2_lmbda 256 \
    --vq_sh3_lmbda 256 \
    --vq_scale_cbsize 8192 \
    --vq_rot_cbsize 8192 \
    --vq_dc_cbsize 8192 \
    --vq_sh1_cbsize 4096 \
    --vq_sh2_cbsize 4096 \
    --vq_sh3_cbsize 4096 \
    --vq_patch_size "${vq_patch}" \
    --sh_mask_lambda "${SH_MASK_LAMBDA}" \
    --sh_mask_lr 0.05 \
    --gs_mask_lambda "${GS_MASK_LAMBDA}" \
    --gs_mask_lr 0.01 \
    --eval

  echo "[$(date '+%F %T')] RENDER ${dataset_key}/${scene}/rate${RATE_IDX}"
  "${PYTHON}" render.py -m "${path_output}" -s "${path_source}" --skip_train

  echo "[$(date '+%F %T')] METRICS ${dataset_key}/${scene}/rate${RATE_IDX}"
  "${PYTHON}" metrics.py -m "${path_output}"

  echo "[$(date '+%F %T')] DONE ${dataset_key}/${scene}/rate${RATE_IDX}"
done

echo "[$(date '+%F %T')] all jobs finished for RATE_IDX=${RATE_IDX}"
