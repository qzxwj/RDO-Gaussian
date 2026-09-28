#!/usr/bin/env bash
set -euo pipefail

# Canon and stage freeze ablations on mipnerf360 room/garden rate2.
# GPU 4/6 only. Wave 1: room two arms. Wave 2: garden two arms.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

source /home/lsh/Documents/miniconda3/etc/profile.d/conda.sh
conda activate rdo_gaussian
PYTHON="${PYTHON:-/home/lsh/Documents/miniconda3/envs/rdo_gaussian/bin/python}"
DATA_ROOT="${DATA_ROOT:-/home/lsh/Data/xwj/data}"

GPUS=(4 6)
SH_MASK_LAMBDA=0.05
GS_MASK_LAMBDA=0.005
VQ_PATCH=16384

mkdir -p output/_logs

gpu_busy() {
  local gpu="$1"
  local used
  used="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)"
  used="$(echo "${used}" | tr -d '[:space:]')"
  [[ -n "${used}" ]]
}

assert_gpus_free() {
  local busy=0
  for gpu in "${GPUS[@]}"; do
    if gpu_busy "${gpu}"; then
      echo "[$(date '+%F %T')] ERROR GPU ${gpu} is occupied" >&2
      nvidia-smi -i "${gpu}" --query-compute-apps=pid,process_name,used_memory --format=csv >&2 || true
      busy=1
    fi
  done
  if [[ "${busy}" -ne 0 ]]; then
    echo "Refuse to remap onto GPU 0/1/2/3. Free 4/6 or stop this script." >&2
    exit 1
  fi
}

common_train_args() {
  local scene="$1"
  local res_arg
  if [[ "${scene}" == "garden" ]]; then
    res_arg="-r=4"
  else
    res_arg="-r=2"
  fi
  echo "${res_arg}"
}

run_job() {
  local gpu="$1"
  local scene="$2"
  local arm="$3"
  shift 3
  local extra_args=("$@")

  local path_source="${DATA_ROOT}/mipnerf360/${scene}"
  local path_output="output/mipnerf360/${scene}/${arm}"
  local log="output/_logs/${arm}_${scene}.log"
  local res_arg
  res_arg="$(common_train_args "${scene}")"

  if [[ ! -d "${path_source}/images" || ! -d "${path_source}/sparse" ]]; then
    echo "[$(date '+%F %T')] ERROR missing data ${path_source}" >&2
    exit 1
  fi

  if [[ -f "${path_output}/results.json" ]]; then
    echo "[$(date '+%F %T')] SKIP ${scene}/${arm} (results.json exists)"
    return 0
  fi

  if [[ -d "${path_output}" ]]; then
    echo "[$(date '+%F %T')] CLEAN incomplete ${path_output}"
    rm -rf "${path_output}"
  fi
  mkdir -p "${path_output}"

  echo "[$(date '+%F %T')] TRAIN ${scene}/${arm} GPU=${gpu} ${res_arg}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
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
      --vq_patch_size "${VQ_PATCH}" \
      --sh_mask_lambda "${SH_MASK_LAMBDA}" \
      --sh_mask_lr 0.05 \
      --gs_mask_lambda "${GS_MASK_LAMBDA}" \
      --gs_mask_lr 0.01 \
      --eval \
      "${extra_args[@]}"

    echo "[$(date '+%F %T')] RENDER ${scene}/${arm}"
    "${PYTHON}" render.py -m "${path_output}" -s "${path_source}" --skip_train

    echo "[$(date '+%F %T')] METRICS ${scene}/${arm}"
    "${PYTHON}" metrics.py -m "${path_output}"

    if [[ ! -f "${path_output}/results.json" ]]; then
      echo "ERROR: missing ${path_output}/results.json" >&2
      exit 1
    fi
    echo "[$(date '+%F %T')] DONE ${scene}/${arm}"
  ) >"${log}" 2>&1
}

run_wave() {
  local scene="$1"
  assert_gpus_free

  echo "[$(date '+%F %T')] wave start scene=${scene}"
  run_job 4 "${scene}" "canon_rate2" \
    --geo_canonicalize --no_absgs --no_freeze_masks_at_vq \
    --vq_start_iter 15001 --rate_constrain_iter 20001 &
  local p4=$!
  run_job 6 "${scene}" "stage_rate2" \
    --vq_start_iter 20001 --rate_constrain_iter 25001 --freeze_masks_at_vq \
    --no_absgs --no_geo_canonicalize &
  local p6=$!

  local failed=0
  if ! wait "${p4}"; then
    echo "[$(date '+%F %T')] FAIL ${scene}/canon_rate2" >&2
    failed=1
  fi
  if ! wait "${p6}"; then
    echo "[$(date '+%F %T')] FAIL ${scene}/stage_rate2" >&2
    failed=1
  fi
  if [[ "${failed}" -ne 0 ]]; then
    exit 1
  fi
  echo "[$(date '+%F %T')] wave done scene=${scene}"
}

echo "[$(date '+%F %T')] launch_canon_stage GPUs=${GPUS[*]} PYTHON=${PYTHON}"
assert_gpus_free
run_wave room
run_wave garden
"${PYTHON}" scripts/compare_rate2.py
echo "[$(date '+%F %T')] all canon/stage jobs finished"
