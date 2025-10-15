#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}"

TPU_PREFIX="${TPU_PREFIX:-taiming-v4-64}"
ZONE="${ZONE:-us-central2-b}"
PROJECT_ID="${PROJECT_ID:-vision-mix}"
DATA_DIR="${DATA_DIR:-/home/terry/gcs-bucket/Distillation/imagenet_tfds}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/terry/gcs-bucket/Distillation/models/vanilla}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-b-vanilla}"
TPU_LOG_DIR="${TPU_LOG_DIR:-/home/terry/tpu_logs}"
WORLD_SIZE="${WORLD_SIZE:-32}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4096}"

if (( GLOBAL_BATCH_SIZE % WORLD_SIZE != 0 )); then
  echo "Global batch size ${GLOBAL_BATCH_SIZE} is not divisible by world size ${WORLD_SIZE}" >&2
  exit 1
fi
PER_CORE_BATCH_SIZE=$(( GLOBAL_BATCH_SIZE / WORLD_SIZE ))

mkdir -p "${TPU_LOG_DIR}"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  if [[ -f "${HOME}/vision_env/bin/activate" ]]; then
    source "${HOME}/vision_env/bin/activate"
  else
    echo "Virtualenv not activated and ~/vision_env not found" >&2
    exit 1
  fi
fi

export TPU_PREFIX ZONE PROJECT_ID DATA_DIR TPU_LOG_DIR
export PJRT_DEVICE="${PJRT_DEVICE:-TPU}"

if [[ -n "${WANDB_API_KEY:-}" ]]; then
  export WANDB_API_KEY
fi

echo "Launching training via multihost_runner_orig"
echo "TPU prefix: ${TPU_PREFIX}"; echo "Zone: ${ZONE}"; echo "Project: ${PROJECT_ID}"
echo "Data dir: ${DATA_DIR}"; echo "Output dir: ${OUTPUT_DIR}"
echo "World size: ${WORLD_SIZE} | Global batch size: ${GLOBAL_BATCH_SIZE} (${PER_CORE_BATCH_SIZE} per core)"

REMOTE_COMMAND=$(cat <<EOF
mkdir -p ${TPU_LOG_DIR}
export TPU_LOG_DIR=${TPU_LOG_DIR}
source ~/vision_env/bin/activate
export PJRT_DEVICE=TPU
export XLA_USE_BF16=1
cd ~/vision
python -u main.py \
    --tpu \
    --model my_vit_b \
    --epochs 300 \
    --drop_path 0.1 \
    --batch_size ${PER_CORE_BATCH_SIZE} \
    --world_size \${WORLD_SIZE} \
    --rank \$RANK \
    --local_rank \$LOCAL_RANK \
    --update_freq 1 \
    --model_ema true \
    --model_ema_eval true \
    --data_set IMNET \
    --data_path ${DATA_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --experiment ${EXPERIMENT_NAME}
EOF
)

if [[ -n "${WANDB_API_KEY:-}" ]]; then
  REMOTE_COMMAND="export WANDB_API_KEY=${WANDB_API_KEY} && ${REMOTE_COMMAND}"
fi

python -u multihost_runner_orig.py \
  --TPU_PREFIX="${TPU_PREFIX}" \
  --PROJECT="${PROJECT_ID}" \
  --ZONE="${ZONE}" \
  --INTERNAL_IP=true \
  --COMMAND="${REMOTE_COMMAND}"
