#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}"

TPU_PREFIX="${TPU_PREFIX:-taiming-v4-64}"
ZONE="${ZONE:-us-central2-b}"
PROJECT_ID="${PROJECT_ID:-vision-mix}"
DATA_DIR="${DATA_DIR:-/home/terry/gcs-bucket/Distillation/imagenet_tfds}"
SAMPLES_PER_LOOP="${SAMPLES_PER_LOOP:-128}"
NUM_LOOPS="${NUM_LOOPS:-32}"
LOG_EVERY="${LOG_EVERY:-4}"
TPU_LOG_DIR="${TPU_LOG_DIR:-/home/terry/tpu_logs}"

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

echo "Running TFDS multihost loader via multihost_runner_orig"
echo "TPU prefix: ${TPU_PREFIX}"; echo "Zone: ${ZONE}"; echo "Project: ${PROJECT_ID}"
echo "Data dir: ${DATA_DIR}"; echo "Loops: ${NUM_LOOPS} x ${SAMPLES_PER_LOOP} (log every ${LOG_EVERY})"

REMOTE_COMMAND="mkdir -p ${TPU_LOG_DIR} && export TPU_LOG_DIR=${TPU_LOG_DIR} && source ~/vision_env/bin/activate && export PJRT_DEVICE=TPU && cd ~/vision && python -u -m tools.test_tfds_loader_multihost --data-dir ${DATA_DIR} --samples-per-loop ${SAMPLES_PER_LOOP} --num-loops ${NUM_LOOPS} --log-every ${LOG_EVERY}"

if [[ -n "${WANDB_API_KEY:-}" ]]; then
  REMOTE_COMMAND="export WANDB_API_KEY=${WANDB_API_KEY} && ${REMOTE_COMMAND}"
fi

python -u multihost_runner_orig.py \
  --TPU_PREFIX="${TPU_PREFIX}" \
  --PROJECT="${PROJECT_ID}" \
  --ZONE="${ZONE}" \
  --INTERNAL_IP=true \
  --COMMAND="${REMOTE_COMMAND}"
