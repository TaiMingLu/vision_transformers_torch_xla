#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

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
    # Ensure the TPU environment is active when run.sh is invoked standalone.
    source "${HOME}/vision_env/bin/activate"
  else
    echo "Virtualenv not activated and ~/vision_env not found" >&2
    exit 1
  fi
fi

# Verify torch_xla is available; install requirements if not.
if ! python - <<'PY' >/dev/null 2>&1
import importlib
spec = importlib.util.find_spec("torch_xla.distributed.xla_dist")
raise SystemExit(0 if spec else 1)
PY
then
  echo "torch_xla.distributed.xla_dist not found; installing python requirements..." >&2
  python -m pip install --upgrade pip >/dev/null
  python -m pip install -r "${SCRIPT_DIR}/requirements.txt"
fi

if ! python - <<'PY' >/dev/null 2>&1
import importlib
spec = importlib.util.find_spec("torch_xla.distributed.xla_dist")
raise SystemExit(0 if spec else 1)
PY
then
  cat >&2 <<'ERR'
Failed to locate torch_xla.distributed.xla_dist even after installing requirements.
Ensure torch-xla is available in this environment (see requirements.txt) and rerun.
ERR
  exit 1
fi

export TPU_PREFIX ZONE PROJECT_ID DATA_DIR TPU_LOG_DIR
export PJRT_DEVICE="${PJRT_DEVICE:-TPU}"

if [[ -n "${WANDB_API_KEY:-}" ]]; then
  export WANDB_API_KEY
fi

echo "Running TFDS multihost loader with TPU=${TPU_PREFIX} (zone=${ZONE}, project=${PROJECT_ID})"
echo "Data dir: ${DATA_DIR}"; echo "Loops: ${NUM_LOOPS} x ${SAMPLES_PER_LOOP} (log every ${LOG_EVERY})"

DIST_ARGS=("--tpu=${TPU_PREFIX}")
[[ -n "${ZONE}" ]] && DIST_ARGS+=("--zone=${ZONE}")
[[ -n "${PROJECT_ID}" ]] && DIST_ARGS+=("--project=${PROJECT_ID}")

python -m torch_xla.distributed.xla_dist \
  "${DIST_ARGS[@]}" \
  -- python -u -m tools.test_tfds_loader_multihost \
      --data-dir "${DATA_DIR}" \
      --samples-per-loop "${SAMPLES_PER_LOOP}" \
      --num-loops "${NUM_LOOPS}" \
      --log-every "${LOG_EVERY}"
