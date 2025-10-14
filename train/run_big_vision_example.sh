#!/usr/bin/env bash
# Example ImageNet training launch using the big_vision TFDS dataloader.
# 
# This script is meant as a reference. Edit the variables below to point to your
# TFDS ImageNet directory and desired output locations before running it on a
# TPU or GPU host.

set -euo pipefail

# ---- user configuration ---------------------------------------------------
TFDS_DATA_DIR="/home/tl0463/storage/scratch_zhuangl/tl0463/language/big_vision/imagenet_tfds"
OUTPUT_DIR="/tmp/vision_transformers_torch_xla/example_run"
WORKDIR="$(dirname "$(realpath "$0")")/.."

# PJRT_DEVICE can be set to "TPU" or "CPU" depending on the target runtime.
export PJRT_DEVICE="TPU"
export XLA_USE_BF16=1

# ---- derived settings -----------------------------------------------------
mkdir -p "${OUTPUT_DIR}"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

python3 -u main.py \
  --data_set IMNET \
  --tfds_data_dir "${TFDS_DATA_DIR}" \
  --tfds_train_split 'train' \
  --tfds_eval_split 'validation' \
  --tfds_cache_raw False \
  --tfds_cache_eval False \
  --big_vision_pp_train 'decode_jpeg_and_inception_crop(224)|flip_lr|value_range(0, 1)|keep("image", "label")' \
  --big_vision_pp_eval 'decode|resize_small(256)|central_crop(224)|value_range(0, 1)|keep("image", "label")' \
  --batch_size 256 \
  --epochs 300 \
  --model vit_tiny_patch16_224 \
  --lr 0.003 \
  --update_freq 1 \
  --output_dir "${OUTPUT_DIR}" \
  --log_dir "${LOG_DIR}" \
  --enable_wandb False \
  --dist_eval True \
  --tpu True
