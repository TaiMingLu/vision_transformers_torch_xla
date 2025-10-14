#!/bin/bash

# TPU v4-64 training script - Most reliable approach
# Replace the variables below with your actual values

TPU_NAME="your-tpu-name"
ZONE="your-zone"
PROJECT_ID="your-project-id"

echo "Starting TPU v4-64 training on all 8 hosts (64 cores total)"
echo "TPU: $TPU_NAME in zone $ZONE"

# Method 3: Use Python multiprocessing launcher (most reliable for v4-64)
gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
  --zone="$ZONE" --project="$PROJECT_ID" --worker=all \
  --command '
#!/bin/bash
cd /home/terry/gcs-bucket/Distillation/vision/train

# Clear any conflicting environment variables
unset MASTER_ADDR MASTER_PORT RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE
unset NCCL_SOCKET_IFNAME NCCL_IB_HCA NCCL_IB_GID_INDEX NCCL_DEBUG

# Set essential TPU environment variables
export USE_TPU=1
export XLA_USE_BF16=1
export PJRT_DEVICE=TPU

# Use the standard XLA multiprocessing launcher
python3 -m torch_xla.distributed.xla_dist \
    --tpu=$TPU_NAME \
    --restart-tpu-pod-server \
    -- \
    python3 main.py \
    --tpu \
    --model my_vit_b \
    --epochs 100 \
    --drop_path 0.1 \
    --batch_size 32 \
    --update_freq 1 \
    --model_ema true \
    --model_ema_eval true \
    --data_set IMNET_HDF5 \
    --data_path /home/terry/gcs-bucket/Distillation/vision/datasets/imagenet-hdf5 \
    --output_dir /home/terry/gcs-bucket/Distillation/vision/models/vanilla \
    --experiment b-vanilla
' 2>&1 | tee ~/train_tpu_v4_64.log

echo "Training completed. Check ~/train_tpu_v4_64.log for details."
