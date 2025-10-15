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
cd /home/terry/vision

# Set essential TPU environment variables
export USE_TPU=1
export XLA_USE_BF16=1
export PJRT_DEVICE=TPU

PJRT_DEVICE=TPU torchrun --nproc_per_node=8 main.py \
    --tpu \
    --model my_vit_b \
    --epochs 300 \
    --drop_path 0.1 \
    --batch_size 512 \
    --update_freq 1 \
    --model_ema true \
    --model_ema_eval true \
    --data_set IMNET \
    --data_path /home/terry/gcs-bucket/Distillation/imagenet_tfds \
    --output_dir /home/terry/gcs-bucket/Distillation/models/vanilla \
    --experiment b-vanilla
' 2>&1 | tee ~/train_tpu_v4_64.log

echo "Training completed. Check ~/train_tpu_v4_64.log for details."
