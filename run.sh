TPU_NAME="taiming-v4-64"
ZONE="us-central2-b"
PROJECT_ID="vision-mix"


python -u multihost_runner_orig.py \
    --TPU_PREFIX=$TPU_PREFIX \
    --INTERNAL_IP=true \
    --COMMAND=$'
    export TPU_LOG_DIR=/home/terry/tpu_logs
    source ~/vision_env/bin/activate
    export WANDB_API_KEY=01126ae90da25bae0d86704140ac978cb9fd9c73
    cd ~/vision
    python3.10 -u -m tools/test_tfds_loader_multihost.py \
        --data-dir /home/terry/gcs-bucket/Distillation/imagenet_tfds \
        --samples-per-loop 128 --num-loops 32'