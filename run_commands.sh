TPU_NAME="taiming-v4-64_000036"
ZONE="us-central2-b"
PROJECT_ID="vision-mix"

chmod 600 ~/.ssh/id_rsa

gcloud compute tpus tpu-vm ssh terry@${TPU_NAME} \
  --project=${PROJECT_ID} --zone=${ZONE} \
  --worker=all \
  --ssh-key-file="~/.ssh/id_rsa" \
  --command='whoami && ls'

gcloud compute tpus tpu-vm ssh terry@${TPU_NAME} \
  --project=${PROJECT_ID} --zone=${ZONE} \
  --worker=all \
  --ssh-key-file="~/.ssh/id_rsa" \
  --command='rm -rf ~/vision && git clone https://github.com/TaiMingLu/vision_transformers_torch_xla.git ~/vision'

gcloud compute tpus tpu-vm ssh terry@${TPU_NAME} \
  --project=${PROJECT_ID} --zone=${ZONE} \
  --worker=all \
  --ssh-key-file="~/.ssh/id_rsa" \
  --command='rm -rf ~/vision_env && python3 -m venv ~/vision_env && source ~/vision_env/bin/activate && python -m pip install -r /home/terry/vision/requirements.txt'
gcloud compute tpus tpu-vm ssh terry@${TPU_NAME} \
  --project=${PROJECT_ID} --zone=${ZONE} \
  --worker=all \
  --ssh-key-file="~/.ssh/id_rsa" \
  --command='source ~/vision_env/bin/activate && python - <<'PY'
import torch_xla.core.xla_model as xm
print(xm.get_xla_supported_devices())
PY
'

# Dat loader testing
gcloud compute tpus tpu-vm ssh terry@${TPU_NAME} \
  --project=${PROJECT_ID} --zone=${ZONE} \
  --worker=all \
  --ssh-key-file="~/.ssh/id_rsa" \
  --command='source ~/vision_env/bin/activate && cd ~/vision &&  python tools/test_tfds_loader.py --data-dir /home/terry/gcs-bucket/Distillation/imagenet_tfds --split train --num-samples 8 --time-it'
gcloud compute tpus tpu-vm ssh terry@${TPU_NAME} \
  --project=${PROJECT_ID} --zone=${ZONE} \
  --worker=all \
  --ssh-key-file="~/.ssh/id_rsa" \
  --command='source ~/vision_env/bin/activate && cd ~/vision &&  PJRT_DEVICE=TPU torchrun --nproc_per_node=8 tools/test_tfds_loader.py --data-dir /home/terry/gcs-bucket/Distillation/imagenet_tfds --split train --num-samples 8 --time-it'

gcloud compute tpus tpu-vm ssh terry@${TPU_NAME} \
    --project=${PROJECT_ID} --zone=${ZONE} --worker=0 \
    --ssh-key-file="~/.ssh/id_rsa" \
    --command="
    export TPU_PREFIX=taiming-v4-64
    source ~/vision_env/bin/activate
     cd ~/vision
     bash run.sh"


gcloud compute tpus tpu-vm ssh terry@${TPU_NAME} \
  --project=${PROJECT_ID} --zone=${ZONE} --worker=0 \
  --ssh-key-file="~/.ssh/id_rsa" \
  --command="
  cd ~/vision
  git pull
  export TPU_PREFIX=taiming-v4-64
  export ZONE=us-central2-b
  export PROJECT_ID=vision-mix
  export GLOBAL_BATCH_SIZE=4096
  source ~/vision_env/bin/activate
  bash run_train.sh"




gcloud compute tpus tpu-vm ssh terry@${TPU_NAME} \
  --project=${PROJECT_ID} --zone=${ZONE} \
  --worker=all \
  --ssh-key-file="~/.ssh/id_rsa" \
  --command='cd ~/vision && git pull && cd ~'
  
gcloud compute tpus tpu-vm ssh terry@${TPU_NAME} \
  --project=${PROJECT_ID} --zone=${ZONE} \
  --worker=all \
  --ssh-key-file="~/.ssh/id_rsa" \
  --command='source ~/vision_env/bin/activate && python3 -m pip install --upgrade timm==0.9.16'









gcloud compute tpus tpu-vm ssh terry@${TPU_NAME} \
    --project=${PROJECT_ID} --zone=${ZONE} --worker=all \
    --ssh-key-file="~/.ssh/id_rsa" \
    --command=$'
#!/bin/bash

source ~/vision_env/bin/activate

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
    --update_freq 1 \
    --model_ema true \
    --model_ema_eval true \
    --data_set IMNET \
    --data_path /home/terry/gcs-bucket/Distillation/imagenet_tfds \
    --output_dir /home/terry/gcs-bucket/Distillation/models/vanilla \
    --experiment b-vanilla
' 2>&1 | tee /home/tl0463/storage/scratch_zhuangl/tl0463/language/logstrain_tpu_v4_64.log
