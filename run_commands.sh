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
  --command='source ~/vision_env/bin/activate && cd ~/vision && python tools/test_tfds_loader.py --data-dir /home/terry/gcs-bucket/Distillation/imagenet_tfds --split train --num-samples 8 --num-workers 8'


gcloud compute tpus tpu-vm ssh terry@${TPU_NAME} \
  --project=${PROJECT_ID} --zone=${ZONE} \
  --worker=all \
  --ssh-key-file="~/.ssh/id_rsa" \
  --command='rm -rf ~/vision && git clone https://github.com/TaiMingLu/vision_transformers_torch_xla.git ~/vision'
  
gcloud compute tpus tpu-vm ssh terry@${TPU_NAME} \
  --project=${PROJECT_ID} --zone=${ZONE} \
  --worker=all \
  --ssh-key-file="~/.ssh/id_rsa" \
  --command='source ~/vision_env/bin/activate && python -m pip install datasets'

