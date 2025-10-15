TPU_NAME="taiming-v4-64_000036"
ZONE="us-central2-b"
PROJECT_ID="vision-mix"

# Run a command on EVERY TPU host in the node/slice
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --project=${PROJECT_ID} --zone=${ZONE} \
  --worker=all \
  --command='whoami'
