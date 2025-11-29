#!/bin/bash
set -e

cd ~/CS224W_Project

export PROJECT_ID=$(gcloud config get-value project)
export REGION="${REGION:-us-central1}"
export GCS_BUCKET="${GCS_BUCKET:-cs224w-2025-mae-gnn-central}"
export IMAGE_URI="gcr.io/${PROJECT_ID}/cs224w-training:latest"
export JOB_NAME="cs224w-hptuning-$(date +%Y%m%d-%H%M%S)"

echo "=========================================="
echo "Hyperparameter Tuning Job Submission"
echo "=========================================="
echo "Project: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "GCS Bucket: ${GCS_BUCKET}"
echo "Image: ${IMAGE_URI}"
echo "Job: ${JOB_NAME}"
echo "Config: hyperparam_config.yaml"
echo ""

echo "Building Docker image with hypertune support..."
docker build -t ${IMAGE_URI} .

echo ""
echo "Pushing to Container Registry..."
docker push ${IMAGE_URI}

echo ""
echo "Creating worker pool spec with environment variables..."

# Create worker pool spec with GCS_BUCKET environment variable
cat > /tmp/worker_pool_spec_${JOB_NAME}.json << WORKER_EOF
{
  "machineSpec": {
    "machineType": "n1-standard-8",
    "acceleratorType": "NVIDIA_TESLA_T4",
    "acceleratorCount": 1
  },
  "replicaCount": 1,
  "containerSpec": {
    "imageUri": "${IMAGE_URI}",
    "env": [
      {
        "name": "GCS_BUCKET",
        "value": "${GCS_BUCKET}"
      }
    ],
    "args": [
      "--gcs_bucket", "${GCS_BUCKET}",
      "--train_prefix", "data_w_pos/plgems_train.pt",
      "--val_prefix", "data_w_pos/plgems_validation.pt",
      "--test_prefix", "data_w_pos/plgems_full_casf2016.pt",
      "--num_epochs", "50",
      "--checkpoint_interval", "10",
      "--cache_dir", "/tmp/pyg_cache",
      "--model_save_path", "/tmp/model"
    ]
  }
}
WORKER_EOF

echo ""
echo "Submitting hyperparameter tuning job..."

gcloud ai hp-tuning-jobs create \
    --region=${REGION} \
    --display-name=${JOB_NAME} \
    --config=hyperparam_config.yaml \
    --worker-pool-spec=/tmp/worker_pool_spec_${JOB_NAME}.json \
    --base-output-directory=gs://${GCS_BUCKET}/hptuning/${JOB_NAME}

echo ""
echo "=========================================="
echo "Hyperparameter Tuning Job Submitted!"
echo "=========================================="
echo "Job Name: ${JOB_NAME}"
echo ""
echo "Monitor:"
echo "  Console: https://console.cloud.google.com/vertex-ai/training/training-pipelines?project=${PROJECT_ID}"
echo ""
echo "Checkpoints location:"
echo "  gs://${GCS_BUCKET}/checkpoints/trial_*/"
echo ""