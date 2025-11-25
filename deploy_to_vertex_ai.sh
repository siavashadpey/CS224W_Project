#!/bin/bash
# deploy_to_vertex_ai.sh - Deploy training to Vertex AI

set -e

cd ~/CS224W_Project

# Configuration
export PROJECT_ID=$(gcloud config get-value project)
export REGION="${REGION:-us-east1}"  # Allow override
export IMAGE_NAME="cs224w-training"
export IMAGE_TAG="latest"
export IMAGE_URI="gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${IMAGE_TAG}"
export JOB_NAME="cs224w-training-$(date +%Y%m%d-%H%M%S)"

# Machine configuration - set USE_GPU=false to use CPU only
export USE_GPU="${USE_GPU:-true}"
export MACHINE_TYPE="${MACHINE_TYPE:-n1-standard-8}"
export GPU_TYPE="${GPU_TYPE:-NVIDIA_TESLA_T4}"

# Training configuration
export DATASET_FILE_PREFIX="00AEPL_"
export GCS_BUCKET="cs224w-2025-mae-gnn-bucket"
export TRAIN_PREFIX="data/GEMS_pytorch_datasets/${DATASET_FILE_PREFIX}train_cleansplit"
export VAL_PREFIX="data/GEMS_pytorch_datasets/${DATASET_FILE_PREFIX}casf2013"
export TEST_PREFIX="data/GEMS_pytorch_datasets/${DATASET_FILE_PREFIX}casf2016_indep"
export NUM_EPOCHS="${NUM_EPOCHS:-1}"
export BATCH_SIZE="32"
export CHECKPOINT_INTERVAL="10"

echo "=========================================="
echo "Vertex AI Deployment"
echo "=========================================="
echo "Project: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "Image: ${IMAGE_URI}"
echo "Job: ${JOB_NAME}"
echo "GPU: ${USE_GPU}"
echo ""

# Build worker spec based on GPU flag
if [ "$USE_GPU" = "true" ]; then
    WORKER_SPEC="machine-type=${MACHINE_TYPE},replica-count=1,accelerator-type=${GPU_TYPE},accelerator-count=1,container-image-uri=${IMAGE_URI}"
    echo "Using GPU: ${GPU_TYPE}"
else
    WORKER_SPEC="machine-type=${MACHINE_TYPE},replica-count=1,container-image-uri=${IMAGE_URI}"
    echo "Using CPU only"
fi

# Step 1: Build Docker image
echo ""
echo "Building Docker image..."
docker build -t ${IMAGE_URI} .

# Step 2: Push to GCR
echo ""
echo "Pushing image to Google Container Registry..."
docker push ${IMAGE_URI}

# Step 3: Submit job
echo ""
echo "Submitting training job to Vertex AI..."
gcloud ai custom-jobs create \
    --region=${REGION} \
    --display-name=${JOB_NAME} \
    --worker-pool-spec=${WORKER_SPEC} \
    --args="--gcs_bucket=${GCS_BUCKET},--train_prefix=${TRAIN_PREFIX},--val_prefix=${VAL_PREFIX},--test_prefix=${TEST_PREFIX},--num_epochs=${NUM_EPOCHS},--batch_size=${BATCH_SIZE},--checkpoint_interval=${CHECKPOINT_INTERVAL},--model_save_path=/gcs/${GCS_BUCKET}/models"

echo ""
echo "=========================================="
echo "Deployment Complete!"
echo "=========================================="
echo "Job Name: ${JOB_NAME}"
echo ""
echo "Monitor job:"
echo "  Console: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=${PROJECT_ID}"
echo "  Logs: gcloud ai custom-jobs stream-logs ${JOB_NAME} --region=${REGION}"
echo ""
EOF

chmod +x deploy_to_vertex_ai.sh