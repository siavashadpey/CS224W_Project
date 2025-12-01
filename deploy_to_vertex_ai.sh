#!/bin/bash
# deploy_to_vertex_ai.sh - Deploy training to Vertex AI

set -e

cd ~/CS224W_Project

# Configuration
export PROJECT_ID=$(gcloud config get-value project)
export REGION="${REGION:-us-central1}"  # Allow override us-east1
export IMAGE_NAME="cs224w-training"
export IMAGE_TAG="latest"
export IMAGE_URI="gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${IMAGE_TAG}"
export JOB_NAME="cs224w-training-$(date +%Y%m%d-%H%M%S)"

# Machine configuration - set USE_GPU=false to use CPU only
export USE_GPU="${USE_GPU:-true}"
export MACHINE_TYPE="${MACHINE_TYPE:-n1-standard-4}"
export GPU_TYPE="${GPU_TYPE:-NVIDIA_TESLA_T4}"

export CHECKPOINT_ID="${CHECKPOINT_ID}"

# export GCS_BUCKET="cs224w-2025-mae-gnn-bucket"
export GCS_BUCKET="cs224w-2025-mae-gnn-central"

# Set GCS bucket based on region (unless explicitly overridden)
if [ -z "${GCS_BUCKET}" ]; then
    case "${REGION}" in
        us-central1)
            export GCS_BUCKET="cs224w-2025-mae-gnn-central"
            ;;
        us-east1)
            export GCS_BUCKET="cs224w-2025-mae-gnn-bucket"
            ;;
        *)
            # Default fallback - extract region prefix
            REGION_PREFIX=$(echo ${REGION} | cut -d'-' -f1-2)
            export GCS_BUCKET="cs224w-2025-mae-gnn-${REGION_PREFIX}"
            echo "Warning: Using auto-generated bucket name for region ${REGION}"
            echo "   Bucket: ${GCS_BUCKET}"
            ;;
    esac
fi

# Training configuration
export TRAIN_PREFIX="data_w_pos/plgems_train.pt"
export VAL_PREFIX="data_w_pos/plgems_validation.pt"
export TEST_PREFIX="data_w_pos/plgems_full_casf2016.pt"
export NUM_EPOCHS="${NUM_EPOCHS:-150}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-10}"
export LEARNING_RATE=${LEARNING_RATE:-0.0001}
export HIDDEN_DIM="${HIDDEN_DIM:-256}"
export NUM_ENCODER_LAYERS="${NUM_ENCODER_LAYERS:-4}"
export NUM_DECODER_LAYERS="${NUM_DECODER_LAYERS:-4}"
export MASKING_RATIO="${MASKING_RATIO:-0.3}"

echo "=========================================="
echo "Vertex AI Deployment"
echo "=========================================="
echo "Project: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "Image: ${IMAGE_URI}"
echo "Job: ${JOB_NAME}"
echo "GPU: ${USE_GPU}"
echo ""

# Verify bucket exists
echo "Checking if GCS bucket exists..."
if gsutil ls gs://${GCS_BUCKET} >/dev/null 2>&1; then
    echo "Bucket exists: gs://${GCS_BUCKET}"
else
    echo "Bucket does not exist: gs://${GCS_BUCKET}"
    echo ""
    echo "Exiting. Please create bucket or set GCS_BUCKET manually."
    exit 1
fi

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

# Step 3: Submit job using Python API
echo ""
echo "Submitting training job to Vertex AI..."


python3 << PYTHON_EOF
from google.cloud import aiplatform

PROJECT_ID = "${PROJECT_ID}"
REGION = "${REGION}"
IMAGE_URI = "${IMAGE_URI}"
JOB_NAME = "${JOB_NAME}"
GCS_BUCKET = "${GCS_BUCKET}"
CHECKPOINT_ID = "${CHECKPOINT_ID}"
MACHINE_TYPE = "${MACHINE_TYPE}"
GPU_TYPE = "${GPU_TYPE}"
USE_GPU = "${USE_GPU}"

# Initialize
aiplatform.init(
    project=PROJECT_ID, 
    location=REGION,
    staging_bucket=f"gs://{GCS_BUCKET}"
)
# Create job
job = aiplatform.CustomContainerTrainingJob(
    display_name=JOB_NAME,
    container_uri=IMAGE_URI,
)

# Submit with proper arg list format
job.run(
    replica_count=1,
    machine_type="${MACHINE_TYPE}",
    accelerator_type="${GPU_TYPE}" if "${USE_GPU}" == "true" else None,
    accelerator_count=1 if "${USE_GPU}" == "true" else 0,
    environment_variables={
        "GCS_BUCKET": GCS_BUCKET,  # Needed for checkpointing
        "CHECKPOINT_ID": CHECKPOINT_ID, 
        "TRAIN_PREFIX": "${TRAIN_PREFIX}",
        "VAL_PREFIX": "${VAL_PREFIX}",
        "TEST_PREFIX": "${TEST_PREFIX}",
    },
    args=[
        "--gcs_bucket", "${GCS_BUCKET}",
        "--train_prefix", "${TRAIN_PREFIX}",
        "--val_prefix", "${VAL_PREFIX}",
        "--test_prefix", "${TEST_PREFIX}",
        "--num_epochs", "${NUM_EPOCHS}",
        "--batch_size", "${BATCH_SIZE}",
        "--hidden_dim", "${HIDDEN_DIM}",
        "--num_encoder_layers", "${NUM_ENCODER_LAYERS}",
        "--num_decoder_layers", "${NUM_DECODER_LAYERS}",
        "--masking_ratio", "${MASKING_RATIO}",
        "--checkpoint_interval", "${CHECKPOINT_INTERVAL}",
        "--cache_dir", "/tmp/pyg_cache",
        "--model_save_path", "/tmp/model",
    ],
    sync=False,
)

print(f"\nJob submitted: {JOB_NAME}")
print(f"   GCS Bucket: {GCS_BUCKET}")
print(f"   Checkpoints will be saved to: gs://{GCS_BUCKET}/checkpoints/trial_{CHECKPOINT_ID}")
PYTHON_EOF

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
echo "Checkpoints will be saved to:"
echo "  gs://${GCS_BUCKET}/checkpoints/"
echo ""