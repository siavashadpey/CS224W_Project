#!/bin/bash
# run_training_docker.sh - Run training in Docker on VM

set -e

cd ~/CS224W_Project

# Configuration
export PROJECT_ID="totemic-phoenix-476721-n5"
export IMAGE_NAME="cs224w-training"
export IMAGE_TAG="latest"
export IMAGE_URI="gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${IMAGE_TAG}"
export SKIP_BUILD="${SKIP_BUILD:-False}"
export GCS_BUCKET="${GCS_BUCKET:-cs224w-2025-mae-gnn-bucket}"
export TRAIN_PREFIX="${TRAIN_PREFIX:-data_w_pos/plgems_train.pt}"
export VAL_PREFIX="${VAL_PREFIX:-data_w_pos/plgems_validation.pt}"
export TEST_PREFIX="${TEST_PREFIX:-data_w_pos/plgems_full_casf2016.pt}"
export NUM_EPOCHS="${NUM_EPOCHS:-10}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export MASKING_RATIO="${MASKING_RATIO:-0.4}"
export POS_SCALE="${POS_SCALE:-0.0}"
export LEARNING_RATE="${LEARNING_RATE:-0.0001}"
export HIDDEN_DIM="${HIDDEN_DIM:-256}"
export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-5}"
export MODEL_SAVE_PATH="${MODEL_SAVE_PATH:-./checkpoints}"
export CACHE_DIR="${CACHE_DIR:-/tmp/pyg_cache}"

echo "=========================================="
echo "Running Training in Docker on VM"
echo "=========================================="
echo "GCS Bucket: ${GCS_BUCKET}"
echo "Epochs: ${NUM_EPOCHS}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Masking Ratio: ${MASKING_RATIO}"
echo "Pos Scale: ${POS_SCALE}"
echo "Hidden Dim: ${HIDDEN_DIM}"
echo "Learning Rate: ${LEARNING_RATE}"
echo ""

# Build image if needed
if ! SKIP_BUILD; then
    echo "Building Docker image..."
    docker build -t ${IMAGE_URI} --no-cache.
    docker push ${IMAGE_URI}
else
    echo "Using existing Docker image"
fi

echo ""
echo "Starting training..."

docker run --rm \
    --gpus all \
    -w /workspace \
    -e GCS_BUCKET="${GCS_BUCKET}" \
    ${IMAGE_URI} \
    --gcs_bucket "${GCS_BUCKET}" \
    --train_prefix "${TRAIN_PREFIX}" \
    --val_prefix "${VAL_PREFIX}" \
    --test_prefix "${TEST_PREFIX}" \
    --num_epochs "${NUM_EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --masking_ratio "${MASKING_RATIO}" \
    --pos_scale "${POS_SCALE}" \
    --learning_rate "${LEARNING_RATE}" \
    --hidden_dim "${HIDDEN_DIM}" \
    --checkpoint_interval "${CHECKPOINT_INTERVAL}" \
    --model_save_path "${MODEL_SAVE_PATH}" \
    --cache_dir "${CACHE_DIR}"

echo ""
echo "Training completed!"
echo "Check checkpoints in: ${MODEL_SAVE_PATH}"
echo "GCS checkpoints at: gs://${GCS_BUCKET}/checkpoints/"
