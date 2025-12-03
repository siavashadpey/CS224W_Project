#!/bin/bash
# run_training_docker.sh - Run training in Docker on VM

set -e

cd ~/CS224W_Project

# Configuration
export GCS_BUCKET="${GCS_BUCKET:-cs224w-2025-mae-gnn-bucket}"
export TRAIN_PREFIX="${TRAIN_PREFIX:-data_w_pos/plgems_train.pt}"
export VAL_PREFIX="${VAL_PREFIX:-data_w_pos/plgems_validation.pt}"
export TEST_PREFIX="${TEST_PREFIX:-data_w_pos/plgems_full_casf2016.pt}"
export NUM_EPOCHS="${NUM_EPOCHS:-10}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export MASKING_RATIO="${MASKING_RATIO:-0.4}"
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
echo "Hidden Dim: ${HIDDEN_DIM}"
echo "Learning Rate: ${LEARNING_RATE}"
echo ""

# Build image if needed
if ! docker images | grep -q cs224w-training; then
    echo "Building Docker image..."
    docker build -t cs224w-training:latest .
else
    echo "Using existing Docker image"
fi

echo ""
echo "Starting training..."

docker run --rm \
    --gpus all \
    -v $(pwd):/workspace \
    -w /workspace \
    -e GCS_BUCKET="${GCS_BUCKET}" \
    cs224w-training:latest \
    --gcs_bucket "${GCS_BUCKET}" \
    --train_prefix "${TRAIN_PREFIX}" \
    --val_prefix "${VAL_PREFIX}" \
    --test_prefix "${TEST_PREFIX}" \
    --num_epochs "${NUM_EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --masking_ratio "${MASKING_RATIO}" \
    --learning_rate "${LEARNING_RATE}" \
    --hidden_dim "${HIDDEN_DIM}" \
    --checkpoint_interval "${CHECKPOINT_INTERVAL}" \
    --model_save_path "${MODEL_SAVE_PATH}" \
    --cache_dir "${CACHE_DIR}"

echo ""
echo "Training completed!"
echo "Check checkpoints in: ${MODEL_SAVE_PATH}"
echo "GCS checkpoints at: gs://${GCS_BUCKET}/checkpoints/"
