#!/bin/bash

# --- Configuration for Local Test ---

# Mock parameters (these paths are used by the Python script to structure the cache)
MOCK_BUCKET="mock-gcs-bucket-name"
MOCK_CACHE_DIR="./local_test_cache"
MOCK_MODEL_SAVE_PATH="./local_model_checkpoints"

# The prefixes need to match the directories *inside* the mock data creator script.
# In a real run, these would be 'data/train', 'data/val', etc., but here we point
# to the mocked directory structure.
MOCK_TRAIN_PREFIX="mock_data/train"
MOCK_VAL_PREFIX="mock_data/val"
MOCK_TEST_PREFIX="mock_data/test"

# --- Setup ---
echo "--- Step 1: Create mock data files ---"
python create_mock_data.py

# To fully mock the behavior of gcs_dataset_loader.py, 
# we need to simulate the 'download' step by copying the mock files
# to the cache location where the loader expects to find them.

echo "--- Step 2: Simulate GCS download into local cache ---"
mkdir -p $MOCK_CACHE_DIR/train
mkdir -p $MOCK_CACHE_DIR/val
mkdir -p $MOCK_CACHE_DIR/test

# Copy mock data files to the cache directory as if they were downloaded
cp ./mock_gcs_root/${MOCK_TRAIN_PREFIX}/*.pt $MOCK_CACHE_DIR/train/
cp ./mock_gcs_root/${MOCK_VAL_PREFIX}/*.pt $MOCK_CACHE_DIR/val/
cp ./mock_gcs_root/${MOCK_TEST_PREFIX}/*.pt $MOCK_CACHE_DIR/test/

echo "Simulated GCS download to: $MOCK_CACHE_DIR"

# --- Execution ---
echo "--- Step 3: Run the training script with mock arguments ---"

# Set the environment variable to tell gcs_dataset_loader.py to skip the GCS API call
export SKIP_GCS_DOWNLOAD="True"

# Note: We must set --force_download=True or --force_download=False based on how
# the local mock is structured. Since we copied the files, we can use False.
# If we don't actually want to run the model training (which will take time),
# we can change the number of epochs to a small number like 2.

python ../scripts/train_masked_autoencoder.py \
    --gcs_bucket "$MOCK_BUCKET" \
    --train_prefix "$MOCK_TRAIN_PREFIX" \
    --val_prefix "$MOCK_VAL_PREFIX" \
    --test_prefix "$MOCK_TEST_PREFIX" \
    --cache_dir "$MOCK_CACHE_DIR" \
    --model_save_path "$MOCK_MODEL_SAVE_PATH" \
    --num_epochs 2 \
    --checkpoint_interval 1

echo "--- Local test run complete. Check logs and $MOCK_MODEL_SAVE_PATH for results. ---"

# --- Cleanup ---
# Uncomment the following lines to clean up the generated files and directories
# echo "--- Step 4: Cleaning up mock files ---"
# rm -rf ./mock_gcs_root
# rm -rf $MOCK_CACHE_DIR
# rm -rf $MOCK_MODEL_SAVE_PATH