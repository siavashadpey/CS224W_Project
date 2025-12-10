"""
Submit training job to Vertex AI.
Set environment variables before running:
    export GCP_PROJECT_ID="your-project-id"
    export GCP_REGION="us-central1"  # optional, defaults to us-central1
"""

from google.cloud import aiplatform
import os
import sys

PROJECT_ID = os.getenv('GCP_PROJECT_ID')
REGION = os.getenv('GCP_REGION', 'us-central1')  # Default if not set

if not PROJECT_ID:
    print("Error: GCP_PROJECT_ID environment variable not set")
    print("Run: export GCP_PROJECT_ID='your-project-id'")
    sys.exit(1)

BUCKET_NAME = f"cs224w-2025-mae-gnn-central"
IMAGE_URI = f"gcr.io/{PROJECT_ID}/cs224w-project_mae:latest"

print(f"Project ID: {PROJECT_ID}")
print(f"Region: {REGION}")
print(f"Image: {IMAGE_URI}")
print(f"Output bucket: gs://{BUCKET_NAME}")
print()

# Initialize Vertex AI
aiplatform.init(
    project=PROJECT_ID,
    location=REGION,
    staging_bucket=f"gs://{BUCKET_NAME}"
)

# Create training job
job = aiplatform.CustomContainerTrainingJob(
    display_name="cs224w-egnn-training-head",
    container_uri=IMAGE_URI,
    command=["python", "scripts/train_prediction_head.py"]
)

gcs_autoencoder_checkpoint_path = f"gs://{BUCKET_NAME}/checkpoints/trial_FULL_17/checkpoint_epoch_889.pt"
num_encoder_layers = 4
encoder_hidden_channels = 256

gcs_load_checkpoint_path = f"gs://{BUCKET_NAME}/checkpoints/2601096567427432448/checkpoint_epoch_45.pt"

print("Submitting job...")
job.run(
    replica_count=1,
    machine_type="n1-standard-4",
    accelerator_type="NVIDIA_TESLA_T4",
    accelerator_count=1,
    base_output_dir=f"gs://{BUCKET_NAME}/outputs",
    environment_variables={
        "GCS_BUCKET": BUCKET_NAME,
        "AIP_MODEL_DIR": "/tmp/model"
    },
    args=[
        "--train_data_path", "/workspace/data/plgems_train.pt",
        "--val_data_path", "/workspace/data/plgems_validation.pt",
        "--test_data_path", "/workspace/data/plgems_full_casf2016.pt",
        #"--pretrained_checkpoint", gcs_autoencoder_checkpoint_path,
        "--model_save_path", "/tmp/model",
        "--num_encoder_layers", str(num_encoder_layers),
        "--encoder_hidden_channels", str(encoder_hidden_channels),
        "--num_epochs", "1000",
        "--batch_size", "128",
        "--learning_rate", "1E-5",
        "--learning_rate_gamma", "1.0",
        "--num_workers", "8",
        "--checkpoint_interval", "5",
        "--head_method", "egnn",
        "--head_hidden_channels", "64",
        "--head_num_layers", "2",
        "--pooling_method", "global_mean_pool",
        #"--freeze_encoder",
        "--load_model_path", gcs_load_checkpoint_path
    ],
    sync=True
)
print()
print("✓ Job submitted successfully!")
print(f"Monitor: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={PROJECT_ID}")
print(f"View logs: https://console.cloud.google.com/logs/query?project={PROJECT_ID}")