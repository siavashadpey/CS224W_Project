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
    display_name="cs224w-egnn-training-mae",
    container_uri=IMAGE_URI,
)

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
        "--model_save_path", "/tmp/model",
        "--num_epochs", "1000",
        "--batch_size", "16",
        "--learning_rate", "1E-4",
        "--learning_rate_gamma", "0.999",
        "--num_workers", "8",
        "--num_encoder_layers", "4",
        "--num_decoder_layers", "4",
        "--hidden_dim", "256",
        "--masking_ratio", "0.3",
        "--checkpoint_interval", "5"
    ],
    sync=True  # Don't wait for completion
)

print()
print("✓ Job submitted successfully!")
print(f"Monitor: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={PROJECT_ID}")
print(f"View logs: https://console.cloud.google.com/logs/query?project={PROJECT_ID}")