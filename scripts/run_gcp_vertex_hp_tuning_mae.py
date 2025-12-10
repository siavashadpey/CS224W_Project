"""
Submit training job to Vertex AI.
Set environment variables before running:
    export GCP_PROJECT_ID="your-project-id"
    export GCP_REGION="us-central1"  # optional, defaults to us-central1
"""

from google.cloud import aiplatform, storage
from google.cloud.aiplatform import hyperparameter_tuning as hpt
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

metric_spec = {
        'val_loss': 'minimize'
    }

const_arguments = [
    "--train_data_path", "/workspace/data/plgems_train.pt",
    "--val_data_path", "/workspace/data/plgems_validation.pt",
    "--test_data_path", "/workspace/data/plgems_full_casf2016.pt",
    "--num_workers", "4",
    "--learning_rate", "1E-4",
    "--learning_rate_gamma", "1.0", # no decay
    "--checkpoint_interval", "5",
    "--model_save_path", "/tmp/model",
    "--num_epochs", "100",
    "--batch_size", "128",
    ]

parameter_spec = {
        'masking_ratio': hpt.DoubleParameterSpec(
            min=0.4,
            max=0.6,
            scale='linear'
        ),
        'hidden_dim': hpt.DiscreteParameterSpec(
            values=[64, 128],
            scale='linear'
        ),
        'num_encoder_layers': hpt.DiscreteParameterSpec(
            values=[3, 4, 5],
            scale='linear'
        ),
        'num_decoder_layers': hpt.DiscreteParameterSpec(
            values=[3, 4, 5],
            scale='linear'
        ),
    }

trial_job_spec = aiplatform.CustomJob(
    display_name="cs224w-egnn-mae-trial",
    base_output_dir=f"gs://{BUCKET_NAME}/outputs",
    worker_pool_specs=[
        {
            "container_spec": {
                "image_uri": IMAGE_URI,
                "args": const_arguments,
                "env": [
                    {
                        "name": "GCS_BUCKET",
                        "value": BUCKET_NAME
                    },
                ],
            },
            "replica_count": 1,
            "machine_spec": {
                "machine_type": "n1-standard-4",
                "accelerator_type": "NVIDIA_TESLA_T4",
                "accelerator_count": 1,
            },
        }
    ],
)
hp_tuning_job = aiplatform.HyperparameterTuningJob(
    display_name="cs224w-egnn-mae-hp-tuning",
    custom_job=trial_job_spec,
    metric_spec=metric_spec,
    parameter_spec=parameter_spec,
    max_trial_count=50,
    parallel_trial_count=4,
    max_failed_trial_count=5,
    search_algorithm=None, # bayesian 
    measurement_selection='best'
    )

print("Submitting job...")
hp_tuning_job.run(
    sync=True  # Don't wait for completion
)


print()
print("✓ Job submitted successfully!")
print(f"View in console: {hp_tuning_job.get_console_url()}")