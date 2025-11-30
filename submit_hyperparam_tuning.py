#!/usr/bin/env python3
"""
Submit hyperparameter tuning job to Vertex AI
Usage: python submit_hyperparam_tuning.py
"""

import argparse
import subprocess
from datetime import datetime
from google.cloud import aiplatform
from google.cloud.aiplatform import hyperparameter_tuning as hpt


def get_gcloud_config(key):
    """Get value from gcloud config"""
    result = subprocess.run(
        ['gcloud', 'config', 'get-value', key],
        capture_output=True,
        text=True
    )
    return result.stdout.strip()


def build_and_push_image(image_uri):
    """Build and push Docker image"""
    print("Building Docker image...")
    subprocess.run(['docker', 'build', '-t', image_uri, '.'], check=True)
    
    print("Pushing to Container Registry...")
    subprocess.run(['docker', 'push', image_uri], check=True)


def submit_hp_job(args):
    """Submit hyperparameter tuning job"""
    
    # Get configuration
    project_id = args.project or get_gcloud_config('project')
    region = args.region
    gcs_bucket = args.gcs_bucket
    image_uri = f"gcr.io/{project_id}/cs224w-training:latest"
    job_name = f"cs224w-hptuning-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    print("=" * 60)
    print("Hyperparameter Tuning Job Submission")
    print("=" * 60)
    print(f"Project: {project_id}")
    print(f"Region: {region}")
    print(f"GCS Bucket: {gcs_bucket}")
    print(f"Image: {image_uri}")
    print(f"Job: {job_name}")
    print()
    
    # Build and push image
    if not args.skip_build:
        build_and_push_image(image_uri)
    else:
        print("Skipping Docker build")
    
    # Initialize Vertex AI
    aiplatform.init(
        project=project_id,
        location=region,
        staging_bucket=f"gs://{gcs_bucket}"
    )
    
    # ============================================
    # HYPERPARAMETER SEARCH SPACE
    # ============================================
    parameter_spec = {
        'learning_rate': hpt.DoubleParameterSpec(
            min=0.00001,
            max=0.001,
            scale='log'
        ),
        'hidden_dim': hpt.DiscreteParameterSpec(
            values=[64, 128, 256],
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
        'masking_ratio': hpt.DoubleParameterSpec(
            min=0.3,
            max=0.6,
            scale='linear'
        ),
        'batch_size': hpt.DiscreteParameterSpec(
            values=[16, 32, 64],
            scale='linear'
        ),
    }
    
    # ============================================
    # OPTIMIZATION METRIC
    # ============================================
    metric_spec = {
        'val_loss': 'minimize'
    }
    
    # ============================================
    # FIXED TRAINING ARGUMENTS
    # ============================================
    training_args = [
        "--gcs_bucket", gcs_bucket,
        "--train_prefix", args.train_prefix,
        "--val_prefix", args.val_prefix,
        "--test_prefix", args.test_prefix,
        "--num_epochs", str(args.num_epochs),
        "--checkpoint_interval", str(args.checkpoint_interval),
        "--cache_dir", "/tmp/pyg_cache",
        "--model_save_path", "/tmp/model",
    ]
    
    # ============================================
    # ENVIRONMENT VARIABLES
    # ============================================
    environment_variables = {
        "GCS_BUCKET": gcs_bucket,
    }
    
    print(f"Hyperparameters to tune: {list(parameter_spec.keys())}")
    print(f"Optimization metric: {metric_spec}")
    print()

    # ============================================
    # CREATE WORKER POOL SPECS
    # ============================================
    # Build worker pool specs that will be used for each trial
    worker_pool_specs = [
        {
            "machine_spec": {
                "machine_type": args.machine_type,
                "accelerator_type": args.accelerator_type,
                "accelerator_count": 1,
            },
            "replica_count": 1,
            "container_spec": {
                "image_uri": image_uri,
                "args": training_args,
                "env": [{"name": k, "value": v} for k, v in environment_variables.items()],
            }
        }
    ]

    # ============================================
    # CREATE CUSTOM CONTAINER JOB
    # ============================================
    custom_job = aiplatform.jobs.CustomJob(
        display_name=f"{job_name}-base",
        worker_pool_specs=worker_pool_specs,
    )
    
    # ============================================
    # CREATE HYPERPARAMETER TUNING JOB
    # ============================================
    
    hp_job = aiplatform.HyperparameterTuningJob(
        display_name=job_name,
        custom_job=custom_job,
        metric_spec=metric_spec,
        parameter_spec=parameter_spec,
        max_trial_count=args.max_trials,
        parallel_trial_count=args.parallel_trials,
        max_failed_trial_count=args.max_failed_trials,
        search_algorithm=None, # 'bayesian',
        measurement_selection='best',
    #    restart_job_on_worker_restart=False,
    #    base_output_dir=f"gs://{gcs_bucket}/hptuning/{job_name}",
    )
    
    print(f"Submitting hyperparameter tuning job...")
    
    # ============================================
    # RUN THE JOB
    # ============================================
    hp_job.run()
    
    print("\n" + "=" * 60)
    print("Hyperparameter Tuning Job Submitted!")
    print("=" * 60)
    print(f"Job Name: {job_name}")
    print(f"Resource: {hp_job.resource_name}")
    print()
    print("Configuration:")
    print(f"  Max trials: {args.max_trials}")
    print(f"  Parallel trials: {args.parallel_trials}")
    print(f"  Parameters: {len(parameter_spec)}")
    print(f"  Epochs per trial: {args.num_epochs}")
    print(f"  Search algorithm: Bayesian optimization")
    print()
    print("Monitor:")
    print(f"  https://console.cloud.google.com/vertex-ai/training/training-pipelines?project={project_id}")
    print()
    print("Checkpoints:")
    print(f"  gs://{gcs_bucket}/checkpoints/trial_*/")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Submit hyperparameter tuning job to Vertex AI'
    )
    
    # GCP settings
    parser.add_argument('--project', type=str)
    parser.add_argument('--region~````1', type=str, default='us-central1')
    parser.add_argument('--gcs_bucket', type=str, default='cs224w-2025-mae-gnn-central')
    
    # Machine settings
    parser.add_argument('--machine_type', type=str, default='n1-standard-8')
    parser.add_argument('--accelerator_type', type=str, default='NVIDIA_TESLA_T4')
    
    # HP tuning settings
    parser.add_argument('--max_trials', type=int, default=20)
    parser.add_argument('--parallel_trials', type=int, default=4)
    parser.add_argument('--max_failed_trials', type=int, default=5)
    
    # Data paths
    parser.add_argument('--train_prefix', type=str, default='data_w_pos/plgems_train.pt')
    parser.add_argument('--val_prefix', type=str, default='data_w_pos/plgems_validation.pt')
    parser.add_argument('--test_prefix', type=str, default='data_w_pos/plgems_full_casf2016.pt')
    
    # Training settings
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--checkpoint_interval', type=int, default=10)
    
    # Build settings
    parser.add_argument('--skip_build', action='store_true')
    
    args = parser.parse_args()
    
    try:
        submit_hp_job(args)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()