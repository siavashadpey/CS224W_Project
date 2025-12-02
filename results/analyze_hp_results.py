#!/usr/bin/env python3
"""
Analyze hyperparameter tuning results
"""
import argparse
from google.cloud.aiplatform_v1 import JobServiceClient
import pandas as pd


def analyze_results(job_name, region):
    """Analyze all trials from HP tuning job"""
    
    client = JobServiceClient(
        client_options={"api_endpoint": f"{region}-aiplatform.googleapis.com"}
    )
    
    # Get job details
    job = client.get_hyperparameter_tuning_job(name=job_name)
    
    # Extract trial data
    data = []
    for trial in job.trials:
        row = {
            'trial_id': trial.id,
            'state': trial.state.name,
        }
        
        # Extract parameters
        for param in trial.parameters:
            row[param.parameter_id] = param.value
        
        # Extract final metric
        if trial.final_measurement:
            for metric in trial.final_measurement.metrics:
                if metric.metric_id == 'val_loss':
                    row['val_loss'] = metric.value
        
        data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Filter successful trials
    df_success = df[df['state'] == 'SUCCEEDED'].copy()
    
    if len(df_success) == 0:
        print("No successful trials!")
        return
    
    # Sort by val_loss
    df_success = df_success.sort_values('val_loss')
    
    print("=" * 80)
    print("Hyperparameter Tuning Results")
    print("=" * 80)
    print(f"\nTotal trials: {len(df)}")
    print(f"Successful trials: {len(df_success)}")
    print(f"Failed trials: {len(df[df['state'] == 'FAILED'])}")
    print(f"Stopped trials: {len(df[df['state'] == 'STOPPED'])}")
    
    print("\n" + "=" * 80)
    print("Top 5 Trials")
    print("=" * 80)
    print(df_success.head(5).to_string(index=False))
    
    print("\n" + "=" * 80)
    print("Parameter Correlations with val_loss")
    print("=" * 80)
    
    param_cols = ['learning_rate', 'hidden_dim', 'num_encoder_layers', 
                  'num_decoder_layers', 'masking_ratio', 'batch_size']
    
    for col in param_cols:
        if col in df_success.columns:
            corr = df_success[col].corr(df_success['val_loss'])
            print(f"  {col:25s}: {corr:6.3f}")
    
    print("\n" + "=" * 80)
    print("Best Configuration")
    print("=" * 80)
    best = df_success.iloc[0]
    print(f"\nTrial ID: {best['trial_id']}")
    print(f"Val Loss: {best['val_loss']:.6f}")
    print("\nHyperparameters:")
    for col in param_cols:
        if col in best:
            print(f"  {col:25s}: {best[col]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_name', type=str, required=True)
    parser.add_argument('--region', type=str, default='us-central1')
    
    args = parser.parse_args()
    analyze_results(args.job_name, args.region)


if __name__ == "__main__":
    main()

# Run analysis
# python analyze_hp_results.py \
#    --job_name "projects/<PROJECT_ID>/locations/us-central1/hyperparameterTuningJobs/<JOB_ID>"

# python analyze_hp_results.py --job_name "projects/totemic-phoenix-476721-n5/locations/us-central1/hyperparameterTuningJobs/1580898210921054208"