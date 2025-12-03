#!/usr/bin/env python3
"""
Extract gradient statistics from Vertex AI training job logs using gcloud CLI
"""
import argparse
import re
import subprocess
import json
from datetime import datetime, timedelta
import pandas as pd
from google.cloud.aiplatform_v1 import JobServiceClient


def run_gcloud_logs(project_id, filter_str, limit=10000):
    """Run gcloud logging read command"""
    cmd = [
        'gcloud', 'logging', 'read',
        filter_str,
        f'--project={project_id}',
        f'--limit={limit}',
        '--format=json'
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running gcloud command: {result.stderr}")
        return []
    
    if not result.stdout.strip():
        print("No logs returned")
        return []
    
    try:
        logs = json.loads(result.stdout)
        print(f"Found {len(logs)} log entries")
        return logs
    except json.JSONDecodeError as e:
        print(f"Error parsing gcloud output: {e}")
        return []


def extract_gradient_from_message(message):
    """Extract exploding gradient percentage from various log formats"""
    
    # Pattern 1: [GRADIENT_STATS] epoch=5, exploding_grad_pct=12.34%
    match1 = re.search(r'\[GRADIENT_STATS\].*epoch=(\d+).*exploding_grad_pct=([\d.]+)%', message)
    if match1:
        return int(match1.group(1)), float(match1.group(2))
    
    # Pattern 2: EXPLODING gradients: 5 (12.34%)
    match2 = re.search(r'EXPLODING gradients:\s*(\d+)\s*\(([\d.]+)%\)', message)
    if match2:
        return None, float(match2.group(2))
    
    # Pattern 3: ExplodingGrad=12.34%
    match3 = re.search(r'ExplodingGrad=([\d.]+)%', message)
    if match3:
        return None, float(match3.group(1))
    
    # Pattern 4: exploding_grad_pct=12.34 (without %)
    match4 = re.search(r'exploding_grad_pct=([\d.]+)(?!%)', message)
    if match4:
        return None, float(match4.group(1))
    
    return None, None


def extract_from_hp_tuning_job(job_name, project_id, region):
    """Extract gradient stats from HP tuning job trials"""
    
    print(f"\n{'='*80}")
    print(f"Analyzing HP Tuning Job: {job_name}")
    print(f"{'='*80}\n")
    
    # Get HP tuning job details
    client = JobServiceClient(
        client_options={"api_endpoint": f"{region}-aiplatform.googleapis.com"}
    )
    
    job = client.get_hyperparameter_tuning_job(name=job_name)
    
    # Extract trial info
    trial_data = []
    
    for trial in job.trials:
        trial_info = {
            'trial_id': trial.id,
            'state': trial.state.name,
        }
        
        # Extract hyperparameters
        for param in trial.parameters:
            trial_info[param.parameter_id] = param.value
        
        # Extract val_loss
        if trial.final_measurement:
            for metric in trial.final_measurement.metrics:
                if metric.metric_id == 'val_loss':
                    trial_info['val_loss'] = metric.value
        
        trial_data.append(trial_info)
    
    # Now get gradient stats from logs for each trial
    job_id = job_name.split('/')[-1]
    
    for trial_info in trial_data:
        trial_id = trial_info['trial_id']
        
        # Query logs for this specific trial - BROADER SEARCH
        filter_str = f'''
resource.type="aiplatform.googleapis.com/CustomJob"
labels.aiplatform_hyperparameter_tuning_job_id="{job_id}"
labels.trial_id="{trial_id}"
(jsonPayload.message=~"GRADIENT_STATS" OR 
 jsonPayload.message=~"EXPLODING gradients" OR
 jsonPayload.message=~"ExplodingGrad")
'''
        
        print(f"Fetching logs for trial {trial_id}...")
        
        logs = run_gcloud_logs(project_id, filter_str, limit=5000)
        
        if logs:
            print(f"  Found {len(logs)} log entries for trial {trial_id}")
        
        exploding_pcts = []
        
        for entry in logs:
            message = entry.get('jsonPayload', {}).get('message', '')
            
            if not message:
                message = entry.get('textPayload', '')
            
            epoch, pct = extract_gradient_from_message(message)
            
            if pct is not None:
                exploding_pcts.append(pct)
        
        # Add gradient stats to trial info
        if exploding_pcts:
            trial_info['avg_exploding_grad_pct'] = sum(exploding_pcts) / len(exploding_pcts)
            trial_info['max_exploding_grad_pct'] = max(exploding_pcts)
            trial_info['min_exploding_grad_pct'] = min(exploding_pcts)
            trial_info['num_epochs_analyzed'] = len(exploding_pcts)
            print(f"  Found {len(exploding_pcts)} gradient stats for trial {trial_id}")
        else:
            trial_info['avg_exploding_grad_pct'] = None
            trial_info['max_exploding_grad_pct'] = None
            trial_info['min_exploding_grad_pct'] = None
            trial_info['num_epochs_analyzed'] = 0
            print(f"  No gradient stats found for trial {trial_id}")
    
    return trial_data


def extract_from_custom_job(job_name, project_id, region):
    """Extract gradient stats from a single custom training job"""
    
    print(f"\n{'='*80}")
    print(f"Analyzing Custom Training Job: {job_name}")
    print(f"{'='*80}\n")
    
    job_id = job_name.split('/')[-1]
    
    # Query logs for this job - BROADER SEARCH
    filter_str = f'''
resource.type="aiplatform.googleapis.com/CustomJob"
resource.labels.job_id="{job_id}"
(jsonPayload.message=~"GRADIENT_STATS" OR 
 jsonPayload.message=~"EXPLODING gradients" OR
 jsonPayload.message=~"ExplodingGrad")
'''
    
    print(f"Fetching logs for job {job_id}...")
    
    logs = run_gcloud_logs(project_id, filter_str, limit=5000)
    
    epoch_stats = []
    
    for entry in logs:
        message = entry.get('jsonPayload', {}).get('message', '')
        
        if not message:
            message = entry.get('textPayload', '')
        
        timestamp = entry.get('timestamp', '')
        
        epoch, pct = extract_gradient_from_message(message)
        
        if pct is not None:
            epoch_stats.append({
                'epoch': epoch if epoch is not None else len(epoch_stats),
                'exploding_grad_pct': pct,
                'timestamp': timestamp
            })
    
    if not epoch_stats:
        print("  No gradient stats found in logs")
        print("  Make sure your training script logs with [GRADIENT_STATS] tag")
        return None
    
    # Sort by epoch
    epoch_stats.sort(key=lambda x: x['epoch'])
    
    # Compute summary
    exploding_pcts = [s['exploding_grad_pct'] for s in epoch_stats]
    summary = {
        'job_id': job_id,
        'num_epochs': len(epoch_stats),
        'avg_exploding_grad_pct': sum(exploding_pcts) / len(exploding_pcts),
        'max_exploding_grad_pct': max(exploding_pcts),
        'min_exploding_grad_pct': min(exploding_pcts),
        'epoch_details': epoch_stats
    }
    
    return summary


def extract_from_recent_jobs(project_id, region, hours=24, job_type='all'):
    """Extract gradient stats from all recent jobs"""
    
    print(f"\n{'='*80}")
    print(f"Analyzing Recent Jobs (last {hours} hours)")
    print(f"{'='*80}\n")
    
    # Calculate time filter
    start_time = datetime.utcnow() - timedelta(hours=hours)
    
    # Query for all jobs with gradient stats - BROADER SEARCH
    if job_type == 'hp_tuning':
        resource_filter = 'resource.type="aiplatform.googleapis.com/HyperparameterTuningJob"'
    elif job_type == 'custom':
        resource_filter = 'resource.type="aiplatform.googleapis.com/CustomJob"'
    else:
        resource_filter = '(resource.type="aiplatform.googleapis.com/CustomJob" OR resource.type="aiplatform.googleapis.com/HyperparameterTuningJob")'
    
    filter_str = f'''
{resource_filter}
timestamp>="{start_time.isoformat()}Z"
(jsonPayload.message=~"GRADIENT_STATS" OR 
 jsonPayload.message=~"EXPLODING gradients" OR
 jsonPayload.message=~"ExplodingGrad")
'''
    
    print(f"Searching logs...")
    
    logs = run_gcloud_logs(project_id, filter_str, limit=10000)
    
    # Group by job_id
    job_stats = {}
    
    for entry in logs:
        message = entry.get('jsonPayload', {}).get('message', '')
        
        if not message:
            message = entry.get('textPayload', '')
        
        # Get job ID from labels
        labels = entry.get('labels', {})
        resource_labels = entry.get('resource', {}).get('labels', {})
        
        job_id = labels.get('job_id') or resource_labels.get('job_id', 'unknown')
        trial_id = labels.get('trial_id', None)
        
        # Create unique key
        key = f"{job_id}"
        if trial_id:
            key = f"{job_id}/trial_{trial_id}"
        
        if key not in job_stats:
            job_stats[key] = {
                'job_id': job_id,
                'trial_id': trial_id,
                'exploding_pcts': [],
                'timestamp': entry.get('timestamp', '')
            }
        
        # Extract gradient percentage
        epoch, pct = extract_gradient_from_message(message)
        
        if pct is not None:
            job_stats[key]['exploding_pcts'].append(pct)
    
    # Summarize
    summaries = []
    for key, stats in job_stats.items():
        if stats['exploding_pcts']:
            summaries.append({
                'job_id': stats['job_id'],
                'trial_id': stats['trial_id'] or 'N/A',
                'num_epochs': len(stats['exploding_pcts']),
                'avg_exploding_grad_pct': sum(stats['exploding_pcts']) / len(stats['exploding_pcts']),
                'max_exploding_grad_pct': max(stats['exploding_pcts']),
                'min_exploding_grad_pct': min(stats['exploding_pcts']),
            })
    
    if not summaries:
        print(" No jobs with gradient stats found")
        print(" Make sure your training script logs with [GRADIENT_STATS] tag")
        return []
    
    return summaries


def main():
    parser = argparse.ArgumentParser(
        description='Extract gradient statistics from Vertex AI training jobs'
    )
    
    parser.add_argument('--project', type=str, required=True,
                       help='GCP project ID')
    parser.add_argument('--region', type=str, default='us-central1',
                       help='GCP region')
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--hp_tuning_job', type=str,
                           help='Full resource name of HP tuning job')
    mode_group.add_argument('--custom_job', type=str,
                           help='Full resource name of custom training job')
    mode_group.add_argument('--recent_jobs', action='store_true',
                           help='Analyze all recent jobs')
    
    parser.add_argument('--hours', type=int, default=24,
                       help='Hours to look back for recent jobs (default: 24)')
    parser.add_argument('--job_type', type=str, default='all',
                       choices=['all', 'hp_tuning', 'custom'],
                       help='Type of jobs to analyze for --recent_jobs')
    parser.add_argument('--output', type=str, default='gradient_stats.csv',
                       help='Output CSV file')
    
    args = parser.parse_args()
    
    # Execute based on mode
    results = None
    
    if args.hp_tuning_job:
        trial_data = extract_from_hp_tuning_job(
            args.hp_tuning_job, 
            args.project, 
            args.region
        )
        df = pd.DataFrame(trial_data)
        
        print(f"\n{'='*80}")
        print("HP TUNING JOB RESULTS")
        print(f"{'='*80}\n")
        print(df.to_string(index=False))
        
        # Correlation analysis
        if 'avg_exploding_grad_pct' in df.columns and df['avg_exploding_grad_pct'].notna().any():
            print(f"\n{'='*80}")
            print("CORRELATION WITH EXPLODING GRADIENTS")
            print(f"{'='*80}\n")
            
            param_cols = ['learning_rate', 'hidden_dim', 'num_encoder_layers', 
                         'num_decoder_layers', 'masking_ratio', 'batch_size']
            
            for col in param_cols:
                if col in df.columns:
                    valid_df = df[[col, 'avg_exploding_grad_pct']].dropna()
                    if len(valid_df) > 1:
                        corr = valid_df[col].corr(valid_df['avg_exploding_grad_pct'])
                        print(f"  {col:25s}: {corr:+6.3f}")
        
        results = df
    
    elif args.custom_job:
        summary = extract_from_custom_job(
            args.custom_job,
            args.project,
            args.region
        )
        
        if summary:
            print(f"\n{'='*80}")
            print("CUSTOM JOB RESULTS")
            print(f"{'='*80}\n")
            print(f"Job ID: {summary['job_id']}")
            print(f"Epochs analyzed: {summary['num_epochs']}")
            print(f"Avg exploding gradient %: {summary['avg_exploding_grad_pct']:.2f}%")
            print(f"Max exploding gradient %: {summary['max_exploding_grad_pct']:.2f}%")
            print(f"Min exploding gradient %: {summary['min_exploding_grad_pct']:.2f}%")
            
            # Epoch-by-epoch details
            print(f"\nEpoch-by-Epoch Details:")
            print("-" * 80)
            epoch_df = pd.DataFrame(summary['epoch_details'])
            print(epoch_df.to_string(index=False))
            
            results = epoch_df
    
    elif args.recent_jobs:
        summaries = extract_from_recent_jobs(
            args.project,
            args.region,
            args.hours,
            args.job_type
        )
        
        if summaries:
            df = pd.DataFrame(summaries)
            df = df.sort_values('avg_exploding_grad_pct', ascending=False)
            
            print(f"\n{'='*80}")
            print(f"RECENT JOBS (last {args.hours} hours)")
            print(f"{'='*80}\n")
            print(df.to_string(index=False))
            
            results = df
    
    # Save to CSV
    if results is not None and not results.empty:
        results.to_csv(args.output, index=False)
        print(f"\nResults saved to: {args.output}")
    else:
        print(f"\nNo results to save")


if __name__ == "__main__":
    main()