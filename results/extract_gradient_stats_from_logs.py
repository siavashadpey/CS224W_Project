#!/usr/bin/env python3
"""
Extract gradient statistics from Vertex AI training job logs
Supports both HP tuning jobs and regular training jobs
"""
import argparse
import re
from google.cloud import logging as cloud_logging
from google.cloud.aiplatform_v1 import JobServiceClient
import pandas as pd
from datetime import datetime, timedelta
import json


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
    log_client = cloud_logging.Client(project=project_id)
    
    job_id = job_name.split('/')[-1]
    
    for trial_info in trial_data:
        trial_id = trial_info['trial_id']
        
        # Query logs for this specific trial
        filter_str = f'''
        resource.type="aiplatform.googleapis.com/CustomJob"
        labels.aiplatform_hyperparameter_tuning_job_id="{job_id}"
        labels.trial_id="{trial_id}"
        jsonPayload.message=~"GRADIENT_STATS|Epoch.*summary"
        '''
        
        print(f"Fetching logs for trial {trial_id}...")
        
        exploding_counts = []
        epoch_summaries = []
        
        for entry in log_client.list_entries(filter_=filter_str, page_size=1000):
            message = entry.payload.get('message', '')
            
            # Pattern 1: [GRADIENT_STATS] epoch=5, exploding_grad_pct=12.34%
            match1 = re.search(r'epoch=(\d+), exploding_grad_pct=([\d.]+)%', message)
            if match1:
                epoch = int(match1.group(1))
                exploding_pct = float(match1.group(2))
                exploding_counts.append(exploding_pct)
            
            # Pattern 2: Epoch summary: ... EXPLODING gradients: 5 (12.34%)
            match2 = re.search(r'EXPLODING gradients:\s*(\d+)\s*\(([\d.]+)%\)', message)
            if match2:
                count = int(match2.group(1))
                pct = float(match2.group(2))
                epoch_summaries.append({'count': count, 'pct': pct})
        
        # Add gradient stats to trial info
        if exploding_counts:
            trial_info['avg_exploding_grad_pct'] = sum(exploding_counts) / len(exploding_counts)
            trial_info['max_exploding_grad_pct'] = max(exploding_counts)
            trial_info['min_exploding_grad_pct'] = min(exploding_counts)
            trial_info['num_epochs_analyzed'] = len(exploding_counts)
        elif epoch_summaries:
            pcts = [s['pct'] for s in epoch_summaries]
            trial_info['avg_exploding_grad_pct'] = sum(pcts) / len(pcts)
            trial_info['max_exploding_grad_pct'] = max(pcts)
            trial_info['min_exploding_grad_pct'] = min(pcts)
            trial_info['num_epochs_analyzed'] = len(pcts)
        else:
            trial_info['avg_exploding_grad_pct'] = None
            trial_info['max_exploding_grad_pct'] = None
            trial_info['min_exploding_grad_pct'] = None
            trial_info['num_epochs_analyzed'] = 0
            print(f"No gradient stats found for trial {trial_id}")
    
    return trial_data


def extract_from_custom_job(job_name, project_id, region):
    """Extract gradient stats from a single custom training job"""
    
    print(f"\n{'='*80}")
    print(f"Analyzing Custom Training Job: {job_name}")
    print(f"{'='*80}\n")
    
    log_client = cloud_logging.Client(project=project_id)
    
    job_id = job_name.split('/')[-1]
    
    # Query logs for this job
    filter_str = f'''
    resource.type="aiplatform.googleapis.com/CustomJob"
    resource.labels.job_id="{job_id}"
    jsonPayload.message=~"GRADIENT_STATS|Epoch.*summary"
    '''
    
    print(f"Fetching logs for job {job_id}...")
    
    epoch_stats = []
    
    for entry in log_client.list_entries(filter_=filter_str, page_size=5000):
        message = entry.payload.get('message', '')
        timestamp = entry.timestamp
        
        # Pattern 1: [GRADIENT_STATS] epoch=5, exploding_grad_pct=12.34%
        match1 = re.search(r'epoch=(\d+), exploding_grad_pct=([\d.]+)%', message)
        if match1:
            epoch = int(match1.group(1))
            exploding_pct = float(match1.group(2))
            epoch_stats.append({
                'epoch': epoch,
                'exploding_grad_pct': exploding_pct,
                'timestamp': timestamp
            })
        
        # Pattern 2: Epoch X summary: valid_batches=Y, ... EXPLODING gradients: Z (W%)
        match2 = re.search(r'Epoch (\d+) summary:.*EXPLODING gradients:\s*(\d+)\s*\(([\d.]+)%\)', message)
        if match2:
            epoch = int(match2.group(1))
            count = int(match2.group(2))
            pct = float(match2.group(3))
            
            # Check if we already have this epoch
            if not any(s['epoch'] == epoch for s in epoch_stats):
                epoch_stats.append({
                    'epoch': epoch,
                    'exploding_grad_pct': pct,
                    'exploding_count': count,
                    'timestamp': timestamp
                })
    
    if not epoch_stats:
        print("No gradient stats found in logs")
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
    
    log_client = cloud_logging.Client(project=project_id)
    
    # Calculate time filter
    start_time = datetime.utcnow() - timedelta(hours=hours)
    
    # Query for all jobs with gradient stats
    if job_type == 'hp_tuning':
        resource_filter = 'resource.type="aiplatform.googleapis.com/HyperparameterTuningJob"'
    elif job_type == 'custom':
        resource_filter = 'resource.type="aiplatform.googleapis.com/CustomJob"'
    else:
        resource_filter = '(resource.type="aiplatform.googleapis.com/CustomJob" OR resource.type="aiplatform.googleapis.com/HyperparameterTuningJob")'
    
    filter_str = f'''
    {resource_filter}
    timestamp>="{start_time.isoformat()}Z"
    jsonPayload.message=~"GRADIENT_STATS|Epoch.*summary"
    '''
    
    print(f"Searching logs with filter:")
    print(f"  {filter_str}\n")
    
    # Group by job_id
    job_stats = {}
    
    for entry in log_client.list_entries(filter_=filter_str, page_size=10000):
        message = entry.payload.get('message', '')
        
        # Get job ID from labels
        job_id = entry.labels.get('job_id') or entry.resource.labels.get('job_id', 'unknown')
        trial_id = entry.labels.get('trial_id', None)
        
        # Create unique key
        key = f"{job_id}"
        if trial_id:
            key = f"{job_id}/trial_{trial_id}"
        
        if key not in job_stats:
            job_stats[key] = {
                'job_id': job_id,
                'trial_id': trial_id,
                'exploding_pcts': [],
                'first_seen': entry.timestamp,
                'last_seen': entry.timestamp
            }
        
        # Update timestamps
        job_stats[key]['last_seen'] = max(job_stats[key]['last_seen'], entry.timestamp)
        job_stats[key]['first_seen'] = min(job_stats[key]['first_seen'], entry.timestamp)
        
        # Extract gradient percentage
        match1 = re.search(r'exploding_grad_pct=([\d.]+)%', message)
        match2 = re.search(r'EXPLODING gradients:\s*\d+\s*\(([\d.]+)%\)', message)
        
        if match1:
            job_stats[key]['exploding_pcts'].append(float(match1.group(1)))
        elif match2:
            job_stats[key]['exploding_pcts'].append(float(match2.group(1)))
    
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
                'duration_minutes': (stats['last_seen'] - stats['first_seen']).total_seconds() / 60
            })
    
    if not summaries:
        print("No jobs with gradient stats found")
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


if __name__ == "__main__":
    main()