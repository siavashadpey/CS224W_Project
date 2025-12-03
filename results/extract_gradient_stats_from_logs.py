#!/usr/bin/env python3
"""
Extract gradient statistics from Vertex AI training job logs
Handles multi-line epoch summaries where epoch info is separate from gradient info
"""
import argparse
import re
import subprocess
import json
from datetime import datetime, timedelta
import pandas as pd
import sys

def run_gcloud_logs(project_id, filter_str, limit=10000):
    """Run gcloud logging read command"""
    cmd = [
        'gcloud', 'logging', 'read',
        filter_str,
        f'--project={project_id}',
        f'--limit={limit}',
        '--format=json',
    ]
    
    print(f"Running gcloud query...")
    print(f"Filter: {filter_str}\n")
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
        return []
    
    if not result.stdout.strip():
        print("No logs returned")
        return []
    
    try:
        logs = json.loads(result.stdout)
        print(f"Retrieved {len(logs)} log entries")
        return logs
    except json.JSONDecodeError as e:
        print(f"ERROR parsing JSON: {e}")
        return []


def extract_gradient_stats_from_logs(logs):
    """
    Extract gradient statistics from log entries
    Handles case where epoch summary and EXPLODING gradients are in separate log lines
    """
    print(f"Processing {len(logs)} logs...")
    
    # Sort logs by timestamp to maintain order
    logs_sorted = sorted(logs, key=lambda x: x.get('timestamp', ''))
    
    results = []
    current_epoch = None
    
    for i, entry in enumerate(logs_sorted):
        # Check all possible fields for the message
        message = None
        
        if 'jsonPayload' in entry and 'message' in entry['jsonPayload']:
            message = entry['jsonPayload']['message']
        elif 'textPayload' in entry:
            message = entry['textPayload']
        elif 'jsonPayload' in entry and 'summary' in entry['jsonPayload']:
            message = entry['jsonPayload']['summary']
        
        if not message:
            continue
        
        timestamp = entry.get('timestamp', '')
        
        # Pattern 1: "Epoch 19 summary:"
        epoch_match = re.search(r'Epoch (\d+)\s+summary:', message)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            continue
        
        # Pattern 2: "Reported val_loss=10.1070, exploding_grad_pct=5.95% at epoch 19"
        report_match = re.search(r'exploding_grad_pct=([\d.]+)%\s+at epoch (\d+)', message)
        if report_match:
            pct = float(report_match.group(1))
            epoch = int(report_match.group(2))
            results.append({
                'epoch': epoch,
                'exploding_grad_pct': pct,
                'timestamp': timestamp,
                'source': 'reported'
            })
            continue
        
        # Pattern 3: "EXPLODING gradients: 53 (6.44%)"
        exploding_match = re.search(r'EXPLODING gradients:\s*(\d+)\s*\(([\d.]+)%\)', message)
        if exploding_match:
            count = int(exploding_match.group(1))
            pct = float(exploding_match.group(2))
            
            # Check if this line also has epoch info
            epoch_in_line = re.search(r'Epoch (\d+)', message)
            
            if epoch_in_line:
                epoch = int(epoch_in_line.group(1))
            elif current_epoch is not None:
                # Use the most recent epoch summary we saw
                epoch = current_epoch
            else:
                # Look backwards in logs for recent epoch summary (within 10 entries)
                epoch = None
                for j in range(max(0, i-10), i):
                    prev_message = None
                    if 'jsonPayload' in logs_sorted[j] and 'message' in logs_sorted[j]['jsonPayload']:
                        prev_message = logs_sorted[j]['jsonPayload']['message']
                    elif 'textPayload' in logs_sorted[j]:
                        prev_message = logs_sorted[j]['textPayload']
                    
                    if prev_message:
                        prev_epoch = re.search(r'Epoch (\d+)\s+summary:', prev_message)
                        if prev_epoch:
                            epoch = int(prev_epoch.group(1))
                            break
            
            if epoch is not None:
                results.append({
                    'epoch': epoch,
                    'exploding_grad_pct': pct,
                    'exploding_count': count,
                    'timestamp': timestamp,
                    'source': 'summary'
                })
                print(f"  Found: Epoch {epoch}, EXPLODING={count} ({pct}%)")
    
    print(f"Found {len(results)} epoch records with gradient stats")
    return results


def get_all_recent_job_ids(project_id, hours=72):
    """Get all job IDs from recent logs that contain EXPLODING gradients"""
    print(f"\n{'='*80}")
    print("Finding Recent Job IDs with Gradient Stats")
    print(f"{'='*80}\n")
    
    start_time = datetime.utcnow() - timedelta(hours=hours)
    timestamp_str = start_time.isoformat() + 'Z'
    
    # Filter on server side for logs containing "EXPLODING gradients"
    filter_str = f'timestamp>="{timestamp_str}" AND "EXPLODING gradients"'
    
    print(f"Looking for logs since: {timestamp_str}")
    print(f"Filtering for: 'EXPLODING gradients'")
    
    logs = run_gcloud_logs(project_id, filter_str, limit=10000)
    
    if not logs:
        print("No logs found with 'EXPLODING gradients'")
        return []
    
    # Extract unique job IDs
    job_ids = set()
    
    for entry in logs:
        resource_labels = entry.get('resource', {}).get('labels', {})
        labels = entry.get('labels', {})
        
        job_id = resource_labels.get('job_id') or labels.get('job_id')
        if job_id:
            job_ids.add(job_id)
    
    job_ids_list = sorted(list(job_ids))
    
    print(f"\nFound {len(job_ids_list)} unique job IDs:")
    for jid in job_ids_list:
        print(f"   - {jid}")
    
    return job_ids_list


def extract_from_job_id(project_id, job_id):
    """Extract gradient stats from a specific job ID"""
    print(f"\n{'='*80}")
    print(f"Job ID: {job_id}")
    print(f"{'='*80}\n")
    
    # Get ALL logs from this job (not just EXPLODING ones)
    # We need the "Epoch X summary:" lines too
    filter_str = f'resource.labels.job_id="{job_id}"'
    
    logs = run_gcloud_logs(project_id, filter_str, limit=20000)
    
    if not logs:
        print(f"  No logs found")
        return None
    
    gradient_stats = extract_gradient_stats_from_logs(logs)
    
    if not gradient_stats:
        print(f"  No gradient stats extracted")
        return None
    
    # Compute summary
    pcts = [s['exploding_grad_pct'] for s in gradient_stats]
    
    summary = {
        'job_id': job_id,
        'num_epochs': len(gradient_stats),
        'avg_exploding_grad_pct': sum(pcts) / len(pcts),
        'max_exploding_grad_pct': max(pcts),
        'min_exploding_grad_pct': min(pcts),
        'epoch_details': gradient_stats
    }
    
    print(f"  Epochs: {len(gradient_stats)}")
    print(f"  Avg exploding: {summary['avg_exploding_grad_pct']:.2f}%")
    print(f"  Max exploding: {summary['max_exploding_grad_pct']:.2f}%")
    print(f"  Min exploding: {summary['min_exploding_grad_pct']:.2f}%")
    
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=str, required=True)
    parser.add_argument('--job_id', type=str, help='Specific job ID')
    parser.add_argument('--recent_jobs', action='store_true', help='Analyze recent jobs')
    parser.add_argument('--hours', type=int, default=72)
    parser.add_argument('--output', type=str, default='gradient_stats.csv')
    parser.add_argument('--output_detailed', type=str, default='gradient_stats_detailed.csv')
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print("Gradient Stats Extraction")
    print(f"{'='*80}")
    print(f"Project: {args.project}")
    print(f"Mode: {'Recent jobs' if args.recent_jobs else 'Single job'}")
    print(f"Output: {args.output}")
    print()
    
    if args.job_id:
        # Analyze single job
        summary = extract_from_job_id(args.project, args.job_id)
        
        if summary:
            print(f"\n{'='*80}")
            print("RESULTS")
            print(f"{'='*80}\n")
            print(f"Job ID: {summary['job_id']}")
            print(f"Epochs: {summary['num_epochs']}")
            print(f"Avg exploding gradient %: {summary['avg_exploding_grad_pct']:.2f}%")
            print(f"Max exploding gradient %: {summary['max_exploding_grad_pct']:.2f}%")
            print(f"Min exploding gradient %: {summary['min_exploding_grad_pct']:.2f}%")
            
            if summary['epoch_details']:
                df = pd.DataFrame(summary['epoch_details'])
                df['job_id'] = summary['job_id']
                df.to_csv(args.output, index=False)
                print(f"\nSaved detailed results to: {args.output}")
        else:
            print("\nNo results to save")
    
    elif args.recent_jobs:
        # Find and analyze all recent jobs
        job_ids = get_all_recent_job_ids(args.project, args.hours)
        
        if not job_ids:
            print("No job IDs found")
            return
        
        # Accumulate results for ALL jobs
        all_summaries = []
        all_detailed_results = []
        
        for i, job_id in enumerate(job_ids, 1):
            print(f"\n[{i}/{len(job_ids)}] Processing job {job_id}...")
            
            summary = extract_from_job_id(args.project, job_id)
            
            if summary:
                all_summaries.append({
                    'job_id': summary['job_id'],
                    'num_epochs': summary['num_epochs'],
                    'avg_exploding_grad_pct': summary['avg_exploding_grad_pct'],
                    'max_exploding_grad_pct': summary['max_exploding_grad_pct'],
                    'min_exploding_grad_pct': summary['min_exploding_grad_pct'],
                })
                
                for epoch_data in summary['epoch_details']:
                    epoch_data_with_job = epoch_data.copy()
                    epoch_data_with_job['job_id'] = summary['job_id']
                    all_detailed_results.append(epoch_data_with_job)
        
        if all_summaries:
            df_summary = pd.DataFrame(all_summaries)
            df_summary = df_summary.sort_values('avg_exploding_grad_pct', ascending=False)
            
            print(f"\n{'='*80}")
            print("ALL JOBS SUMMARY")
            print(f"{'='*80}\n")
            print(df_summary.to_string(index=False))
            
            df_summary.to_csv(args.output, index=False)
            print(f"\nSummary saved to: {args.output}")
            
            if all_detailed_results:
                df_detailed = pd.DataFrame(all_detailed_results)
                df_detailed = df_detailed.sort_values(['job_id', 'epoch'])
                df_detailed.to_csv(args.output_detailed, index=False)
                print(f"Detailed results saved to: {args.output_detailed}")
                print(f"Total epochs across all jobs: {len(all_detailed_results)}")
        else:
            print("\nNo results found for any jobs")
    else:
        print("ERROR: Must specify --job_id or --recent_jobs")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)