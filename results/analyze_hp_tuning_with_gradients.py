#!/usr/bin/env python3
"""
Analyze HP tuning results including exploding gradient metrics
"""
import argparse
from google.cloud.aiplatform_v1 import JobServiceClient
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def analyze_hp_results(job_name, region):
    """Analyze HP tuning results with gradient metrics"""
    
    client = JobServiceClient(
        client_options={"api_endpoint": f"{region}-aiplatform.googleapis.com"}
    )
    
    # Get job details
    job = client.get_hyperparameter_tuning_job(name=job_name)
    
    if not job.trials:
        print("No trials found!")
        return
    
    # Extract trial data
    data = []
    for trial in job.trials:
        if trial.state.name != 'SUCCEEDED':
            continue
            
        row = {
            'trial_id': trial.id,
        }
        
        # Extract parameters
        for param in trial.parameters:
            row[param.parameter_id] = param.value
        
        # Extract BOTH metrics
        if trial.final_measurement:
            for metric in trial.final_measurement.metrics:
                if metric.metric_id == 'val_loss':
                    row['val_loss'] = metric.value
                elif metric.metric_id == 'exploding_grad_pct':
                    row['exploding_grad_pct'] = metric.value
        
        data.append(row)
    
    df = pd.DataFrame(data)
    
    if len(df) == 0:
        print("No successful trials!")
        return
    
    # Sort by val_loss
    df = df.sort_values('val_loss')
    
    print("=" * 120)
    print("HYPERPARAMETER TUNING RESULTS WITH GRADIENT ANALYSIS")
    print("=" * 120)
    print(f"\nTotal successful trials: {len(df)}")
    
    # Top 10 by val_loss
    print("\n" + "=" * 120)
    print("Top 10 Trials by Validation Loss")
    print("=" * 120)
    print(df.head(10).to_string(index=False))
    
    # Top 10 by exploding gradients (lowest)
    if 'exploding_grad_pct' in df.columns:
        print("\n" + "=" * 120)
        print("Top 10 Trials by Lowest Exploding Gradient %")
        print("=" * 120)
        df_by_grads = df.sort_values('exploding_grad_pct')
        print(df_by_grads.head(10).to_string(index=False))
        
        # Correlation analysis
        print("\n" + "=" * 120)
        print("Correlation with Exploding Gradient %")
        print("=" * 120)
        
        param_cols = ['learning_rate', 'hidden_dim', 'num_encoder_layers', 
                      'num_decoder_layers', 'masking_ratio', 'batch_size']
        
        for col in param_cols:
            if col in df.columns:
                corr = df[col].corr(df['exploding_grad_pct'])
                print(f"  {col:25s}: {corr:6.3f}")
        
        # Sweet spot analysis
        print("\n" + "=" * 120)
        print("Sweet Spot Analysis (val_loss < median AND exploding_grad_pct < 5%)")
        print("=" * 120)
        
        median_val_loss = df['val_loss'].median()
        sweet_spot = df[(df['val_loss'] < median_val_loss) & 
                        (df['exploding_grad_pct'] < 5.0)]
        
        if len(sweet_spot) > 0:
            print(f"\nFound {len(sweet_spot)} configurations in sweet spot:")
            print(sweet_spot.sort_values('val_loss').to_string(index=False))
        else:
            print("\nNo configurations found in sweet spot!")
            print("Consider:")
            print("  - Lower learning rates")
            print("  - Smaller model sizes")
            print("  - Lower masking ratios")
    
    # Best overall configuration
    print("\n" + "=" * 120)
    print("BEST CONFIGURATION (Lowest val_loss)")
    print("=" * 120)
    best = df.iloc[0]
    print(f"\nTrial ID: {best['trial_id']}")
    print(f"Val Loss: {best['val_loss']:.6f}")
    if 'exploding_grad_pct' in best:
        print(f"Exploding Gradient %: {best['exploding_grad_pct']:.2f}%")
    print("\nHyperparameters:")
    for col in param_cols:
        if col in best:
            print(f"  {col:25s}: {best[col]}")
    
    return df


def plot_gradient_analysis(df, output_dir):
    """Create plots for gradient analysis"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if 'exploding_grad_pct' not in df.columns:
        print("No exploding gradient data to plot")
        return
    
    # Plot 1: Scatter plot - val_loss vs exploding_grad_pct
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(df['exploding_grad_pct'], df['val_loss'], 
                        c=df['learning_rate'], cmap='viridis', s=100, alpha=0.6)
    ax.set_xlabel('Exploding Gradient %', fontsize=12)
    ax.set_ylabel('Validation Loss', fontsize=12)
    ax.set_title('Validation Loss vs Exploding Gradients (colored by learning rate)', fontsize=14)
    ax.grid(alpha=0.3)
    plt.colorbar(scatter, label='Learning Rate')
    
    # Add sweet spot rectangle
    median_loss = df['val_loss'].median()
    ax.axhline(median_loss, color='r', linestyle='--', alpha=0.5, label='Median val_loss')
    ax.axvline(5.0, color='g', linestyle='--', alpha=0.5, label='5% exploding threshold')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'val_loss_vs_exploding_gradients.png', dpi=150)
    print(f"Saved: {output_dir / 'val_loss_vs_exploding_gradients.png'}")
    
    # Plot 2: Box plots for each hyperparameter
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    param_cols = ['learning_rate', 'hidden_dim', 'num_encoder_layers', 
                  'num_decoder_layers', 'masking_ratio', 'batch_size']
    
    for i, col in enumerate(param_cols):
        if col in df.columns:
            df_grouped = df.groupby(col)['exploding_grad_pct'].apply(list)
            
            axes[i].boxplot(df_grouped.values, labels=df_grouped.index)
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Exploding Gradient %')
            axes[i].set_title(f'Exploding Gradients by {col}')
            axes[i].axhline(5.0, color='r', linestyle='--', alpha=0.5)
            axes[i].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'exploding_gradients_by_param.png', dpi=150)
    print(f"Saved: {output_dir / 'exploding_gradients_by_param.png'}")
    
    # Plot 3: Heatmap - correlation matrix
    param_cols_with_metrics = param_cols + ['val_loss', 'exploding_grad_pct']
    cols_present = [col for col in param_cols_with_metrics if col in df.columns]
    
    if len(cols_present) > 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        corr_matrix = df[cols_present].corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, ax=ax, cbar_kws={'label': 'Correlation'})
        ax.set_title('Correlation Matrix: Hyperparameters vs Metrics')
        plt.tight_layout()
        plt.savefig(output_dir / 'correlation_heatmap.png', dpi=150)
        print(f"Saved: {output_dir / 'correlation_heatmap.png'}")
    
    plt.close('all')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_name', type=str, required=True,
                       help='Full resource name of HP tuning job')
    parser.add_argument('--region', type=str, default='us-central1')
    parser.add_argument('--output_dir', type=str, default='./hp_gradient_analysis',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    print("Fetching HP tuning results...")
    df = analyze_hp_results(args.job_name, args.region)
    
    if df is not None and len(df) > 0:
        print("\nGenerating plots...")
        plot_gradient_analysis(df, args.output_dir)
        
        # Save CSV
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        df.to_csv(output_dir / 'hp_tuning_results.csv', index=False)
        print(f"\nSaved results to: {output_dir / 'hp_tuning_results.csv'}")


if __name__ == "__main__":
    main()