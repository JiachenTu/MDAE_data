#!/usr/bin/env python3
"""
Comprehensive Analysis Pipeline for MDAE Benchmarking
======================================================
This script processes raw WandB data and generates comprehensive analysis
including all benchmarks, modalities, and method comparisons.

Usage:
    python run_comprehensive_analysis.py [--input-dir RAW_DATA_DIR] [--output-dir OUTPUT_DIR]

Default paths:
    Input: raw_data/20250811/
    Output: processed_data/comprehensive_analysis/
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import warnings
import sys
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Default paths
DEFAULT_RAW_DATA_DIR = Path('/home/t-jiachentu/repos/benchmarking/misc/data/raw_data/20250811')
DEFAULT_OUTPUT_DIR = Path('/home/t-jiachentu/repos/benchmarking/misc/data/processed_data/comprehensive_analysis')

# Benchmark name mapping for display
BENCHMARK_MAPPING = {
    'upenn_gbm_age_group': 'UPenn-GBM: Age',
    'upenn_gbm_gender': 'UPenn-GBM: Gender',
    'upenn_gbm_survival_18month': 'UPenn-GBM: Surv_18M',
    'upenn_gbm_survival_1year': 'UPenn-GBM: Surv_1Y',
    'upenn_gbm_survival_2year': 'UPenn-GBM: Surv_2Y',
    'brats18_lgg_vs_hgg': 'BraTS18: LGG_vs_HGG',
    'brats23_gli_vs_men': 'BraTS23: Glio_vs_Men',
    'brats23_gli_vs_met': 'BraTS23: Glio_vs_Met',
    'brats23_men_vs_met': 'BraTS23: Men_vs_Met',
    'tcga_gbm_dss_1year': 'TCGA-GBM: DSS_1Y',
    'tcga_gbm_pfi_1year': 'TCGA-GBM: PFI_1Y',
    'ucsf_pdgm_idh_classification': 'UCSF-PDGM: IDH',
    'upenn_gbm_gtr_status': 'UPenn-GBM: GTR',
    'upenn_gbm_idh1_status': 'UPenn-GBM: IDH1',
    'rsna_miccai_mgmt_methylation': 'RSNA-MICCAI: MGMT'
}

# Method recognition patterns - Order matters!
METHOD_PATTERNS = {
    # Our methods (most specific patterns first)
    'MDAE': r'^(resenc_MDAETrainer_RandomMask_Flow_BS48_2000ep|resenc_MDAE_pretrained|resenc_MDAE_scratch)',
    'MDAE (TC)': r'^(resenc_time_conditioned|resenc_multimodal_mm_mdae)',
    
    # SSL baselines (specific patterns)
    'SimCLR': r'^resenc_SimCLR',
    'VoCo': r'^resenc_VoCo',
    'MG': r'^resenc_MG',
    'SwinUNETR': r'^resenc_SwinUNETR',
    'VF': r'^resenc_VF',
    'S3D': r'^resenc_S3D',
    
    # Foundation models
    'BrainIAC': r'^brainiac_pretrained',
    'MRI-Core': r'^mri_core',
    'BrainMVP': r'^brainmvp',
    
    # Other baselines
    'DinoV2': r'^dinov2',
    'ResNet-50': r'^brainiac_scratch',
    
    # Generic MAE pattern - MUST BE LAST
    'MAE': r'^resenc_pretrained_'
}

# Metrics to extract
METRICS = ['Test_AUROC', 'Test_AP', 'Test_F1', 'Test_Balanced_Accuracy']
THRESHOLD_INDEPENDENT = ['Test_AUROC', 'Test_AP']
THRESHOLD_DEPENDENT = ['Test_F1', 'Test_Balanced_Accuracy']

# Visualization settings
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def identify_method(run_name: str) -> Optional[str]:
    """Identify method from run name using patterns."""
    import re
    for method, pattern in METHOD_PATTERNS.items():
        if re.match(pattern, run_name):
            return method
    return None

def load_benchmark_data(benchmark_dir: Path, verbose: bool = False) -> pd.DataFrame:
    """Load and process data for a single benchmark."""
    csv_path = benchmark_dir / 'runs_summary.csv'
    
    if not csv_path.exists():
        if verbose:
            print(f"  Warning: {csv_path} not found")
        return pd.DataFrame()
    
    df = pd.read_csv(csv_path)
    
    # Extract method from run_name
    df['Method'] = df['run_name'].apply(identify_method)
    
    # Use modality from CSV if available, normalize to lowercase
    if 'modality' in df.columns:
        # Normalize modality names to lowercase
        df['Modality'] = df['modality'].str.lower()
        # Handle special cases for consistency
        df['Modality'] = df['Modality'].replace({
            't1c': 't1ce',  # Normalize t1c to t1ce
            't2w': 't2',     # Normalize t2w to t2
            't1w': 't1',     # Normalize t1w to t1
            't1wce': 't1ce', # Normalize t1wce to t1ce
            't2f': 'flair',  # T2-FLAIR
        })
    else:
        # No modality column - shouldn't happen with proper data
        df['Modality'] = 'unknown'
    
    # Filter valid methods only
    df = df[df['Method'].notna()]
    
    # For multiple runs of same method/modality, select best by Test_AUROC
    df = df.sort_values('Test_AUROC', ascending=False).groupby(['Method', 'Modality']).first().reset_index()
    
    return df

# ============================================================================
# PROCESSING FUNCTIONS
# ============================================================================

def process_benchmark(benchmark_name: str, benchmark_dir: Path, output_dir: Path, verbose: bool = False) -> Dict:
    """Process a single benchmark with all modalities."""
    if verbose:
        print(f"\nProcessing: {benchmark_name}")
    
    # Load data
    df = load_benchmark_data(benchmark_dir, verbose)
    if df.empty:
        if verbose:
            print(f"  No data found for {benchmark_name}")
        return {}
    
    # Get available modalities (excluding multimodal and unknown unless only option)
    modalities = df['Modality'].unique()
    valid_modalities = [m for m in modalities if m not in ['multimodal', 'unknown']]
    if not valid_modalities and 'unknown' in modalities:
        valid_modalities = ['unknown']  # Use unknown if it's the only option
    
    if verbose:
        print(f"  Found {len(valid_modalities)} modalities: {', '.join(valid_modalities)}")
    
    # Create output directory structure
    bench_output = output_dir / benchmark_name
    bench_output.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Process each valid modality
    for modality in valid_modalities:
        mod_df = df[df['Modality'] == modality].copy()
        if mod_df.empty:
            continue
            
        mod_output = bench_output / modality
        mod_output.mkdir(exist_ok=True)
        
        # Save metrics table
        metrics_df = mod_df[['Method'] + METRICS].copy()
        metrics_df = metrics_df.sort_values('Test_AUROC', ascending=False)
        metrics_df.to_csv(mod_output / 'metrics_table.csv', index=False)
        
        # Create visualizations
        create_modality_visualizations(metrics_df, mod_output, modality)
        
        results[modality] = metrics_df
    
    # Create cross-modality comparison if multiple modalities exist
    if len(results) > 1:
        create_cross_modality_comparison(results, bench_output, benchmark_name)
    
    return results

def create_modality_visualizations(df: pd.DataFrame, output_dir: Path, modality: str):
    """Create both threshold-independent and all metrics visualizations."""
    
    # 1. Threshold-independent metrics (AUROC & AP)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # AUROC
    ax = axes[0]
    colors = ['#e74c3c' if 'MDAE' in m else '#3498db' for m in df['Method']]
    bars = ax.barh(range(len(df)), df['Test_AUROC'], color=colors)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['Method'])
    ax.set_xlabel('Test AUROC')
    ax.set_title(f'{modality.upper()} - Test AUROC')
    ax.set_xlim([0, 1])
    ax.grid(True, alpha=0.3)
    
    # Add values
    for i, (val, bar) in enumerate(zip(df['Test_AUROC'], bars)):
        if not pd.isna(val):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{val:.3f}', va='center', fontsize=9)
    
    # AP
    ax = axes[1]
    bars = ax.barh(range(len(df)), df['Test_AP'], color=colors)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['Method'])
    ax.set_xlabel('Test AP')
    ax.set_title(f'{modality.upper()} - Test AP')
    ax.set_xlim([0, 1])
    ax.grid(True, alpha=0.3)
    
    # Add values
    for i, (val, bar) in enumerate(zip(df['Test_AP'], bars)):
        if not pd.isna(val):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{val:.3f}', va='center', fontsize=9)
    
    plt.suptitle(f'Threshold-Independent Metrics - {modality.upper()}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'threshold_independent_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. All metrics visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for idx, metric in enumerate(METRICS):
        ax = axes[idx // 2, idx % 2]
        
        values = df[metric].values
        colors = ['#e74c3c' if 'MDAE' in m else '#3498db' for m in df['Method']]
        
        bars = ax.barh(range(len(df)), values, color=colors)
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df['Method'])
        ax.set_xlabel(metric.replace('Test_', '').replace('_', ' '))
        ax.set_title(f'{modality.upper()} - {metric.replace("Test_", "")}')
        ax.set_xlim([0, 1])
        ax.grid(True, alpha=0.3)
        
        # Add values
        for i, (val, bar) in enumerate(zip(values, bars)):
            if not pd.isna(val):
                ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                       f'{val:.3f}', va='center', fontsize=9)
    
    plt.suptitle(f'All Metrics - {modality.upper()}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'all_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_cross_modality_comparison(results: Dict, output_dir: Path, benchmark_name: str):
    """Create cross-modality comparison for a benchmark."""
    
    # Prepare data for comparison
    comparison_data = []
    
    for modality, df in results.items():
        for _, row in df.iterrows():
            for metric in METRICS:
                comparison_data.append({
                    'Modality': modality,
                    'Method': row['Method'],
                    'Metric': metric.replace('Test_', ''),
                    'Value': row[metric]
                })
    
    comp_df = pd.DataFrame(comparison_data)
    
    # Create comparison visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    for idx, metric in enumerate(['AUROC', 'AP', 'F1', 'Balanced_Accuracy']):
        ax = axes[idx // 2, idx % 2]
        
        metric_df = comp_df[comp_df['Metric'] == metric]
        pivot_df = metric_df.pivot(index='Method', columns='Modality', values='Value')
        
        # Heatmap
        sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='RdYlBu_r', 
                   vmin=0, vmax=1, ax=ax, cbar_kws={'label': metric})
        ax.set_title(f'{metric} Across Modalities')
        ax.set_xlabel('Modality')
        ax.set_ylabel('Method')
    
    plt.suptitle(f'Cross-Modality Comparison - {benchmark_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'cross_modality_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save comparison table
    pivot_df = comp_df.pivot(index=['Method', 'Metric'], columns='Modality', values='Value')
    pivot_df.to_csv(output_dir / 'cross_modality_table.csv')

def combine_mdae_variants(all_results: Dict, verbose: bool = False) -> Dict:
    """Add MDAE (Combined) row with best performance between MDAE and MDAE-TC variants."""
    if verbose:
        print("\nCombining MDAE variants...")
    
    for benchmark, modality_results in all_results.items():
        for modality, df in modality_results.items():
            # Check if both MDAE and MDAE (TC) exist
            if 'MDAE' in df['Method'].values and 'MDAE (TC)' in df['Method'].values:
                mdae_row = df[df['Method'] == 'MDAE'].iloc[0]
                mdae_tc_row = df[df['Method'] == 'MDAE (TC)'].iloc[0]
                
                # Create MDAE (Combined) row with best values based on AUROC
                combined_row = mdae_row.copy()
                combined_row['Method'] = 'MDAE (Combined)'
                
                # Take the variant with best AUROC
                if mdae_tc_row['Test_AUROC'] > mdae_row['Test_AUROC']:
                    combined_row[METRICS] = mdae_tc_row[METRICS].values
                    if verbose:
                        print(f"  {benchmark}/{modality}: Using MDAE (TC) for Combined")
                else:
                    if verbose:
                        print(f"  {benchmark}/{modality}: Using MDAE for Combined")
                
                # Add the combined row to dataframe
                df = pd.concat([df, pd.DataFrame([combined_row])], ignore_index=True)
                all_results[benchmark][modality] = df
                
            elif 'MDAE (TC)' in df['Method'].values and 'MDAE' not in df['Method'].values:
                # If only MDAE (TC) exists, create MDAE (Combined) from it
                mdae_tc_row = df[df['Method'] == 'MDAE (TC)'].iloc[0]
                combined_row = mdae_tc_row.copy()
                combined_row['Method'] = 'MDAE (Combined)'
                
                df = pd.concat([df, pd.DataFrame([combined_row])], ignore_index=True)
                all_results[benchmark][modality] = df
                
            elif 'MDAE' in df['Method'].values and 'MDAE (TC)' not in df['Method'].values:
                # If only MDAE exists, create MDAE (Combined) from it
                mdae_row = df[df['Method'] == 'MDAE'].iloc[0]
                combined_row = mdae_row.copy()
                combined_row['Method'] = 'MDAE (Combined)'
                
                df = pd.concat([df, pd.DataFrame([combined_row])], ignore_index=True)
                all_results[benchmark][modality] = df
    
    return all_results

def create_overall_analysis(all_results: Dict, output_dir: Path, verbose: bool = False):
    """Create overall analysis across all benchmarks and modalities."""
    
    if verbose:
        print("\n" + "="*60)
        print("CREATING OVERALL ANALYSIS")
        print("="*60)
    
    # Check if we have any results
    if not all_results:
        print("No results to analyze!")
        return
    
    # Aggregate all results
    overall_data = []
    
    for benchmark, modality_results in all_results.items():
        for modality, df in modality_results.items():
            for _, row in df.iterrows():
                for metric in METRICS:
                    overall_data.append({
                        'Benchmark': benchmark,
                        'Modality': modality,
                        'Method': row['Method'],
                        'Metric': metric,
                        'Value': row[metric]
                    })
    
    overall_df = pd.DataFrame(overall_data)
    
    # Check if we have data
    if overall_df.empty:
        print("No data to analyze!")
        return
    
    # Calculate summary statistics
    summary_stats = []
    
    for method in overall_df['Method'].unique():
        method_df = overall_df[overall_df['Method'] == method]
        
        for metric in METRICS:
            metric_df = method_df[method_df['Metric'] == metric]
            values = metric_df['Value'].dropna()
            
            if len(values) > 0:
                summary_stats.append({
                    'Method': method,
                    'Metric': metric,
                    'Mean': values.mean(),
                    'Std': values.std(),
                    'Median': values.median(),
                    'Min': values.min(),
                    'Max': values.max(),
                    'Count': len(values)
                })
    
    summary_df = pd.DataFrame(summary_stats)
    
    # Pivot for easier viewing
    pivot_mean = summary_df.pivot(index='Method', columns='Metric', values='Mean')
    pivot_std = summary_df.pivot(index='Method', columns='Metric', values='Std')
    
    # Sort by Test_AUROC
    pivot_mean = pivot_mean.sort_values('Test_AUROC', ascending=False)
    
    # Save tables
    pivot_mean.to_csv(output_dir / 'overall_mean_metrics.csv')
    pivot_std.to_csv(output_dir / 'overall_std_metrics.csv')
    summary_df.to_csv(output_dir / 'overall_summary_statistics.csv')
    
    # Create comprehensive table with mean ± std
    comprehensive_table = pd.DataFrame(index=pivot_mean.index)
    
    for metric in METRICS:
        if metric in pivot_mean.columns:
            mean_vals = pivot_mean[metric]
            std_vals = pivot_std[metric]
            comprehensive_table[metric] = mean_vals.apply(lambda x: f'{x:.3f}') + ' ± ' + std_vals.apply(lambda x: f'{x:.3f}')
    
    comprehensive_table.to_csv(output_dir / 'comprehensive_metrics_table.csv')
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for idx, metric in enumerate(METRICS):
        ax = axes[idx // 2, idx % 2]
        
        metric_means = pivot_mean[metric].sort_values(ascending=False)
        colors = ['#e74c3c' if 'MDAE' in m else '#3498db' for m in metric_means.index]
        
        bars = ax.barh(range(len(metric_means)), metric_means.values, color=colors)
        ax.set_yticks(range(len(metric_means)))
        ax.set_yticklabels(metric_means.index)
        ax.set_xlabel(metric.replace('Test_', '').replace('_', ' '))
        ax.set_title(f'Mean {metric.replace("Test_", "")} Across All Benchmarks')
        ax.set_xlim([0, 1])
        ax.grid(True, alpha=0.3)
        
        # Add values
        for i, val in enumerate(metric_means.values):
            if not pd.isna(val):
                ax.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=9)
    
    plt.suptitle('Overall Performance Across All Benchmarks and Modalities', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'overall_performance.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print summary
    if verbose:
        print("\nTop Methods by Mean Test AUROC:")
        print(pivot_mean['Test_AUROC'].head(10).to_string())
        
        # Print MDAE performance if available
        for mdae_variant in ['MDAE (Combined)', 'MDAE', 'MDAE (TC)']:
            if mdae_variant in pivot_mean.index:
                print(f"\n{mdae_variant} Performance Summary:")
                print(f"  Mean AUROC: {pivot_mean.loc[mdae_variant, 'Test_AUROC']:.3f}")
                print(f"  Mean AP: {pivot_mean.loc[mdae_variant, 'Test_AP']:.3f}")
                print(f"  Mean F1: {pivot_mean.loc[mdae_variant, 'Test_F1']:.3f}")
                print(f"  Mean Balanced Accuracy: {pivot_mean.loc[mdae_variant, 'Test_Balanced_Accuracy']:.3f}")

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main(raw_data_dir: Path, output_dir: Path, verbose: bool = True):
    """Main processing pipeline."""
    
    print("="*60)
    print("COMPREHENSIVE MDAE BENCHMARKING ANALYSIS")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    print(f"Input directory: {raw_data_dir}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process all benchmarks
    all_results = {}
    
    for benchmark_dir in sorted(raw_data_dir.iterdir()):
        if not benchmark_dir.is_dir():
            continue
        
        benchmark_name = benchmark_dir.name
        
        # Skip if not in mapping
        if benchmark_name not in BENCHMARK_MAPPING:
            if verbose:
                print(f"\nSkipping unmapped benchmark: {benchmark_name}")
            continue
        
        display_name = BENCHMARK_MAPPING[benchmark_name]
        results = process_benchmark(display_name, benchmark_dir, output_dir / 'benchmarks', verbose)
        
        if results:
            all_results[display_name] = results
    
    # Combine MDAE variants
    all_results = combine_mdae_variants(all_results, verbose)
    
    # Re-save individual benchmark results with MDAE (Combined)
    if verbose:
        print("\nSaving updated benchmark results with MDAE (Combined)...")
    
    for benchmark_name, modality_results in all_results.items():
        bench_output = output_dir / 'benchmarks' / benchmark_name
        for modality, df in modality_results.items():
            mod_output = bench_output / modality
            if mod_output.exists():
                # Re-save metrics table with combined row
                df_sorted = df.sort_values('Test_AUROC', ascending=False)
                df_sorted.to_csv(mod_output / 'metrics_table.csv', index=False)
    
    # Create overall analysis
    create_overall_analysis(all_results, output_dir, verbose)
    
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"Results saved to: {output_dir}")
    print(f"  - Benchmark results: {output_dir}/benchmarks/")
    print(f"  - Overall analysis: {output_dir}/")
    
    # Summary statistics
    total_benchmarks = len(all_results)
    total_modalities = sum(len(mod_results) for mod_results in all_results.values())
    
    print(f"\nProcessed:")
    print(f"  - {total_benchmarks} benchmarks")
    print(f"  - {total_modalities} total modality combinations")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run comprehensive MDAE benchmarking analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--input-dir',
        type=Path,
        default=DEFAULT_RAW_DATA_DIR,
        help=f'Input directory with raw data (default: {DEFAULT_RAW_DATA_DIR})'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f'Output directory for results (default: {DEFAULT_OUTPUT_DIR})'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not args.input_dir.exists():
        print(f"Error: Input directory does not exist: {args.input_dir}")
        sys.exit(1)
    
    # Run the analysis
    main(args.input_dir, args.output_dir, verbose=not args.quiet)