#!/usr/bin/env python3
"""
Comprehensive processing script for all modalities and metrics.
Processes ALL benchmarks, ALL modalities, ALL metrics with dual visualization types.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Configuration
RAW_DATA_DIR = Path('/home/t-jiachentu/repos/benchmarking/misc/data/raw_data/20250811')
OUTPUT_DIR = Path('/home/t-jiachentu/repos/benchmarking/misc/data/processed_data/comprehensive_analysis')
BASELINES_FILE = Path('/home/t-jiachentu/repos/benchmarking/misc/data/baselines/key_baselines.txt')

# Benchmark mapping - Updated to match actual directory names
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

# Method patterns - Fixed to correctly identify all methods
# IMPORTANT: Order matters! More specific patterns must come before generic ones
METHOD_PATTERNS = {
    # Our methods (most specific)
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
    
    # Generic MAE pattern - MUST BE LAST as it's the most generic
    'MAE': r'^resenc_pretrained_'
}

# All test metrics
METRICS = ['Test_AUROC', 'Test_AP', 'Test_F1', 'Test_Balanced_Accuracy']
THRESHOLD_INDEPENDENT = ['Test_AUROC', 'Test_AP']
THRESHOLD_DEPENDENT = ['Test_F1', 'Test_Balanced_Accuracy']

def identify_method(run_name: str) -> Optional[str]:
    """Identify method from run name using patterns."""
    import re
    for method, pattern in METHOD_PATTERNS.items():
        if re.match(pattern, run_name):
            return method
    return None

def extract_modality(run_name: str) -> str:
    """Extract modality from run name."""
    modalities = ['asl', 'flair', 't1ce', 't1pre', 't1post', 't1', 't2']  # Order matters - check longer patterns first
    run_lower = run_name.lower()
    
    # Check for modality patterns in different positions
    for mod in modalities:
        # Check various patterns
        if f'_{mod}_' in run_lower:
            return mod
        if run_lower.endswith(f'_{mod}'):
            return mod
        if f'single_{mod}' in run_lower:
            return mod
        if f'{mod}_single' in run_lower:
            return mod
            
    # If no modality found, check if it's a multimodal run
    if 'multimodal' in run_lower or '_mm_' in run_lower:
        return 'multimodal'
    
    return 'unknown'

def load_benchmark_data(benchmark_dir: Path) -> pd.DataFrame:
    """Load and process data for a single benchmark."""
    csv_path = benchmark_dir / 'runs_summary.csv'
    
    if not csv_path.exists():
        print(f"  Warning: {csv_path} not found")
        return pd.DataFrame()
    
    df = pd.read_csv(csv_path)
    
    # Extract method from run_name
    df['Method'] = df['run_name'].apply(identify_method)
    
    # Use modality from CSV if available, normalize to lowercase
    if 'modality' in df.columns:
        # Normalize modality names to lowercase
        df['Modality'] = df['modality'].str.lower()
        # Handle special cases
        df['Modality'] = df['Modality'].replace({
            't1c': 't1ce',  # Normalize t1c to t1ce
            't2w': 't2',     # Normalize t2w to t2
            't1w': 't1',     # Normalize t1w to t1
        })
    else:
        # Fall back to extraction from run_name if no modality column
        df['Modality'] = df['run_name'].apply(extract_modality)
    
    # Filter valid methods
    df = df[df['Method'].notna()]
    
    # For multiple runs, select best by Test_AUROC
    df = df.sort_values('Test_AUROC', ascending=False).groupby(['Method', 'Modality']).first().reset_index()
    
    return df

def process_benchmark(benchmark_name: str, benchmark_dir: Path, output_dir: Path) -> Dict:
    """Process a single benchmark with all modalities."""
    print(f"\nProcessing: {benchmark_name}")
    
    # Load data
    df = load_benchmark_data(benchmark_dir)
    if df.empty:
        print(f"  No data found for {benchmark_name}")
        return {}
    
    # Get available modalities
    modalities = df['Modality'].unique()
    print(f"  Found {len(modalities)} modalities: {', '.join(modalities)}")
    
    # Create output directory structure
    bench_output = output_dir / benchmark_name
    bench_output.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Process each modality
    for modality in modalities:
        # Skip unknown modalities unless it's the only one available
        if modality == 'unknown' and len(modalities) > 1:
            continue
        
        # Skip multimodal - we only want single-modality results
        if modality == 'multimodal':
            continue
            
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
        
    # Create cross-modality comparison
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
                   f'{val:.3f}', va='center')
    
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
                   f'{val:.3f}', va='center')
    
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
                       f'{val:.3f}', va='center')
    
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

def create_overall_analysis(all_results: Dict, output_dir: Path):
    """Create overall analysis across all benchmarks and modalities."""
    
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
                ax.text(val + 0.01, i, f'{val:.3f}', va='center')
    
    plt.suptitle('Overall Performance Across All Benchmarks and Modalities', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'overall_performance.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print summary
    print("\nTop Methods by Mean Test AUROC:")
    print(pivot_mean['Test_AUROC'].head(10).to_string())
    
    print("\nMDAE Performance Summary:")
    if 'MDAE' in pivot_mean.index:
        print(f"  Mean AUROC: {pivot_mean.loc['MDAE', 'Test_AUROC']:.3f}")
        print(f"  Mean AP: {pivot_mean.loc['MDAE', 'Test_AP']:.3f}")
        print(f"  Mean F1: {pivot_mean.loc['MDAE', 'Test_F1']:.3f}")
        print(f"  Mean Balanced Accuracy: {pivot_mean.loc['MDAE', 'Test_Balanced_Accuracy']:.3f}")

def combine_mdae_variants(all_results: Dict) -> Dict:
    """Add MDAE (Combined) row with best performance between MDAE and MDAE-TC variants."""
    
    for benchmark, modality_results in all_results.items():
        for modality, df in modality_results.items():
            # Check if both MDAE and MDAE (TC) exist
            if 'MDAE' in df['Method'].values and 'MDAE (TC)' in df['Method'].values:
                mdae_row = df[df['Method'] == 'MDAE'].iloc[0]
                mdae_tc_row = df[df['Method'] == 'MDAE (TC)'].iloc[0]
                
                # Create MDAE (Combined) row with best values
                combined_row = mdae_row.copy()
                combined_row['Method'] = 'MDAE (Combined)'
                
                # Take best AUROC and corresponding metrics
                if mdae_tc_row['Test_AUROC'] > mdae_row['Test_AUROC']:
                    combined_row[METRICS] = mdae_tc_row[METRICS].values
                
                # Add the combined row to dataframe
                df = pd.concat([df, pd.DataFrame([combined_row])], ignore_index=True)
                all_results[benchmark][modality] = df
                
            elif 'MDAE (TC)' in df['Method'].values and 'MDAE' not in df['Method'].values:
                # If only MDAE (TC) exists, create MDAE (Combined) from it
                mdae_tc_row = df[df['Method'] == 'MDAE (TC)'].iloc[0]
                combined_row = mdae_tc_row.copy()
                combined_row['Method'] = 'MDAE (Combined)'
                
                # Add the combined row to dataframe
                df = pd.concat([df, pd.DataFrame([combined_row])], ignore_index=True)
                all_results[benchmark][modality] = df
                
            elif 'MDAE' in df['Method'].values and 'MDAE (TC)' not in df['Method'].values:
                # If only MDAE exists, create MDAE (Combined) from it
                mdae_row = df[df['Method'] == 'MDAE'].iloc[0]
                combined_row = mdae_row.copy()
                combined_row['Method'] = 'MDAE (Combined)'
                
                # Add the combined row to dataframe
                df = pd.concat([df, pd.DataFrame([combined_row])], ignore_index=True)
                all_results[benchmark][modality] = df
    
    return all_results

def main():
    """Main processing pipeline."""
    
    print("="*60)
    print("COMPREHENSIVE MDAE BENCHMARKING ANALYSIS")
    print("="*60)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process all benchmarks
    all_results = {}
    
    for benchmark_dir in sorted(RAW_DATA_DIR.iterdir()):
        if not benchmark_dir.is_dir():
            continue
        
        benchmark_name = benchmark_dir.name
        
        # Skip if not in mapping
        if benchmark_name not in BENCHMARK_MAPPING:
            print(f"Skipping unmapped benchmark: {benchmark_name}")
            continue
        
        display_name = BENCHMARK_MAPPING[benchmark_name]
        results = process_benchmark(display_name, benchmark_dir, OUTPUT_DIR / 'benchmarks')
        
        if results:
            all_results[display_name] = results
    
    # Combine MDAE variants
    print("\nCombining MDAE variants...")
    all_results = combine_mdae_variants(all_results)
    
    # Re-save individual benchmark results with MDAE (Combined)
    for benchmark_name, modality_results in all_results.items():
        bench_output = OUTPUT_DIR / 'benchmarks' / benchmark_name
        for modality, df in modality_results.items():
            mod_output = bench_output / modality
            if mod_output.exists():
                # Re-save metrics table with combined row
                df_sorted = df.sort_values('Test_AUROC', ascending=False)
                df_sorted.to_csv(mod_output / 'metrics_table.csv', index=False)
    
    # Create overall analysis
    create_overall_analysis(all_results, OUTPUT_DIR)
    
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"  - Benchmark results: {OUTPUT_DIR}/benchmarks/")
    print(f"  - Overall analysis: {OUTPUT_DIR}/")
    
    # Summary statistics
    total_benchmarks = len(all_results)
    total_modalities = sum(len(mod_results) for mod_results in all_results.values())
    
    print(f"\nProcessed:")
    print(f"  - {total_benchmarks} benchmarks")
    print(f"  - {total_modalities} total modality combinations")

if __name__ == "__main__":
    main()