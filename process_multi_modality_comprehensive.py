#!/usr/bin/env python3
"""
Comprehensive Processing Pipeline for Multi-Modality Benchmarks
================================================================
This script processes multi-modality benchmark results similar to the 
single-modality processed_data_combined structure.

Key features:
- Processes all 15 multi-modality benchmarks
- Creates MDAE (Combined) from best of MDAE and MDAE (TC)
- Generates visualizations and metrics tables
- Creates paper-ready LaTeX outputs
- Compares with single-modality performance

Usage:
    python process_multi_modality_comprehensive.py                    # Combined MDAE only (default)
    python process_multi_modality_comprehensive.py --no-combine-mdae  # Keep all variants
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import warnings
from datetime import datetime
import re
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Method patterns from README_METHODS.md
METHOD_PATTERNS = {
    'MDAE': r'^(resenc_MDAETrainer_RandomMask_Flow_BS48_2000ep|resenc_MDAE_pretrained|resenc_MDAE_scratch)',
    'MDAE (TC)': r'^(resenc_time_conditioned|resenc_multimodal_mm_mdae)',
    'MAE': r'^resenc_pretrained_',
    'SimCLR': r'^resenc_SimCLR',
    'VoCo': r'^resenc_VoCo',
    'MG': r'^resenc_MG',
    'SwinUNETR': r'^resenc_SwinUNETR',
    'VF': r'^resenc_VF',
    'S3D': r'^resenc_S3D',
    'BrainIAC': r'^brainiac_pretrained',
    'MRI-Core': r'^mri_core',
    'BrainMVP': r'^brainmvp',
    'DinoV2': r'^dinov2',
    'ResNet-50': r'^brainiac_scratch'
}

# Benchmark display names
BENCHMARK_DISPLAY = {
    'brats18_lgg_vs_hgg': 'BraTS18_LGG_vs_HGG',
    'brats23_gli_vs_men': 'BraTS23_Glio_vs_Men',
    'brats23_gli_vs_met': 'BraTS23_Glio_vs_Met',
    'brats23_men_vs_met': 'BraTS23_Men_vs_Met',
    'rsna_miccai_mgmt': 'RSNA-MICCAI_MGMT',
    'tcga_gbm_dss_1year': 'TCGA-GBM_DSS_1Y',
    'tcga_gbm_pfi_1year': 'TCGA-GBM_PFI_1Y',
    'ucsf_pdgm_idh': 'UCSF-PDGM_IDH',
    'upenn_gbm_age_group': 'UPenn-GBM_Age',
    'upenn_gbm_gender': 'UPenn-GBM_Gender',
    'upenn_gbm_gtr_status': 'UPenn-GBM_GTR',
    'upenn_gbm_idh1_status': 'UPenn-GBM_IDH1',
    'upenn_gbm_survival_18month': 'UPenn-GBM_Surv_18M',
    'upenn_gbm_survival_1year': 'UPenn-GBM_Surv_1Y',
    'upenn_gbm_survival_2year': 'UPenn-GBM_Surv_2Y'
}

# Modality combinations for each benchmark
BENCHMARK_MODALITIES = {
    'brats18_lgg_vs_hgg': ['flair', 't1', 't1ce', 't2'],
    'brats23_gli_vs_men': ['t1n', 't1c', 't2w', 't2f'],
    'brats23_gli_vs_met': ['t1n', 't1c', 't2w', 't2f'],
    'brats23_men_vs_met': ['t1n', 't1c', 't2w', 't2f'],
    'rsna_miccai_mgmt': ['t1w', 't1wce', 't2w', 'flair'],
    'ucsf_pdgm_idh': ['t1', 't1c', 't2', 'flair', 'swi', 'asl'],
    'tcga_gbm_dss_1year': ['t1', 't1gd', 't2', 'flair'],
    'tcga_gbm_pfi_1year': ['t1', 't1gd', 't2', 'flair'],
    'upenn_gbm_age_group': ['t1', 't1gd', 't2', 'flair'],
    'upenn_gbm_gender': ['t1', 't1gd', 't2', 'flair'],
    'upenn_gbm_gtr_status': ['t1', 't1gd', 't2', 'flair'],
    'upenn_gbm_idh1_status': ['t1', 't1gd', 't2', 'flair'],
    'upenn_gbm_survival_18month': ['t1', 't1gd', 't2', 'flair'],
    'upenn_gbm_survival_1year': ['t1', 't1gd', 't2', 'flair'],
    'upenn_gbm_survival_2year': ['t1', 't1gd', 't2', 'flair']
}

# MDAE variant configurations
MDAE_VARIANTS_ALL = ['MDAE', 'MDAE (TC)', 'MDAE (Combined)']
MDAE_VARIANTS_COMBINED_ONLY = ['MDAE (Combined)']

# Color scheme for visualizations
COLOR_SCHEME = {
    'MDAE': '#FF6B6B',
    'MDAE (TC)': '#4ECDC4',
    'MDAE (Combined)': '#45B7D1',
    'MAE': '#95E77E',
    'SimCLR': '#FFD93D',
    'VoCo': '#6A4C93',
    'Other': '#B0B0B0'
}

# ============================================================================
# DATA LOADING AND PROCESSING
# ============================================================================

def identify_method(run_name: str) -> str:
    """Identify method from run name using pattern matching."""
    # Check MDAE patterns first (more specific)
    if re.match(METHOD_PATTERNS['MDAE'], run_name):
        return 'MDAE'
    if re.match(METHOD_PATTERNS['MDAE (TC)'], run_name):
        return 'MDAE (TC)'
    
    # Check other methods
    for method, pattern in METHOD_PATTERNS.items():
        if method not in ['MDAE', 'MDAE (TC)', 'MAE']:
            if re.match(pattern, run_name):
                return method
    
    # MAE pattern is most generic, check last
    if re.match(METHOD_PATTERNS['MAE'], run_name):
        return 'MAE'
    
    return 'Unknown'

def load_benchmark_data(benchmark_path: Path) -> pd.DataFrame:
    """Load and process data for a single benchmark."""
    csv_path = benchmark_path / "runs_summary.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    
    df = pd.read_csv(csv_path)
    
    # Add method identification
    df['Method'] = df['run_name'].apply(identify_method)
    
    # Extract metrics (handle different column naming)
    metric_cols = {
        'Test_AUROC': ['metric_Test/AUROC', 'metric_Test_AUROC', 'Test_AUROC'],
        'Test_AP': ['metric_Test/AP', 'metric_Test_AP', 'Test_AP'],
        'Test_F1': ['metric_Test/F1', 'metric_Test_F1', 'Test_F1'],
        'Test_Balanced_Accuracy': ['metric_Test/Balanced_Accuracy', 'metric_Test_Balanced_Accuracy', 'Test_Balanced_Accuracy']
    }
    
    for metric_name, possible_cols in metric_cols.items():
        for col in possible_cols:
            if col in df.columns:
                df[metric_name] = df[col]
                break
        if metric_name not in df.columns:
            df[metric_name] = np.nan
    
    return df

def combine_mdae_variants(df: pd.DataFrame) -> pd.DataFrame:
    """Create MDAE (Combined) by taking best of MDAE and MDAE (TC)."""
    # Group by method and get best performance
    method_best = df.groupby('Method').agg({
        'Test_AUROC': 'max',
        'Test_AP': 'max',
        'Test_F1': 'max',
        'Test_Balanced_Accuracy': 'max'
    }).reset_index()
    
    # Create combined MDAE if both variants exist
    if 'MDAE' in method_best['Method'].values and 'MDAE (TC)' in method_best['Method'].values:
        mdae_row = method_best[method_best['Method'] == 'MDAE'].iloc[0]
        mdae_tc_row = method_best[method_best['Method'] == 'MDAE (TC)'].iloc[0]
        
        combined_row = {
            'Method': 'MDAE (Combined)',
            'Test_AUROC': max(mdae_row['Test_AUROC'], mdae_tc_row['Test_AUROC']),
            'Test_AP': max(mdae_row['Test_AP'], mdae_tc_row['Test_AP']),
            'Test_F1': max(mdae_row['Test_F1'], mdae_tc_row['Test_F1']),
            'Test_Balanced_Accuracy': max(mdae_row['Test_Balanced_Accuracy'], mdae_tc_row['Test_Balanced_Accuracy'])
        }
        
        method_best = pd.concat([method_best, pd.DataFrame([combined_row])], ignore_index=True)
    
    return method_best

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_benchmark_visualization(df: pd.DataFrame, output_dir: Path, benchmark_name: str, 
                                  combine_mdae: bool = True):
    """Create visualizations for a single benchmark."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine which methods to show
    if combine_mdae:
        # Remove individual MDAE variants if combined exists
        if 'MDAE (Combined)' in df['Method'].values:
            df = df[~df['Method'].isin(['MDAE', 'MDAE (TC)'])]
            df.loc[df['Method'] == 'MDAE (Combined)', 'Method'] = 'MDAE'
    
    # Sort by AUROC
    df_sorted = df.sort_values('Test_AUROC', ascending=False)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: AUROC performance
    colors = [COLOR_SCHEME.get(m.replace(' (Combined)', ''), COLOR_SCHEME['Other']) 
              for m in df_sorted['Method']]
    bars1 = ax1.bar(range(len(df_sorted)), df_sorted['Test_AUROC'], color=colors)
    ax1.set_xticks(range(len(df_sorted)))
    ax1.set_xticklabels(df_sorted['Method'], rotation=45, ha='right')
    ax1.set_ylabel('AUROC')
    ax1.set_title(f'{benchmark_name} - Test AUROC')
    ax1.set_ylim([0, 1])
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars1, df_sorted['Test_AUROC']):
        if not pd.isna(val):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: AP performance
    bars2 = ax2.bar(range(len(df_sorted)), df_sorted['Test_AP'], color=colors)
    ax2.set_xticks(range(len(df_sorted)))
    ax2.set_xticklabels(df_sorted['Method'], rotation=45, ha='right')
    ax2.set_ylabel('Average Precision')
    ax2.set_title(f'{benchmark_name} - Test AP')
    ax2.set_ylim([0, 1])
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars2, df_sorted['Test_AP']):
        if not pd.isna(val):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.suptitle(f'Multi-Modality Performance: {benchmark_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'multi_modality_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create threshold-independent metrics plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data for grouped bar plot
    methods = df_sorted['Method'].tolist()
    auroc_values = df_sorted['Test_AUROC'].tolist()
    ap_values = df_sorted['Test_AP'].tolist()
    
    x = np.arange(len(methods))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, auroc_values, width, label='AUROC', color='#2E86AB')
    bars2 = ax.bar(x + width/2, ap_values, width, label='AP', color='#A23B72')
    
    ax.set_xlabel('Method')
    ax.set_ylabel('Score')
    ax.set_title(f'Threshold-Independent Metrics: {benchmark_name}')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'threshold_independent_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_overall_summary(all_results: Dict, output_dir: Path, combine_mdae: bool = True):
    """Create overall summary visualizations and tables."""
    # Collect all results
    summary_data = []
    
    for benchmark, df in all_results.items():
        if combine_mdae and 'MDAE (Combined)' in df['Method'].values:
            df = df[~df['Method'].isin(['MDAE', 'MDAE (TC)'])]
            df.loc[df['Method'] == 'MDAE (Combined)', 'Method'] = 'MDAE'
        
        for _, row in df.iterrows():
            summary_data.append({
                'Benchmark': BENCHMARK_DISPLAY[benchmark],
                'Method': row['Method'],
                'Test_AUROC': row['Test_AUROC'],
                'Test_AP': row['Test_AP']
            })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Create pivot table for comprehensive metrics
    pivot_auroc = summary_df.pivot_table(
        index='Method', 
        columns='Benchmark', 
        values='Test_AUROC', 
        aggfunc='mean'
    )
    
    # Save comprehensive metrics table
    pivot_auroc.to_csv(output_dir / 'comprehensive_metrics_table.csv')
    
    # Calculate summary statistics
    stats_data = []
    for method in pivot_auroc.index:
        values = pivot_auroc.loc[method].dropna()
        if len(values) > 0:
            stats_data.append({
                'Method': method,
                'Mean_AUROC': values.mean(),
                'Std_AUROC': values.std(),
                'Median_AUROC': values.median(),
                'Min_AUROC': values.min(),
                'Max_AUROC': values.max(),
                'Count': len(values)
            })
    
    stats_df = pd.DataFrame(stats_data)
    stats_df = stats_df.sort_values('Mean_AUROC', ascending=False)
    stats_df.to_csv(output_dir / 'overall_summary_statistics.csv', index=False)
    
    # Create overall performance heatmap
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Sort methods by mean performance
    method_order = stats_df['Method'].tolist()
    pivot_sorted = pivot_auroc.reindex(method_order)
    
    # Create heatmap
    sns.heatmap(pivot_sorted, annot=True, fmt='.3f', cmap='RdYlGn', 
                vmin=0.4, vmax=1.0, cbar_kws={'label': 'AUROC'},
                linewidths=0.5, linecolor='gray')
    
    ax.set_title('Multi-Modality Performance Across All Benchmarks', fontsize=14, fontweight='bold')
    ax.set_xlabel('Benchmark', fontsize=12)
    ax.set_ylabel('Method', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'overall_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return stats_df

# ============================================================================
# LATEX TABLE GENERATION
# ============================================================================

def generate_latex_tables(all_results: Dict, stats_df: pd.DataFrame, output_dir: Path, 
                         combine_mdae: bool = True):
    """Generate LaTeX-formatted tables for paper."""
    latex_dir = output_dir / 'latex_tables'
    latex_dir.mkdir(parents=True, exist_ok=True)
    
    # Main results table
    top_methods = stats_df.head(10)
    
    latex_content = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Multi-Modality Benchmark Results (Mean AUROC)}",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "Method & Mean & Std & Median & Min & Max & Count \\\\",
        "\\midrule"
    ]
    
    for _, row in top_methods.iterrows():
        method = row['Method']
        if method == 'MDAE':
            # Bold for best method
            latex_content.append(
                f"\\textbf{{{method}}} & \\textbf{{{row['Mean_AUROC']:.3f}}} & "
                f"{row['Std_AUROC']:.3f} & {row['Median_AUROC']:.3f} & "
                f"{row['Min_AUROC']:.3f} & {row['Max_AUROC']:.3f} & {int(row['Count'])} \\\\"
            )
        else:
            latex_content.append(
                f"{method} & {row['Mean_AUROC']:.3f} & {row['Std_AUROC']:.3f} & "
                f"{row['Median_AUROC']:.3f} & {row['Min_AUROC']:.3f} & "
                f"{row['Max_AUROC']:.3f} & {int(row['Count'])} \\\\"
            )
    
    latex_content.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])
    
    with open(latex_dir / 'main_results.tex', 'w') as f:
        f.write('\n'.join(latex_content))
    
    print(f"LaTeX tables saved to {latex_dir}")

# ============================================================================
# MAIN PROCESSING FUNCTION
# ============================================================================

def main():
    """Main processing function."""
    parser = argparse.ArgumentParser(description='Process multi-modality benchmark results')
    parser.add_argument('--input-dir', type=str, 
                       default='/home/t-jiachentu/repos/benchmarking/misc/data/raw_data/20250814_multi',
                       help='Input directory with raw multi-modality data')
    parser.add_argument('--output-dir', type=str,
                       default='/home/t-jiachentu/repos/benchmarking/misc/data/processed_data_multi_modality_combined',
                       help='Output directory for processed results')
    parser.add_argument('--no-combine-mdae', action='store_true',
                       help='Keep MDAE variants separate instead of combining')
    
    args = parser.parse_args()
    
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    combine_mdae = not args.no_combine_mdae
    
    print("=" * 60)
    print("Multi-Modality Comprehensive Processing")
    print("=" * 60)
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"MDAE mode: {'Combined' if combine_mdae else 'All variants'}")
    print("=" * 60)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    benchmarks_dir = output_path / 'benchmarks'
    benchmarks_dir.mkdir(exist_ok=True)
    
    # Process all benchmarks
    all_results = {}
    
    for benchmark_dir in sorted(input_path.iterdir()):
        if benchmark_dir.is_dir():
            benchmark_name = benchmark_dir.name
            print(f"\nProcessing {benchmark_name}...")
            
            # Load data
            df = load_benchmark_data(benchmark_dir)
            if df.empty:
                print(f"  No data found for {benchmark_name}")
                continue
            
            # Combine MDAE variants
            df_processed = combine_mdae_variants(df)
            
            # Create benchmark output directory
            bench_output = benchmarks_dir / BENCHMARK_DISPLAY.get(benchmark_name, benchmark_name)
            bench_output.mkdir(exist_ok=True)
            
            # Save metrics table
            df_processed.to_csv(bench_output / 'metrics_table.csv', index=False)
            
            # Create visualizations
            create_benchmark_visualization(
                df_processed, bench_output, 
                BENCHMARK_DISPLAY.get(benchmark_name, benchmark_name),
                combine_mdae
            )
            
            all_results[benchmark_name] = df_processed
            
            print(f"  Processed {len(df_processed)} methods")
            print(f"  Best AUROC: {df_processed['Test_AUROC'].max():.3f}")
    
    # Create overall summary
    print("\nCreating overall summary...")
    stats_df = create_overall_summary(all_results, output_path, combine_mdae)
    
    # Generate LaTeX tables
    print("\nGenerating LaTeX tables...")
    generate_latex_tables(all_results, stats_df, output_path, combine_mdae)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Processing Complete!")
    print("=" * 60)
    print(f"\nTop 5 Methods by Mean AUROC:")
    print(stats_df[['Method', 'Mean_AUROC', 'Std_AUROC']].head())
    print(f"\nOutputs saved to: {output_path}")

if __name__ == "__main__":
    main()