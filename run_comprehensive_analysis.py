#!/usr/bin/env python3
"""
Enhanced Comprehensive Analysis Pipeline for MDAE Benchmarking
===============================================================
This script processes raw WandB data with improved modality handling,
MDAE combination logic, and visualization highlighting.

Key improvements:
- Fixes T2F/FLAIR duplication issue
- Creates MDAE (Combined) from best of MDAE and MDAE (TC)
- Highlights MDAE variants in visualizations
- Generates paper-ready outputs

Usage:
    python run_comprehensive_analysis_enhanced.py
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
import sys
from datetime import datetime
import re
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Default paths
DEFAULT_RAW_DATA_DIR = Path('/home/jtu9/Documents/MDAE/MDAE_data/raw_data/20250811')
DEFAULT_OUTPUT_DIR = Path('/home/jtu9/Documents/MDAE/MDAE_data/processed_data/comprehensive_analysis')

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

# Modality standardization mapping - comprehensive to avoid duplicates
MODALITY_MAPPING = {
    # T2F variations → FLAIR (critical for BraTS23)
    't2f': 'flair',
    'T2f': 'flair',
    'T2F': 'flair',
    
    # T1CE variations (critical for RSNA-MICCAI)
    't1wce': 't1ce',
    'T1wce': 't1ce', 
    'T1WCE': 't1ce',
    't1c': 't1ce',
    'T1c': 't1ce',
    'T1C': 't1ce',
    
    # T1 variations
    't1w': 't1',
    'T1w': 't1',
    'T1W': 't1',
    
    # T2 variations
    't2w': 't2',
    'T2w': 't2',
    'T2W': 't2',
    
    # Standard names (keep as-is)
    'FLAIR': 'flair',
    'flair': 'flair',
    'Flair': 'flair',
    't1': 't1',
    'T1': 't1',
    't2': 't2',
    'T2': 't2',
    't1ce': 't1ce',
    'T1ce': 't1ce',
    'T1CE': 't1ce',
    't1n': 't1n',
    'T1n': 't1n',
    'T1N': 't1n',
    't1gd': 't1gd',
    'T1gd': 't1gd',
    'T1GD': 't1gd',
    'ASL': 'asl',
    'asl': 'asl',
    'SWI': 'swi',
    'swi': 'swi',
    'mixed_contrasts': 'mixed_contrasts',
    'MIXED_CONTRASTS': 'mixed_contrasts'
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

# MDAE variants for highlighting
MDAE_VARIANTS = ['MDAE', 'MDAE (TC)', 'MDAE (Combined)']

# Color scheme for visualizations
COLOR_SCHEME = {
    'MDAE': '#FF6B6B',           # Red
    'MDAE (TC)': '#4ECDC4',       # Teal
    'MDAE (Combined)': '#45B7D1', # Blue
    # Other methods will use grey scale
}

# Metrics to extract
METRICS = ['Test_AUROC', 'Test_AP', 'Test_F1', 'Test_Balanced_Accuracy']
THRESHOLD_INDEPENDENT = ['Test_AUROC', 'Test_AP']
THRESHOLD_DEPENDENT = ['Test_F1', 'Test_Balanced_Accuracy']

# Visualization settings
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'sans-serif'
sns.set_style("whitegrid")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def standardize_modality(modality: str) -> str:
    """Standardize modality names to lowercase and handle mapping."""
    modality_lower = modality.lower()
    return MODALITY_MAPPING.get(modality, MODALITY_MAPPING.get(modality_lower, modality_lower))

def identify_method(run_name: str) -> Optional[str]:
    """Identify method from run name using patterns."""
    for method, pattern in METHOD_PATTERNS.items():
        if re.match(pattern, run_name):
            return method
    return None

def load_benchmark_data(benchmark_dir: Path, verbose: bool = False) -> pd.DataFrame:
    """Load and process data for a single benchmark with modality standardization."""
    csv_path = benchmark_dir / 'runs_summary.csv'
    
    if not csv_path.exists():
        if verbose:
            print(f"  Warning: {csv_path} not found")
        return pd.DataFrame()
    
    df = pd.read_csv(csv_path)
    
    # Extract method from run_name
    df['Method'] = df['run_name'].apply(identify_method)
    
    # Standardize modality names
    if 'modality' in df.columns:
        df['Modality'] = df['modality'].apply(standardize_modality)
    else:
        df['Modality'] = 'unknown'
    
    # Filter valid methods only
    df = df[df['Method'].notna()]
    
    # For multiple runs of same method/modality, select best by Test_AUROC
    df = df.sort_values('Test_AUROC', ascending=False).groupby(['Method', 'Modality']).first().reset_index()
    
    return df

def create_mdae_combined(df: pd.DataFrame) -> pd.DataFrame:
    """Create MDAE (Combined) by taking best of MDAE and MDAE (TC) for each modality."""
    combined_rows = []
    
    # Get unique modalities
    modalities = df['Modality'].unique()
    
    for modality in modalities:
        mod_df = df[df['Modality'] == modality]
        
        # Get MDAE and MDAE (TC) results for this modality
        mdae_row = mod_df[mod_df['Method'] == 'MDAE']
        mdae_tc_row = mod_df[mod_df['Method'] == 'MDAE (TC)']
        
        if not mdae_row.empty or not mdae_tc_row.empty:
            # Create combined row
            if mdae_row.empty:
                combined = mdae_tc_row.copy()
            elif mdae_tc_row.empty:
                combined = mdae_row.copy()
            else:
                # Take best AUROC
                if mdae_row.iloc[0]['Test_AUROC'] >= mdae_tc_row.iloc[0]['Test_AUROC']:
                    combined = mdae_row.copy()
                else:
                    combined = mdae_tc_row.copy()
            
            combined['Method'] = 'MDAE (Combined)'
            combined_rows.append(combined)
    
    if combined_rows:
        combined_df = pd.concat(combined_rows, ignore_index=True)
        df = pd.concat([df, combined_df], ignore_index=True)
    
    return df

def detect_and_merge_duplicates(df: pd.DataFrame, benchmark_name: str, verbose: bool = False) -> pd.DataFrame:
    """Detect and merge duplicate modalities after standardization."""
    duplicates_found = []
    rows_before = len(df)
    
    # After standardization, check if we have multiple entries for the same method/modality combination
    for method in df['Method'].unique():
        method_df = df[df['Method'] == method]
        mod_counts = method_df['Modality'].value_counts()
        
        for modality, count in mod_counts.items():
            if count > 1:
                duplicates_found.append(f"{method} has {count} entries for {modality}")
                # Keep only the best one by Test_AUROC
                best_idx = method_df[method_df['Modality'] == modality].nlargest(1, 'Test_AUROC').index
                drop_idx = method_df[(method_df['Modality'] == modality) & (~method_df.index.isin(best_idx))].index
                df = df.drop(drop_idx)
    
    rows_after = len(df)
    
    if duplicates_found and verbose:
        print(f"\n  Duplicates detected and merged in {benchmark_name}:")
        for dup in duplicates_found:
            print(f"    - {dup}")
        print(f"    Rows: {rows_before} → {rows_after} (removed {rows_before - rows_after} duplicates)")
    elif verbose and rows_before != rows_after:
        print(f"  Removed {rows_before - rows_after} duplicate entries")
    
    return df

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_modality_visualizations(df: pd.DataFrame, output_dir: Path, modality: str, benchmark_name: str):
    """Create visualizations for a single modality with MDAE highlighting."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sort by Test_AUROC
    df = df.sort_values('Test_AUROC', ascending=False)
    
    # Save metrics table
    df[['Method'] + METRICS].to_csv(output_dir / 'metrics_table.csv', index=False)
    
    # Create color palette
    colors = []
    for method in df['Method']:
        if method in COLOR_SCHEME:
            colors.append(COLOR_SCHEME[method])
        else:
            colors.append('#95A5A6')  # Grey for other methods
    
    # Create threshold-independent metrics plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # AUROC plot
    bars1 = ax1.barh(df['Method'], df['Test_AUROC'], color=colors)
    ax1.set_xlabel('Test AUROC', fontsize=12, fontweight='bold')
    ax1.set_title(f'{benchmark_name} - {modality.upper()}\nTest AUROC', fontsize=14, fontweight='bold')
    ax1.set_xlim([0, 1])
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars1, df['Test_AUROC'])):
        if not pd.isna(val):
            ax1.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{val:.3f}', va='center', fontsize=10)
    
    # Highlight MDAE variants
    for i, method in enumerate(df['Method']):
        if method in MDAE_VARIANTS:
            bars1[i].set_edgecolor('black')
            bars1[i].set_linewidth(2)
    
    # AP plot
    bars2 = ax2.barh(df['Method'], df['Test_AP'], color=colors)
    ax2.set_xlabel('Test AP', fontsize=12, fontweight='bold')
    ax2.set_title(f'{benchmark_name} - {modality.upper()}\nTest Average Precision', fontsize=14, fontweight='bold')
    ax2.set_xlim([0, 1])
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars2, df['Test_AP'])):
        if not pd.isna(val):
            ax2.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{val:.3f}', va='center', fontsize=10)
    
    # Highlight MDAE variants
    for i, method in enumerate(df['Method']):
        if method in MDAE_VARIANTS:
            bars2[i].set_edgecolor('black')
            bars2[i].set_linewidth(2)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'threshold_independent_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create all metrics plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for idx, metric in enumerate(METRICS):
        ax = axes[idx // 2, idx % 2]
        bars = ax.barh(df['Method'], df[metric], color=colors)
        ax.set_xlabel(metric.replace('_', ' '), fontsize=11, fontweight='bold')
        ax.set_title(f'{metric.replace("_", " ")}', fontsize=12, fontweight='bold')
        ax.set_xlim([0, 1])
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, df[metric])):
            if not pd.isna(val):
                ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                       f'{val:.3f}', va='center', fontsize=9)
        
        # Highlight MDAE variants
        for i, method in enumerate(df['Method']):
            if method in MDAE_VARIANTS:
                bars[i].set_edgecolor('black')
                bars[i].set_linewidth(2)
    
    fig.suptitle(f'{benchmark_name} - {modality.upper()}\nAll Metrics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'all_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_cross_modality_comparison(results: Dict, output_dir: Path, benchmark_name: str):
    """Create cross-modality comparison with MDAE highlighting."""
    # Prepare data for comparison
    all_methods = set()
    for modality_df in results.values():
        all_methods.update(modality_df['Method'].unique())
    
    all_methods = sorted(list(all_methods))
    modalities = sorted(results.keys())
    
    # Create matrix for AUROC
    auroc_matrix = pd.DataFrame(index=all_methods, columns=modalities)
    
    for modality, df in results.items():
        for _, row in df.iterrows():
            auroc_matrix.loc[row['Method'], modality] = row['Test_AUROC']
    
    # Save cross-modality table
    auroc_matrix.to_csv(output_dir / 'cross_modality_table.csv')
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(max(10, len(modalities)), max(8, len(all_methods) * 0.4)))
    
    # Convert to numeric for plotting
    plot_data = auroc_matrix.astype(float)
    
    # Create custom colormap
    sns.heatmap(plot_data, annot=True, fmt='.3f', cmap='RdYlGn', 
                vmin=0.5, vmax=1.0, ax=ax, cbar_kws={'label': 'Test AUROC'})
    
    # Highlight MDAE variants with bold labels
    y_labels = []
    for method in all_methods:
        if method in MDAE_VARIANTS:
            y_labels.append(f'**{method}**')
        else:
            y_labels.append(method)
    
    ax.set_yticklabels(y_labels, rotation=0)
    ax.set_xticklabels(modalities, rotation=45, ha='right')
    ax.set_title(f'{benchmark_name}\nCross-Modality AUROC Comparison', fontsize=14, fontweight='bold')
    
    # Add border for MDAE variants
    for i, method in enumerate(all_methods):
        if method in MDAE_VARIANTS:
            for j in range(len(modalities)):
                rect = plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='black', lw=2)
                ax.add_patch(rect)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cross_modality_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

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
    
    # Detect and merge duplicates
    df = detect_and_merge_duplicates(df, benchmark_name, verbose)
    
    # Create MDAE (Combined)
    df = create_mdae_combined(df)
    
    # Get display name
    display_name = BENCHMARK_MAPPING.get(benchmark_name, benchmark_name)
    
    # Get available modalities
    modalities = df['Modality'].unique()
    valid_modalities = [m for m in modalities if m not in ['multimodal', 'unknown']]
    if not valid_modalities and 'unknown' in modalities:
        valid_modalities = ['unknown']
    
    if verbose:
        print(f"  Found {len(valid_modalities)} modalities: {', '.join(valid_modalities)}")
        # Check for MDAE variants
        mdae_methods = df[df['Method'].isin(MDAE_VARIANTS)]['Method'].unique()
        print(f"  MDAE variants present: {', '.join(mdae_methods)}")
    
    # Create output directory structure
    bench_output = output_dir / 'benchmarks' / display_name.replace(':', '').replace(' ', '_')
    bench_output.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Process each valid modality
    for modality in valid_modalities:
        mod_df = df[df['Modality'] == modality].copy()
        if mod_df.empty:
            continue
            
        mod_output = bench_output / modality
        mod_output.mkdir(exist_ok=True)
        
        # Create visualizations
        create_modality_visualizations(mod_df, mod_output, modality, display_name)
        
        results[modality] = mod_df
    
    # Create cross-modality comparison if multiple modalities exist
    if len(results) > 1:
        create_cross_modality_comparison(results, bench_output, display_name)
    
    return results

def create_overall_summary(all_results: Dict, output_dir: Path, verbose: bool = False):
    """Create overall summary statistics and visualizations."""
    if verbose:
        print("\nCreating overall summary...")
    
    # Collect all results
    all_data = []
    for benchmark_name, benchmark_results in all_results.items():
        display_name = BENCHMARK_MAPPING.get(benchmark_name, benchmark_name)
        for modality, df in benchmark_results.items():
            for _, row in df.iterrows():
                data_point = {
                    'Benchmark': display_name,
                    'Modality': modality,
                    'Method': row['Method'],
                    'Test_AUROC': row['Test_AUROC'],
                    'Test_AP': row.get('Test_AP', np.nan)
                }
                all_data.append(data_point)
    
    df_all = pd.DataFrame(all_data)
    
    # Create comprehensive metrics table
    pivot_auroc = df_all.pivot_table(index='Method', columns='Benchmark', values='Test_AUROC', aggfunc='mean')
    pivot_auroc.to_csv(output_dir / 'comprehensive_metrics_table.csv')
    
    # Calculate summary statistics
    summary_stats = []
    for method in pivot_auroc.index:
        values = pivot_auroc.loc[method].dropna()
        stats = {
            'Method': method,
            'Mean_AUROC': values.mean(),
            'Std_AUROC': values.std(),
            'Median_AUROC': values.median(),
            'Min_AUROC': values.min(),
            'Max_AUROC': values.max(),
            'Count': len(values)
        }
        summary_stats.append(stats)
    
    stats_df = pd.DataFrame(summary_stats)
    stats_df = stats_df.sort_values('Mean_AUROC', ascending=False)
    stats_df.to_csv(output_dir / 'overall_summary_statistics.csv', index=False)
    
    # Create overall performance plot with MDAE highlighting
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Prepare colors
    colors = []
    for method in stats_df['Method']:
        if method in COLOR_SCHEME:
            colors.append(COLOR_SCHEME[method])
        else:
            colors.append('#95A5A6')
    
    # Create bar plot
    bars = ax.barh(range(len(stats_df)), stats_df['Mean_AUROC'], color=colors)
    
    # Add error bars
    ax.errorbar(stats_df['Mean_AUROC'], range(len(stats_df)), 
                xerr=stats_df['Std_AUROC'], fmt='none', color='black', capsize=3)
    
    # Customize
    ax.set_yticks(range(len(stats_df)))
    ax.set_yticklabels(stats_df['Method'])
    ax.set_xlabel('Mean Test AUROC (± std)', fontsize=12, fontweight='bold')
    ax.set_title('Overall Performance Across All Benchmarks', fontsize=14, fontweight='bold')
    ax.set_xlim([0, 1])
    
    # Add value labels
    for i, (mean, std) in enumerate(zip(stats_df['Mean_AUROC'], stats_df['Std_AUROC'])):
        ax.text(mean + std + 0.01, i, f'{mean:.3f}±{std:.3f}', 
                va='center', fontsize=10)
    
    # Highlight MDAE variants
    for i, method in enumerate(stats_df['Method']):
        if method in MDAE_VARIANTS:
            bars[i].set_edgecolor('black')
            bars[i].set_linewidth(2)
            ax.get_yticklabels()[i].set_weight('bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'overall_performance.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    if verbose:
        print("\nTop 5 Methods by Mean AUROC:")
        print(stats_df.head()[['Method', 'Mean_AUROC', 'Std_AUROC']].to_string(index=False))

# ============================================================================
# MAIN PROCESSING FUNCTION
# ============================================================================

def main():
    """Main processing function."""
    parser = argparse.ArgumentParser(description='Enhanced MDAE Benchmarking Analysis')
    parser.add_argument('--input-dir', type=Path, default=DEFAULT_RAW_DATA_DIR,
                       help='Input directory with raw data')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR,
                       help='Output directory for processed results')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    print("="*70)
    print("ENHANCED MDAE COMPREHENSIVE ANALYSIS")
    print("="*70)
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print("="*70)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all benchmark directories
    benchmark_dirs = [d for d in args.input_dir.iterdir() if d.is_dir()]
    
    if not benchmark_dirs:
        print("No benchmark directories found!")
        return
    
    print(f"Found {len(benchmark_dirs)} benchmarks to process")
    
    # Process all benchmarks
    all_results = {}
    
    for benchmark_dir in tqdm(sorted(benchmark_dirs), desc="Processing benchmarks"):
        benchmark_name = benchmark_dir.name
        results = process_benchmark(benchmark_name, benchmark_dir, args.output_dir, args.verbose)
        if results:
            all_results[benchmark_name] = results
    
    # Create overall summary
    create_overall_summary(all_results, args.output_dir, args.verbose)
    
    print("\n" + "="*70)
    print("PROCESSING COMPLETE")
    print(f"Results saved to: {args.output_dir}")
    print("="*70)
    
    # Print summary of MDAE performance
    stats_path = args.output_dir / 'overall_summary_statistics.csv'
    if stats_path.exists():
        stats_df = pd.read_csv(stats_path)
        mdae_stats = stats_df[stats_df['Method'].isin(MDAE_VARIANTS)]
        if not mdae_stats.empty:
            print("\nMDAE Performance Summary:")
            for _, row in mdae_stats.iterrows():
                print(f"  {row['Method']}: {row['Mean_AUROC']:.3f} ± {row['Std_AUROC']:.3f}")

if __name__ == "__main__":
    main()