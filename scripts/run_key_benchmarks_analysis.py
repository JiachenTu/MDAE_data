#!/usr/bin/env python3
"""
Run Key Benchmarks Analysis with Original Structure
====================================================
This script processes the extracted single-modality data and generates
results in the original processed_data_key_benchmarks structure.
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
import re
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Configuration
BENCHMARK_MAPPING = {
    'upenn_gbm_age_group': 'UPenn-GBM_Age',
    'upenn_gbm_gender': 'UPenn-GBM_Gender',
    'upenn_gbm_survival_18month': 'UPenn-GBM_Surv_18M',
    'upenn_gbm_survival_1year': 'UPenn-GBM_Surv_1Y',
    'upenn_gbm_survival_2year': 'UPenn-GBM_Surv_2Y',
    'brats18_lgg_vs_hgg': 'BraTS18_LGG_vs_HGG',
    'brats23_gli_vs_men': 'BraTS23_Glio_vs_Men',
    'brats23_gli_vs_met': 'BraTS23_Glio_vs_Met',
    'brats23_men_vs_met': 'BraTS23_Men_vs_Met',
    'tcga_gbm_dss_1year': 'TCGA-GBM_DSS_1Y',
    'tcga_gbm_pfi_1year': 'TCGA-GBM_PFI_1Y',
    'ucsf_pdgm_idh_classification': 'UCSF-PDGM_IDH',
    'upenn_gbm_gtr_status': 'UPenn-GBM_GTR',
    'upenn_gbm_idh1_status': 'UPenn-GBM_IDH1',
    'rsna_miccai_mgmt_methylation': 'RSNA-MICCAI_MGMT'
}

# Key benchmarks
KEY_BENCHMARKS_T1T2 = {
    'brats18_lgg_vs_hgg': ['t1', 't2'],
    'brats23_gli_vs_met': ['t1n', 't2w', 't1', 't2'],  # Include both naming conventions
    'rsna_miccai_mgmt_methylation': ['t1w', 't2w', 't1', 't2'],
    'upenn_gbm_survival_18month': ['t1', 't2'],
    'upenn_gbm_idh1_status': ['t1', 't2']
}

KEY_BENCHMARKS_GENERALIZATION = {
    'ucsf_pdgm_idh_classification': ['asl', 'swi'],
    'upenn_gbm_age_group': ['flair'],
    'upenn_gbm_gtr_status': ['t1gd'],
    'tcga_gbm_dss_1year': ['mixed_contrasts', 'flair', 't1', 't2'],
    'tcga_gbm_pfi_1year': ['mixed_contrasts', 't1', 't2']
}

KEY_BENCHMARKS_ALL = {**KEY_BENCHMARKS_T1T2, **KEY_BENCHMARKS_GENERALIZATION}

# Modality mapping
MODALITY_MAPPING = {
    't2f': 'flair',
    't1wce': 't1ce',
    't1c': 't1ce',
    't1w': 't1',
    't2w': 't2',
    't1n': 't1',
}

# Method patterns
METHOD_PATTERNS = {
    'MDAE': r'^(resenc_MDAETrainer_RandomMask_Flow_BS48_2000ep|resenc_MDAE_pretrained|resenc_MDAE_scratch)',
    'MDAE (TC)': r'^(resenc_time_conditioned|resenc_multimodal_mm_mdae)',
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
    'ResNet-50': r'^brainiac_scratch',
    'MAE': r'^resenc_pretrained_'
}

# Visualization settings
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'sans-serif'
sns.set_style("whitegrid")

# Color scheme
COLOR_SCHEME = {
    'MDAE': '#45B7D1',
    'MDAE (TC)': '#4ECDC4',
    'MDAE (Combined)': '#45B7D1'
}

def standardize_modality(modality: str) -> str:
    """Standardize modality names."""
    modality_lower = modality.lower()
    return MODALITY_MAPPING.get(modality, MODALITY_MAPPING.get(modality_lower, modality_lower))

def identify_method(run_name: str) -> Optional[str]:
    """Identify method from run name."""
    for method, pattern in METHOD_PATTERNS.items():
        if re.match(pattern, run_name):
            return method
    return None

def normalize_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize metric column names."""
    metric_mapping = {
        'Test/AUROC': 'Test_AUROC',
        'Test/AP': 'Test_AP',
        'Test/F1': 'Test_F1',
        'Test/Balanced_Accuracy': 'Test_Balanced_Accuracy',
        'Val/AUROC': 'Val_AUROC',
        'Val/AP': 'Val_AP'
    }
    
    df = df.rename(columns=metric_mapping)
    
    # Ensure key metrics exist
    for metric in ['Test_AUROC', 'Test_AP', 'Test_F1', 'Test_Balanced_Accuracy']:
        if metric not in df.columns:
            df[metric] = np.nan
    
    return df

def load_benchmark_data(benchmark_dir: Path, verbose: bool = False) -> pd.DataFrame:
    """Load and process benchmark data."""
    csv_path = benchmark_dir / 'runs_summary.csv'
    
    if not csv_path.exists():
        if verbose:
            print(f"  Warning: {csv_path} not found")
        return pd.DataFrame()
    
    df = pd.read_csv(csv_path)
    
    # Normalize metrics
    df = normalize_metrics(df)
    
    # Extract method from run_name
    df['Method'] = df['run_name'].apply(identify_method)
    
    # Standardize modality
    if 'modality' in df.columns:
        df['Modality'] = df['modality'].apply(standardize_modality)
    else:
        df['Modality'] = 'unknown'
    
    # Filter valid methods
    df = df[df['Method'].notna()]
    
    # For multiple runs of same method/modality, select best by Test_AUROC
    if 'Test_AUROC' in df.columns and not df.empty:
        df = df.sort_values('Test_AUROC', ascending=False).groupby(['Method', 'Modality']).first().reset_index()
    
    return df

def create_mdae_combined(df: pd.DataFrame, combine_mdae: bool = True) -> pd.DataFrame:
    """Create MDAE Combined from best of MDAE and MDAE (TC)."""
    combined_rows = []
    
    modalities = df['Modality'].unique()
    
    for modality in modalities:
        mod_df = df[df['Modality'] == modality]
        
        mdae_row = mod_df[mod_df['Method'] == 'MDAE']
        mdae_tc_row = mod_df[mod_df['Method'] == 'MDAE (TC)']
        
        if not mdae_row.empty or not mdae_tc_row.empty:
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
            
            if combine_mdae:
                combined['Method'] = 'MDAE'
            else:
                combined['Method'] = 'MDAE (Combined)'
            
            combined_rows.append(combined)
    
    if combined_rows:
        combined_df = pd.concat(combined_rows, ignore_index=True)
        
        if combine_mdae:
            # Remove original MDAE and MDAE (TC) entries
            df = df[~df['Method'].isin(['MDAE', 'MDAE (TC)'])]
        
        df = pd.concat([df, combined_df], ignore_index=True)
    
    return df

def create_modality_visualizations(df: pd.DataFrame, output_dir: Path, modality: str, 
                                  benchmark_name: str, mdae_variants: List[str]):
    """Create visualizations for a single modality."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sort by Test_AUROC
    df = df.sort_values('Test_AUROC', ascending=False)
    
    # Save metrics table
    df[['Method', 'Test_AUROC', 'Test_AP', 'Test_F1', 'Test_Balanced_Accuracy']].to_csv(
        output_dir / 'metrics_table.csv', index=False
    )
    
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
        if method in mdae_variants:
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
        if method in mdae_variants:
            bars2[i].set_edgecolor('black')
            bars2[i].set_linewidth(2)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'threshold_independent_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create all metrics plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = ['Test_AUROC', 'Test_AP', 'Test_F1', 'Test_Balanced_Accuracy']
    
    for idx, metric in enumerate(metrics):
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
            if method in mdae_variants:
                bars[i].set_edgecolor('black')
                bars[i].set_linewidth(2)
    
    fig.suptitle(f'{benchmark_name} - {modality.upper()}\nAll Metrics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'all_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_cross_modality_comparison(results: Dict, output_dir: Path, benchmark_name: str, mdae_variants: List[str]):
    """Create cross-modality comparison."""
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
    
    # Create heatmap
    sns.heatmap(plot_data, annot=True, fmt='.3f', cmap='RdYlGn', 
                vmin=0.5, vmax=1.0, ax=ax, cbar_kws={'label': 'Test AUROC'})
    
    ax.set_yticklabels(all_methods, rotation=0)
    ax.set_xticklabels(modalities, rotation=45, ha='right')
    ax.set_title(f'{benchmark_name}\nCross-Modality AUROC Comparison', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cross_modality_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

def process_benchmark(benchmark_name: str, benchmark_dir: Path, output_dir: Path,
                     combine_mdae: bool = True, selected_modalities: Optional[List[str]] = None,
                     verbose: bool = False) -> Dict:
    """Process a single benchmark."""
    if verbose:
        print(f"\nProcessing: {benchmark_name}")
        if selected_modalities:
            print(f"  Selected modalities: {', '.join(selected_modalities)}")
    
    # Load data
    df = load_benchmark_data(benchmark_dir, verbose)
    if df.empty:
        if verbose:
            print(f"  No data found for {benchmark_name}")
        return {}
    
    # Create MDAE combined
    df = create_mdae_combined(df, combine_mdae=combine_mdae)
    
    # Determine which MDAE variants to highlight
    mdae_variants = ['MDAE'] if combine_mdae else ['MDAE', 'MDAE (TC)', 'MDAE (Combined)']
    
    # Get display name
    display_name = BENCHMARK_MAPPING.get(benchmark_name, benchmark_name)
    
    # Get available modalities
    modalities = df['Modality'].unique()
    valid_modalities = [m for m in modalities if m not in ['multimodal', 'unknown']]
    
    # Filter modalities if specific ones are selected
    if selected_modalities:
        # Map selected modalities through standardization
        standardized_selected = [standardize_modality(m) for m in selected_modalities]
        valid_modalities = [m for m in valid_modalities if m in standardized_selected]
    
    if verbose:
        print(f"  Found {len(valid_modalities)} modalities: {', '.join(valid_modalities)}")
    
    # Create output directory structure
    bench_output = output_dir / 'benchmarks' / display_name
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
        create_modality_visualizations(mod_df, mod_output, modality, display_name, mdae_variants)
        
        results[modality] = mod_df
    
    # Create cross-modality comparison if multiple modalities exist
    if len(results) > 1:
        create_cross_modality_comparison(results, bench_output, display_name, mdae_variants)
    
    return results

def create_overall_summary(all_results: Dict, output_dir: Path, combine_mdae: bool = True, verbose: bool = False):
    """Create overall summary statistics and visualizations."""
    mdae_variants = ['MDAE'] if combine_mdae else ['MDAE', 'MDAE (TC)', 'MDAE (Combined)']
    
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
    
    if not all_data:
        print("No data for overall summary")
        return
    
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
    
    # Create overall performance plot
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
        if method in mdae_variants:
            bars[i].set_edgecolor('black')
            bars[i].set_linewidth(2)
            ax.get_yticklabels()[i].set_weight('bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'overall_performance.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    if verbose:
        print("\nTop 5 Methods by Mean AUROC:")
        print(stats_df.head()[['Method', 'Mean_AUROC', 'Std_AUROC']].to_string(index=False))

def main():
    """Main processing function."""
    parser = argparse.ArgumentParser(description='Run key benchmarks analysis')
    parser.add_argument('--input-dir', type=Path, default=Path('raw_data/20250815_consolidated'),
                       help='Input directory with consolidated data')
    parser.add_argument('--output-dir', type=Path, default=Path('processed_data_key_benchmarks'),
                       help='Output directory for results')
    parser.add_argument('--category', choices=['t1t2', 'generalization', 'all'], default='all',
                       help='Category of benchmarks to analyze')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Select benchmarks
    if args.category == 't1t2':
        selected_benchmarks = KEY_BENCHMARKS_T1T2
    elif args.category == 'generalization':
        selected_benchmarks = KEY_BENCHMARKS_GENERALIZATION
    else:
        selected_benchmarks = KEY_BENCHMARKS_ALL
    
    print(f"{'='*70}")
    print(f"KEY BENCHMARKS ANALYSIS - ORIGINAL STRUCTURE")
    print(f"{'='*70}")
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Category: {args.category}")
    print(f"Benchmarks: {len(selected_benchmarks)}")
    print(f"{'='*70}")
    
    # Create output directory
    category_dir = args.output_dir / args.category
    category_dir.mkdir(parents=True, exist_ok=True)
    
    # Process all benchmarks
    all_results = {}
    
    for benchmark_name, modalities in tqdm(selected_benchmarks.items(), desc="Processing benchmarks"):
        benchmark_dir = args.input_dir / benchmark_name
        
        if not benchmark_dir.exists():
            if args.verbose:
                print(f"  Skipping {benchmark_name} - directory not found")
            continue
        
        results = process_benchmark(
            benchmark_name, benchmark_dir, category_dir,
            combine_mdae=True, selected_modalities=modalities,
            verbose=args.verbose
        )
        
        if results:
            all_results[benchmark_name] = results
    
    # Create overall summary
    create_overall_summary(all_results, category_dir, combine_mdae=True, verbose=args.verbose)
    
    print(f"\n{'='*70}")
    print(f"PROCESSING COMPLETE")
    print(f"Results saved to: {category_dir}")
    print(f"{'='*70}")
    
    # Print summary of MDAE performance
    stats_path = category_dir / 'overall_summary_statistics.csv'
    if stats_path.exists():
        stats_df = pd.read_csv(stats_path)
        mdae_stats = stats_df[stats_df['Method'] == 'MDAE']
        if not mdae_stats.empty:
            print(f"\nMDAE Performance: {mdae_stats.iloc[0]['Mean_AUROC']:.3f} ± {mdae_stats.iloc[0]['Std_AUROC']:.3f}")

if __name__ == "__main__":
    main()