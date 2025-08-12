#!/usr/bin/env python3
"""
Fixed version: Process MDAE benchmark results with corrected pattern recognition
and create AUROC/AP focused visualizations.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import re

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

# FIXED METHOD PATTERNS - INCLUDING ALL MDAE VARIANTS (ResNet-based only)
METHOD_PATTERNS = {
    # Our Methods - FIXED to include all ResNet-based MDAE variants
    'MDAE': r'^(resenc_MDAETrainer_RandomMask_Flow_BS48_2000ep|resenc_MDAE_pretrained|resenc_MDAE_scratch)',
    'MDAE (TC)': r'^(resenc_time_conditioned|resenc_multimodal_mm_mdae)',
    
    # SSL Baselines (specific patterns before generic ones)
    'SimCLR': r'^resenc_SimCLR',
    'VoCo': r'^resenc_VoCo',
    'MG': r'^resenc_MG',
    'SwinUNETR': r'^resenc_SwinUNETR',
    'VF': r'^resenc_VF',
    'S3D': r'^resenc_S3D',
    'MAE': r'^resenc_pretrained_',  # Most generic pattern last
    
    # Foundation Models
    'BrainIAC': r'^brainiac_pretrained',
    'MRI-Core': r'^mri_core',
    'BrainMVP': r'^brainmvp',
    
    # Other Baselines
    'DinoV2': r'^dinov2',
    'ResNet-50': r'^brainiac_scratch',
}

# Modality mappings
MODALITY_NAMES = {
    'T1': 'T1-weighted',
    'T1c': 'T1-contrast',
    'T1C': 'T1-contrast',
    'T1n': 'T1-native',
    'T1N': 'T1-native',
    'T2': 'T2-weighted',
    'T2f': 'T2-FLAIR',
    'T2w': 'T2-weighted',
    'FLAIR': 'T2-FLAIR',
    'ASL': 'ASL',
    'SWI': 'SWI',
    'T1GD': 'T1-Gd',
    'T1W': 'T1-weighted',
    'T1WCE': 'T1-contrast',
    'MIXED_CONTRASTS': 'Mixed'
}

# Benchmark info
BENCHMARK_INFO = {
    'brats18_lgg_vs_hgg': 'BraTS18 LGG vs HGG',
    'brats23_gli_vs_men': 'BraTS23 Glioma vs Meningioma',
    'brats23_gli_vs_met': 'BraTS23 Glioma vs Metastasis',
    'brats23_men_vs_met': 'BraTS23 Meningioma vs Metastasis',
    'rsna_miccai_mgmt_methylation': 'MGMT Methylation',
    'tcga_gbm_dss_1year': 'TCGA-GBM DSS 1-Year',
    'tcga_gbm_pfi_1year': 'TCGA-GBM PFI 1-Year',
    'ucsf_pdgm_idh_classification': 'IDH Classification',
    'upenn_gbm_age_group': 'Age Group',
    'upenn_gbm_gender': 'Gender',
    'upenn_gbm_gtr_status': 'GTR Status',
    'upenn_gbm_idh1_status': 'IDH1 Status',
    'upenn_gbm_survival_18month': 'Survival 18-Month',
    'upenn_gbm_survival_1year': 'Survival 1-Year',
    'upenn_gbm_survival_2year': 'Survival 2-Year',
}


def identify_method(run_name: str) -> str:
    """Identify which method a run belongs to."""
    for method, pattern in METHOD_PATTERNS.items():
        if re.match(pattern, run_name):
            return method
    return 'Unknown'


def normalize_modality(modality: str) -> str:
    """Normalize modality names."""
    if pd.isna(modality):
        return 'Unknown'
    
    modality = str(modality).upper()
    
    # Normalize variations
    if modality == 'T2W':
        return 'T2'
    elif modality in ['T2F', 'FLAIR']:
        return 'T2f'
    elif modality in ['T1CE', 'T1WCE']:
        return 'T1C'
    elif modality == 'T1GD':
        return 'T1GD'
    
    return modality


def process_benchmark_modality(benchmark_name: str, modality: str, df: pd.DataFrame) -> Dict:
    """Process data for a specific modality within a benchmark."""
    
    # Filter for specific modality
    modality_df = df[df['modality_normalized'] == modality].copy()
    
    if modality_df.empty:
        return None
    
    # Get best run for each method (focus on AUROC and AP)
    metrics = ['Test_AUROC', 'Test_AP', 'Test_F1', 'Test_Balanced_Accuracy']
    best_runs = []
    
    for method in modality_df['Method'].unique():
        method_df = modality_df[modality_df['Method'] == method]
        if not method_df.empty and not method_df['Test_AUROC'].isna().all():
            best_idx = method_df['Test_AUROC'].idxmax()
            best_run = method_df.loc[best_idx]
            best_runs.append({
                'Method': method,
                'Test_AUROC': best_run['Test_AUROC'],
                'Test_AP': best_run['Test_AP'],
                'Test_F1': best_run['Test_F1'],
                'Test_Balanced_Accuracy': best_run['Test_Balanced_Accuracy'],
                'run_name': best_run['run_name']
            })
    
    if not best_runs:
        return None
    
    results_df = pd.DataFrame(best_runs)
    
    # Combine MDAE variants
    mdae_combined = {'Method': 'MDAE (Combined)'}
    mdae_row = results_df[results_df['Method'] == 'MDAE'].iloc[0] if 'MDAE' in results_df['Method'].values else None
    mdae_tc_row = results_df[results_df['Method'] == 'MDAE (TC)'].iloc[0] if 'MDAE (TC)' in results_df['Method'].values else None
    
    if mdae_row is not None and mdae_tc_row is not None:
        if mdae_row['Test_AUROC'] >= mdae_tc_row['Test_AUROC']:
            for metric in metrics:
                mdae_combined[metric] = mdae_row[metric]
            mdae_combined['Best_Variant'] = 'MDAE'
            mdae_combined['run_name'] = mdae_row['run_name']
        else:
            for metric in metrics:
                mdae_combined[metric] = mdae_tc_row[metric]
            mdae_combined['Best_Variant'] = 'MDAE (TC)'
            mdae_combined['run_name'] = mdae_tc_row['run_name']
    elif mdae_row is not None:
        for metric in metrics:
            mdae_combined[metric] = mdae_row[metric]
        mdae_combined['Best_Variant'] = 'MDAE'
        mdae_combined['run_name'] = mdae_row['run_name']
    elif mdae_tc_row is not None:
        for metric in metrics:
            mdae_combined[metric] = mdae_tc_row[metric]
        mdae_combined['Best_Variant'] = 'MDAE (TC)'
        mdae_combined['run_name'] = mdae_tc_row['run_name']
    
    if 'Best_Variant' in mdae_combined:
        results_df = pd.concat([results_df, pd.DataFrame([mdae_combined])], ignore_index=True)
    
    # Sort by Test_AUROC
    results_df = results_df.sort_values('Test_AUROC', ascending=False)
    
    return results_df


def create_auroc_ap_visualizations(results_df: pd.DataFrame, modality: str, 
                                   benchmark_name: str, output_dir: Path):
    """Create AUROC and AP focused visualizations."""
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Define colors
    def get_color(method):
        if 'MDAE' in method:
            return '#2E7D32'  # Green
        elif method in ['MAE', 'SimCLR', 'VoCo', 'MG', 'SwinUNETR', 'VF', 'S3D']:
            return '#1976D2'  # Blue
        elif method in ['BrainIAC', 'MRI-Core', 'BrainMVP']:
            return '#7B1FA2'  # Purple
        else:
            return '#757575'  # Gray
    
    # Plot 1: AUROC Rankings
    ax1 = axes[0]
    auroc_sorted = results_df.sort_values('Test_AUROC', ascending=False)
    colors = [get_color(m) for m in auroc_sorted['Method']]
    
    bars = ax1.barh(range(len(auroc_sorted)), auroc_sorted['Test_AUROC'], color=colors, alpha=0.8)
    ax1.set_yticks(range(len(auroc_sorted)))
    ax1.set_yticklabels(auroc_sorted['Method'])
    ax1.set_xlabel('Test AUROC')
    ax1.set_title(f'AUROC Rankings - {MODALITY_NAMES.get(modality, modality)}')
    ax1.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Random')
    
    # Add value labels and rank
    for i, (bar, val) in enumerate(zip(bars, auroc_sorted['Test_AUROC'])):
        if not pd.isna(val):
            ax1.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'#{i+1}: {val:.3f}', ha='left', va='center', fontsize=9)
    
    ax1.set_xlim([0.3, 1.05])
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Plot 2: AP Rankings
    ax2 = axes[1]
    # Filter out NaN AP values
    ap_data = results_df.dropna(subset=['Test_AP']).sort_values('Test_AP', ascending=False)
    
    if not ap_data.empty:
        colors = [get_color(m) for m in ap_data['Method']]
        bars = ax2.barh(range(len(ap_data)), ap_data['Test_AP'], color=colors, alpha=0.8)
        ax2.set_yticks(range(len(ap_data)))
        ax2.set_yticklabels(ap_data['Method'])
        ax2.set_xlabel('Test AP (Average Precision)')
        ax2.set_title(f'AP Rankings - {MODALITY_NAMES.get(modality, modality)}')
        ax2.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Random')
        
        # Add value labels and rank
        for i, (bar, val) in enumerate(zip(bars, ap_data['Test_AP'])):
            if not pd.isna(val):
                ax2.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                        f'#{i+1}: {val:.3f}', ha='left', va='center', fontsize=9)
        
        ax2.set_xlim([0.3, 1.05])
        ax2.grid(True, alpha=0.3, axis='x')
    else:
        ax2.text(0.5, 0.5, 'No AP data available', ha='center', va='center', fontsize=12)
        ax2.set_title(f'AP Rankings - {MODALITY_NAMES.get(modality, modality)}')
    
    # Plot 3: AUROC vs AP Scatter
    ax3 = axes[2]
    
    # Filter data with both AUROC and AP
    scatter_data = results_df.dropna(subset=['Test_AUROC', 'Test_AP'])
    
    if not scatter_data.empty:
        for _, row in scatter_data.iterrows():
            color = get_color(row['Method'])
            ax3.scatter(row['Test_AUROC'], row['Test_AP'], 
                       color=color, s=100, alpha=0.7, edgecolors='black', linewidth=1)
            
            # Add labels for MDAE methods
            if 'MDAE' in row['Method']:
                ax3.annotate(row['Method'], 
                           (row['Test_AUROC'], row['Test_AP']),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, fontweight='bold')
        
        # Add diagonal line (perfect correlation)
        min_val = min(scatter_data['Test_AUROC'].min(), scatter_data['Test_AP'].min())
        max_val = max(scatter_data['Test_AUROC'].max(), scatter_data['Test_AP'].max())
        ax3.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, label='y=x')
        
        # Calculate and display correlation
        if len(scatter_data) > 1:
            correlation = scatter_data['Test_AUROC'].corr(scatter_data['Test_AP'])
            ax3.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                    transform=ax3.transAxes, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax3.set_xlabel('Test AUROC')
        ax3.set_ylabel('Test AP')
        ax3.set_title(f'AUROC vs AP - {MODALITY_NAMES.get(modality, modality)}')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim([0.3, 1.0])
        ax3.set_ylim([0.3, 1.0])
    else:
        ax3.text(0.5, 0.5, 'Insufficient data for scatter plot', 
                ha='center', va='center', fontsize=12)
        ax3.set_title(f'AUROC vs AP - {MODALITY_NAMES.get(modality, modality)}')
    
    plt.suptitle(f"{BENCHMARK_INFO[benchmark_name]} - Threshold-Independent Metrics", 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save figure
    fig_path = output_dir / 'auroc_ap_analysis.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()


def process_benchmark_with_fixed_patterns(benchmark_name: str, data_dir: Path, output_base: Path):
    """Process benchmark with fixed MDAE pattern recognition."""
    
    # Load raw data
    csv_path = data_dir / benchmark_name / 'runs_summary.csv'
    if not csv_path.exists():
        print(f"Warning: {csv_path} not found")
        return None
    
    df = pd.read_csv(csv_path)
    
    # Add method identification with FIXED patterns
    df['Method'] = df['run_name'].apply(identify_method)
    df = df[df['Method'] != 'Unknown']
    
    # Normalize modality names
    df['modality_normalized'] = df['modality'].apply(normalize_modality)
    
    # Get unique modalities
    modalities = df['modality_normalized'].unique()
    print(f"  Found modalities: {', '.join(sorted(modalities))}")
    
    # Create directories
    benchmark_dir = output_base / benchmark_name
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    modalities_dir = benchmark_dir / 'modalities'
    modalities_dir.mkdir(exist_ok=True)
    
    # Process each modality
    modality_results = {}
    
    for modality in sorted(modalities):
        if modality == 'Unknown':
            continue
            
        modality_dir = modalities_dir / modality
        modality_dir.mkdir(exist_ok=True)
        
        # Process modality data
        results_df = process_benchmark_modality(benchmark_name, modality, df)
        
        if results_df is not None and not results_df.empty:
            modality_results[modality] = results_df
            
            # Save metrics
            metrics_path = modality_dir / 'metrics.csv'
            results_df.to_csv(metrics_path, index=False)
            
            # Save threshold-independent metrics only
            ti_metrics = results_df[['Method', 'Test_AUROC', 'Test_AP', 'run_name']].copy()
            ti_path = modality_dir / 'threshold_independent_metrics.csv'
            ti_metrics.to_csv(ti_path, index=False)
            
            # Create AUROC/AP visualizations
            create_auroc_ap_visualizations(results_df, modality, benchmark_name, modality_dir)
            
            # Print MDAE performance
            mdae_data = results_df[results_df['Method'] == 'MDAE (Combined)']
            if not mdae_data.empty:
                mdae_row = mdae_data.iloc[0]
                mdae_rank = (results_df['Test_AUROC'] > mdae_row['Test_AUROC']).sum() + 1
                ap_str = f"{mdae_row['Test_AP']:.3f}" if not pd.isna(mdae_row['Test_AP']) else "N/A"
                print(f"    - {modality}: MDAE rank #{mdae_rank}, "
                      f"AUROC={mdae_row['Test_AUROC']:.3f}, "
                      f"AP={ap_str}")
                
                # Special check for the ASL modality in IDH classification
                if benchmark_name == 'ucsf_pdgm_idh_classification' and modality == 'ASL':
                    print(f"      [VERIFICATION] MDAE run: {mdae_row.get('run_name', 'Unknown')}")
    
    return modality_results


def main():
    """Main processing function with fixed patterns."""
    
    # Setup paths
    data_dir = Path('/home/t-jiachentu/repos/benchmarking/misc/data/raw_data/20250811')
    output_base = Path('/home/t-jiachentu/repos/benchmarking/misc/data/processed_data/benchmark_results_fixed')
    output_base.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("PROCESSING WITH FIXED MDAE PATTERNS")
    print("="*60)
    print("\nPattern Updates Applied:")
    print("  - MDAE now includes: resenc_MDAE_pretrained, resenc_MDAE_scratch")
    print("  - MDAE (TC) includes: time_conditioned, multimodal_mm_mdae")
    print("  - Focus on ResNet-based models only")
    print()
    
    # Process key benchmarks first to verify fix
    test_benchmarks = ['ucsf_pdgm_idh_classification', 'brats23_gli_vs_met']
    
    for benchmark_name in test_benchmarks:
        print(f"\nProcessing {benchmark_name} (verification)...")
        process_benchmark_with_fixed_patterns(benchmark_name, data_dir, output_base)
    
    # Process all remaining benchmarks
    all_benchmarks = list(BENCHMARK_INFO.keys())
    remaining = [b for b in all_benchmarks if b not in test_benchmarks]
    
    for benchmark_name in remaining:
        print(f"\nProcessing {benchmark_name}...")
        process_benchmark_with_fixed_patterns(benchmark_name, data_dir, output_base)
    
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {output_base}")
    print("\nKey improvements with fixed patterns:")
    print("  - MDAE variants properly recognized")
    print("  - AUROC/AP focused visualizations created")
    print("  - Threshold-independent metrics separated")


if __name__ == "__main__":
    main()