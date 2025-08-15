#!/usr/bin/env python3
"""
Analyze Key Benchmarks for Single-Modality Data
================================================
This script analyzes the extracted single-modality data and generates
results for key benchmarks.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import re
import argparse
from tqdm import tqdm
import json

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11

# Benchmark display names
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

# Key benchmarks for analysis
KEY_BENCHMARKS_T1T2 = {
    'brats18_lgg_vs_hgg': ['t1', 't2'],
    'brats23_gli_vs_met': ['t1n', 't2w'],  # Use actual modality names
    'rsna_miccai_mgmt_methylation': ['t1w', 't2w'],
    'upenn_gbm_survival_18month': ['t1', 't2'],
    'upenn_gbm_idh1_status': ['t1', 't2']
}

KEY_BENCHMARKS_GENERALIZATION = {
    'ucsf_pdgm_idh_classification': ['asl', 'swi'],
    'upenn_gbm_age_group': ['flair'],
    'upenn_gbm_gtr_status': ['t1gd'],
    'tcga_gbm_dss_1year': ['flair'],  # Use available modality
    'tcga_gbm_pfi_1year': ['t1']  # Use available modality
}

KEY_BENCHMARKS_ALL = {**KEY_BENCHMARKS_T1T2, **KEY_BENCHMARKS_GENERALIZATION}

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

# MDAE color scheme
COLOR_SCHEME = {
    'MDAE': '#45B7D1',
    'MDAE (TC)': '#4ECDC4',
    'MDAE (Combined)': '#45B7D1'
}

def identify_method(run_name: str) -> Optional[str]:
    """Identify method from run name."""
    for method, pattern in METHOD_PATTERNS.items():
        if re.match(pattern, run_name):
            return method
    return None

def normalize_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize metric column names."""
    # Map old names to new names
    metric_mapping = {
        'Test/AUROC': 'Test_AUROC',
        'Test/AP': 'Test_AP',
        'Test/F1': 'Test_F1',
        'Test/Balanced_Accuracy': 'Test_Balanced_Accuracy',
        'Val/AUROC': 'Val_AUROC',
        'Val/AP': 'Val_AP',
        'Val/F1': 'Val_F1',
        'Val/Balanced_Accuracy': 'Val_Balanced_Accuracy'
    }
    
    # Rename columns
    df = df.rename(columns=metric_mapping)
    
    # Ensure key metrics exist
    for metric in ['Test_AUROC', 'Test_AP', 'Test_F1', 'Test_Balanced_Accuracy']:
        if metric not in df.columns:
            df[metric] = np.nan
    
    return df

def load_benchmark_data(benchmark_dir: Path, modalities: List[str]) -> pd.DataFrame:
    """Load and process benchmark data for specific modalities."""
    csv_path = benchmark_dir / 'runs_summary.csv'
    
    if not csv_path.exists():
        print(f"  Warning: {csv_path} not found")
        return pd.DataFrame()
    
    df = pd.read_csv(csv_path)
    
    # Normalize metrics
    df = normalize_metrics(df)
    
    # Identify methods
    df['Method'] = df['run_name'].apply(identify_method)
    
    # Filter valid methods
    df = df[df['Method'].notna()]
    
    # Filter to selected modalities
    if modalities and 'modality' in df.columns:
        df = df[df['modality'].isin(modalities)]
    
    # For each method/modality, keep best run by Test_AUROC
    if not df.empty and 'Test_AUROC' in df.columns:
        df = df.sort_values('Test_AUROC', ascending=False).groupby(['Method', 'modality']).first().reset_index()
    
    return df

def create_mdae_combined(df: pd.DataFrame) -> pd.DataFrame:
    """Create MDAE Combined from best of MDAE and MDAE (TC)."""
    if 'modality' not in df.columns:
        return df
    
    combined_rows = []
    
    for modality in df['modality'].unique():
        mod_df = df[df['modality'] == modality]
        
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
            
            combined['Method'] = 'MDAE'  # Label as just MDAE
            combined_rows.append(combined)
    
    if combined_rows:
        combined_df = pd.concat(combined_rows, ignore_index=True)
        # Remove original MDAE and MDAE (TC)
        df = df[~df['Method'].isin(['MDAE', 'MDAE (TC)'])]
        df = pd.concat([df, combined_df], ignore_index=True)
    
    return df

def create_summary_table(all_results: Dict, category: str) -> pd.DataFrame:
    """Create summary table for benchmarks."""
    rows = []
    
    for benchmark_name, benchmark_data in all_results.items():
        if benchmark_data.empty:
            continue
            
        display_name = BENCHMARK_MAPPING.get(benchmark_name, benchmark_name)
        
        # Get unique methods
        methods = benchmark_data['Method'].unique()
        
        for method in methods:
            method_data = benchmark_data[benchmark_data['Method'] == method]
            
            # Calculate average across modalities
            metrics = {}
            for metric in ['Test_AUROC', 'Test_AP', 'Test_F1', 'Test_Balanced_Accuracy']:
                if metric in method_data.columns:
                    values = method_data[metric].dropna()
                    if len(values) > 0:
                        metrics[metric] = values.mean()
                    else:
                        metrics[metric] = np.nan
            
            rows.append({
                'Benchmark': display_name,
                'Method': method,
                **metrics
            })
    
    if not rows:
        return pd.DataFrame()
    
    df = pd.DataFrame(rows)
    
    # Create pivot table for better visualization
    if 'Test_AUROC' in df.columns:
        pivot_df = df.pivot_table(
            index='Method',
            columns='Benchmark',
            values='Test_AUROC',
            aggfunc='mean'
        )
        
        # Add average column
        pivot_df['Average'] = pivot_df.mean(axis=1)
        
        # Sort by average
        pivot_df = pivot_df.sort_values('Average', ascending=False)
        
        return pivot_df
    
    return df

def create_visualizations(all_results: Dict, output_dir: Path, category: str):
    """Create visualizations for key benchmarks."""
    if not all_results:
        return
    
    # Create summary plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = ['Test_AUROC', 'Test_AP', 'Test_F1', 'Test_Balanced_Accuracy']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        # Collect data for this metric
        plot_data = []
        for benchmark_name, benchmark_data in all_results.items():
            if benchmark_data.empty or metric not in benchmark_data.columns:
                continue
            
            display_name = BENCHMARK_MAPPING.get(benchmark_name, benchmark_name)
            
            # Calculate average per method
            method_avgs = benchmark_data.groupby('Method')[metric].mean()
            
            for method, value in method_avgs.items():
                if not pd.isna(value):
                    plot_data.append({
                        'Benchmark': display_name[:20],  # Truncate long names
                        'Method': method,
                        'Value': value
                    })
        
        if plot_data:
            plot_df = pd.DataFrame(plot_data)
            
            # Get top methods
            top_methods = plot_df.groupby('Method')['Value'].mean().nlargest(10).index
            plot_df = plot_df[plot_df['Method'].isin(top_methods)]
            
            # Create grouped bar plot
            pivot = plot_df.pivot_table(index='Method', columns='Benchmark', values='Value')
            pivot.plot(kind='barh', ax=ax, width=0.8)
            
            ax.set_xlabel(metric.replace('_', ' '))
            ax.set_title(metric.replace('_', ' '))
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            ax.set_xlim([0, 1])
    
    plt.suptitle(f'Key Benchmarks Performance - {category.upper()}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f'key_benchmarks_{category}_performance.png', dpi=150, bbox_inches='tight')
    plt.close()

def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(description='Analyze key benchmarks')
    parser.add_argument('--input-dir', type=Path, default=Path('raw_data/20250815_consolidated'),
                       help='Input directory with consolidated data')
    parser.add_argument('--output-dir', type=Path, default=Path('processed_data_key_benchmarks_20250815'),
                       help='Output directory for results')
    parser.add_argument('--category', choices=['t1t2', 'generalization', 'all'], default='all',
                       help='Category of benchmarks to analyze')
    
    args = parser.parse_args()
    
    # Select benchmarks
    if args.category == 't1t2':
        selected_benchmarks = KEY_BENCHMARKS_T1T2
    elif args.category == 'generalization':
        selected_benchmarks = KEY_BENCHMARKS_GENERALIZATION
    else:
        selected_benchmarks = KEY_BENCHMARKS_ALL
    
    print(f"{'='*60}")
    print(f"KEY BENCHMARKS ANALYSIS")
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Category: {args.category}")
    print(f"Benchmarks: {len(selected_benchmarks)}")
    print(f"{'='*60}\n")
    
    # Create output directories
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / args.category).mkdir(exist_ok=True)
    
    # Process benchmarks
    all_results = {}
    
    for benchmark_name, modalities in selected_benchmarks.items():
        print(f"Processing {benchmark_name}...")
        print(f"  Modalities: {', '.join(modalities)}")
        
        benchmark_dir = args.input_dir / benchmark_name
        if not benchmark_dir.exists():
            print(f"  ⚠ Directory not found")
            continue
        
        # Load data
        df = load_benchmark_data(benchmark_dir, modalities)
        
        if df.empty:
            print(f"  ⚠ No data found")
            continue
        
        # Create MDAE combined
        df = create_mdae_combined(df)
        
        print(f"  ✓ Loaded {len(df)} method-modality combinations")
        
        # Check MDAE performance
        mdae_data = df[df['Method'] == 'MDAE']
        if not mdae_data.empty and 'Test_AUROC' in mdae_data.columns:
            avg_auroc = mdae_data['Test_AUROC'].mean()
            print(f"  MDAE Average AUROC: {avg_auroc:.3f}")
        
        all_results[benchmark_name] = df
    
    # Create summary table
    print(f"\n{'='*60}")
    print("CREATING SUMMARY TABLE")
    print(f"{'='*60}")
    
    summary_df = create_summary_table(all_results, args.category)
    
    if not summary_df.empty:
        # Save summary
        summary_path = args.output_dir / args.category / f'summary_{args.category}.csv'
        summary_df.to_csv(summary_path)
        print(f"Summary saved to: {summary_path}")
        
        # Print top methods
        print("\nTop Methods by Average AUROC:")
        print(summary_df.head(10))
        
        # Create visualizations
        create_visualizations(all_results, args.output_dir / args.category, args.category)
        print(f"\nVisualizations saved to: {args.output_dir / args.category}")
    else:
        print("No data to summarize")
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()