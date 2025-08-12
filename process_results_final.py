#!/usr/bin/env python3
"""
Final version: Process MDAE benchmark results with clean visualizations
and comprehensive AUROC/AP metrics table.
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
plt.rcParams['figure.figsize'] = (12, 6)
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
    'T1': 'T1',
    'T1c': 'T1c',
    'T1C': 'T1c',
    'T1n': 'T1n',
    'T1N': 'T1n',
    'T2': 'T2',
    'T2f': 'T2f',
    'T2w': 'T2',
    'T2W': 'T2',
    'FLAIR': 'T2f',
    'ASL': 'ASL',
    'SWI': 'SWI',
    'T1GD': 'T1GD',
    'T1W': 'T1',
    'T1WCE': 'T1c',
    'MIXED_CONTRASTS': 'Mixed'
}

# Benchmark info
BENCHMARK_INFO = {
    'brats18_lgg_vs_hgg': 'BraTS18_LGG_HGG',
    'brats23_gli_vs_men': 'BraTS23_Gli_Men',
    'brats23_gli_vs_met': 'BraTS23_Gli_Met',
    'brats23_men_vs_met': 'BraTS23_Men_Met',
    'rsna_miccai_mgmt_methylation': 'MGMT',
    'tcga_gbm_dss_1year': 'TCGA_DSS_1Y',
    'tcga_gbm_pfi_1year': 'TCGA_PFI_1Y',
    'ucsf_pdgm_idh_classification': 'IDH',
    'upenn_gbm_age_group': 'Age',
    'upenn_gbm_gender': 'Gender',
    'upenn_gbm_gtr_status': 'GTR',
    'upenn_gbm_idh1_status': 'IDH1',
    'upenn_gbm_survival_18month': 'Surv_18M',
    'upenn_gbm_survival_1year': 'Surv_1Y',
    'upenn_gbm_survival_2year': 'Surv_2Y',
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
    
    modality_str = str(modality).upper()
    return MODALITY_NAMES.get(modality_str, modality_str)


def create_clean_visualizations(results_df: pd.DataFrame, modality: str, 
                               benchmark_name: str, output_dir: Path):
    """Create clean AUROC and AP visualizations without ranking numbers."""
    
    # Create figure with 2 subplots (removed correlation plot)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
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
    ax1.set_title(f'AUROC - {modality}', fontsize=12, fontweight='bold')
    ax1.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Random')
    
    # Add value labels (without ranking)
    for bar, val in zip(bars, auroc_sorted['Test_AUROC']):
        if not pd.isna(val):
            ax1.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{val:.3f}', ha='left', va='center', fontsize=9)
    
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
        ax2.set_title(f'Average Precision - {modality}', fontsize=12, fontweight='bold')
        ax2.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Random')
        
        # Add value labels (without ranking)
        for bar, val in zip(bars, ap_data['Test_AP']):
            if not pd.isna(val):
                ax2.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{val:.3f}', ha='left', va='center', fontsize=9)
        
        ax2.set_xlim([0.3, 1.05])
        ax2.grid(True, alpha=0.3, axis='x')
    else:
        ax2.text(0.5, 0.5, 'No AP data available', ha='center', va='center', fontsize=12)
        ax2.set_title(f'Average Precision - {modality}', fontsize=12, fontweight='bold')
    
    plt.suptitle(f"{BENCHMARK_INFO[benchmark_name]}", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save figure
    fig_path = output_dir / 'auroc_ap_analysis.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()


def process_benchmark_modality(benchmark_name: str, modality: str, df: pd.DataFrame) -> pd.DataFrame:
    """Process data for a specific modality within a benchmark."""
    
    # Filter for specific modality
    modality_df = df[df['modality_normalized'] == modality].copy()
    
    if modality_df.empty:
        return None
    
    # Get best run for each method (focus on AUROC and AP)
    metrics = ['Test_AUROC', 'Test_AP']
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


def create_comprehensive_metrics_table(all_results: Dict) -> pd.DataFrame:
    """Create comprehensive table with AUROC and AP for all methods across benchmarks."""
    
    # Collect all unique methods
    all_methods = set()
    for benchmark_data in all_results.values():
        for modality_data in benchmark_data.values():
            if modality_data is not None:
                all_methods.update(modality_data['Method'].values)
    
    # Remove individual MDAE variants if combined exists
    if 'MDAE (Combined)' in all_methods:
        all_methods.discard('MDAE')
        all_methods.discard('MDAE (TC)')
    
    all_methods = sorted(all_methods)
    
    # Build comprehensive table
    rows = []
    for method in all_methods:
        row = {'Method': method.replace(' (Combined)', '')}
        
        auroc_values = []
        ap_values = []
        
        for benchmark_name, benchmark_data in all_results.items():
            for modality, modality_data in benchmark_data.items():
                if modality_data is not None:
                    method_row = modality_data[modality_data['Method'] == method]
                    if not method_row.empty:
                        col_name = f"{BENCHMARK_INFO[benchmark_name]}_{modality}"
                        auroc = method_row.iloc[0]['Test_AUROC']
                        ap = method_row.iloc[0]['Test_AP']
                        
                        row[f"{col_name}_AUROC"] = auroc
                        row[f"{col_name}_AP"] = ap
                        
                        if not pd.isna(auroc):
                            auroc_values.append(auroc)
                        if not pd.isna(ap):
                            ap_values.append(ap)
        
        # Calculate summary statistics
        if auroc_values:
            row['Mean_AUROC'] = np.mean(auroc_values)
            row['Std_AUROC'] = np.std(auroc_values)
            row['Count_AUROC'] = len(auroc_values)
        
        if ap_values:
            row['Mean_AP'] = np.mean(ap_values)
            row['Std_AP'] = np.std(ap_values)
            row['Count_AP'] = len(ap_values)
        
        rows.append(row)
    
    # Create DataFrame and sort by Mean_AUROC
    comprehensive_df = pd.DataFrame(rows)
    if 'Mean_AUROC' in comprehensive_df.columns:
        comprehensive_df = comprehensive_df.sort_values('Mean_AUROC', ascending=False)
    
    return comprehensive_df


def create_summary_table(comprehensive_df: pd.DataFrame) -> pd.DataFrame:
    """Create a summary table with key benchmarks only."""
    
    # Select key columns for summary
    summary_cols = ['Method']
    
    # Key benchmarks to include in summary
    key_benchmarks = [
        'BraTS23_Gli_Met_T2f',  # Tumor type
        'IDH_ASL',  # Molecular marker
        'Age_T2',  # Clinical feature
        'Surv_1Y_T2',  # Survival
    ]
    
    for benchmark in key_benchmarks:
        if f"{benchmark}_AUROC" in comprehensive_df.columns:
            summary_cols.append(f"{benchmark}_AUROC")
        if f"{benchmark}_AP" in comprehensive_df.columns:
            summary_cols.append(f"{benchmark}_AP")
    
    # Add mean values
    summary_cols.extend(['Mean_AUROC', 'Mean_AP'])
    
    # Filter columns that exist
    summary_cols = [col for col in summary_cols if col in comprehensive_df.columns]
    
    summary_df = comprehensive_df[summary_cols].copy()
    
    # Round values
    numeric_cols = summary_df.select_dtypes(include=[np.number]).columns
    summary_df[numeric_cols] = summary_df[numeric_cols].round(3)
    
    return summary_df


def generate_latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    """Generate LaTeX table from DataFrame."""
    
    # Round numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_rounded = df.copy()
    for col in numeric_cols:
        df_rounded[col] = df_rounded[col].round(3)
    
    # Replace NaN with -
    df_rounded = df_rounded.fillna('-')
    
    # Simplify column names for LaTeX
    col_mapping = {col: col.replace('_', ' ') for col in df.columns}
    df_rounded = df_rounded.rename(columns=col_mapping)
    
    # Generate LaTeX
    latex = "\\begin{table}[ht]\n"
    latex += "\\centering\n"
    latex += f"\\caption{{{caption}}}\n"
    latex += f"\\label{{{label}}}\n"
    latex += "\\resizebox{\\textwidth}{!}{%\n"
    latex += df_rounded.to_latex(index=False, escape=False)
    latex += "}\n"
    latex += "\\end{table}\n"
    
    return latex


def main():
    """Main processing function."""
    
    # Setup paths
    data_dir = Path('/home/t-jiachentu/repos/benchmarking/misc/data/raw_data/20250811')
    output_base = Path('/home/t-jiachentu/repos/benchmarking/misc/data/processed_data/benchmark_results_final')
    output_base.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("FINAL PROCESSING WITH CLEAN VISUALIZATIONS")
    print("="*60)
    
    all_results = {}
    
    # Process all benchmarks
    for benchmark_name in BENCHMARK_INFO.keys():
        print(f"\nProcessing {benchmark_name}...")
        
        # Load raw data
        csv_path = data_dir / benchmark_name / 'runs_summary.csv'
        if not csv_path.exists():
            print(f"  Warning: {csv_path} not found")
            continue
        
        df = pd.read_csv(csv_path)
        
        # Add method identification
        df['Method'] = df['run_name'].apply(identify_method)
        df = df[df['Method'] != 'Unknown']
        
        # Normalize modality names
        df['modality_normalized'] = df['modality'].apply(normalize_modality)
        
        # Get unique modalities
        modalities = df['modality_normalized'].unique()
        modalities = [m for m in modalities if m != 'Unknown']
        
        # Create directories
        benchmark_dir = output_base / benchmark_name
        benchmark_dir.mkdir(parents=True, exist_ok=True)
        modalities_dir = benchmark_dir / 'modalities'
        modalities_dir.mkdir(exist_ok=True)
        
        # Process each modality
        benchmark_results = {}
        
        for modality in sorted(modalities):
            modality_dir = modalities_dir / modality
            modality_dir.mkdir(exist_ok=True)
            
            # Process modality data
            results_df = process_benchmark_modality(benchmark_name, modality, df)
            
            if results_df is not None and not results_df.empty:
                benchmark_results[modality] = results_df
                
                # Save metrics
                metrics_path = modality_dir / 'auroc_ap_metrics.csv'
                results_df.to_csv(metrics_path, index=False)
                
                # Create clean visualizations
                create_clean_visualizations(results_df, modality, benchmark_name, modality_dir)
                
                # Print MDAE performance
                mdae_data = results_df[results_df['Method'] == 'MDAE (Combined)']
                if not mdae_data.empty:
                    mdae_row = mdae_data.iloc[0]
                    ap_str = f"{mdae_row['Test_AP']:.3f}" if not pd.isna(mdae_row['Test_AP']) else "N/A"
                    print(f"  {modality:10s}: MDAE AUROC={mdae_row['Test_AUROC']:.3f}, AP={ap_str}")
        
        all_results[benchmark_name] = benchmark_results
    
    # Create comprehensive metrics table
    print("\n" + "="*60)
    print("CREATING COMPREHENSIVE METRICS TABLE")
    print("="*60)
    
    comprehensive_df = create_comprehensive_metrics_table(all_results)
    
    # Save comprehensive table
    comprehensive_path = output_base / 'comprehensive_auroc_ap_table.csv'
    comprehensive_df.to_csv(comprehensive_path, index=False)
    print(f"Comprehensive table saved to: {comprehensive_path}")
    
    # Create and save summary table
    summary_df = create_summary_table(comprehensive_df)
    summary_path = output_base / 'summary_metrics_table.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary table saved to: {summary_path}")
    
    # Print top methods
    print("\nTop Methods by Mean AUROC:")
    print(summary_df[['Method', 'Mean_AUROC', 'Mean_AP']].head(10).to_string(index=False))
    
    # Generate LaTeX tables
    latex_comprehensive = generate_latex_table(
        summary_df.head(10),
        "Test AUROC and AP performance across key benchmarks",
        "tab:auroc_ap_results"
    )
    
    latex_path = output_base / 'latex_tables.tex'
    with open(latex_path, 'w') as f:
        f.write(latex_comprehensive)
    
    print(f"\nLaTeX tables saved to: {latex_path}")
    
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()