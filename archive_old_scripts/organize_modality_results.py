#!/usr/bin/env python3
"""
Organize benchmark results by modality (T1, T1c, T2, T2f/FLAIR, etc.)
for comprehensive modality-specific analysis.
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
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Define modality mappings
MODALITY_NAMES = {
    'T1': 'T1-weighted',
    'T1c': 'T1-contrast enhanced',
    'T1n': 'T1-native',
    'T2': 'T2-weighted',
    'T2f': 'T2-FLAIR',
    'T2w': 'T2-weighted',
    'FLAIR': 'T2-FLAIR'
}

# Method patterns
METHOD_PATTERNS = {
    'MDAE': r'^resenc_MDAETrainer_RandomMask_Flow_BS48_2000ep_pretrained',
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
    'ResNet-50': r'^brainiac_scratch',
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
    modality = modality.upper()
    # Map T2W to T2, T2F to FLAIR, etc.
    if modality == 'T2W':
        return 'T2'
    elif modality in ['T2F', 'FLAIR']:
        return 'T2f'
    elif modality == 'T1CE':
        return 'T1c'
    return modality


def process_modality_data(benchmark_name: str, modality: str, df: pd.DataFrame) -> Dict:
    """Process data for a specific modality within a benchmark."""
    
    # Filter for specific modality
    modality_df = df[df['modality_normalized'] == modality].copy()
    
    if modality_df.empty:
        return None
    
    # Get best run for each method
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
        else:
            for metric in metrics:
                mdae_combined[metric] = mdae_tc_row[metric]
            mdae_combined['Best_Variant'] = 'MDAE (TC)'
    elif mdae_row is not None:
        for metric in metrics:
            mdae_combined[metric] = mdae_row[metric]
        mdae_combined['Best_Variant'] = 'MDAE'
    elif mdae_tc_row is not None:
        for metric in metrics:
            mdae_combined[metric] = mdae_tc_row[metric]
        mdae_combined['Best_Variant'] = 'MDAE (TC)'
    
    if 'Best_Variant' in mdae_combined:
        results_df = pd.concat([results_df, pd.DataFrame([mdae_combined])], ignore_index=True)
    
    # Sort by Test_AUROC
    results_df = results_df.sort_values('Test_AUROC', ascending=False)
    
    # Create summary
    summary = {
        'benchmark': benchmark_name,
        'modality': modality,
        'modality_name': MODALITY_NAMES.get(modality, modality),
        'num_runs': len(modality_df),
        'num_methods': len(results_df),
        'best_method': results_df.iloc[0]['Method'] if not results_df.empty else None,
        'best_auroc': float(results_df.iloc[0]['Test_AUROC']) if not results_df.empty else None,
        'mdae_performance': {},
        'top_5_methods': [],
        'statistics': {}
    }
    
    # MDAE performance
    if 'MDAE (Combined)' in results_df['Method'].values:
        mdae_row = results_df[results_df['Method'] == 'MDAE (Combined)'].iloc[0]
        summary['mdae_performance'] = {
            'auroc': float(mdae_row['Test_AUROC']) if not pd.isna(mdae_row['Test_AUROC']) else None,
            'ap': float(mdae_row['Test_AP']) if not pd.isna(mdae_row['Test_AP']) else None,
            'f1': float(mdae_row['Test_F1']) if not pd.isna(mdae_row['Test_F1']) else None,
            'balanced_accuracy': float(mdae_row['Test_Balanced_Accuracy']) if not pd.isna(mdae_row['Test_Balanced_Accuracy']) else None,
            'rank': int((results_df['Test_AUROC'] > mdae_row['Test_AUROC']).sum() + 1),
            'best_variant': mdae_row.get('Best_Variant', 'Unknown')
        }
    
    # Top 5 methods
    for _, row in results_df.head(5).iterrows():
        summary['top_5_methods'].append({
            'method': row['Method'],
            'auroc': float(row['Test_AUROC']) if not pd.isna(row['Test_AUROC']) else None
        })
    
    # Statistics
    auroc_values = results_df['Test_AUROC'].dropna()
    if not auroc_values.empty:
        summary['statistics'] = {
            'mean_auroc': float(auroc_values.mean()),
            'std_auroc': float(auroc_values.std()),
            'median_auroc': float(auroc_values.median()),
            'min_auroc': float(auroc_values.min()),
            'max_auroc': float(auroc_values.max())
        }
    
    return {'results_df': results_df, 'summary': summary}


def create_modality_visualization(results_df: pd.DataFrame, modality: str, 
                                 benchmark_name: str, output_dir: Path):
    """Create visualization for a specific modality."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Define colors
    colors = []
    for method in results_df['Method']:
        if 'MDAE' in method:
            colors.append('#2E7D32')  # Green
        elif method in ['MAE', 'SimCLR', 'VoCo', 'MG', 'SwinUNETR', 'VF', 'S3D']:
            colors.append('#1976D2')  # Blue
        elif method in ['BrainIAC', 'MRI-Core', 'BrainMVP']:
            colors.append('#7B1FA2')  # Purple
        else:
            colors.append('#757575')  # Gray
    
    # Plot 1: AUROC comparison
    ax1 = axes[0]
    bars = ax1.barh(range(len(results_df)), results_df['Test_AUROC'], color=colors, alpha=0.8)
    ax1.set_yticks(range(len(results_df)))
    ax1.set_yticklabels(results_df['Method'])
    ax1.set_xlabel('Test AUROC')
    ax1.set_title(f'{MODALITY_NAMES.get(modality, modality)} - Test AUROC')
    ax1.axvline(x=0.5, color='red', linestyle='--', alpha=0.5)
    
    # Add value labels
    for bar, val in zip(bars, results_df['Test_AUROC']):
        if not pd.isna(val):
            ax1.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{val:.3f}', ha='left', va='center', fontsize=9)
    
    ax1.set_xlim([0.3, 1.05])
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Plot 2: All metrics for top 5
    ax2 = axes[1]
    top_5 = results_df.head(5)
    metrics = ['Test_AUROC', 'Test_AP', 'Test_F1', 'Test_Balanced_Accuracy']
    metric_labels = ['AUROC', 'AP', 'F1', 'Bal. Acc']
    
    x = np.arange(len(top_5))
    width = 0.2
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        values = top_5[metric].fillna(0).values
        ax2.bar(x + i * width, values, width, label=label, alpha=0.8)
    
    ax2.set_xlabel('Method')
    ax2.set_ylabel('Score')
    ax2.set_title(f'All Metrics - Top 5 Methods')
    ax2.set_xticks(x + width * 1.5)
    ax2.set_xticklabels(top_5['Method'], rotation=45, ha='right')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([0, 1.05])
    
    plt.suptitle(f"{BENCHMARK_INFO[benchmark_name]} - {modality} Modality", 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save figure
    fig_path = output_dir / 'visualization.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_cross_modality_comparison(benchmark_name: str, modality_results: Dict, output_dir: Path):
    """Create cross-modality comparison visualizations and tables."""
    
    # Prepare data for comparison
    comparison_data = []
    
    for modality, data in modality_results.items():
        if data and data['summary']['mdae_performance']:
            comparison_data.append({
                'Modality': modality,
                'MDAE_AUROC': data['summary']['mdae_performance'].get('auroc', np.nan),
                'MDAE_Rank': data['summary']['mdae_performance'].get('rank', np.nan),
                'Best_Method': data['summary']['best_method'],
                'Best_AUROC': data['summary']['best_auroc'],
                'Num_Methods': data['summary']['num_methods']
            })
    
    if not comparison_data:
        return
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('MDAE_AUROC', ascending=False)
    
    # Save comparison table
    comparison_df.to_csv(output_dir / 'cross_modality_comparison.csv', index=False)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: MDAE performance across modalities
    ax1 = axes[0, 0]
    bars = ax1.bar(range(len(comparison_df)), comparison_df['MDAE_AUROC'], 
                   color='#2E7D32', alpha=0.8)
    ax1.set_xticks(range(len(comparison_df)))
    ax1.set_xticklabels(comparison_df['Modality'], rotation=45)
    ax1.set_ylabel('MDAE AUROC')
    ax1.set_title('MDAE Performance Across Modalities')
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    ax1.set_ylim([0.4, 1.0])
    
    # Add value labels
    for bar, val in zip(bars, comparison_df['MDAE_AUROC']):
        if not pd.isna(val):
            ax1.text(bar.get_x() + bar.get_width()/2, val + 0.01, 
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: MDAE rank across modalities
    ax2 = axes[0, 1]
    bars = ax2.bar(range(len(comparison_df)), comparison_df['MDAE_Rank'], 
                   color='#1976D2', alpha=0.8)
    ax2.set_xticks(range(len(comparison_df)))
    ax2.set_xticklabels(comparison_df['Modality'], rotation=45)
    ax2.set_ylabel('MDAE Rank')
    ax2.set_title('MDAE Ranking Across Modalities')
    ax2.invert_yaxis()  # Lower rank is better
    ax2.set_ylim([10, 0])
    
    # Add value labels
    for bar, val in zip(bars, comparison_df['MDAE_Rank']):
        if not pd.isna(val):
            ax2.text(bar.get_x() + bar.get_width()/2, val - 0.2, 
                    f'#{int(val)}', ha='center', va='top', fontsize=9)
    
    # Plot 3: Method × Modality heatmap
    ax3 = axes[1, 0]
    
    # Create matrix for heatmap
    methods = ['MDAE (Combined)', 'MAE', 'SimCLR', 'VoCo', 'BrainIAC']
    modalities = sorted(modality_results.keys())
    
    heatmap_data = []
    for method in methods:
        row = []
        for modality in modalities:
            if modality_results[modality]:
                df = modality_results[modality]['results_df']
                method_data = df[df['Method'] == method]
                if not method_data.empty:
                    row.append(method_data.iloc[0]['Test_AUROC'])
                else:
                    row.append(np.nan)
            else:
                row.append(np.nan)
        heatmap_data.append(row)
    
    heatmap_df = pd.DataFrame(heatmap_data, index=methods, columns=modalities)
    sns.heatmap(heatmap_df, annot=True, fmt='.3f', cmap='RdYlGn', 
                vmin=0.5, vmax=1.0, ax=ax3, cbar_kws={'label': 'AUROC'})
    ax3.set_title('Method Performance Across Modalities')
    ax3.set_xlabel('Modality')
    ax3.set_ylabel('Method')
    
    # Plot 4: Best method for each modality
    ax4 = axes[1, 1]
    
    # Count best methods
    best_counts = comparison_df['Best_Method'].value_counts()
    colors_pie = []
    for method in best_counts.index:
        if 'MDAE' in method:
            colors_pie.append('#2E7D32')
        elif method in ['MAE', 'SimCLR', 'VoCo', 'MG', 'SwinUNETR', 'VF', 'S3D']:
            colors_pie.append('#1976D2')
        else:
            colors_pie.append('#7B1FA2')
    
    wedges, texts, autotexts = ax4.pie(best_counts.values, labels=best_counts.index, 
                                        colors=colors_pie, autopct='%1.0f%%',
                                        startangle=90)
    ax4.set_title('Best Methods Distribution Across Modalities')
    
    # Overall title
    plt.suptitle(f"{BENCHMARK_INFO[benchmark_name]} - Cross-Modality Analysis", 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save figure
    fig_path = output_dir / 'cross_modality_analysis.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return comparison_df


def process_benchmark_modalities(benchmark_name: str, data_dir: Path, output_base: Path):
    """Process all modalities for a single benchmark."""
    
    # Load raw data
    csv_path = data_dir / benchmark_name / 'runs_summary.csv'
    if not csv_path.exists():
        print(f"Warning: {csv_path} not found")
        return None
    
    df = pd.read_csv(csv_path)
    
    # Add method identification
    df['Method'] = df['run_name'].apply(identify_method)
    df = df[df['Method'] != 'Unknown']
    
    # Normalize modality names
    df['modality_normalized'] = df['modality'].apply(normalize_modality)
    
    # Get unique modalities
    modalities = df['modality_normalized'].unique()
    print(f"  Found modalities: {', '.join(sorted(modalities))}")
    
    # Create modalities directory
    benchmark_dir = output_base / benchmark_name
    modalities_dir = benchmark_dir / 'modalities'
    modalities_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each modality
    modality_results = {}
    
    for modality in sorted(modalities):
        modality_dir = modalities_dir / modality
        modality_dir.mkdir(exist_ok=True)
        
        # Process modality data
        result = process_modality_data(benchmark_name, modality, df)
        
        if result:
            modality_results[modality] = result
            
            # Save metrics
            metrics_path = modality_dir / 'metrics.csv'
            result['results_df'].to_csv(metrics_path, index=False)
            
            # Save summary
            summary_path = modality_dir / 'summary.json'
            with open(summary_path, 'w') as f:
                json.dump(result['summary'], f, indent=2)
            
            # Create visualization
            create_modality_visualization(result['results_df'], modality, 
                                        benchmark_name, modality_dir)
            
            print(f"    - {modality}: MDAE rank #{result['summary']['mdae_performance'].get('rank', 'N/A')}, "
                  f"AUROC={result['summary']['mdae_performance'].get('auroc', 'N/A'):.3f}")
    
    # Create cross-modality comparison
    if modality_results:
        comparison_df = create_cross_modality_comparison(benchmark_name, modality_results, modalities_dir)
        
        # Find best modality for MDAE
        if comparison_df is not None and not comparison_df.empty:
            best_modality = comparison_df.iloc[0]['Modality']
            best_auroc = comparison_df.iloc[0]['MDAE_AUROC']
            print(f"  Best modality for MDAE: {best_modality} (AUROC={best_auroc:.3f})")
    
    return modality_results


def create_master_modality_table(all_results: Dict, output_dir: Path):
    """Create master table showing best modality for each benchmark."""
    
    master_data = []
    
    for benchmark, modality_results in all_results.items():
        if not modality_results:
            continue
            
        # Find best modality for MDAE
        best_modality = None
        best_auroc = 0
        
        for modality, data in modality_results.items():
            if data and data['summary']['mdae_performance']:
                auroc = data['summary']['mdae_performance'].get('auroc', 0)
                if auroc > best_auroc:
                    best_auroc = auroc
                    best_modality = modality
        
        if best_modality:
            master_data.append({
                'Benchmark': BENCHMARK_INFO[benchmark],
                'Best_Modality': best_modality,
                'MDAE_AUROC': best_auroc,
                'Num_Modalities': len(modality_results)
            })
    
    if master_data:
        master_df = pd.DataFrame(master_data)
        master_df = master_df.sort_values('MDAE_AUROC', ascending=False)
        
        # Save master table
        master_df.to_csv(output_dir / 'master_modality_summary.csv', index=False)
        
        # Print summary
        print("\nBest Modality Summary:")
        for _, row in master_df.iterrows():
            print(f"  {row['Benchmark']:30s}: {row['Best_Modality']:5s} (AUROC={row['MDAE_AUROC']:.3f})")
        
        return master_df
    
    return None


def main():
    """Main processing function."""
    
    # Setup paths
    data_dir = Path('/home/t-jiachentu/repos/benchmarking/misc/data/raw_data/20250811')
    output_base = Path('/home/t-jiachentu/repos/benchmarking/misc/data/processed_data/benchmark_results')
    
    print("="*60)
    print("ORGANIZING RESULTS BY MODALITY")
    print("="*60)
    
    # Process each benchmark
    all_results = {}
    benchmarks = list(BENCHMARK_INFO.keys())
    
    for benchmark_name in benchmarks:
        print(f"\nProcessing {benchmark_name}...")
        modality_results = process_benchmark_modalities(benchmark_name, data_dir, output_base)
        if modality_results:
            all_results[benchmark_name] = modality_results
    
    # Create master modality summary
    print("\n" + "="*60)
    print("CREATING MASTER MODALITY SUMMARY")
    print("="*60)
    
    master_df = create_master_modality_table(all_results, output_base)
    
    # Summary statistics
    print("\n" + "="*60)
    print("MODALITY STATISTICS")
    print("="*60)
    
    # Count modality preferences
    modality_counts = {}
    for benchmark, modality_results in all_results.items():
        for modality in modality_results.keys():
            if modality not in modality_counts:
                modality_counts[modality] = 0
            modality_counts[modality] += 1
    
    print("\nModality availability across benchmarks:")
    for modality, count in sorted(modality_counts.items(), key=lambda x: -x[1]):
        print(f"  {modality:5s}: {count:2d} benchmarks ({count/len(benchmarks)*100:.1f}%)")
    
    # Best modality distribution
    if master_df is not None:
        best_modality_counts = master_df['Best_Modality'].value_counts()
        print("\nBest modality distribution (for MDAE):")
        for modality, count in best_modality_counts.items():
            print(f"  {modality:5s}: {count:2d} benchmarks ({count/len(master_df)*100:.1f}%)")
    
    print("\n✓ Modality organization complete!")
    print(f"Results saved to: {output_base}/[benchmark]/modalities/")


if __name__ == "__main__":
    main()