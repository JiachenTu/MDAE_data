#!/usr/bin/env python3
"""
Organize MDAE benchmark results into specialized per-benchmark directories
with comprehensive metrics and visualizations.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import re

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

# Define benchmark metadata
BENCHMARK_INFO = {
    'brats18_lgg_vs_hgg': {
        'name': 'BraTS18 LGG vs HGG',
        'category': 'Tumor Grade',
        'task': 'Binary Classification',
        'description': 'Low-grade vs High-grade Glioma'
    },
    'brats23_gli_vs_men': {
        'name': 'BraTS23 Glioma vs Meningioma',
        'category': 'Tumor Type',
        'task': 'Binary Classification',
        'description': 'Glioma vs Meningioma differentiation'
    },
    'brats23_gli_vs_met': {
        'name': 'BraTS23 Glioma vs Metastasis',
        'category': 'Tumor Type',
        'task': 'Binary Classification',
        'description': 'Glioma vs Brain Metastasis'
    },
    'brats23_men_vs_met': {
        'name': 'BraTS23 Meningioma vs Metastasis',
        'category': 'Tumor Type',
        'task': 'Binary Classification',
        'description': 'Meningioma vs Brain Metastasis'
    },
    'rsna_miccai_mgmt_methylation': {
        'name': 'MGMT Methylation Status',
        'category': 'Molecular Marker',
        'task': 'Binary Classification',
        'description': 'MGMT promoter methylation prediction'
    },
    'tcga_gbm_dss_1year': {
        'name': 'TCGA-GBM DSS 1-Year',
        'category': 'Survival Prediction',
        'task': 'Binary Classification',
        'description': 'Disease-specific survival at 1 year'
    },
    'tcga_gbm_pfi_1year': {
        'name': 'TCGA-GBM PFI 1-Year',
        'category': 'Survival Prediction',
        'task': 'Binary Classification',
        'description': 'Progression-free interval at 1 year'
    },
    'ucsf_pdgm_idh_classification': {
        'name': 'IDH Mutation Status',
        'category': 'Molecular Marker',
        'task': 'Binary Classification',
        'description': 'IDH1/2 mutation status prediction'
    },
    'upenn_gbm_age_group': {
        'name': 'UPenn-GBM Age Group',
        'category': 'Clinical Feature',
        'task': 'Binary Classification',
        'description': 'Age group classification'
    },
    'upenn_gbm_gender': {
        'name': 'UPenn-GBM Gender',
        'category': 'Clinical Feature',
        'task': 'Binary Classification',
        'description': 'Gender prediction from MRI'
    },
    'upenn_gbm_gtr_status': {
        'name': 'UPenn-GBM GTR Status',
        'category': 'Clinical Feature',
        'task': 'Binary Classification',
        'description': 'Gross total resection achievement'
    },
    'upenn_gbm_idh1_status': {
        'name': 'UPenn-GBM IDH1 Status',
        'category': 'Molecular Marker',
        'task': 'Binary Classification',
        'description': 'IDH1 mutation status'
    },
    'upenn_gbm_survival_18month': {
        'name': 'UPenn-GBM Survival 18-Month',
        'category': 'Survival Prediction',
        'task': 'Binary Classification',
        'description': 'Overall survival at 18 months'
    },
    'upenn_gbm_survival_1year': {
        'name': 'UPenn-GBM Survival 1-Year',
        'category': 'Survival Prediction',
        'task': 'Binary Classification',
        'description': 'Overall survival at 1 year'
    },
    'upenn_gbm_survival_2year': {
        'name': 'UPenn-GBM Survival 2-Year',
        'category': 'Survival Prediction',
        'task': 'Binary Classification',
        'description': 'Overall survival at 2 years'
    }
}

# Method patterns for identification
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


def identify_method(run_name: str) -> str:
    """Identify which method a run belongs to based on its name."""
    for method, pattern in METHOD_PATTERNS.items():
        if re.match(pattern, run_name):
            return method
    return 'Unknown'


def process_single_benchmark(benchmark_name: str, data_dir: Path, output_dir: Path):
    """Process a single benchmark and create its specialized directory."""
    
    # Create benchmark directory
    benchmark_dir = output_dir / benchmark_name
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    
    # Load raw data
    csv_path = data_dir / benchmark_name / 'runs_summary.csv'
    if not csv_path.exists():
        print(f"Warning: {csv_path} not found")
        return None
    
    df = pd.read_csv(csv_path)
    
    # Add method identification
    df['Method'] = df['run_name'].apply(identify_method)
    df = df[df['Method'] != 'Unknown']
    df = df.dropna(subset=['Test_AUROC'])
    
    # Get best run for each method (all metrics)
    metrics = ['Test_AUROC', 'Test_AP', 'Test_F1', 'Test_Balanced_Accuracy']
    best_runs = []
    
    for method in df['Method'].unique():
        method_df = df[df['Method'] == method]
        if not method_df.empty:
            best_idx = method_df['Test_AUROC'].idxmax()
            best_run = method_df.loc[best_idx]
            best_runs.append({
                'Method': method,
                'Test_AUROC': best_run['Test_AUROC'],
                'Test_AP': best_run['Test_AP'],
                'Test_F1': best_run['Test_F1'],
                'Test_Balanced_Accuracy': best_run['Test_Balanced_Accuracy']
            })
    
    if not best_runs:
        return None
    
    results_df = pd.DataFrame(best_runs)
    
    # Combine MDAE variants
    mdae_combined = {'Method': 'MDAE (Combined)'}
    if 'MDAE' in results_df['Method'].values and 'MDAE (TC)' in results_df['Method'].values:
        mdae_row = results_df[results_df['Method'] == 'MDAE'].iloc[0]
        mdae_tc_row = results_df[results_df['Method'] == 'MDAE (TC)'].iloc[0]
        
        # Take best AUROC, then use all metrics from that variant
        if mdae_row['Test_AUROC'] >= mdae_tc_row['Test_AUROC']:
            best_variant = 'MDAE'
            for metric in metrics:
                mdae_combined[metric] = mdae_row[metric]
        else:
            best_variant = 'MDAE (TC)'
            for metric in metrics:
                mdae_combined[metric] = mdae_tc_row[metric]
        
        mdae_combined['Best_Variant'] = best_variant
    elif 'MDAE' in results_df['Method'].values:
        mdae_row = results_df[results_df['Method'] == 'MDAE'].iloc[0]
        for metric in metrics:
            mdae_combined[metric] = mdae_row[metric]
        mdae_combined['Best_Variant'] = 'MDAE'
    elif 'MDAE (TC)' in results_df['Method'].values:
        mdae_tc_row = results_df[results_df['Method'] == 'MDAE (TC)'].iloc[0]
        for metric in metrics:
            mdae_combined[metric] = mdae_tc_row[metric]
        mdae_combined['Best_Variant'] = 'MDAE (TC)'
    
    # Add combined MDAE to results
    results_df = pd.concat([results_df, pd.DataFrame([mdae_combined])], ignore_index=True)
    
    # Sort by Test_AUROC
    results_df = results_df.sort_values('Test_AUROC', ascending=False)
    
    # Save metrics table
    metrics_path = benchmark_dir / 'metrics.csv'
    results_df.to_csv(metrics_path, index=False)
    
    # Create summary JSON
    summary = {
        'benchmark': benchmark_name,
        'info': BENCHMARK_INFO.get(benchmark_name, {}),
        'best_method': results_df.iloc[0]['Method'],
        'best_auroc': float(results_df.iloc[0]['Test_AUROC']),
        'mdae_performance': {},
        'top_5_methods': [],
        'statistics': {}
    }
    
    # MDAE performance
    if 'MDAE (Combined)' in results_df['Method'].values:
        mdae_row = results_df[results_df['Method'] == 'MDAE (Combined)'].iloc[0]
        summary['mdae_performance'] = {
            'auroc': float(mdae_row['Test_AUROC']),
            'ap': float(mdae_row['Test_AP']) if not pd.isna(mdae_row['Test_AP']) else None,
            'f1': float(mdae_row['Test_F1']) if not pd.isna(mdae_row['Test_F1']) else None,
            'balanced_accuracy': float(mdae_row['Test_Balanced_Accuracy']) if not pd.isna(mdae_row['Test_Balanced_Accuracy']) else None,
            'rank': int((results_df['Test_AUROC'] > mdae_row['Test_AUROC']).sum() + 1),
            'best_variant': mdae_row.get('Best_Variant', 'MDAE')
        }
    
    # Top 5 methods
    for _, row in results_df.head(5).iterrows():
        summary['top_5_methods'].append({
            'method': row['Method'],
            'auroc': float(row['Test_AUROC'])
        })
    
    # Statistics
    auroc_values = results_df['Test_AUROC'].dropna()
    summary['statistics'] = {
        'mean_auroc': float(auroc_values.mean()),
        'std_auroc': float(auroc_values.std()),
        'median_auroc': float(auroc_values.median()),
        'num_methods': len(auroc_values)
    }
    
    # Save summary
    summary_path = benchmark_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Create visualization
    create_benchmark_visualization(results_df, benchmark_name, benchmark_dir)
    
    # Create LaTeX table
    create_benchmark_latex_table(results_df, benchmark_name, benchmark_dir)
    
    return summary


def create_benchmark_visualization(results_df: pd.DataFrame, benchmark_name: str, output_dir: Path):
    """Create visualization for a single benchmark."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Define colors
    colors = []
    for method in results_df['Method']:
        if 'MDAE' in method:
            colors.append('#2E7D32')  # Green for MDAE
        elif method in ['MAE', 'SimCLR', 'VoCo', 'MG', 'SwinUNETR', 'VF', 'S3D']:
            colors.append('#1976D2')  # Blue for SSL
        elif method in ['BrainIAC', 'MRI-Core', 'BrainMVP']:
            colors.append('#7B1FA2')  # Purple for foundation models
        else:
            colors.append('#757575')  # Gray for others
    
    # Plot 1: AUROC comparison
    ax1 = axes[0]
    bars = ax1.barh(range(len(results_df)), results_df['Test_AUROC'], color=colors, alpha=0.8)
    ax1.set_yticks(range(len(results_df)))
    ax1.set_yticklabels(results_df['Method'])
    ax1.set_xlabel('Test AUROC')
    ax1.set_title(f'{BENCHMARK_INFO[benchmark_name]["name"]}\nTest AUROC Performance')
    ax1.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Random')
    
    # Add value labels
    for bar, val in zip(bars, results_df['Test_AUROC']):
        ax1.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', ha='left', va='center', fontsize=9)
    
    ax1.set_xlim([0.3, 1.05])
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Plot 2: All metrics comparison (for top 5 methods)
    ax2 = axes[1]
    top_5 = results_df.head(5)
    metrics = ['Test_AUROC', 'Test_AP', 'Test_F1', 'Test_Balanced_Accuracy']
    metric_labels = ['AUROC', 'AP', 'F1', 'Balanced Acc']
    
    x = np.arange(len(top_5))
    width = 0.2
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        values = top_5[metric].fillna(0).values
        ax2.bar(x + i * width, values, width, label=label, alpha=0.8)
    
    ax2.set_xlabel('Method')
    ax2.set_ylabel('Score')
    ax2.set_title('All Metrics Comparison (Top 5 Methods)')
    ax2.set_xticks(x + width * 1.5)
    ax2.set_xticklabels(top_5['Method'], rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([0, 1.05])
    
    plt.suptitle(f"Benchmark: {BENCHMARK_INFO[benchmark_name]['description']}", 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save figure
    fig_path = output_dir / 'visualization.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_benchmark_latex_table(results_df: pd.DataFrame, benchmark_name: str, output_dir: Path):
    """Create LaTeX table for a single benchmark."""
    
    # Prepare data for LaTeX
    latex_df = results_df[['Method', 'Test_AUROC', 'Test_AP', 'Test_F1', 'Test_Balanced_Accuracy']].copy()
    
    # Round values
    for col in ['Test_AUROC', 'Test_AP', 'Test_F1', 'Test_Balanced_Accuracy']:
        latex_df[col] = latex_df[col].apply(lambda x: f'{x:.3f}' if not pd.isna(x) else '-')
    
    # Highlight best values
    for col in ['Test_AUROC', 'Test_AP', 'Test_F1', 'Test_Balanced_Accuracy']:
        col_values = results_df[col].dropna()
        if len(col_values) > 0:
            max_val = col_values.max()
            max_idx = results_df[results_df[col] == max_val].index[0]
            latex_df.loc[max_idx, col] = f'\\textbf{{{latex_df.loc[max_idx, col]}}}'
    
    # Create LaTeX string
    latex = f"""\\begin{{table}}[ht]
\\centering
\\caption{{{BENCHMARK_INFO[benchmark_name]['name']} - Performance Metrics}}
\\label{{tab:{benchmark_name}}}
\\begin{{tabular}}{{lcccc}}
\\toprule
Method & AUROC & AP & F1 & Balanced Acc \\\\
\\midrule
"""
    
    for _, row in latex_df.iterrows():
        latex += f"{row['Method']} & {row['Test_AUROC']} & {row['Test_AP']} & {row['Test_F1']} & {row['Test_Balanced_Accuracy']} \\\\\n"
    
    latex += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    
    # Save LaTeX table
    latex_path = output_dir / 'latex_table.tex'
    with open(latex_path, 'w') as f:
        f.write(latex)


def create_master_comparison_table(summaries: List[Dict], output_dir: Path):
    """Create comprehensive comparison table across all benchmarks."""
    
    # Prepare data for master table
    master_data = []
    
    for summary in summaries:
        row = {
            'Benchmark': BENCHMARK_INFO[summary['benchmark']]['name'],
            'Category': BENCHMARK_INFO[summary['benchmark']]['category']
        }
        
        # Add MDAE performance
        if summary['mdae_performance']:
            row['MDAE_AUROC'] = summary['mdae_performance']['auroc']
            row['MDAE_Rank'] = summary['mdae_performance']['rank']
        else:
            row['MDAE_AUROC'] = np.nan
            row['MDAE_Rank'] = np.nan
        
        # Add top competitors
        for i, method_info in enumerate(summary['top_5_methods'][:5]):
            if method_info['method'] != 'MDAE (Combined)':
                row[f'Top_{i+1}_Method'] = method_info['method']
                row[f'Top_{i+1}_AUROC'] = method_info['auroc']
        
        master_data.append(row)
    
    master_df = pd.DataFrame(master_data)
    
    # Save master table
    master_df.to_csv(output_dir / 'master_comparison_table.csv', index=False)
    
    # Create category summary
    category_summary = master_df.groupby('Category')['MDAE_AUROC'].agg(['mean', 'std', 'count'])
    category_summary.to_csv(output_dir / 'category_summary.csv')
    
    return master_df


def create_comprehensive_report(summaries: List[Dict], output_dir: Path):
    """Create comprehensive HTML report."""
    
    html = """<!DOCTYPE html>
<html>
<head>
    <title>MDAE Benchmark Results Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #2E7D32; }
        h2 { color: #1976D2; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .best { background-color: #E8F5E9; font-weight: bold; }
        .category { margin: 30px 0; padding: 20px; border-left: 4px solid #2E7D32; background: #f9f9f9; }
    </style>
</head>
<body>
    <h1>MDAE Comprehensive Benchmark Results</h1>
"""
    
    # Group by category
    categories = {}
    for summary in summaries:
        cat = BENCHMARK_INFO[summary['benchmark']]['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(summary)
    
    # Add category sections
    for category, cat_summaries in categories.items():
        html += f"""
    <div class="category">
        <h2>{category}</h2>
        <table>
            <tr>
                <th>Benchmark</th>
                <th>MDAE AUROC</th>
                <th>MDAE Rank</th>
                <th>Best Method</th>
                <th>Best AUROC</th>
            </tr>
"""
        for s in cat_summaries:
            mdae_auroc = s['mdae_performance'].get('auroc', '-') if s['mdae_performance'] else '-'
            mdae_rank = s['mdae_performance'].get('rank', '-') if s['mdae_performance'] else '-'
            is_best = 'best' if s['best_method'] == 'MDAE (Combined)' else ''
            
            auroc_str = f"{mdae_auroc:.3f}" if isinstance(mdae_auroc, float) else str(mdae_auroc)
            html += f"""
            <tr class="{is_best}">
                <td>{BENCHMARK_INFO[s['benchmark']]['name']}</td>
                <td>{auroc_str}</td>
                <td>{mdae_rank}</td>
                <td>{s['best_method']}</td>
                <td>{s['best_auroc']:.3f}</td>
            </tr>
"""
        html += """
        </table>
    </div>
"""
    
    html += """
</body>
</html>
"""
    
    # Save HTML report
    report_path = output_dir / 'comprehensive_report.html'
    with open(report_path, 'w') as f:
        f.write(html)


def main():
    """Main processing function."""
    
    # Setup paths
    data_dir = Path('/home/t-jiachentu/repos/benchmarking/misc/data/raw_data/20250811')
    output_base = Path('/home/t-jiachentu/repos/benchmarking/misc/data/processed_data/benchmark_results')
    output_base.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("ORGANIZING BENCHMARK-SPECIFIC RESULTS")
    print("="*60)
    
    # Process each benchmark
    summaries = []
    for benchmark_name in BENCHMARK_INFO.keys():
        print(f"\nProcessing {benchmark_name}...")
        summary = process_single_benchmark(benchmark_name, data_dir, output_base)
        if summary:
            summaries.append(summary)
            print(f"  ✓ Created results directory: {output_base / benchmark_name}")
            print(f"  - MDAE Rank: {summary['mdae_performance'].get('rank', 'N/A')}")
            print(f"  - MDAE AUROC: {summary['mdae_performance'].get('auroc', 'N/A'):.3f}")
    
    # Create master comparison table
    print("\nCreating master comparison table...")
    master_df = create_master_comparison_table(summaries, output_base)
    
    # Create comprehensive report
    print("Creating comprehensive HTML report...")
    create_comprehensive_report(summaries, output_base)
    
    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    # Count rankings
    rank_counts = {1: 0, 2: 0, 3: 0}
    for s in summaries:
        if s['mdae_performance']:
            rank = s['mdae_performance']['rank']
            if rank in rank_counts:
                rank_counts[rank] += 1
    
    print(f"\nMDAE Rankings across {len(summaries)} benchmarks:")
    print(f"  #1: {rank_counts[1]} benchmarks")
    print(f"  #2: {rank_counts[2]} benchmarks")
    print(f"  #3: {rank_counts[3]} benchmarks")
    print(f"  Top-3: {sum(rank_counts.values())} benchmarks ({sum(rank_counts.values())/len(summaries)*100:.1f}%)")
    
    # Category performance
    print("\nMDAE Performance by Category:")
    categories = {}
    for s in summaries:
        cat = BENCHMARK_INFO[s['benchmark']]['category']
        if cat not in categories:
            categories[cat] = []
        if s['mdae_performance']:
            categories[cat].append(s['mdae_performance']['auroc'])
    
    for cat, scores in categories.items():
        if scores:
            print(f"  {cat:20s}: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
    
    print("\nOutput files created:")
    print(f"  - Individual benchmark directories: {output_base}/[benchmark_name]/")
    print(f"  - Master comparison table: {output_base}/master_comparison_table.csv")
    print(f"  - Category summary: {output_base}/category_summary.csv")
    print(f"  - HTML report: {output_base}/comprehensive_report.html")
    
    print("\n✓ Organization complete!")


if __name__ == "__main__":
    main()