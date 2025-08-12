#!/usr/bin/env python3
"""
Process MDAE benchmarking results for ICLR 2026 paper.
Extracts best results for each method based on Test_AUROC and generates analysis.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import re
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Define method patterns based on key_baselines.txt
# Order matters! More specific patterns should come first
METHOD_PATTERNS = {
    # Our Methods (most specific patterns first)
    'MDAE': r'^resenc_MDAETrainer_RandomMask_Flow_BS48_2000ep_pretrained',
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

# Define benchmark names mapping
BENCHMARK_NAMES = {
    'brats18_lgg_vs_hgg': 'BraTS18 LGG vs HGG',
    'brats23_gli_vs_men': 'BraTS23 Glioma vs Meningioma',
    'brats23_gli_vs_met': 'BraTS23 Glioma vs Metastasis',
    'brats23_men_vs_met': 'BraTS23 Meningioma vs Metastasis',
    'rsna_miccai_mgmt_methylation': 'MGMT Methylation',
    'tcga_gbm_dss_1year': 'TCGA-GBM DSS 1-Year',
    'tcga_gbm_pfi_1year': 'TCGA-GBM PFI 1-Year',
    'ucsf_pdgm_idh_classification': 'IDH Classification',
    'upenn_gbm_age_group': 'UPenn-GBM Age Group',
    'upenn_gbm_gender': 'UPenn-GBM Gender',
    'upenn_gbm_gtr_status': 'UPenn-GBM GTR Status',
    'upenn_gbm_idh1_status': 'UPenn-GBM IDH1 Status',
    'upenn_gbm_survival_18month': 'UPenn-GBM Survival 18-Month',
    'upenn_gbm_survival_1year': 'UPenn-GBM Survival 1-Year',
    'upenn_gbm_survival_2year': 'UPenn-GBM Survival 2-Year',
}

# Test metrics to extract
TEST_METRICS = ['Test_AUROC', 'Test_AP', 'Test_F1', 'Test_Balanced_Accuracy']


def identify_method(run_name: str) -> str:
    """Identify which method a run belongs to based on its name."""
    for method, pattern in METHOD_PATTERNS.items():
        if re.match(pattern, run_name):
            return method
    return 'Unknown'


def process_benchmark_data(csv_path: Path) -> pd.DataFrame:
    """Process a single benchmark CSV file and extract best results per method."""
    df = pd.read_csv(csv_path)
    
    # Add method column
    df['Method'] = df['run_name'].apply(identify_method)
    
    # Filter out unknown methods
    df = df[df['Method'] != 'Unknown']
    
    # Filter out rows with NaN Test_AUROC
    df = df.dropna(subset=['Test_AUROC'])
    
    # Group by method and get the run with highest Test_AUROC
    best_runs = []
    for method in df['Method'].unique():
        method_df = df[df['Method'] == method]
        if not method_df.empty and not method_df['Test_AUROC'].isna().all():
            best_run = method_df.loc[method_df['Test_AUROC'].idxmax()]
            best_runs.append(best_run)
    
    if best_runs:
        result_df = pd.DataFrame(best_runs)
        return result_df[['Method'] + TEST_METRICS]
    else:
        return pd.DataFrame()


def process_all_benchmarks(data_dir: Path) -> Dict[str, pd.DataFrame]:
    """Process all benchmark datasets."""
    results = {}
    
    for benchmark_dir in sorted(data_dir.iterdir()):
        if benchmark_dir.is_dir():
            csv_path = benchmark_dir / 'runs_summary.csv'
            if csv_path.exists():
                benchmark_name = benchmark_dir.name
                print(f"Processing {benchmark_name}...")
                results[benchmark_name] = process_benchmark_data(csv_path)
    
    return results


def create_summary_table(results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Create a summary table with all methods across all benchmarks."""
    all_methods = set()
    for df in results.values():
        if not df.empty:
            all_methods.update(df['Method'].unique())
    
    all_methods = sorted(all_methods)
    benchmarks = sorted(results.keys())
    
    # Create table for each metric
    summary_tables = {}
    
    for metric in TEST_METRICS:
        table_data = []
        for method in all_methods:
            row = {'Method': method}
            for benchmark in benchmarks:
                if not results[benchmark].empty:
                    method_data = results[benchmark][results[benchmark]['Method'] == method]
                    if not method_data.empty:
                        row[benchmark] = method_data.iloc[0][metric]
                    else:
                        row[benchmark] = np.nan
                else:
                    row[benchmark] = np.nan
            table_data.append(row)
        
        summary_tables[metric] = pd.DataFrame(table_data)
    
    return summary_tables


def calculate_statistics(summary_tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Calculate average performance and statistics for each method."""
    auroc_table = summary_tables['Test_AUROC']
    
    # Calculate statistics
    stats = []
    for _, row in auroc_table.iterrows():
        method = row['Method']
        values = row.drop('Method').values
        valid_values = values[~pd.isna(values)]
        
        if len(valid_values) > 0:
            stats.append({
                'Method': method,
                'Mean_AUROC': np.mean(valid_values),
                'Std_AUROC': np.std(valid_values),
                'Median_AUROC': np.median(valid_values),
                'Count': len(valid_values),
                'Min_AUROC': np.min(valid_values),
                'Max_AUROC': np.max(valid_values)
            })
    
    stats_df = pd.DataFrame(stats)
    stats_df = stats_df.sort_values('Mean_AUROC', ascending=False)
    
    return stats_df


def generate_latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    """Generate LaTeX table from DataFrame."""
    # Round numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_rounded = df.copy()
    for col in numeric_cols:
        df_rounded[col] = df_rounded[col].round(3)
    
    # Replace NaN with -
    df_rounded = df_rounded.fillna('-')
    
    # Generate LaTeX
    latex = df_rounded.to_latex(index=False, escape=False)
    
    # Add caption and label
    latex = latex.replace('\\begin{table}', f'\\begin{{table}}[ht]\n\\centering\n\\caption{{{caption}}}\n\\label{{{label}}}')
    
    return latex


def main():
    """Main processing function."""
    data_dir = Path('/home/t-jiachentu/repos/benchmarking/misc/data/raw_data/20250811')
    output_dir = Path('/home/t-jiachentu/repos/benchmarking/misc/data/processed_data')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process all benchmarks
    print("=" * 50)
    print("Processing MDAE Benchmarking Results")
    print("=" * 50)
    
    results = process_all_benchmarks(data_dir)
    
    # Create summary tables
    print("\nCreating summary tables...")
    summary_tables = create_summary_table(results)
    
    # Calculate statistics
    print("Calculating statistics...")
    stats_df = calculate_statistics(summary_tables)
    
    # Save results
    print("\nSaving results...")
    
    # Save summary tables to CSV files instead
    for metric, df in summary_tables.items():
        # Clean column names for display
        df_display = df.copy()
        df_display.columns = ['Method'] + [BENCHMARK_NAMES.get(col, col) for col in df.columns[1:]]
        csv_path = output_dir / f'{metric.lower().replace("test_", "")}_results.csv'
        df_display.to_csv(csv_path, index=False)
        print(f"{metric} results saved to: {csv_path}")
    
    # Save statistics
    stats_csv_path = output_dir / 'statistics.csv'
    stats_df.to_csv(stats_csv_path, index=False)
    print(f"Statistics saved to: {stats_csv_path}")
    
    # Save AUROC table as CSV for easier processing
    auroc_csv_path = output_dir / 'auroc_results.csv'
    summary_tables['Test_AUROC'].to_csv(auroc_csv_path, index=False)
    print(f"AUROC results saved to: {auroc_csv_path}")
    
    # Print summary statistics
    print("\n" + "=" * 50)
    print("SUMMARY STATISTICS (Test AUROC)")
    print("=" * 50)
    print(stats_df.to_string(index=False))
    
    # Calculate improvements
    print("\n" + "=" * 50)
    print("PERFORMANCE IMPROVEMENTS")
    print("=" * 50)
    
    if 'MDAE' in stats_df['Method'].values and 'MDAE (TC)' in stats_df['Method'].values:
        mdae_auroc = stats_df[stats_df['Method'] == 'MDAE']['Mean_AUROC'].values[0]
        mdae_tc_auroc = stats_df[stats_df['Method'] == 'MDAE (TC)']['Mean_AUROC'].values[0]
        
        print(f"\nMDAE Mean AUROC: {mdae_auroc:.3f}")
        print(f"MDAE (TC) Mean AUROC: {mdae_tc_auroc:.3f}")
        
        # Compare with baselines
        baseline_methods = ['MAE', 'SimCLR', 'VoCo', 'BrainIAC', 'MRI-Core', 'BrainMVP']
        print("\nImprovements over baselines:")
        for baseline in baseline_methods:
            if baseline in stats_df['Method'].values:
                baseline_auroc = stats_df[stats_df['Method'] == baseline]['Mean_AUROC'].values[0]
                mdae_improvement = (mdae_auroc - baseline_auroc) * 100
                mdae_tc_improvement = (mdae_tc_auroc - baseline_auroc) * 100
                print(f"  vs {baseline:12s}: MDAE +{mdae_improvement:5.2f}%, MDAE(TC) +{mdae_tc_improvement:5.2f}%")
    
    # Generate LaTeX tables
    print("\n" + "=" * 50)
    print("GENERATING LATEX TABLES")
    print("=" * 50)
    
    # Main results table (AUROC)
    auroc_table = summary_tables['Test_AUROC'].copy()
    auroc_table.columns = ['Method'] + [BENCHMARK_NAMES.get(col, col) for col in auroc_table.columns[1:]]
    
    # Select key methods for paper
    key_methods = ['MDAE (TC)', 'MDAE', 'MAE', 'SimCLR', 'VoCo', 'BrainIAC', 'DinoV2']
    auroc_paper = auroc_table[auroc_table['Method'].isin(key_methods)]
    
    latex_table = generate_latex_table(auroc_paper, 
                                       "Test AUROC results across all benchmarks",
                                       "tab:main_results")
    
    latex_path = output_dir / 'latex_tables.tex'
    with open(latex_path, 'w') as f:
        f.write(latex_table)
    
    print(f"LaTeX tables saved to: {latex_path}")
    
    print("\n" + "=" * 50)
    print("PROCESSING COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    main()