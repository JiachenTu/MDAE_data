#!/usr/bin/env python3
"""
Create publication-ready tables and visualizations for MDAE paper.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def load_results():
    """Load processed results."""
    processed_dir = Path('/home/t-jiachentu/repos/benchmarking/misc/data/processed_data')
    
    # Load AUROC results
    auroc_df = pd.read_csv(processed_dir / 'auroc_results.csv')
    stats_df = pd.read_csv(processed_dir / 'statistics.csv')
    
    # Load other metrics
    ap_df = pd.read_csv(processed_dir / 'ap_results.csv')
    f1_df = pd.read_csv(processed_dir / 'f1_results.csv')
    ba_df = pd.read_csv(processed_dir / 'balanced_accuracy_results.csv')
    
    return auroc_df, ap_df, f1_df, ba_df, stats_df


def create_main_results_table(auroc_df, ap_df, f1_df, ba_df):
    """Create the main results table for the paper."""
    
    # Select key methods
    key_methods = ['MDAE (TC)', 'MDAE', 'MAE', 'SimCLR', 'VoCo', 'MG', 'SwinUNETR', 
                   'VF', 'S3D', 'BrainIAC', 'MRI-Core', 'BrainMVP']
    
    # Group benchmarks by category
    tumor_type = ['brats23_gli_vs_men', 'brats23_gli_vs_met', 'brats23_men_vs_met']
    tumor_grade = ['brats18_lgg_vs_hgg']
    molecular = ['rsna_miccai_mgmt_methylation', 'ucsf_pdgm_idh_classification', 
                 'upenn_gbm_idh1_status']
    survival = ['tcga_gbm_dss_1year', 'tcga_gbm_pfi_1year', 
                'upenn_gbm_survival_18month', 'upenn_gbm_survival_1year', 
                'upenn_gbm_survival_2year']
    other = ['upenn_gbm_age_group', 'upenn_gbm_gender', 'upenn_gbm_gtr_status']
    
    # Calculate average for each category
    results = []
    for method in key_methods:
        if method in auroc_df['Method'].values:
            row_data = auroc_df[auroc_df['Method'] == method].iloc[0]
            
            # Calculate category averages
            tumor_type_avg = row_data[tumor_type].mean()
            tumor_grade_avg = row_data[tumor_grade].mean()
            molecular_avg = row_data[molecular].mean()
            survival_avg = row_data[survival].mean()
            other_avg = row_data[other].mean()
            overall_avg = row_data.drop('Method').mean()
            
            results.append({
                'Method': method,
                'Tumor Type': tumor_type_avg,
                'Tumor Grade': tumor_grade_avg,
                'Molecular': molecular_avg,
                'Survival': survival_avg,
                'Other': other_avg,
                'Overall': overall_avg
            })
    
    results_df = pd.DataFrame(results)
    
    # Sort by overall performance
    results_df = results_df.sort_values('Overall', ascending=False)
    
    return results_df


def create_detailed_table(auroc_df):
    """Create detailed results table with all benchmarks."""
    
    # Select key methods
    key_methods = ['MDAE (TC)', 'MDAE', 'MAE', 'SimCLR', 'VoCo', 'BrainIAC']
    
    # Filter and transpose
    filtered_df = auroc_df[auroc_df['Method'].isin(key_methods)]
    
    # Rename columns for clarity
    column_mapping = {
        'brats18_lgg_vs_hgg': 'LGG vs HGG',
        'brats23_gli_vs_men': 'Gli vs Men',
        'brats23_gli_vs_met': 'Gli vs Met',
        'brats23_men_vs_met': 'Men vs Met',
        'rsna_miccai_mgmt_methylation': 'MGMT',
        'tcga_gbm_dss_1year': 'DSS 1Y',
        'tcga_gbm_pfi_1year': 'PFI 1Y',
        'ucsf_pdgm_idh_classification': 'IDH',
        'upenn_gbm_age_group': 'Age',
        'upenn_gbm_gender': 'Gender',
        'upenn_gbm_gtr_status': 'GTR',
        'upenn_gbm_idh1_status': 'IDH1',
        'upenn_gbm_survival_18month': 'Surv 18M',
        'upenn_gbm_survival_1year': 'Surv 1Y',
        'upenn_gbm_survival_2year': 'Surv 2Y'
    }
    
    filtered_df = filtered_df.copy()
    filtered_df.rename(columns=column_mapping, inplace=True)
    
    return filtered_df


def format_latex_table(df, bold_best=True):
    """Format DataFrame as LaTeX table with best values in bold."""
    
    if bold_best:
        # Find best value in each column (excluding Method column)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Create a copy for formatting
        df_formatted = df.copy()
        
        for col in numeric_cols:
            max_val = df[col].max()
            # Format values, bolding the best
            df_formatted[col] = df[col].apply(
                lambda x: f"\\textbf{{{x:.3f}}}" if x == max_val and not pd.isna(x) else f"{x:.3f}" if not pd.isna(x) else "-"
            )
    else:
        df_formatted = df.copy()
        numeric_cols = df_formatted.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df_formatted[col] = df_formatted[col].apply(
                lambda x: f"{x:.3f}" if not pd.isna(x) else "-"
            )
    
    return df_formatted


def create_improvement_analysis(stats_df):
    """Analyze improvements of MDAE over baselines."""
    
    # Get MDAE and MDAE (TC) performance
    mdae_perf = stats_df[stats_df['Method'] == 'MDAE']['Mean_AUROC'].values[0]
    mdae_tc_perf = stats_df[stats_df['Method'] == 'MDAE (TC)']['Mean_AUROC'].values[0]
    
    # Calculate improvements
    improvements = []
    
    for _, row in stats_df.iterrows():
        if row['Method'] not in ['MDAE', 'MDAE (TC)']:
            baseline_perf = row['Mean_AUROC']
            mdae_imp = ((mdae_perf - baseline_perf) / baseline_perf) * 100
            mdae_tc_imp = ((mdae_tc_perf - baseline_perf) / baseline_perf) * 100
            
            improvements.append({
                'Baseline': row['Method'],
                'Baseline AUROC': baseline_perf,
                'MDAE Improvement (%)': mdae_imp,
                'MDAE (TC) Improvement (%)': mdae_tc_imp
            })
    
    imp_df = pd.DataFrame(improvements)
    imp_df = imp_df.sort_values('MDAE Improvement (%)', ascending=False)
    
    return imp_df


def create_visualization(auroc_df, stats_df):
    """Create visualization comparing methods."""
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Bar plot of average performance
    ax1 = axes[0, 0]
    top_methods = stats_df.head(10).sort_values('Mean_AUROC')
    ax1.barh(top_methods['Method'], top_methods['Mean_AUROC'])
    ax1.set_xlabel('Mean AUROC')
    ax1.set_title('Average Performance Across All Tasks')
    ax1.axvline(x=0.5, color='red', linestyle='--', alpha=0.5)
    
    # Highlight our methods
    colors = ['green' if m in ['MDAE', 'MDAE (TC)'] else 'steelblue' 
              for m in top_methods['Method']]
    bars = ax1.barh(top_methods['Method'], top_methods['Mean_AUROC'], color=colors)
    
    # 2. Heatmap of results
    ax2 = axes[0, 1]
    key_methods = ['MDAE (TC)', 'MDAE', 'MAE', 'SimCLR', 'VoCo', 'BrainIAC']
    heatmap_data = auroc_df[auroc_df['Method'].isin(key_methods)].set_index('Method')
    heatmap_data = heatmap_data.iloc[:, :8]  # Select first 8 benchmarks for visibility
    
    sns.heatmap(heatmap_data.astype(float), annot=True, fmt='.2f', cmap='RdYlGn', 
                vmin=0.5, vmax=1.0, ax=ax2, cbar_kws={'label': 'AUROC'})
    ax2.set_title('Performance Heatmap (Selected Benchmarks)')
    ax2.set_xlabel('')
    
    # 3. Box plot comparison
    ax3 = axes[1, 0]
    box_data = []
    box_labels = []
    for method in ['MDAE (TC)', 'MDAE', 'MAE', 'SimCLR', 'VoCo', 'BrainIAC']:
        if method in auroc_df['Method'].values:
            values = auroc_df[auroc_df['Method'] == method].iloc[0, 1:].dropna().values
            box_data.append(values)
            box_labels.append(method)
    
    bp = ax3.boxplot(box_data, labels=box_labels, patch_artist=True)
    for i, box in enumerate(bp['boxes']):
        if box_labels[i] in ['MDAE', 'MDAE (TC)']:
            box.set_facecolor('lightgreen')
        else:
            box.set_facecolor('lightblue')
    
    ax3.set_ylabel('AUROC')
    ax3.set_title('Performance Distribution Across Tasks')
    ax3.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    ax3.grid(True, alpha=0.3)
    
    # 4. Improvement plot
    ax4 = axes[1, 1]
    improvements = []
    baseline_methods = ['MAE', 'SimCLR', 'VoCo', 'BrainIAC', 'MRI-Core', 'BrainMVP']
    
    mdae_mean = stats_df[stats_df['Method'] == 'MDAE']['Mean_AUROC'].values[0]
    mdae_tc_mean = stats_df[stats_df['Method'] == 'MDAE (TC)']['Mean_AUROC'].values[0]
    
    for baseline in baseline_methods:
        if baseline in stats_df['Method'].values:
            baseline_mean = stats_df[stats_df['Method'] == baseline]['Mean_AUROC'].values[0]
            improvements.append({
                'Method': baseline,
                'MDAE': (mdae_mean - baseline_mean) * 100,
                'MDAE (TC)': (mdae_tc_mean - baseline_mean) * 100
            })
    
    imp_df = pd.DataFrame(improvements)
    x = np.arange(len(imp_df))
    width = 0.35
    
    ax4.bar(x - width/2, imp_df['MDAE'], width, label='MDAE', color='green', alpha=0.7)
    ax4.bar(x + width/2, imp_df['MDAE (TC)'], width, label='MDAE (TC)', color='darkgreen', alpha=0.7)
    
    ax4.set_xlabel('Baseline Method')
    ax4.set_ylabel('Improvement (%)')
    ax4.set_title('AUROC Improvement over Baselines')
    ax4.set_xticks(x)
    ax4.set_xticklabels(imp_df['Method'], rotation=45)
    ax4.legend()
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig


def main():
    """Main function to generate all tables and figures."""
    
    output_dir = Path('/home/t-jiachentu/repos/benchmarking/misc/data/processed_data')
    
    # Load results
    print("Loading results...")
    auroc_df, ap_df, f1_df, ba_df, stats_df = load_results()
    
    # Create main results table
    print("\nCreating main results table...")
    main_table = create_main_results_table(auroc_df, ap_df, f1_df, ba_df)
    print("\nMain Results Table (AUROC by Category):")
    print(main_table.to_string(index=False, float_format='%.3f'))
    
    # Create detailed table
    print("\nCreating detailed results table...")
    detailed_table = create_detailed_table(auroc_df)
    
    # Format for LaTeX
    main_table_latex = format_latex_table(main_table)
    detailed_table_latex = format_latex_table(detailed_table)
    
    # Save LaTeX tables
    latex_output = output_dir / 'paper_tables.tex'
    with open(latex_output, 'w') as f:
        f.write("% Main Results Table (Category Averages)\n")
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write("\\caption{Average test AUROC performance across benchmark categories. ")
        f.write("Bold indicates best performance in each category.}\n")
        f.write("\\label{tab:main_results}\n")
        f.write("\\resizebox{\\textwidth}{!}{%\n")
        f.write(main_table_latex.to_latex(index=False, escape=False))
        f.write("}\n")
        f.write("\\end{table}\n\n")
        
        f.write("% Detailed Results Table\n")
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write("\\caption{Detailed test AUROC results for all benchmarks. ")
        f.write("Bold indicates best performance in each benchmark.}\n")
        f.write("\\label{tab:detailed_results}\n")
        f.write("\\resizebox{\\textwidth}{!}{%\n")
        f.write(detailed_table_latex.to_latex(index=False, escape=False))
        f.write("}\n")
        f.write("\\end{table}\n")
    
    print(f"\nLaTeX tables saved to: {latex_output}")
    
    # Create improvement analysis
    print("\nAnalyzing improvements...")
    improvements = create_improvement_analysis(stats_df)
    print("\nImprovement Analysis:")
    print(improvements.to_string(index=False, float_format='%.2f'))
    
    # Save improvement analysis
    improvements.to_csv(output_dir / 'improvement_analysis.csv', index=False)
    
    # Create visualizations
    print("\nCreating visualizations...")
    fig = create_visualization(auroc_df, stats_df)
    fig.savefig(output_dir / 'mdae_results_visualization.png', dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_dir / 'mdae_results_visualization.png'}")
    
    # Summary for paper
    print("\n" + "="*60)
    print("SUMMARY FOR PAPER")
    print("="*60)
    
    mdae_mean = stats_df[stats_df['Method'] == 'MDAE']['Mean_AUROC'].values[0]
    mdae_tc_mean = stats_df[stats_df['Method'] == 'MDAE (TC)']['Mean_AUROC'].values[0]
    
    print(f"\nMDAE Average AUROC: {mdae_mean:.1%}")
    print(f"MDAE (TC) Average AUROC: {mdae_tc_mean:.1%}")
    
    print("\nKey Improvements (MDAE with Time Conditioning):")
    key_baselines = ['MAE', 'SimCLR', 'VoCo', 'BrainIAC']
    for baseline in key_baselines:
        if baseline in stats_df['Method'].values:
            baseline_mean = stats_df[stats_df['Method'] == baseline]['Mean_AUROC'].values[0]
            improvement = (mdae_tc_mean - baseline_mean) * 100
            print(f"  vs {baseline:10s}: +{improvement:.1f}% absolute improvement")
    
    print("\nAverage improvement over SSL baselines: ", end="")
    ssl_baselines = ['MAE', 'SimCLR', 'VoCo', 'MG', 'SwinUNETR', 'VF', 'S3D']
    ssl_improvements = []
    for baseline in ssl_baselines:
        if baseline in stats_df['Method'].values:
            baseline_mean = stats_df[stats_df['Method'] == baseline]['Mean_AUROC'].values[0]
            ssl_improvements.append((mdae_tc_mean - baseline_mean) * 100)
    print(f"{np.mean(ssl_improvements):.1f}%")
    
    print("\nAverage improvement over foundation models: ", end="")
    fm_baselines = ['BrainIAC', 'MRI-Core', 'BrainMVP']
    fm_improvements = []
    for baseline in fm_baselines:
        if baseline in stats_df['Method'].values:
            baseline_mean = stats_df[stats_df['Method'] == baseline]['Mean_AUROC'].values[0]
            fm_improvements.append((mdae_tc_mean - baseline_mean) * 100)
    print(f"{np.mean(fm_improvements):.1f}%")


if __name__ == "__main__":
    main()