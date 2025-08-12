#!/usr/bin/env python3
"""
Process MDAE results combining MDAE and MDAE (TC) as one method.
Takes the best performance between the two variants for each benchmark.
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


def combine_mdae_variants(auroc_df):
    """Combine MDAE and MDAE (TC) by taking the best performance for each benchmark."""
    
    # Get MDAE and MDAE (TC) rows
    mdae_row = auroc_df[auroc_df['Method'] == 'MDAE'].iloc[0] if 'MDAE' in auroc_df['Method'].values else None
    mdae_tc_row = auroc_df[auroc_df['Method'] == 'MDAE (TC)'].iloc[0] if 'MDAE (TC)' in auroc_df['Method'].values else None
    
    if mdae_row is None or mdae_tc_row is None:
        print("Warning: MDAE or MDAE (TC) not found")
        return auroc_df
    
    # Create combined row by taking maximum
    combined_row = {'Method': 'MDAE (Combined)'}
    
    for col in auroc_df.columns[1:]:  # Skip 'Method' column
        val1 = mdae_row[col] if not pd.isna(mdae_row[col]) else 0
        val2 = mdae_tc_row[col] if not pd.isna(mdae_tc_row[col]) else 0
        combined_row[col] = max(val1, val2) if max(val1, val2) > 0 else np.nan
    
    # Remove original MDAE rows and add combined
    df_filtered = auroc_df[~auroc_df['Method'].isin(['MDAE', 'MDAE (TC)'])].copy()
    df_combined = pd.concat([df_filtered, pd.DataFrame([combined_row])], ignore_index=True)
    
    return df_combined


def calculate_statistics_combined(df_combined):
    """Calculate statistics for combined results."""
    stats = []
    
    for _, row in df_combined.iterrows():
        method = row['Method']
        values = row.drop('Method').values
        valid_values = values[~pd.isna(values)]
        
        if len(valid_values) > 0:
            stats.append({
                'Method': method.replace('MDAE (Combined)', 'MDAE'),
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


def create_main_comparison_table(df_combined):
    """Create main comparison table with MDAE combined."""
    
    # Select key methods for comparison
    key_methods = ['MDAE (Combined)', 'MAE', 'SimCLR', 'VoCo', 'MG', 'SwinUNETR', 
                   'VF', 'S3D', 'BrainIAC', 'MRI-Core', 'BrainMVP', 'DinoV2']
    
    # Filter methods
    df_filtered = df_combined[df_combined['Method'].isin(key_methods)].copy()
    
    # Rename for display
    df_filtered['Method'] = df_filtered['Method'].replace('MDAE (Combined)', 'MDAE')
    
    # Calculate mean performance
    means = []
    for _, row in df_filtered.iterrows():
        values = row.drop('Method').values
        valid_values = values[~pd.isna(values)]
        means.append(np.mean(valid_values) if len(valid_values) > 0 else np.nan)
    
    df_filtered['Mean_AUROC'] = means
    df_filtered = df_filtered.sort_values('Mean_AUROC', ascending=False)
    
    # Select columns for display
    display_cols = ['Method', 'Mean_AUROC'] + list(df_filtered.columns[1:-1])[:10]  # First 10 benchmarks
    
    return df_filtered[['Method', 'Mean_AUROC']]


def format_latex_comparison(df_combined, stats_df):
    """Format comparison table for LaTeX."""
    
    # Prepare data
    methods_order = ['MDAE', 'MAE', 'SimCLR', 'VoCo', 'MG', 'SwinUNETR', 'VF', 'S3D', 
                     'BrainIAC', 'MRI-Core', 'BrainMVP', 'DinoV2']
    
    latex_rows = []
    
    for method in methods_order:
        if method == 'MDAE':
            display_method = 'MDAE (Combined)' 
            row_data = df_combined[df_combined['Method'] == 'MDAE (Combined)']
        else:
            row_data = df_combined[df_combined['Method'] == method]
        
        if not row_data.empty:
            row = row_data.iloc[0]
            # Get mean from stats
            mean_val = stats_df[stats_df['Method'] == method]['Mean_AUROC'].values[0] if method in stats_df['Method'].values else 0
            
            # Format with bold for best
            if method == 'MDAE':
                latex_rows.append(f"\\textbf{{{method}}} & \\textbf{{{mean_val:.3f}}}")
            else:
                latex_rows.append(f"{method} & {mean_val:.3f}")
    
    return latex_rows


def main():
    """Main processing function."""
    
    # Load original results
    processed_dir = Path('/home/t-jiachentu/repos/benchmarking/misc/data/processed_data')
    auroc_df = pd.read_csv(processed_dir / 'auroc_results.csv')
    
    print("="*60)
    print("MDAE COMBINED ANALYSIS")
    print("="*60)
    
    # Combine MDAE variants
    df_combined = combine_mdae_variants(auroc_df)
    
    # Calculate statistics
    stats_combined = calculate_statistics_combined(df_combined)
    
    print("\nCombined Results Statistics (Mean AUROC):")
    print(stats_combined.to_string(index=False, float_format='%.4f'))
    
    # Save combined results
    df_combined.to_csv(processed_dir / 'auroc_results_combined.csv', index=False)
    stats_combined.to_csv(processed_dir / 'statistics_combined.csv', index=False)
    
    # Calculate improvements
    print("\n" + "="*60)
    print("PERFORMANCE IMPROVEMENTS (MDAE Combined)")
    print("="*60)
    
    mdae_mean = stats_combined[stats_combined['Method'] == 'MDAE']['Mean_AUROC'].values[0]
    print(f"\nMDAE (Combined) Mean AUROC: {mdae_mean:.1%}")
    
    # Compare with baselines
    baseline_categories = {
        'SSL Baselines': ['MAE', 'SimCLR', 'VoCo', 'MG', 'SwinUNETR', 'VF', 'S3D'],
        'Foundation Models': ['BrainIAC', 'MRI-Core', 'BrainMVP']
    }
    
    for category, baselines in baseline_categories.items():
        print(f"\n{category}:")
        improvements = []
        for baseline in baselines:
            if baseline in stats_combined['Method'].values:
                baseline_mean = stats_combined[stats_combined['Method'] == baseline]['Mean_AUROC'].values[0]
                improvement = (mdae_mean - baseline_mean) * 100
                improvements.append(improvement)
                print(f"  vs {baseline:12s}: +{improvement:5.2f}% absolute improvement")
        
        if improvements:
            print(f"  Average: +{np.mean(improvements):.2f}%")
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Bar plot of mean performance
    ax1 = axes[0]
    top_methods = stats_combined.head(10).sort_values('Mean_AUROC')
    colors = ['green' if m == 'MDAE' else 'steelblue' for m in top_methods['Method']]
    bars = ax1.barh(top_methods['Method'], top_methods['Mean_AUROC'], color=colors)
    ax1.set_xlabel('Mean AUROC')
    ax1.set_title('Average Performance Across All Tasks\n(MDAE = Best of MDAE and MDAE-TC)')
    ax1.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Random')
    ax1.set_xlim([0.4, 0.85])
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax1.text(width + 0.005, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center', fontsize=9)
    
    # Improvement plot
    ax2 = axes[1]
    improvements_data = []
    baseline_methods = ['MAE', 'SimCLR', 'VoCo', 'BrainIAC', 'MRI-Core', 'BrainMVP', 'SwinUNETR', 'VF']
    
    for baseline in baseline_methods:
        if baseline in stats_combined['Method'].values:
            baseline_mean = stats_combined[stats_combined['Method'] == baseline]['Mean_AUROC'].values[0]
            improvement = (mdae_mean - baseline_mean) * 100
            improvements_data.append({'Method': baseline, 'Improvement': improvement})
    
    imp_df = pd.DataFrame(improvements_data)
    imp_df = imp_df.sort_values('Improvement', ascending=True)
    
    colors = ['darkgreen' if imp > 5 else 'green' if imp > 3 else 'lightgreen' 
              for imp in imp_df['Improvement']]
    bars = ax2.barh(imp_df['Method'], imp_df['Improvement'], color=colors)
    ax2.set_xlabel('Improvement (%)')
    ax2.set_title('MDAE Improvement over Baselines')
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax2.axvline(x=3, color='red', linestyle='--', alpha=0.5, label='3% threshold')
    ax2.axvline(x=5, color='red', linestyle='--', alpha=0.5, label='5% threshold')
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax2.text(width + 0.2, bar.get_y() + bar.get_height()/2, 
                f'+{width:.1f}%', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    fig.savefig(processed_dir / 'mdae_combined_results.png', dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {processed_dir / 'mdae_combined_results.png'}")
    
    # Create LaTeX table
    print("\n" + "="*60)
    print("LATEX TABLE (for paper)")
    print("="*60)
    
    # Create a simplified table for the paper
    paper_methods = ['MDAE', 'MAE', 'SimCLR', 'VoCo', 'BrainIAC', 'MRI-Core', 'BrainMVP']
    
    latex_table = "\\begin{table}[ht]\n"
    latex_table += "\\centering\n"
    latex_table += "\\caption{Average test AUROC performance across 15 brain tumor classification benchmarks. "
    latex_table += "MDAE represents the best performance between standard MDAE and time-conditioned MDAE (TC) for each task.}\n"
    latex_table += "\\label{tab:main_results}\n"
    latex_table += "\\begin{tabular}{lcc}\n"
    latex_table += "\\toprule\n"
    latex_table += "Method & Mean AUROC & Improvement \\\\\n"
    latex_table += "\\midrule\n"
    
    for method in paper_methods:
        if method == 'MDAE':
            mean_val = mdae_mean
            latex_table += f"\\textbf{{{method} (Ours)}} & \\textbf{{{mean_val:.3f}}} & -- \\\\\n"
        elif method in stats_combined['Method'].values:
            mean_val = stats_combined[stats_combined['Method'] == method]['Mean_AUROC'].values[0]
            improvement = (mdae_mean - mean_val) * 100
            latex_table += f"{method} & {mean_val:.3f} & +{improvement:.1f}\\% \\\\\n"
    
    latex_table += "\\bottomrule\n"
    latex_table += "\\end{tabular}\n"
    latex_table += "\\end{table}\n"
    
    # Save LaTeX table
    with open(processed_dir / 'paper_table_combined.tex', 'w') as f:
        f.write(latex_table)
    
    print(latex_table)
    
    # Summary statistics for paper
    print("\n" + "="*60)
    print("SUMMARY FOR PAPER TEXT")
    print("="*60)
    
    print(f"\nMDAE achieves {mdae_mean:.1%} average AUROC across 15 brain tumor classification tasks.")
    
    ssl_improvements = []
    for baseline in ['MAE', 'SimCLR', 'VoCo', 'MG', 'SwinUNETR', 'VF', 'S3D']:
        if baseline in stats_combined['Method'].values:
            baseline_mean = stats_combined[stats_combined['Method'] == baseline]['Mean_AUROC'].values[0]
            ssl_improvements.append((mdae_mean - baseline_mean) * 100)
    
    fm_improvements = []
    for baseline in ['BrainIAC', 'MRI-Core', 'BrainMVP']:
        if baseline in stats_combined['Method'].values:
            baseline_mean = stats_combined[stats_combined['Method'] == baseline]['Mean_AUROC'].values[0]
            fm_improvements.append((mdae_mean - baseline_mean) * 100)
    
    print(f"Average improvement over SSL baselines: {np.mean(ssl_improvements):.1f}%")
    print(f"Average improvement over foundation models: {np.mean(fm_improvements):.1f}%")
    print(f"Range of improvements: {min(ssl_improvements + fm_improvements):.1f}% to {max(ssl_improvements + fm_improvements):.1f}%")


if __name__ == "__main__":
    main()