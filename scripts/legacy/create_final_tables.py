#!/usr/bin/env python3
"""
Create final comprehensive tables with AUROC and AP metrics for all methods.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def create_comprehensive_excel_tables():
    """Create Excel-friendly comprehensive tables."""
    
    # Load the comprehensive CSV
    input_path = Path('/home/t-jiachentu/repos/benchmarking/misc/data/processed_data/benchmark_results_final/comprehensive_auroc_ap_table.csv')
    output_dir = Path('/home/t-jiachentu/repos/benchmarking/misc/data/processed_data/benchmark_results_final')
    
    comprehensive_df = pd.read_csv(input_path)
    
    # Create separate AUROC and AP tables
    auroc_cols = ['Method'] + [col for col in comprehensive_df.columns if 'AUROC' in col]
    ap_cols = ['Method'] + [col for col in comprehensive_df.columns if 'AP' in col and 'Mean' not in col and 'Std' not in col and 'Count' not in col]
    
    auroc_df = comprehensive_df[auroc_cols].copy()
    ap_df = comprehensive_df[ap_cols].copy()
    
    # Round values
    for col in auroc_df.columns[1:]:
        if col in auroc_df.columns:
            auroc_df[col] = auroc_df[col].round(3)
    
    for col in ap_df.columns[1:]:
        if col in ap_df.columns:
            ap_df[col] = ap_df[col].round(3)
    
    # Save to CSV
    auroc_df.to_csv(output_dir / 'all_methods_auroc_table.csv', index=False)
    ap_df.to_csv(output_dir / 'all_methods_ap_table.csv', index=False)
    
    # Create a compact summary table
    summary_cols = ['Method', 'Mean_AUROC', 'Std_AUROC', 'Mean_AP', 'Std_AP']
    summary_df = comprehensive_df[summary_cols].copy()
    summary_df = summary_df.round(3)
    summary_df = summary_df.sort_values('Mean_AUROC', ascending=False)
    
    # Add ranking
    summary_df['AUROC_Rank'] = range(1, len(summary_df) + 1)
    
    # Save summary
    summary_df.to_csv(output_dir / 'methods_summary_statistics.csv', index=False)
    
    print("="*60)
    print("COMPREHENSIVE TABLES CREATED")
    print("="*60)
    
    print("\nMethods Summary (Top 10):")
    print(summary_df.head(10).to_string(index=False))
    
    print("\nFiles created:")
    print(f"  - all_methods_auroc_table.csv")
    print(f"  - all_methods_ap_table.csv") 
    print(f"  - methods_summary_statistics.csv")
    
    # Create LaTeX table for paper
    create_paper_table(summary_df.head(12), output_dir)


def create_paper_table(df: pd.DataFrame, output_dir: Path):
    """Create LaTeX table for paper."""
    
    # Prepare data for LaTeX
    latex_df = df[['Method', 'Mean_AUROC', 'Std_AUROC', 'Mean_AP', 'Std_AP']].copy()
    
    # Format with mean ± std
    latex_df['AUROC'] = latex_df.apply(lambda x: f"{x['Mean_AUROC']:.3f} ± {x['Std_AUROC']:.3f}", axis=1)
    latex_df['AP'] = latex_df.apply(lambda x: f"{x['Mean_AP']:.3f} ± {x['Std_AP']:.3f}", axis=1)
    
    # Keep only formatted columns
    latex_df = latex_df[['Method', 'AUROC', 'AP']]
    
    # Bold MDAE
    latex_df.loc[latex_df['Method'] == 'MDAE', 'Method'] = '\\textbf{MDAE (Ours)}'
    latex_df.loc[latex_df['Method'] == 'MDAE', 'AUROC'] = '\\textbf{' + latex_df.loc[latex_df['Method'] == '\\textbf{MDAE (Ours)}', 'AUROC'].values[0] + '}'
    latex_df.loc[latex_df['Method'] == '\\textbf{MDAE (Ours)}', 'AP'] = '\\textbf{' + latex_df.loc[latex_df['Method'] == '\\textbf{MDAE (Ours)}', 'AP'].values[0] + '}'
    
    # Generate LaTeX
    latex = """\\begin{table}[ht]
\\centering
\\caption{Average test AUROC and AP performance across all benchmarks and modalities. Results show mean ± standard deviation across all evaluation settings.}
\\label{tab:comprehensive_metrics}
\\begin{tabular}{lcc}
\\toprule
Method & AUROC & AP \\\\
\\midrule
"""
    
    for _, row in latex_df.iterrows():
        latex += f"{row['Method']} & {row['AUROC']} & {row['AP']} \\\\\n"
    
    latex += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    
    # Save LaTeX table
    latex_path = output_dir / 'paper_comprehensive_table.tex'
    with open(latex_path, 'w') as f:
        f.write(latex)
    
    print(f"\nLaTeX table saved to: paper_comprehensive_table.tex")


if __name__ == "__main__":
    create_comprehensive_excel_tables()