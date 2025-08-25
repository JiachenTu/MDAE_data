#!/usr/bin/env python3
"""
Generate formatted LaTeX tables with best and second-best highlighting
"""

import pandas as pd
import numpy as np
from pathlib import Path

def format_value(value, is_best=False, is_second=False):
    """Format a value with LaTeX markup."""
    if pd.isna(value) or value == '-':
        return '-'
    
    formatted = f"{value:.3f}" if isinstance(value, (int, float)) else str(value)
    
    if is_best:
        return f"\\textbf{{{formatted}}}"
    elif is_second:
        return f"\\underline{{{formatted}}}"
    else:
        return formatted

def generate_formatted_latex(df, benchmark_name, label):
    """Generate formatted LaTeX table with best/second-best highlighting."""
    if df.empty:
        return ""
    
    # Start LaTeX table
    latex_str = f"% {benchmark_name} - Complete Baselines with Best/Second-Best Highlighting\n"
    latex_str += "\\begin{table}[h!]\n\\centering\n"
    latex_str += f"\\caption{{{benchmark_name} - All Baselines}}\n"
    latex_str += f"\\label{{tab:{label}}}\n"
    latex_str += "\\begin{tabular}{lcccc}\n\\hline\n"
    latex_str += "Method & AUROC & AP & F1 & Bal. Acc. \\\\\n\\hline\n"
    
    # Get metrics columns
    metrics = ['Test_AUROC', 'Test_AP', 'Test_F1', 'Test_Balanced_Accuracy']
    
    # Find best and second-best for each metric (excluding failed/no data)
    best_indices = {}
    second_indices = {}
    
    for metric in metrics:
        valid_values = df[df['Status'] == 'Success'][metric].dropna()
        if len(valid_values) >= 1:
            sorted_values = valid_values.sort_values(ascending=False)
            best_indices[metric] = sorted_values.index[0] if len(sorted_values) > 0 else None
            second_indices[metric] = sorted_values.index[1] if len(sorted_values) > 1 else None
    
    # Generate rows
    for idx, row in df.iterrows():
        method = row['Method']
        
        # Clean up method name
        if method == 'MDAE (Combined)':
            method = 'MDAE'
        
        # Check if row should be commented (for methods with no successful runs)
        should_comment = row['Status'] != 'Success'
        comment_prefix = "% " if should_comment else ""
        
        # Format each metric value
        formatted_values = []
        for metric in metrics:
            value = row[metric]
            is_best = idx == best_indices.get(metric)
            is_second = idx == second_indices.get(metric)
            formatted_values.append(format_value(value, is_best, is_second))
        
        # Special case: always bold MDAE for AUROC if it's best
        if 'MDAE' in method and idx == best_indices.get('Test_AUROC'):
            method = f"\\textbf{{{method}}}"
        
        # Build row
        latex_str += f"{comment_prefix}{method} & {formatted_values[0]} & {formatted_values[1]} & {formatted_values[2]} & {formatted_values[3]} \\\\\n"
    
    latex_str += "\\hline\n\\end{tabular}\n\\end{table}\n"
    return latex_str

def main():
    """Generate formatted LaTeX tables."""
    # Load data
    brats_df = pd.read_csv('processed_data_20250815_complete/brats18_complete_baselines.csv')
    ucsf_df = pd.read_csv('processed_data_20250815_complete/ucsf_complete_baselines.csv')
    
    # Generate LaTeX
    latex_content = ""
    latex_content += generate_formatted_latex(brats_df, "BraTS18 LGG vs HGG", "brats18_complete_baselines")
    latex_content += "\n"
    latex_content += generate_formatted_latex(ucsf_df, "UCSF-PDGM IDH Classification", "ucsf_complete_baselines")
    
    # Save to file
    output_path = Path('processed_data_20250815_complete/latex_tables/complete_baselines_auto_formatted.tex')
    with open(output_path, 'w') as f:
        f.write(latex_content)
    
    print(f"Formatted LaTeX tables saved to: {output_path}")
    
    # Print summary
    print("\nFormatting Summary:")
    print("-" * 40)
    
    for df, name in [(brats_df, 'BraTS18'), (ucsf_df, 'UCSF')]:
        print(f"\n{name}:")
        successful = df[df['Status'] == 'Success']
        
        for metric, display in [('Test_AUROC', 'AUROC'), ('Test_AP', 'AP'), 
                                ('Test_F1', 'F1'), ('Test_Balanced_Accuracy', 'Bal.Acc')]:
            if metric in successful.columns:
                valid = successful[metric].dropna()
                if len(valid) > 0:
                    best_idx = valid.idxmax()
                    best_method = df.loc[best_idx, 'Method']
                    best_value = valid.max()
                    
                    if len(valid) > 1:
                        sorted_valid = valid.sort_values(ascending=False)
                        second_idx = sorted_valid.index[1]
                        second_method = df.loc[second_idx, 'Method']
                        second_value = sorted_valid.iloc[1]
                        print(f"  {display}: Best={best_method} ({best_value:.3f}), "
                              f"Second={second_method} ({second_value:.3f})")
                    else:
                        print(f"  {display}: Best={best_method} ({best_value:.3f})")

if __name__ == "__main__":
    main()