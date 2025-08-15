#!/usr/bin/env python3
"""
Generate Publication-Ready Tables for Ablation Studies
Creates LaTeX, Markdown, and CSV formatted tables
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from scipy import stats

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_raw_data():
    """Load all raw CSV data files"""
    data_dir = Path("ablation_analysis/raw_data_extracted")
    
    data = {
        'noise_corruption': pd.read_csv(data_dir / "noise_corruption_raw.csv"),
        'masking_ratio': pd.read_csv(data_dir / "masking_ratio_raw.csv"),
        'masking_type': pd.read_csv(data_dir / "masking_type_raw.csv"),
        'flowmdae': pd.read_csv(data_dir / "flowmdae_raw.csv"),
        'summary': pd.read_csv(data_dir / "summary_statistics.csv")
    }
    
    return data

def create_noise_corruption_table(df):
    """Create noise corruption performance table"""
    # Create pivot table
    pivot = df.pivot_table(
        values='test_auroc',
        index='benchmark',
        columns='noise_type',
        aggfunc=['mean', 'std', 'count']
    )
    
    # Format the table
    result = pd.DataFrame()
    for noise_type in ['Flow', 'VE', 'VP']:
        if ('mean', noise_type) in pivot.columns:
            mean_val = pivot[('mean', noise_type)]
            std_val = pivot[('std', noise_type)]
            count_val = pivot[('count', noise_type)]
            
            # Format as mean ± std (n)
            result[noise_type] = mean_val.apply(lambda x: f'{x:.3f}') + ' ± ' + \
                                 std_val.apply(lambda x: f'{x:.3f}') + \
                                 ' (' + count_val.apply(lambda x: f'{int(x)}' if not pd.isna(x) else '0') + ')'
    
    # Add best column
    mean_only = df.pivot_table(values='test_auroc', index='benchmark', columns='noise_type', aggfunc='mean')
    result['Best'] = mean_only.idxmax(axis=1)
    
    return result

def create_masking_ratio_table(df):
    """Create masking ratio performance table"""
    # Create pivot table
    pivot = df.pivot_table(
        values='test_auroc',
        index='benchmark',
        columns='masking_ratio',
        aggfunc=['mean', 'std', 'count']
    )
    
    # Format the table
    result = pd.DataFrame()
    for ratio in [75, 80, 85, 90, 95]:
        if ('mean', ratio) in pivot.columns:
            mean_val = pivot[('mean', ratio)]
            std_val = pivot[('std', ratio)]
            count_val = pivot[('count', ratio)]
            
            # Format as mean ± std
            formatted = []
            for m, s, c in zip(mean_val, std_val, count_val):
                if pd.isna(m):
                    formatted.append('-')
                else:
                    formatted.append(f'{m:.3f} ± {s:.3f} ({int(c)})')
            result[f'{ratio}%'] = formatted
    
    # Add optimal ratio
    mean_only = df.pivot_table(values='test_auroc', index='benchmark', columns='masking_ratio', aggfunc='mean')
    if not mean_only.empty:
        result['Optimal'] = mean_only.idxmax(axis=1).apply(lambda x: f'{x}%')
    
    return result

def create_flowmdae_top_configs_table(df):
    """Create table of top FlowMDAE configurations"""
    # Get top 15 configurations
    top_configs = df.nlargest(15, 'test_auroc')[
        ['benchmark', 'modality', 'masking_ratio', 'noise_level', 'test_auroc', 'test_ap', 'test_f1']
    ].copy()
    
    # Format configuration column
    top_configs['Configuration'] = 'M' + top_configs['masking_ratio'].astype(str) + '_N' + top_configs['noise_level'].astype(str)
    
    # Select and rename columns
    result = top_configs[['Configuration', 'benchmark', 'modality', 'test_auroc', 'test_ap', 'test_f1']].copy()
    result.columns = ['Config', 'Benchmark', 'Modality', 'AUROC', 'AP', 'F1']
    
    # Format numerical columns
    for col in ['AUROC', 'AP', 'F1']:
        result[col] = result[col].apply(lambda x: f'{x:.4f}' if pd.notna(x) else '-')
    
    return result

def create_summary_comparison_table(data):
    """Create overall summary comparison table"""
    summary = []
    
    # Noise Corruption
    nc_data = data['noise_corruption']
    for noise_type in ['Flow', 'VE', 'VP']:
        subset = nc_data[nc_data['noise_type'] == noise_type]['test_auroc'].dropna()
        if len(subset) > 0:
            summary.append({
                'Ablation Type': f'Noise: {noise_type}',
                'N': len(subset),
                'Mean AUROC': f'{subset.mean():.4f}',
                'Std Dev': f'{subset.std():.4f}',
                'Min': f'{subset.min():.4f}',
                'Max': f'{subset.max():.4f}',
                'Median': f'{subset.median():.4f}'
            })
    
    # Masking Ratio
    mr_data = data['masking_ratio']
    for ratio in sorted(mr_data['masking_ratio'].unique()):
        subset = mr_data[mr_data['masking_ratio'] == ratio]['test_auroc'].dropna()
        if len(subset) > 0:
            summary.append({
                'Ablation Type': f'Mask Ratio: {ratio}%',
                'N': len(subset),
                'Mean AUROC': f'{subset.mean():.4f}',
                'Std Dev': f'{subset.std():.4f}',
                'Min': f'{subset.min():.4f}',
                'Max': f'{subset.max():.4f}',
                'Median': f'{subset.median():.4f}'
            })
    
    # Masking Type
    mt_data = data['masking_type']
    for mask_type in mt_data['masking_type'].unique():
        subset = mt_data[mt_data['masking_type'] == mask_type]['test_auroc'].dropna()
        if len(subset) > 0:
            summary.append({
                'Ablation Type': f'Mask Type: {mask_type}',
                'N': len(subset),
                'Mean AUROC': f'{subset.mean():.4f}',
                'Std Dev': f'{subset.std():.4f}',
                'Min': f'{subset.min():.4f}',
                'Max': f'{subset.max():.4f}',
                'Median': f'{subset.median():.4f}'
            })
    
    return pd.DataFrame(summary)

def create_best_configs_per_benchmark_table(data):
    """Create table showing best configuration for each benchmark"""
    results = []
    
    for benchmark in data['noise_corruption']['benchmark'].unique():
        best_scores = {}
        
        # Best noise corruption
        nc_subset = data['noise_corruption'][data['noise_corruption']['benchmark'] == benchmark]
        if not nc_subset.empty:
            nc_grouped = nc_subset.groupby('noise_type')['test_auroc'].mean()
            if not nc_grouped.empty:
                best_noise = nc_grouped.idxmax()
                best_scores['Noise'] = f'{best_noise} ({nc_grouped.max():.3f})'
        
        # Best masking ratio
        mr_subset = data['masking_ratio'][data['masking_ratio']['benchmark'] == benchmark]
        if not mr_subset.empty:
            mr_grouped = mr_subset.groupby('masking_ratio')['test_auroc'].mean()
            if not mr_grouped.empty:
                best_ratio = mr_grouped.idxmax()
                best_scores['Ratio'] = f'{best_ratio}% ({mr_grouped.max():.3f})'
        
        # Best masking type
        mt_subset = data['masking_type'][data['masking_type']['benchmark'] == benchmark]
        if not mt_subset.empty:
            mt_grouped = mt_subset.groupby('masking_type')['test_auroc'].mean()
            if not mt_grouped.empty:
                best_type = mt_grouped.idxmax()
                best_scores['Type'] = f'{best_type} ({mt_grouped.max():.3f})'
        
        # Best FlowMDAE
        fm_subset = data['flowmdae'][data['flowmdae']['benchmark'] == benchmark]
        if not fm_subset.empty:
            best_flow = fm_subset.nlargest(1, 'test_auroc').iloc[0]
            best_scores['FlowMDAE'] = f"M{best_flow['masking_ratio']}_N{best_flow['noise_level']} ({best_flow['test_auroc']:.3f})"
        
        if best_scores:
            results.append({
                'Benchmark': benchmark,
                **{k: v if k in best_scores else '-' for k, v in 
                   {'Noise': '-', 'Ratio': '-', 'Type': '-', 'FlowMDAE': '-'}.items()}
            })
            results[-1].update(best_scores)
    
    return pd.DataFrame(results)

def save_as_latex(df, filepath, caption="", label=""):
    """Save DataFrame as LaTeX table"""
    latex_str = df.to_latex(
        index=True,
        escape=False,
        column_format='l' + 'c' * len(df.columns),
        caption=caption,
        label=label
    )
    
    # Add booktabs formatting
    latex_str = latex_str.replace('\\toprule', '\\toprule\n\\midrule')
    latex_str = latex_str.replace('\\bottomrule', '\\midrule\n\\bottomrule')
    
    with open(filepath, 'w') as f:
        f.write('\\begin{table}[htbp]\n')
        f.write('\\centering\n')
        f.write(latex_str)
        f.write('\\end{table}\n')

def save_as_markdown(df, filepath, title=""):
    """Save DataFrame as Markdown table"""
    with open(filepath, 'w') as f:
        if title:
            f.write(f"# {title}\n\n")
        f.write(df.to_markdown(index=True))
        f.write('\n')

def main():
    """Main table generation pipeline"""
    logger.info("Starting table generation...")
    
    # Load data
    data = load_raw_data()
    
    # Create output directories
    output_base = Path("ablation_analysis/tables")
    latex_dir = output_base / "latex"
    markdown_dir = output_base / "markdown"
    csv_dir = output_base / "csv"
    
    for dir_path in [latex_dir, markdown_dir, csv_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Generate tables
    tables = {}
    
    # 1. Noise Corruption Table
    logger.info("Creating noise corruption table...")
    tables['noise_corruption'] = create_noise_corruption_table(data['noise_corruption'])
    
    # 2. Masking Ratio Table
    logger.info("Creating masking ratio table...")
    tables['masking_ratio'] = create_masking_ratio_table(data['masking_ratio'])
    
    # 3. Top FlowMDAE Configurations
    logger.info("Creating FlowMDAE top configurations table...")
    tables['flowmdae_top'] = create_flowmdae_top_configs_table(data['flowmdae'])
    
    # 4. Summary Comparison
    logger.info("Creating summary comparison table...")
    tables['summary_comparison'] = create_summary_comparison_table(data)
    
    # 5. Best Configurations per Benchmark
    logger.info("Creating best configurations table...")
    tables['best_configs'] = create_best_configs_per_benchmark_table(data)
    
    # Save all tables in different formats
    for name, df in tables.items():
        # CSV
        df.to_csv(csv_dir / f"{name}.csv")
        
        # LaTeX
        save_as_latex(
            df, 
            latex_dir / f"{name}.tex",
            caption=f"Ablation Study Results: {name.replace('_', ' ').title()}",
            label=f"tab:{name}"
        )
        
        # Markdown
        save_as_markdown(
            df,
            markdown_dir / f"{name}.md",
            title=name.replace('_', ' ').title()
        )
        
        logger.info(f"  Saved {name} table in all formats")
    
    # Create master LaTeX file
    master_latex = latex_dir / "all_tables.tex"
    with open(master_latex, 'w') as f:
        f.write("% Master file with all ablation study tables\n\n")
        f.write("\\documentclass{article}\n")
        f.write("\\usepackage{booktabs}\n")
        f.write("\\usepackage{array}\n")
        f.write("\\usepackage{caption}\n")
        f.write("\\begin{document}\n\n")
        
        for name in tables.keys():
            f.write(f"\\input{{{name}.tex}}\n")
            f.write("\\clearpage\n\n")
        
        f.write("\\end{document}\n")
    
    # Create master Markdown file
    master_md = markdown_dir / "ALL_TABLES.md"
    with open(master_md, 'w') as f:
        f.write("# Ablation Study - All Tables\n\n")
        f.write("## Table of Contents\n\n")
        
        for name in tables.keys():
            title = name.replace('_', ' ').title()
            f.write(f"- [{title}](#{name.replace('_', '-')})\n")
        
        f.write("\n---\n\n")
        
        for name, df in tables.items():
            title = name.replace('_', ' ').title()
            f.write(f"## {title}\n\n")
            f.write(df.to_markdown(index=True))
            f.write("\n\n---\n\n")
    
    logger.info("\n" + "="*60)
    logger.info("TABLE GENERATION COMPLETE")
    logger.info("="*60)
    logger.info(f"Output directory: {output_base}")
    logger.info("\nGenerated tables:")
    for name in tables.keys():
        logger.info(f"  - {name}")
    logger.info("\nFormats created:")
    logger.info("  - CSV files")
    logger.info("  - LaTeX files (with master document)")
    logger.info("  - Markdown files (with combined document)")
    
    # Display summary
    logger.info("\nSummary Statistics:")
    print(data['summary'].to_string(index=False))
    
    return tables

if __name__ == "__main__":
    tables = main()