#!/usr/bin/env python3
"""
Analyze Ablation Studies and Generate Matrices
Creates performance matrices and visualizations for different ablation types
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, List, Tuple
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_ablation_data(file_path: str = "ablation_analysis/raw_data/ablations_only.json") -> List[Dict]:
    """Load the ablation data from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_performance_metrics(run: Dict) -> Dict:
    """Extract key performance metrics from a run"""
    metrics = run.get('metrics', {})
    summary = run.get('summary', {})
    
    # Try different metric naming conventions from both metrics and summary
    auroc = (metrics.get('Test_AUROC') or 
             metrics.get('metric_Test/AUROC') or
             summary.get('Test/AUROC') or
             summary.get('Test_AUROC') or
             summary.get('metric_Test/AUROC'))
    
    ap = (metrics.get('Test_AP') or 
          metrics.get('metric_Test/AP') or
          summary.get('Test/AP') or
          summary.get('Test_AP') or
          summary.get('metric_Test/AP'))
    
    f1 = (metrics.get('Test_F1') or 
          metrics.get('metric_Test/F1') or
          summary.get('Test/F1') or
          summary.get('Test_F1') or
          summary.get('metric_Test/F1'))
    
    return {
        'auroc': auroc,
        'ap': ap,
        'f1': f1,
        'run_name': run.get('name', ''),
        'run_id': run.get('id', ''),
        'state': run.get('state', '')
    }

def create_noise_corruption_matrix(data: List[Dict]) -> pd.DataFrame:
    """Create performance matrix for noise corruption ablations"""
    logger.info("Creating Noise Corruption performance matrix...")
    
    results = []
    for project in data:
        benchmark = project['benchmark']
        modality = project['modality']
        
        for run in project['ablation_runs']:
            ablation = run['ablation_parsed']
            if ablation['ablation_type'] == 'noise_corruption':
                metrics = extract_performance_metrics(run)
                if metrics['auroc'] is not None:  # Only include runs with valid metrics
                    results.append({
                        'benchmark': benchmark,
                        'modality': modality,
                        'noise_type': ablation['noise_type'],
                        'masking_ratio': ablation['masking_ratio'],
                        'auroc': metrics['auroc'],
                        'ap': metrics['ap'],
                        'f1': metrics['f1']
                    })
    
    df = pd.DataFrame(results)
    
    # Create pivot table for AUROC by benchmark and noise type
    if not df.empty:
        pivot = df.pivot_table(
            values='auroc',
            index='benchmark',
            columns='noise_type',
            aggfunc='mean'
        )
        return pivot
    return pd.DataFrame()

def create_masking_type_matrix(data: List[Dict]) -> pd.DataFrame:
    """Create performance matrix for masking type ablations"""
    logger.info("Creating Masking Type performance matrix...")
    
    results = []
    for project in data:
        benchmark = project['benchmark']
        modality = project['modality']
        
        for run in project['ablation_runs']:
            ablation = run['ablation_parsed']
            if ablation['ablation_type'] == 'masking_type':
                metrics = extract_performance_metrics(run)
                if metrics['auroc'] is not None:
                    results.append({
                        'benchmark': benchmark,
                        'modality': modality,
                        'masking_type': ablation.get('masking_type', 'Unknown'),
                        'auroc': metrics['auroc'],
                        'ap': metrics['ap'],
                        'f1': metrics['f1']
                    })
    
    df = pd.DataFrame(results)
    
    if not df.empty:
        pivot = df.pivot_table(
            values='auroc',
            index='benchmark',
            columns='masking_type',
            aggfunc='mean'
        )
        return pivot
    return pd.DataFrame()

def create_masking_ratio_matrix(data: List[Dict]) -> pd.DataFrame:
    """Create performance matrix for masking ratio ablations"""
    logger.info("Creating Masking Ratio performance matrix...")
    
    results = []
    for project in data:
        benchmark = project['benchmark']
        modality = project['modality']
        
        for run in project['ablation_runs']:
            ablation = run['ablation_parsed']
            if ablation['ablation_type'] == 'masking_ratio':
                metrics = extract_performance_metrics(run)
                if metrics['auroc'] is not None:
                    results.append({
                        'benchmark': benchmark,
                        'modality': modality,
                        'masking_ratio': ablation['masking_ratio'],
                        'noise_type': ablation.get('noise_type', 'Flow'),
                        'auroc': metrics['auroc'],
                        'ap': metrics['ap'],
                        'f1': metrics['f1']
                    })
    
    df = pd.DataFrame(results)
    
    if not df.empty:
        pivot = df.pivot_table(
            values='auroc',
            index='benchmark',
            columns='masking_ratio',
            aggfunc='mean'
        )
        return pivot
    return pd.DataFrame()

def create_flowmdae_grid(data: List[Dict]) -> pd.DataFrame:
    """Create performance grid for FlowMDAE ablations"""
    logger.info("Creating FlowMDAE parameter grid...")
    
    results = []
    for project in data:
        benchmark = project['benchmark']
        modality = project['modality']
        
        for run in project['ablation_runs']:
            ablation = run['ablation_parsed']
            if ablation['ablation_type'] == 'flowmdae':
                metrics = extract_performance_metrics(run)
                if metrics['auroc'] is not None:
                    results.append({
                        'benchmark': benchmark,
                        'modality': modality,
                        'masking_ratio': ablation['masking_ratio'],
                        'noise_level': ablation['noise_level'],
                        'auroc': metrics['auroc'],
                        'ap': metrics['ap']
                    })
    
    df = pd.DataFrame(results)
    
    if not df.empty:
        # Create 2D grid for masking vs noise
        pivot = df.pivot_table(
            values='auroc',
            index='masking_ratio',
            columns='noise_level',
            aggfunc='mean'
        )
        return pivot
    return pd.DataFrame()

def visualize_noise_corruption(matrix: pd.DataFrame, output_dir: str):
    """Create visualization for noise corruption ablation"""
    if matrix.empty:
        logger.warning("No data for noise corruption visualization")
        return
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(matrix, annot=True, fmt='.3f', cmap='viridis', cbar_kws={'label': 'AUROC'})
    plt.title('Noise Corruption Type Ablation (AUROC)', fontsize=14, fontweight='bold')
    plt.xlabel('Noise Type', fontsize=12)
    plt.ylabel('Benchmark', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/noise_corruption_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create bar chart
    plt.figure(figsize=(10, 6))
    matrix.plot(kind='bar', width=0.8)
    plt.title('Noise Corruption Performance by Benchmark', fontsize=14, fontweight='bold')
    plt.xlabel('Benchmark', fontsize=12)
    plt.ylabel('AUROC', fontsize=12)
    plt.legend(title='Noise Type', loc='best')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/noise_corruption_bars.png', dpi=150, bbox_inches='tight')
    plt.close()

def visualize_masking_ratio(matrix: pd.DataFrame, output_dir: str):
    """Create visualization for masking ratio ablation"""
    if matrix.empty:
        logger.warning("No data for masking ratio visualization")
        return
    
    # Line plot showing trends
    plt.figure(figsize=(10, 6))
    for benchmark in matrix.index:
        plt.plot(matrix.columns, matrix.loc[benchmark], marker='o', label=benchmark, linewidth=2)
    
    plt.title('Masking Ratio Ablation Performance', fontsize=14, fontweight='bold')
    plt.xlabel('Masking Ratio (%)', fontsize=12)
    plt.ylabel('AUROC', fontsize=12)
    plt.legend(title='Benchmark', loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/masking_ratio_trends.png', dpi=150, bbox_inches='tight')
    plt.close()

def visualize_flowmdae_grid(matrix: pd.DataFrame, output_dir: str):
    """Create visualization for FlowMDAE parameter grid"""
    if matrix.empty:
        logger.warning("No data for FlowMDAE visualization")
        return
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt='.3f', cmap='coolwarm', cbar_kws={'label': 'AUROC'})
    plt.title('FlowMDAE Parameter Grid (AUROC)', fontsize=14, fontweight='bold')
    plt.xlabel('Noise Level (%)', fontsize=12)
    plt.ylabel('Masking Ratio (%)', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/flowmdae_parameter_grid.png', dpi=150, bbox_inches='tight')
    plt.close()

def generate_summary_statistics(data: List[Dict]) -> pd.DataFrame:
    """Generate summary statistics for all ablation types"""
    logger.info("Generating summary statistics...")
    
    stats = []
    ablation_types = ['noise_corruption', 'masking_type', 'masking_ratio', 'flowmdae']
    
    for abl_type in ablation_types:
        type_runs = []
        for project in data:
            for run in project['ablation_runs']:
                if run['ablation_parsed']['ablation_type'] == abl_type:
                    metrics = extract_performance_metrics(run)
                    if metrics['auroc'] is not None:
                        type_runs.append(metrics['auroc'])
        
        if type_runs:
            stats.append({
                'ablation_type': abl_type,
                'count': len(type_runs),
                'mean_auroc': np.mean(type_runs),
                'std_auroc': np.std(type_runs),
                'min_auroc': np.min(type_runs),
                'max_auroc': np.max(type_runs),
                'median_auroc': np.median(type_runs)
            })
    
    return pd.DataFrame(stats)

def main():
    """Main analysis pipeline"""
    logger.info("Starting ablation analysis...")
    
    # Create output directories
    output_base = "ablation_analysis/analysis"
    os.makedirs(output_base, exist_ok=True)
    
    for subdir in ['noise_corruption', 'masking_type', 'masking_ratio', 'flowmdae', 'summary']:
        os.makedirs(f"{output_base}/{subdir}", exist_ok=True)
    
    # Load data
    data = load_ablation_data()
    logger.info(f"Loaded {len(data)} projects with ablation data")
    
    # Generate matrices
    noise_matrix = create_noise_corruption_matrix(data)
    masking_type_matrix = create_masking_type_matrix(data)
    masking_ratio_matrix = create_masking_ratio_matrix(data)
    flowmdae_grid = create_flowmdae_grid(data)
    
    # Save matrices as CSV
    if not noise_matrix.empty:
        noise_matrix.to_csv(f"{output_base}/noise_corruption/performance_matrix.csv")
        logger.info(f"Noise corruption matrix: {noise_matrix.shape}")
    
    if not masking_type_matrix.empty:
        masking_type_matrix.to_csv(f"{output_base}/masking_type/performance_matrix.csv")
        logger.info(f"Masking type matrix: {masking_type_matrix.shape}")
    
    if not masking_ratio_matrix.empty:
        masking_ratio_matrix.to_csv(f"{output_base}/masking_ratio/performance_matrix.csv")
        logger.info(f"Masking ratio matrix: {masking_ratio_matrix.shape}")
    
    if not flowmdae_grid.empty:
        flowmdae_grid.to_csv(f"{output_base}/flowmdae/parameter_grid.csv")
        logger.info(f"FlowMDAE grid: {flowmdae_grid.shape}")
    
    # Generate visualizations
    visualize_noise_corruption(noise_matrix, f"{output_base}/noise_corruption")
    visualize_masking_ratio(masking_ratio_matrix, f"{output_base}/masking_ratio")
    visualize_flowmdae_grid(flowmdae_grid, f"{output_base}/flowmdae")
    
    # Generate summary statistics
    summary_stats = generate_summary_statistics(data)
    summary_stats.to_csv(f"{output_base}/summary/ablation_statistics.csv", index=False)
    logger.info("\nSummary Statistics:")
    print(summary_stats.to_string())
    
    # Create overall summary report
    report_lines = [
        "# Ablation Study Analysis Report",
        f"\n## Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        "\n## Overview",
        f"- Total projects analyzed: {len(data)}",
        f"- Total ablation runs: {sum(len(p['ablation_runs']) for p in data)}",
        "\n## Key Findings",
        "\n### Noise Corruption Analysis"
    ]
    
    if not noise_matrix.empty:
        best_noise = noise_matrix.mean().idxmax()
        report_lines.append(f"- Best performing noise type (average): {best_noise} ({noise_matrix.mean()[best_noise]:.3f} AUROC)")
        report_lines.append(f"- Benchmarks tested: {', '.join(noise_matrix.index.tolist())}")
    
    report_lines.append("\n### Masking Ratio Analysis")
    if not masking_ratio_matrix.empty:
        best_ratio = masking_ratio_matrix.mean().idxmax()
        report_lines.append(f"- Optimal masking ratio (average): {best_ratio}% ({masking_ratio_matrix.mean()[best_ratio]:.3f} AUROC)")
    
    report_lines.append("\n### FlowMDAE Parameter Analysis")
    if not flowmdae_grid.empty:
        max_val = flowmdae_grid.max().max()
        max_pos = flowmdae_grid.stack().idxmax()
        report_lines.append(f"- Best parameter combination: Masking={max_pos[0]}%, Noise={max_pos[1]}% ({max_val:.3f} AUROC)")
    
    # Write report
    with open(f"{output_base}/summary/ANALYSIS_REPORT.md", 'w') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"\n✅ Analysis complete! Results saved to {output_base}/")
    logger.info(f"Generated {len([f for f in os.listdir(f'{output_base}/noise_corruption') if f.endswith('.png')])} visualizations")
    
    return summary_stats

if __name__ == "__main__":
    stats = main()