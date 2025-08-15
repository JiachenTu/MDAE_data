#!/usr/bin/env python3
"""
Generate Enhanced Ablation Study Visualizations
Creates publication-quality figures from extracted raw data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from scipy import stats
from scipy.interpolate import griddata
import warnings
warnings.filterwarnings('ignore')

# Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def load_raw_data():
    """Load all raw CSV data files"""
    data_dir = Path("ablation_analysis/raw_data_extracted")
    
    data = {
        'noise_corruption': pd.read_csv(data_dir / "noise_corruption_raw.csv"),
        'masking_ratio': pd.read_csv(data_dir / "masking_ratio_raw.csv"),
        'masking_type': pd.read_csv(data_dir / "masking_type_raw.csv"),
        'flowmdae': pd.read_csv(data_dir / "flowmdae_raw.csv")
    }
    
    return data

def visualize_noise_corruption(df, output_dir):
    """Create enhanced noise corruption visualizations - UPenn datasets focus"""
    logger.info("Creating noise corruption visualizations (UPenn focus)...")
    output_path = Path(output_dir) / "noise_corruption"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Filter to only UPenn datasets
    df = df[df['benchmark'].str.contains('upenn', case=False)].copy()
    logger.info(f"Filtered to {len(df)} UPenn runs from {df['benchmark'].nunique()} benchmarks")
    
    # 1. Grouped bar chart by benchmark
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Prepare data
    pivot_data = df.pivot_table(
        values='test_auroc',
        index='benchmark',
        columns='noise_type',
        aggfunc='mean'
    )
    
    # Sort by mean performance
    pivot_data['mean'] = pivot_data.mean(axis=1)
    pivot_data = pivot_data.sort_values('mean', ascending=True).drop('mean', axis=1)
    
    # Create grouped bar chart
    pivot_data.plot(kind='barh', ax=ax, width=0.8)
    
    ax.set_xlabel('AUROC', fontsize=12, fontweight='bold')
    ax.set_ylabel('Benchmark', fontsize=12, fontweight='bold')
    ax.set_title('Noise Corruption Type Performance by Benchmark', fontsize=14, fontweight='bold')
    ax.legend(title='Noise Type', title_fontsize=11, fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim([0, 1])
    
    # Add value labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path / 'noise_corruption_grouped_bars.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Heatmap with annotations
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlGn', 
                center=0.6, vmin=0.3, vmax=0.9,
                cbar_kws={'label': 'AUROC'}, ax=ax,
                linewidths=0.5, linecolor='gray')
    
    ax.set_title('Noise Corruption Type Performance Heatmap', fontsize=14, fontweight='bold')
    ax.set_xlabel('Noise Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Benchmark', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path / 'noise_corruption_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Box plot showing distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create box plot
    df.boxplot(column='test_auroc', by='noise_type', ax=ax, 
               patch_artist=True, medianprops={'color': 'red', 'linewidth': 2})
    
    ax.set_xlabel('Noise Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('AUROC', fontsize=12, fontweight='bold')
    ax.set_title('Noise Corruption Type Performance Distribution', fontsize=14, fontweight='bold')
    plt.suptitle('')  # Remove automatic title
    ax.grid(True, alpha=0.3)
    
    # Add mean markers
    means = df.groupby('noise_type')['test_auroc'].mean()
    positions = range(1, len(means) + 1)
    ax.scatter(positions, means.values, color='blue', s=100, zorder=5, marker='D', label='Mean')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path / 'noise_corruption_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved noise corruption visualizations to {output_path}")

def visualize_masking_ratio(df, output_dir):
    """Create enhanced masking ratio visualizations"""
    logger.info("Creating masking ratio visualizations...")
    output_path = Path(output_dir) / "masking_ratio"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Group by benchmark and masking ratio
    grouped = df.groupby(['benchmark', 'masking_ratio'])['test_auroc'].agg(['mean', 'std', 'count']).reset_index()
    
    # Create figure with trend analysis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Line plot with confidence intervals
    has_ci = False
    for benchmark in grouped['benchmark'].unique():
        bench_data = grouped[grouped['benchmark'] == benchmark].sort_values('masking_ratio')
        
        # Calculate 95% confidence intervals
        ci = 1.96 * bench_data['std'] / np.sqrt(bench_data['count'])
        
        line = ax1.plot(bench_data['masking_ratio'], bench_data['mean'], 
                       marker='o', linewidth=2.5, label=benchmark, markersize=9)[0]
        
        # Add confidence interval shading
        fill = ax1.fill_between(bench_data['masking_ratio'], 
                                bench_data['mean'] - ci, 
                                bench_data['mean'] + ci, 
                                alpha=0.15, color=line.get_color())
        if not has_ci:
            # Add to legend only once
            fill.set_label('95% Confidence Interval')
            has_ci = True
    
    ax1.set_xlabel('Masking Ratio (%)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('AUROC', fontsize=12, fontweight='bold')
    ax1.set_title('Masking Ratio Performance Trends with 95% CI', fontsize=14, fontweight='bold')
    ax1.legend(title='Legend', loc='best', fontsize=9, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xticks([75, 80, 85, 90, 95])
    ax1.set_ylim([0.4, 0.9])
    
    # 2. Performance degradation analysis
    overall_means = df.groupby('masking_ratio')['test_auroc'].mean().sort_index()
    optimal_ratio = overall_means.idxmax()
    
    ax2.bar(overall_means.index, overall_means.values, width=3, alpha=0.7, edgecolor='black')
    ax2.axhline(y=overall_means.max(), color='r', linestyle='--', alpha=0.5, label=f'Best: {optimal_ratio}%')
    
    # Add polynomial fit
    z = np.polyfit(overall_means.index, overall_means.values, 2)
    p = np.poly1d(z)
    x_smooth = np.linspace(75, 95, 100)
    ax2.plot(x_smooth, p(x_smooth), 'b-', linewidth=2, alpha=0.8, label='Polynomial Fit')
    
    ax2.set_xlabel('Masking Ratio (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Mean AUROC', fontsize=12, fontweight='bold')
    ax2.set_title('Overall Masking Ratio Performance', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks([75, 80, 85, 90, 95])
    
    # Add value labels
    for i, (ratio, value) in enumerate(overall_means.items()):
        ax2.text(ratio, value + 0.005, f'{value:.3f}', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path / 'masking_ratio_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved masking ratio visualizations to {output_path}")

def visualize_masking_type(df, output_dir):
    """Create enhanced masking type visualizations"""
    logger.info("Creating masking type visualizations...")
    output_path = Path(output_dir) / "masking_type"
    output_path.mkdir(parents=True, exist_ok=True)
    
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
    
    # Side-by-side comparison with improved aesthetics
    masking_stats = df.groupby('masking_type')['test_auroc'].agg(['mean', 'std', 'count'])
    
    colors = ['#2E86AB', '#A23B72']
    bars = ax1.bar(masking_stats.index, masking_stats['mean'], 
                   yerr=masking_stats['std'], capsize=8,
                   color=colors, alpha=0.8, edgecolor='black', linewidth=2,
                   error_kw={'linewidth': 2, 'ecolor': 'black'})
    
    ax1.set_xlabel('Masking Type', fontsize=12, fontweight='bold')
    ax1.set_ylabel('AUROC', fontsize=12, fontweight='bold')
    ax1.set_title('Masking Type Performance Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 0.85])
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Add value labels
    for bar, (idx, row) in zip(bars, masking_stats.iterrows()):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + row['std'] + 0.02,
                f'{height:.3f}\n(n={int(row["count"])})',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add statistical test annotation
    block_scores = df[df['masking_type'] == 'Block']['test_auroc'].values
    random_scores = df[df['masking_type'] == 'Random']['test_auroc'].values
    
    if len(block_scores) > 0 and len(random_scores) > 0:
        t_stat, p_value = stats.ttest_ind(block_scores, random_scores)
        significance = 'p < 0.05' if p_value < 0.05 else 'p = n.s.'
        
        # Add bracket and p-value
        y_max = max(masking_stats['mean'] + masking_stats['std']) + 0.05
        ax1.plot([0, 1], [y_max, y_max], 'k-', linewidth=1.5)
        ax1.plot([0, 0], [y_max - 0.01, y_max], 'k-', linewidth=1.5)
        ax1.plot([1, 1], [y_max - 0.01, y_max], 'k-', linewidth=1.5)
        ax1.text(0.5, y_max + 0.01, f'{significance}\n(p={p_value:.3f})',
                ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.savefig(output_path / 'masking_type_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved masking type visualizations to {output_path}")

def visualize_flowmdae_grid(df, output_dir):
    """Create enhanced FlowMDAE (Flow SDE) parameter grid visualization"""
    logger.info("Creating FlowMDAE (Flow SDE) parameter grid visualization...")
    output_path = Path(output_dir) / "flowmdae"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for grid
    grid_data = df.pivot_table(
        values='test_auroc',
        index='noise_level',  # Y-axis: Max Noise Corruption Level
        columns='masking_ratio',  # X-axis: Masking Ratio
        aggfunc='mean'
    )
    
    # Sort axes - X-axis from small to large, Y-axis reversed (25 at bottom, 100 at top)
    grid_data = grid_data.sort_index(ascending=False)  # Reverse for correct display
    grid_data = grid_data[sorted(grid_data.columns)]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(14, 10))
    
    # Main heatmap
    ax_main = plt.subplot2grid((4, 4), (1, 0), colspan=3, rowspan=3)
    
    # Create enhanced heatmap with improved aesthetics
    sns.heatmap(grid_data, annot=True, fmt='.3f', 
                cmap='RdYlGn', center=0.6,
                cbar_kws={'label': 'AUROC', 'shrink': 0.8, 'pad': 0.02},
                linewidths=1.5, linecolor='white',
                vmin=0.45, vmax=0.70, ax=ax_main,
                annot_kws={'fontsize': 10, 'fontweight': 'bold'})
    
    # Highlight best cell
    max_val = grid_data.max().max()
    max_pos = grid_data.stack().idxmax()
    y_pos = list(grid_data.index).index(max_pos[0])
    x_pos = list(grid_data.columns).index(max_pos[1])
    ax_main.add_patch(plt.Rectangle((x_pos, y_pos), 1, 1, 
                                    fill=False, edgecolor='blue', lw=3))
    
    ax_main.set_xlabel('Masking Ratio (%)', fontsize=12, fontweight='bold')
    ax_main.set_ylabel('Max Noise Corruption Level (%)', fontsize=12, fontweight='bold')
    ax_main.set_title('MDAE (Flow SDE) Parameter Grid - AUROC Performance', 
                     fontsize=14, fontweight='bold')
    
    # Marginal plot for masking ratio (top)
    ax_top = plt.subplot2grid((4, 4), (0, 0), colspan=3)
    masking_means = grid_data.mean(axis=0)
    ax_top.bar(range(len(masking_means)), masking_means.values, 
               color='skyblue', alpha=0.7, edgecolor='black')
    ax_top.set_xticks(range(len(masking_means)))
    ax_top.set_xticklabels([f'{int(x)}%' for x in masking_means.index])
    ax_top.set_ylabel('Mean\nAUROC', fontsize=10)
    ax_top.set_title('Average Performance by Masking Ratio', fontsize=11)
    ax_top.grid(True, alpha=0.3, axis='y')
    ax_top.set_ylim([0.5, 0.65])
    
    # Add value labels
    for i, v in enumerate(masking_means.values):
        ax_top.text(i, v + 0.005, f'{v:.3f}', ha='center', fontsize=9)
    
    # Marginal plot for noise level (right)
    ax_right = plt.subplot2grid((4, 4), (1, 3), rowspan=3)
    noise_means = grid_data.mean(axis=1)
    ax_right.barh(range(len(noise_means)), noise_means.values, 
                  color='lightcoral', alpha=0.7, edgecolor='black')
    ax_right.set_yticks(range(len(noise_means)))
    ax_right.set_yticklabels([f'{int(y)}%' for y in noise_means.index])
    ax_right.set_xlabel('Mean AUROC', fontsize=10)
    ax_right.set_title('Average Performance\nby Noise Level', fontsize=11)
    ax_right.grid(True, alpha=0.3, axis='x')
    ax_right.set_xlim([0.5, 0.65])
    
    # Add value labels
    for i, v in enumerate(noise_means.values):
        ax_right.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=9)
    
    # Add annotation for best configuration
    best_config_text = (f"Optimal Configuration:\n"
                       f"Masking: {max_pos[1]}%\n"
                       f"Noise: {max_pos[0]}%\n"
                       f"AUROC: {max_val:.3f}")
    
    ax_text = plt.subplot2grid((4, 4), (0, 3))
    ax_text.text(0.5, 0.5, best_config_text, 
                ha='center', va='center', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    ax_text.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path / 'flowmdae_parameter_grid_enhanced.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create contour plot for smooth visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use the same pivot table data as the heatmap for consistency
    contour_data = df.pivot_table(
        values='test_auroc',
        index='noise_level',
        columns='masking_ratio',
        aggfunc='mean'
    )
    
    # Create meshgrid from actual data points
    X, Y = np.meshgrid(sorted(contour_data.columns), sorted(contour_data.index))
    Z = contour_data.values
    
    # Create contour plot with consistent data
    contour = ax.contourf(X, Y, Z, levels=15, cmap='RdYlGn', vmin=0.45, vmax=0.70)
    contour_lines = ax.contour(X, Y, Z, levels=8, colors='black', alpha=0.4, linewidths=0.5)
    ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.3f')
    
    # Add colorbar
    cbar = plt.colorbar(contour, ax=ax, label='AUROC', shrink=0.9)
    
    # Add actual data points as markers
    for i, row in df.iterrows():
        ax.scatter(row['masking_ratio'], row['noise_level'], 
                  c=[row['test_auroc']], s=120, 
                  cmap='RdYlGn', vmin=0.45, vmax=0.70,
                  edgecolor='white', linewidth=2, zorder=5)
    
    ax.set_xlabel('Masking Ratio (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Max Noise Corruption Level (%)', fontsize=12, fontweight='bold')
    ax.set_title('MDAE (Flow SDE) Performance Landscape', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'flowmdae_contour_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved FlowMDAE visualizations to {output_path}")

def create_combined_summary_figure(data, output_dir):
    """Create a combined summary figure for all ablation types"""
    logger.info("Creating combined summary figure...")
    output_path = Path(output_dir) / "summary"
    output_path.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Noise Corruption Summary
    ax = axes[0, 0]
    noise_means = data['noise_corruption'].groupby('noise_type')['test_auroc'].mean().sort_values()
    ax.barh(noise_means.index, noise_means.values, color='steelblue', alpha=0.7)
    ax.set_xlabel('Mean AUROC', fontsize=11)
    ax.set_title('Noise Corruption Types', fontsize=12, fontweight='bold')
    ax.set_xlim([0.5, 0.7])
    for i, v in enumerate(noise_means.values):
        ax.text(v + 0.005, i, f'{v:.3f}', va='center')
    
    # 2. Masking Ratio Summary
    ax = axes[0, 1]
    ratio_means = data['masking_ratio'].groupby('masking_ratio')['test_auroc'].mean()
    ax.plot(ratio_means.index, ratio_means.values, 'o-', linewidth=2, markersize=8, color='green')
    ax.set_xlabel('Masking Ratio (%)', fontsize=11)
    ax.set_ylabel('Mean AUROC', fontsize=11)
    ax.set_title('Masking Ratio Effect', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks([75, 80, 85, 90, 95])
    
    # 3. Masking Type Summary
    ax = axes[1, 0]
    type_means = data['masking_type'].groupby('masking_type')['test_auroc'].mean()
    ax.bar(type_means.index, type_means.values, color=['coral', 'lightblue'], alpha=0.7)
    ax.set_ylabel('Mean AUROC', fontsize=11)
    ax.set_title('Masking Types', fontsize=12, fontweight='bold')
    ax.set_ylim([0.5, 0.7])
    for i, (idx, v) in enumerate(type_means.items()):
        ax.text(i, v + 0.005, f'{v:.3f}', ha='center')
    
    # 4. FlowMDAE Best Configs
    ax = axes[1, 1]
    flowmdae_top = data['flowmdae'].nlargest(10, 'test_auroc')[['param_combo', 'test_auroc']]
    ax.barh(range(len(flowmdae_top)), flowmdae_top['test_auroc'].values, color='purple', alpha=0.7)
    ax.set_yticks(range(len(flowmdae_top)))
    ax.set_yticklabels(flowmdae_top['param_combo'].values, fontsize=9)
    ax.set_xlabel('AUROC', fontsize=11)
    ax.set_title('Top 10 FlowMDAE Configurations', fontsize=12, fontweight='bold')
    ax.set_xlim([0.5, 0.85])
    
    plt.suptitle('Ablation Study Summary - All Types', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path / 'combined_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved combined summary to {output_path}")

def main():
    """Main visualization pipeline"""
    logger.info("Starting enhanced visualization generation...")
    
    # Load raw data
    data = load_raw_data()
    
    # Create output directory
    output_dir = Path("ablation_analysis/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate visualizations for each ablation type
    visualize_noise_corruption(data['noise_corruption'], output_dir)
    visualize_masking_ratio(data['masking_ratio'], output_dir)
    visualize_masking_type(data['masking_type'], output_dir)
    visualize_flowmdae_grid(data['flowmdae'], output_dir)
    
    # Create combined summary
    create_combined_summary_figure(data, output_dir)
    
    logger.info("\n" + "="*60)
    logger.info("VISUALIZATION GENERATION COMPLETE")
    logger.info("="*60)
    logger.info(f"All visualizations saved to: {output_dir}")
    logger.info("\nGenerated visualizations:")
    logger.info("  - Noise Corruption: 3 figures")
    logger.info("  - Masking Ratio: 1 figure")
    logger.info("  - Masking Type: 1 figure")
    logger.info("  - FlowMDAE: 2 figures (grid + contour)")
    logger.info("  - Combined Summary: 1 figure")
    
    return True

if __name__ == "__main__":
    main()