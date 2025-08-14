#!/usr/bin/env python3
"""
Create publication-ready box plot showing MDAE performance distribution across benchmarks.

This script generates a box plot visualization comparing MDAE against various baselines
across 15 brain tumor classification benchmarks. The plot uses specific configurations
optimized for clarity and statistical insight.

Configuration Details:
- Notches: Shows 95% confidence intervals around medians for statistical comparison
- Whiskers: Set to 1.0×IQR (instead of default 1.5) for better outlier visibility
- Colors: Green (MDAE), Blue (SSL methods), Purple (Foundation models), Gray (Others)
- Mean markers: Red diamonds show arithmetic mean alongside median

Author: MDAE Team
Date: 2024
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for publication-quality figure
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11


def create_final_boxplot():
    """
    Create final clean box plot with notches and shorter whiskers.
    
    Returns:
        matplotlib.figure.Figure: The generated box plot figure
    
    Box Plot Elements:
        - Box: Contains middle 50% of data (Q1 to Q3)
        - Notch: 95% confidence interval around median
        - Median: Black horizontal line inside box
        - Mean: Red diamond marker
        - Whiskers: Extend to values within 1.0×IQR
        - Outliers: Circles beyond whisker range
    """
    
    # Load data from processed results
    data_dir = Path('/home/t-jiachentu/repos/benchmarking/misc/data/processed_data_combined')
    metrics_df = pd.read_csv(data_dir / 'comprehensive_metrics_table.csv')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Define color scheme
    mdae_color = '#2E7D32'  # Dark green for MDAE
    ssl_color = '#1976D2'   # Blue for SSL baselines
    fm_color = '#7B1FA2'    # Purple for foundation models
    other_color = '#757575'  # Gray for others
    
    # Prepare box plot data
    box_methods = ['MDAE', 'BrainIAC', 'VoCo', 'MAE', 'SimCLR', 'MRI-Core', 'BrainMVP', 'DinoV2']
    box_data = []
    box_labels = []
    box_colors = []
    
    for method in box_methods:
        if method in ['MAE', 'SimCLR', 'VoCo']:
            color = ssl_color
        elif method in ['BrainIAC', 'MRI-Core', 'BrainMVP']:
            color = fm_color
        elif method == 'MDAE':
            color = mdae_color
        else:
            color = other_color
            
        if method in metrics_df['Method'].values:
            # Get all benchmark values for this method
            row = metrics_df[metrics_df['Method'] == method].iloc[0]
            values = []
            for col in metrics_df.columns:
                if col != 'Method' and pd.notna(row[col]):
                    values.append(row[col])
            
            if values:
                box_data.append(values)
                box_labels.append(method)
                box_colors.append(color)
    
    # Create box plot with custom configuration
    bp = ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True,
                     notch=True,           # Shows 95% CI around median for statistical comparison
                     whis=1.0,             # Whiskers at 1.0×IQR (tighter than default 1.5)
                     showmeans=True,       # Display mean as red diamond
                     meanprops=dict(marker='D', markerfacecolor='red', 
                                   markersize=7, markeredgecolor='darkred', alpha=0.8))
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    
    # Customize whiskers and caps
    for whisker in bp['whiskers']:
        whisker.set(color='#4A4A4A', linewidth=1.2)
    for cap in bp['caps']:
        cap.set(color='#4A4A4A', linewidth=1.2)
    for median in bp['medians']:
        median.set(color='black', linewidth=2)
    
    # Labels and title
    ax.set_ylabel('Test AUROC', fontsize=13, fontweight='bold')
    ax.set_xlabel('Method', fontsize=13, fontweight='bold')
    ax.set_title('Performance Distribution Across 15 Brain Tumor Classification Benchmarks', 
                 fontsize=14, fontweight='bold', pad=15)
    
    # Add random baseline
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
    
    # Set y-axis limits
    ax.set_ylim([0.3, 1.02])
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add mean values below each box
    for i, data in enumerate(box_data):
        mean_val = np.mean(data)
        ax.text(i+1, 0.32, f'{mean_val:.3f}', ha='center', va='center', 
                fontsize=10, fontweight='bold')
    
    # Add legend for colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=mdae_color, alpha=0.8, label='MDAE (Ours)'),
        Patch(facecolor=ssl_color, alpha=0.8, label='SSL Baselines'),
        Patch(facecolor=fm_color, alpha=0.8, label='Foundation Models'),
        Patch(facecolor=other_color, alpha=0.8, label='Others')
    ]
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9)
    
    plt.tight_layout()
    return fig


def main():
    """
    Main function to generate and save the box plot visualization.
    
    Outputs:
        - boxplot_final.png: High-resolution PNG (300 DPI) for presentations
        - boxplot_final.pdf: Vector format for publication
    """
    
    output_dir = Path('/home/t-jiachentu/repos/benchmarking/misc/data/processed_data_combined')
    
    print("Creating MDAE performance box plot...")
    print("Configuration: Notches (95% CI), 1.0×IQR whiskers, mean markers")
    
    # Generate the plot
    fig = create_final_boxplot()
    
    # Save in multiple formats
    output_path = output_dir / 'boxplot_final.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ PNG saved: {output_path}")
    
    pdf_path = output_dir / 'boxplot_final.pdf'
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"✓ PDF saved: {pdf_path}")
    
    print("\nBox plot generation complete!")


if __name__ == "__main__":
    main()