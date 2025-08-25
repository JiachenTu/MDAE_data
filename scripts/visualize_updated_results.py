#!/usr/bin/env python3
"""
Visualize the updated multi-modality results
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load data
new_summary = pd.read_csv('processed_data_20250815_multi/overall_summary_statistics.csv')
new_comprehensive = pd.read_csv('processed_data_20250815_multi/comprehensive_metrics_table.csv')

# Create figure with subplots
fig = plt.figure(figsize=(20, 12))

# 1. Overall Method Performance
ax1 = plt.subplot(2, 3, 1)
top_methods = new_summary.nlargest(10, 'Mean_AUROC')
bars = ax1.barh(range(len(top_methods)), top_methods['Mean_AUROC'], xerr=top_methods['Std_AUROC'])
ax1.set_yticks(range(len(top_methods)))
ax1.set_yticklabels(top_methods['Method'])
ax1.set_xlabel('Mean AUROC')
ax1.set_title('Top 10 Methods by Mean AUROC\n(2025-08-15 Update)', fontsize=12, fontweight='bold')
ax1.set_xlim([0, 1])
ax1.grid(axis='x', alpha=0.3)

# Color the bars
colors = ['#2E86AB' if m == 'MDAE' else '#A23B72' if m == 'MAE' else '#F18F01' for m in top_methods['Method']]
for bar, color in zip(bars, colors):
    bar.set_color(color)

# 2. Method Performance Distribution
ax2 = plt.subplot(2, 3, 2)
methods_to_plot = ['MDAE', 'MAE', 'SimCLR', 'VoCo', 'SwinUNETR']
benchmark_cols = [col for col in new_comprehensive.columns if col != 'Method']

data_for_violin = []
labels_for_violin = []
for method in methods_to_plot:
    method_data = new_comprehensive[new_comprehensive['Method'] == method]
    if not method_data.empty:
        values = method_data[benchmark_cols].values.flatten()
        values = values[~np.isnan(values)]
        if len(values) > 0:
            data_for_violin.append(values)
            labels_for_violin.append(method)

parts = ax2.violinplot(data_for_violin, positions=range(len(labels_for_violin)), 
                       showmeans=True, showmedians=True)
ax2.set_xticks(range(len(labels_for_violin)))
ax2.set_xticklabels(labels_for_violin, rotation=45)
ax2.set_ylabel('AUROC')
ax2.set_title('Performance Distribution Across Benchmarks\n(Top 5 Methods)', fontsize=12, fontweight='bold')
ax2.set_ylim([0, 1.05])
ax2.grid(axis='y', alpha=0.3)

# 3. Benchmark-wise Best Performance
ax3 = plt.subplot(2, 3, 3)
benchmark_best = {}
for col in benchmark_cols:
    col_data = new_comprehensive[col].dropna()
    if len(col_data) > 0:
        benchmark_best[col] = col_data.max()

if benchmark_best:
    bench_names = list(benchmark_best.keys())
    bench_values = list(benchmark_best.values())
    
    # Shorten names for display
    bench_names_short = [name.replace('_', ' ').replace('BraTS', 'B').replace('UPenn-GBM', 'UPenn')[:15] 
                         for name in bench_names]
    
    bars = ax3.bar(range(len(bench_names_short)), bench_values)
    ax3.set_xticks(range(len(bench_names_short)))
    ax3.set_xticklabels(bench_names_short, rotation=90, ha='right', fontsize=8)
    ax3.set_ylabel('Best AUROC')
    ax3.set_title('Best Performance per Benchmark', fontsize=12, fontweight='bold')
    ax3.set_ylim([0, 1.05])
    ax3.axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='0.9 threshold')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # Color bars by performance
    for bar, val in zip(bars, bench_values):
        if val >= 0.9:
            bar.set_color('#2E86AB')
        elif val >= 0.8:
            bar.set_color('#A23B72')
        elif val >= 0.7:
            bar.set_color('#F18F01')
        else:
            bar.set_color('#C73E1D')

# 4. MDAE Performance Across Benchmarks
ax4 = plt.subplot(2, 3, 4)
mdae_data = new_comprehensive[new_comprehensive['Method'] == 'MDAE']
if not mdae_data.empty:
    mdae_values = []
    mdae_labels = []
    for col in benchmark_cols:
        if not pd.isna(mdae_data[col].values[0]):
            mdae_values.append(mdae_data[col].values[0])
            mdae_labels.append(col.replace('_', ' ').replace('BraTS', 'B').replace('UPenn-GBM', 'UPenn')[:15])
    
    if mdae_values:
        bars = ax4.barh(range(len(mdae_values)), mdae_values)
        ax4.set_yticks(range(len(mdae_values)))
        ax4.set_yticklabels(mdae_labels, fontsize=8)
        ax4.set_xlabel('AUROC')
        ax4.set_title('MDAE Performance Across Benchmarks', fontsize=12, fontweight='bold')
        ax4.set_xlim([0, 1.05])
        ax4.axvline(x=0.8115, color='r', linestyle='--', alpha=0.5, label=f'Mean: {0.8115:.3f}')
        ax4.legend()
        ax4.grid(axis='x', alpha=0.3)
        
        # Color bars
        for bar, val in zip(bars, mdae_values):
            if val >= 0.9:
                bar.set_color('#2E86AB')
            elif val >= 0.8:
                bar.set_color('#A23B72')
            elif val >= 0.7:
                bar.set_color('#F18F01')
            else:
                bar.set_color('#C73E1D')

# 5. Method Comparison Heatmap
ax5 = plt.subplot(2, 3, 5)
# Select top methods for heatmap
top_methods_list = new_summary.nlargest(8, 'Mean_AUROC')['Method'].tolist()
heatmap_data = []
for method in top_methods_list:
    method_row = new_comprehensive[new_comprehensive['Method'] == method]
    if not method_row.empty:
        row_values = []
        for col in benchmark_cols[:10]:  # Limit to 10 benchmarks for readability
            val = method_row[col].values[0] if not method_row[col].isna().all() else np.nan
            row_values.append(val)
        heatmap_data.append(row_values)

if heatmap_data:
    im = ax5.imshow(heatmap_data, cmap='RdYlGn', vmin=0.3, vmax=1.0, aspect='auto')
    ax5.set_xticks(range(len(benchmark_cols[:10])))
    ax5.set_xticklabels([col.replace('_', ' ')[:12] for col in benchmark_cols[:10]], 
                        rotation=45, ha='right', fontsize=8)
    ax5.set_yticks(range(len(top_methods_list)))
    ax5.set_yticklabels(top_methods_list)
    ax5.set_title('Performance Heatmap (Top Methods)', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax5, fraction=0.046, pad=0.04)

# 6. Summary Statistics
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

# Create summary text
summary_text = f"""
📊 MULTI-MODALITY ANALYSIS SUMMARY
{'='*40}
📅 Date: 2025-08-15
📁 Benchmarks: 15
🔬 Methods Evaluated: {len(new_summary)}

🏆 TOP 5 METHODS:
1. {new_summary.iloc[0]['Method']:<10} {new_summary.iloc[0]['Mean_AUROC']:.4f} (±{new_summary.iloc[0]['Std_AUROC']:.4f})
2. {new_summary.iloc[1]['Method']:<10} {new_summary.iloc[1]['Mean_AUROC']:.4f} (±{new_summary.iloc[1]['Std_AUROC']:.4f})
3. {new_summary.iloc[2]['Method']:<10} {new_summary.iloc[2]['Mean_AUROC']:.4f} (±{new_summary.iloc[2]['Std_AUROC']:.4f})
4. {new_summary.iloc[3]['Method']:<10} {new_summary.iloc[3]['Mean_AUROC']:.4f} (±{new_summary.iloc[3]['Std_AUROC']:.4f})
5. {new_summary.iloc[4]['Method']:<10} {new_summary.iloc[4]['Mean_AUROC']:.4f} (±{new_summary.iloc[4]['Std_AUROC']:.4f})

📈 KEY INSIGHTS:
• MDAE maintains top performance (0.8115)
• MAE shows improvement to 0.7930
• SimCLR improved to 0.7712
• 5 methods achieve mean AUROC > 0.75
• Best single benchmark: 1.000 AUROC

🔄 UPDATES FROM PREVIOUS:
• New data extracted from WandB
• All 15 benchmarks processed
• Comprehensive metrics generated
"""

ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, 
         fontsize=10, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Overall title
fig.suptitle('Multi-Modality Benchmark Results - Updated Analysis (2025-08-15)', 
             fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout()
plt.savefig('processed_data_20250815_multi/updated_analysis_visualization.png', 
            dpi=300, bbox_inches='tight')
plt.show()

print("✅ Visualization saved to: processed_data_20250815_multi/updated_analysis_visualization.png")