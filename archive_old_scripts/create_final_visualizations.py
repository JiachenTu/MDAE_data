#!/usr/bin/env python3
"""
Create final comprehensive visualizations for MDAE paper with combined results.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for publication-quality figures
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10


def load_combined_results():
    """Load combined MDAE results."""
    processed_dir = Path('/home/t-jiachentu/repos/benchmarking/misc/data/processed_data')
    
    # Load combined results
    auroc_df = pd.read_csv(processed_dir / 'auroc_results_combined.csv')
    stats_df = pd.read_csv(processed_dir / 'statistics_combined.csv')
    
    return auroc_df, stats_df


def create_comprehensive_visualization(auroc_df, stats_df):
    """Create comprehensive 4-panel visualization."""
    
    # Create figure with 4 subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.25, wspace=0.3)
    
    # Define color scheme
    mdae_color = '#2E7D32'  # Dark green for MDAE
    ssl_color = '#1976D2'   # Blue for SSL baselines
    fm_color = '#7B1FA2'    # Purple for foundation models
    other_color = '#757575'  # Gray for others
    
    # 1. Bar plot of average performance
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Prepare data
    top_methods = stats_df.head(12).sort_values('Mean_AUROC')
    
    # Assign colors based on method type
    colors = []
    for method in top_methods['Method']:
        if method == 'MDAE':
            colors.append(mdae_color)
        elif method in ['MAE', 'SimCLR', 'VoCo', 'MG', 'SwinUNETR', 'VF', 'S3D']:
            colors.append(ssl_color)
        elif method in ['BrainIAC', 'MRI-Core', 'BrainMVP']:
            colors.append(fm_color)
        else:
            colors.append(other_color)
    
    bars = ax1.barh(range(len(top_methods)), top_methods['Mean_AUROC'], color=colors, alpha=0.8)
    ax1.set_yticks(range(len(top_methods)))
    ax1.set_yticklabels(top_methods['Method'])
    ax1.set_xlabel('Mean AUROC')
    ax1.set_title('(a) Average Performance Across All Tasks', fontweight='bold', pad=10)
    ax1.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Random baseline')
    ax1.set_xlim([0.45, 0.85])
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, top_methods['Mean_AUROC'])):
        ax1.text(val + 0.005, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', ha='left', va='center', fontsize=9)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=mdae_color, alpha=0.8, label='MDAE (Ours)'),
        Patch(facecolor=ssl_color, alpha=0.8, label='SSL Baselines'),
        Patch(facecolor=fm_color, alpha=0.8, label='Foundation Models'),
        Patch(facecolor=other_color, alpha=0.8, label='Others')
    ]
    ax1.legend(handles=legend_elements, loc='lower right')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # 2. Heatmap of results
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Select key methods and benchmarks for heatmap
    key_methods = ['MDAE', 'BrainIAC', 'VoCo', 'MAE', 'SimCLR', 'MRI-Core', 'BrainMVP']
    key_benchmarks = [
        'brats23_gli_vs_men', 'brats23_gli_vs_met', 'brats23_men_vs_met',
        'rsna_miccai_mgmt_methylation', 'ucsf_pdgm_idh_classification',
        'upenn_gbm_survival_1year', 'upenn_gbm_survival_2year',
        'upenn_gbm_gtr_status'
    ]
    
    # Create heatmap data
    heatmap_data = []
    for method in key_methods:
        if method == 'MDAE':
            method_name = 'MDAE (Combined)'
        else:
            method_name = method
        
        if method_name in auroc_df['Method'].values:
            row = auroc_df[auroc_df['Method'] == method_name].iloc[0]
            heatmap_row = [row[bench] if bench in row.index else np.nan for bench in key_benchmarks]
            heatmap_data.append(heatmap_row)
        else:
            heatmap_data.append([np.nan] * len(key_benchmarks))
    
    heatmap_df = pd.DataFrame(heatmap_data, index=key_methods, columns=[
        'Gli vs Men', 'Gli vs Met', 'Men vs Met', 'MGMT', 'IDH', 'Surv 1Y', 'Surv 2Y', 'GTR'
    ])
    
    # Create heatmap
    sns.heatmap(heatmap_df.astype(float), annot=True, fmt='.2f', cmap='RdYlGn', 
                vmin=0.5, vmax=1.0, ax=ax2, cbar_kws={'label': 'AUROC'})
    ax2.set_title('(b) Performance Heatmap (Selected Benchmarks)', fontweight='bold', pad=10)
    ax2.set_xlabel('')
    ax2.set_ylabel('Method')
    
    # 3. Box plot comparison - Performance Distribution Across Tasks
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Prepare box plot data
    box_methods = ['MDAE', 'BrainIAC', 'VoCo', 'MAE', 'SimCLR', 'MRI-Core', 'BrainMVP', 'DinoV2']
    box_data = []
    box_labels = []
    box_colors = []
    
    for method in box_methods:
        if method == 'MDAE':
            method_name = 'MDAE (Combined)'
            color = mdae_color
        elif method in ['MAE', 'SimCLR', 'VoCo']:
            method_name = method
            color = ssl_color
        elif method in ['BrainIAC', 'MRI-Core', 'BrainMVP']:
            method_name = method
            color = fm_color
        else:
            method_name = method
            color = other_color
            
        if method_name in auroc_df['Method'].values:
            values = auroc_df[auroc_df['Method'] == method_name].iloc[0, 1:].dropna().values
            box_data.append(values)
            box_labels.append(method)
            box_colors.append(color)
    
    # Create box plot
    bp = ax3.boxplot(box_data, tick_labels=box_labels, patch_artist=True,
                      notch=True, showmeans=True,
                      meanprops=dict(marker='D', markerfacecolor='red', markersize=6))
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Customize whiskers and caps
    for whisker in bp['whiskers']:
        whisker.set(color='#8B8B8B', linewidth=1.2)
    for cap in bp['caps']:
        cap.set(color='#8B8B8B', linewidth=1.2)
    for median in bp['medians']:
        median.set(color='black', linewidth=2)
    
    ax3.set_ylabel('AUROC')
    ax3.set_xlabel('Method')
    ax3.set_title('(c) Performance Distribution Across Tasks', fontweight='bold', pad=10)
    ax3.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random baseline')
    ax3.set_ylim([0.25, 1.05])
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add mean values as text
    for i, (data, label) in enumerate(zip(box_data, box_labels)):
        mean_val = np.mean(data)
        ax3.text(i+1, 0.3, f'{mean_val:.3f}', ha='center', va='center', 
                fontsize=9, fontweight='bold')
    
    # 4. Improvement plot
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Calculate improvements
    mdae_mean = stats_df[stats_df['Method'] == 'MDAE']['Mean_AUROC'].values[0]
    
    improvements = []
    baseline_methods = ['MAE', 'SimCLR', 'VoCo', 'MG', 'SwinUNETR', 'VF', 'BrainIAC', 'MRI-Core', 'BrainMVP']
    
    for baseline in baseline_methods:
        if baseline in stats_df['Method'].values:
            baseline_mean = stats_df[stats_df['Method'] == baseline]['Mean_AUROC'].values[0]
            improvements.append({
                'Method': baseline,
                'Improvement': (mdae_mean - baseline_mean) * 100,
                'Type': 'SSL' if baseline in ['MAE', 'SimCLR', 'VoCo', 'MG', 'SwinUNETR', 'VF'] else 'FM'
            })
    
    imp_df = pd.DataFrame(improvements)
    imp_df = imp_df.sort_values('Improvement')
    
    # Assign colors
    bar_colors = [ssl_color if t == 'SSL' else fm_color for t in imp_df['Type']]
    
    bars = ax4.barh(range(len(imp_df)), imp_df['Improvement'], color=bar_colors, alpha=0.8)
    ax4.set_yticks(range(len(imp_df)))
    ax4.set_yticklabels(imp_df['Method'])
    ax4.set_xlabel('Improvement (%)')
    ax4.set_title('(d) MDAE Improvement over Baselines', fontweight='bold', pad=10)
    ax4.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax4.axvline(x=3, color='red', linestyle='--', alpha=0.5, label='3% target')
    ax4.axvline(x=5, color='red', linestyle=':', alpha=0.5, label='5% target')
    
    # Add value labels
    for bar, val in zip(bars, imp_df['Improvement']):
        ax4.text(val + 0.3, bar.get_y() + bar.get_height()/2, 
                f'+{val:.1f}%', ha='left', va='center', fontsize=9, fontweight='bold')
    
    ax4.legend(loc='lower right')
    ax4.grid(True, alpha=0.3, axis='x')
    ax4.set_xlim([-1, max(imp_df['Improvement']) + 2])
    
    # Add overall title
    fig.suptitle('MDAE Performance Analysis Across 15 Brain Tumor Classification Benchmarks', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Add summary statistics as text
    textstr = f'MDAE Mean AUROC: {mdae_mean:.1%}\n'
    textstr += f'Avg. improvement over SSL: {imp_df[imp_df["Type"]=="SSL"]["Improvement"].mean():.1f}%\n'
    textstr += f'Avg. improvement over FM: {imp_df[imp_df["Type"]=="FM"]["Improvement"].mean():.1f}%'
    
    fig.text(0.02, 0.02, textstr, fontsize=11, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    return fig


def create_detailed_performance_table(auroc_df):
    """Create a detailed performance comparison table."""
    
    # Select methods to display
    methods_order = ['MDAE (Combined)', 'BrainIAC', 'VoCo', 'DinoV2', 'MAE', 
                     'SimCLR', 'MRI-Core', 'BrainMVP']
    
    # Filter and reorder
    df_filtered = auroc_df[auroc_df['Method'].isin(methods_order)]
    df_filtered = df_filtered.set_index('Method').loc[methods_order].reset_index()
    
    # Rename MDAE (Combined) to MDAE
    df_filtered['Method'] = df_filtered['Method'].replace('MDAE (Combined)', 'MDAE')
    
    # Calculate mean for each method
    means = []
    for _, row in df_filtered.iterrows():
        values = row.drop('Method').values
        valid_values = values[~pd.isna(values)]
        means.append(np.mean(valid_values) if len(valid_values) > 0 else np.nan)
    
    df_filtered['Mean'] = means
    
    # Round values
    numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df_filtered[col] = df_filtered[col].round(3)
    
    return df_filtered


def main():
    """Main function to generate final visualizations."""
    
    output_dir = Path('/home/t-jiachentu/repos/benchmarking/misc/data/processed_data')
    
    # Load combined results
    print("Loading combined MDAE results...")
    auroc_df, stats_df = load_combined_results()
    
    # Create comprehensive visualization
    print("Creating comprehensive visualization...")
    fig = create_comprehensive_visualization(auroc_df, stats_df)
    
    # Save the figure
    output_path = output_dir / 'mdae_final_visualization.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    
    # Also save as PDF for paper
    pdf_path = output_dir / 'mdae_final_visualization.pdf'
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"PDF version saved to: {pdf_path}")
    
    # Create detailed table
    print("\nCreating detailed performance table...")
    detailed_table = create_detailed_performance_table(auroc_df)
    
    # Save detailed table
    csv_path = output_dir / 'detailed_performance_table.csv'
    detailed_table.to_csv(csv_path, index=False)
    print(f"Detailed table saved to: {csv_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("VISUALIZATION SUMMARY")
    print("="*60)
    
    mdae_stats = stats_df[stats_df['Method'] == 'MDAE'].iloc[0]
    print(f"\nMDAE Performance:")
    print(f"  Mean AUROC: {mdae_stats['Mean_AUROC']:.3f}")
    print(f"  Std AUROC: {mdae_stats['Std_AUROC']:.3f}")
    print(f"  Min AUROC: {mdae_stats['Min_AUROC']:.3f}")
    print(f"  Max AUROC: {mdae_stats['Max_AUROC']:.3f}")
    
    print("\nTop 5 Methods by Mean AUROC:")
    for i, row in stats_df.head(5).iterrows():
        print(f"  {i+1}. {row['Method']:15s}: {row['Mean_AUROC']:.3f}")
    
    print("\nFiles generated:")
    print(f"  - {output_path}")
    print(f"  - {pdf_path}")
    print(f"  - {csv_path}")


if __name__ == "__main__":
    main()