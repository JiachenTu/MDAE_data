#!/usr/bin/env python3
"""
Generate contour plots for FlowMDAE ablation analysis.
Creates visualizations for both test and validation metrics across all modalities.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def create_contour_visualization(data_grid, masking_ratios, noise_levels, 
                                modality, metric_type, metric_name, output_dir):
    """
    Create comprehensive contour visualization for a single metric.
    
    Args:
        data_grid: 2D array of metric values (noise x masking)
        masking_ratios: List of masking ratio values
        noise_levels: List of noise level values
        modality: MRI modality name
        metric_type: 'Test' or 'Val'
        metric_name: Name of the metric (e.g., 'AUROC')
        output_dir: Directory to save plots
    """
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(f'{modality} - {metric_type} {metric_name}\nMasking Ratio vs Noise Corruption Level', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Prepare labels
    masking_labels = [f'M{m}' for m in masking_ratios]
    noise_labels = [f'N{n}' for n in noise_levels]
    
    # 1. Contour plot
    ax1 = plt.subplot(2, 3, 1)
    X, Y = np.meshgrid(range(len(masking_ratios)), range(len(noise_levels)))
    
    # Handle NaN values by interpolation for smoother contours
    data_filled = np.nan_to_num(data_grid, nan=np.nanmean(data_grid))
    
    # Create contour plot
    levels = 15
    contour = ax1.contourf(X, Y, data_filled, levels=levels, cmap='viridis', alpha=0.8)
    contour_lines = ax1.contour(X, Y, data_filled, levels=10, colors='white', 
                                alpha=0.6, linewidths=0.8)
    ax1.clabel(contour_lines, inline=True, fontsize=8, fmt='%.3f')
    
    # Add data points and values
    for i in range(len(noise_levels)):
        for j in range(len(masking_ratios)):
            if not np.isnan(data_grid[i, j]):
                ax1.plot(j, i, 'ko', markersize=6)
                ax1.text(j, i, f'{data_grid[i,j]:.3f}', ha='center', va='center', 
                        color='white', fontweight='bold', fontsize=9)
    
    ax1.set_xticks(range(len(masking_ratios)))
    ax1.set_yticks(range(len(noise_levels)))
    ax1.set_xticklabels(masking_labels)
    ax1.set_yticklabels(noise_labels)
    ax1.set_xlabel('Masking Ratio (%)', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Noise Corruption Level (%)', fontweight='bold', fontsize=12)
    ax1.set_title('Contour Plot with Values', fontweight='bold', fontsize=14)
    plt.colorbar(contour, ax=ax1, label=metric_name, shrink=0.8)
    
    # 2. Heatmap
    ax2 = plt.subplot(2, 3, 2)
    sns.heatmap(data_grid, annot=True, fmt='.3f', cmap='viridis', 
                xticklabels=masking_labels, yticklabels=noise_labels,
                cbar_kws={'label': metric_name}, ax=ax2, 
                annot_kws={'fontweight': 'bold', 'fontsize': 10},
                linewidths=0.5, linecolor='gray')
    ax2.set_xlabel('Masking Ratio (%)', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Noise Corruption Level (%)', fontweight='bold', fontsize=12)
    ax2.set_title('Heatmap Visualization', fontweight='bold', fontsize=14)
    
    # 3. 3D Surface plot
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    X_3d, Y_3d = np.meshgrid(range(len(masking_ratios)), range(len(noise_levels)))
    surface = ax3.plot_surface(X_3d, Y_3d, data_filled, cmap='viridis', 
                              alpha=0.8, edgecolor='none', antialiased=True)
    
    # Add wireframe for better visibility
    ax3.plot_wireframe(X_3d, Y_3d, data_filled, color='black', 
                      alpha=0.2, linewidth=0.5)
    
    ax3.set_xticks(range(len(masking_ratios)))
    ax3.set_yticks(range(len(noise_levels)))
    ax3.set_xticklabels(masking_labels)
    ax3.set_yticklabels(noise_labels)
    ax3.set_xlabel('Masking Ratio', fontweight='bold', labelpad=10)
    ax3.set_ylabel('Noise Level', fontweight='bold', labelpad=10)
    ax3.set_zlabel(metric_name, fontweight='bold', labelpad=10)
    ax3.set_title('3D Surface Plot', fontweight='bold', fontsize=14)
    ax3.view_init(elev=20, azim=45)
    
    # 4. Line plots for trends across masking ratios
    ax4 = plt.subplot(2, 3, 4)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']
    
    for i, noise in enumerate(noise_labels):
        ax4.plot(range(len(masking_ratios)), data_grid[i, :], 
                color=colors[i], marker=markers[i], linewidth=2.5, markersize=10,
                label=f'{noise}', markerfacecolor='white', markeredgewidth=2)
    
    ax4.set_xticks(range(len(masking_ratios)))
    ax4.set_xticklabels(masking_labels)
    ax4.set_xlabel('Masking Ratio (%)', fontweight='bold', fontsize=12)
    ax4.set_ylabel(metric_name, fontweight='bold', fontsize=12)
    ax4.set_title('Performance vs Masking Ratio', fontweight='bold', fontsize=14)
    ax4.legend(title='Noise Level', title_fontsize=11, fontsize=10, loc='best')
    ax4.grid(True, alpha=0.3)
    
    # 5. Line plots for trends across noise levels
    ax5 = plt.subplot(2, 3, 5)
    
    for j, mask in enumerate(masking_labels):
        ax5.plot(range(len(noise_levels)), data_grid[:, j], 
                color=colors[j], marker=markers[j], linewidth=2.5, markersize=10,
                label=f'{mask}', markerfacecolor='white', markeredgewidth=2)
    
    ax5.set_xticks(range(len(noise_levels)))
    ax5.set_xticklabels(noise_labels)
    ax5.set_xlabel('Noise Corruption Level (%)', fontweight='bold', fontsize=12)
    ax5.set_ylabel(metric_name, fontweight='bold', fontsize=12)
    ax5.set_title('Performance vs Noise Level', fontweight='bold', fontsize=14)
    ax5.legend(title='Masking Ratio', title_fontsize=11, fontsize=10, loc='best')
    ax5.grid(True, alpha=0.3)
    
    # 6. Statistics and best configuration
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # Calculate statistics
    valid_data = data_grid[~np.isnan(data_grid)]
    if len(valid_data) > 0:
        best_idx = np.nanargmax(data_grid)
        best_i, best_j = np.unravel_index(best_idx, data_grid.shape)
        best_value = data_grid[best_i, best_j]
        
        worst_idx = np.nanargmin(data_grid)
        worst_i, worst_j = np.unravel_index(worst_idx, data_grid.shape)
        worst_value = data_grid[worst_i, worst_j]
        
        stats_text = f"""
        STATISTICS SUMMARY
        {'='*40}
        
        Best Performance:
          Value: {best_value:.4f}
          Config: {masking_labels[best_j]}, {noise_labels[best_i]}
          
        Worst Performance:
          Value: {worst_value:.4f}
          Config: {masking_labels[worst_j]}, {noise_labels[worst_i]}
        
        Overall Statistics:
          Mean: {np.nanmean(data_grid):.4f}
          Std Dev: {np.nanstd(data_grid):.4f}
          Median: {np.nanmedian(data_grid):.4f}
          Range: {worst_value:.4f} - {best_value:.4f}
        
        Per Masking Ratio (Mean):
        """
        for j, mask in enumerate(masking_labels):
            mean_val = np.nanmean(data_grid[:, j])
            stats_text += f"  {mask}: {mean_val:.4f}\n"
        
        stats_text += "\nPer Noise Level (Mean):\n"
        for i, noise in enumerate(noise_labels):
            mean_val = np.nanmean(data_grid[i, :])
            stats_text += f"  {noise}: {mean_val:.4f}\n"
        
        ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, 
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Add best config annotation to line plot
        ax4.annotate(f'Best: {best_value:.3f}\n({masking_labels[best_j]}, {noise_labels[best_i]})',
                    xy=(best_j, best_value), xytext=(best_j+0.3, best_value+0.03),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2),
                    fontsize=10, fontweight='bold', color='red',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    
    # Save figure
    output_path = output_dir / f'{modality}_{metric_type.lower()}_{metric_name.lower().replace("/", "_")}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path.name}")

def main():
    # Load processed data
    data_dir = Path('/home/t-jiachentu/repos/benchmarking/misc/data/ablation/contour_analysis/data')
    output_base = Path('/home/t-jiachentu/repos/benchmarking/misc/data/ablation/contour_analysis/plots')
    
    print("Loading processed grids...")
    data = np.load(data_dir / 'processed_grids.npz', allow_pickle=True)
    grids = data['grids'].item()
    masking_ratios = data['masking_ratios']
    noise_levels = data['noise_levels']
    modalities = data['modalities']
    
    # Metrics to visualize
    test_metrics = ['Test_AUROC', 'Test_AP', 'Test_F1', 'Test_Balanced_Accuracy']
    val_metrics = ['Val_AUROC', 'Val_AP', 'Val_F1', 'Val_Balanced_Accuracy']
    
    print("\n" + "="*60)
    print("GENERATING CONTOUR PLOTS")
    print("="*60)
    
    # Generate plots for each modality
    for modality in modalities:
        print(f"\nProcessing {modality}:")
        
        # Test metrics
        print("  Test metrics:")
        for metric in test_metrics:
            if metric in grids[modality]:
                metric_name = metric.replace('Test_', '').replace('_', ' ')
                create_contour_visualization(
                    grids[modality][metric],
                    masking_ratios, noise_levels,
                    modality, 'Test', metric_name,
                    output_base / 'test_metrics'
                )
        
        # Validation metrics
        print("  Validation metrics:")
        for metric in val_metrics:
            if metric in grids[modality]:
                metric_name = metric.replace('Val_', '').replace('_', ' ')
                create_contour_visualization(
                    grids[modality][metric],
                    masking_ratios, noise_levels,
                    modality, 'Val', metric_name,
                    output_base / 'val_metrics'
                )
    
    print("\n" + "="*60)
    print("PLOT GENERATION COMPLETE")
    print("="*60)
    print(f"All plots saved to: {output_base}")
    print("  - Test metrics: {output_base}/test_metrics/")
    print("  - Validation metrics: {output_base}/val_metrics/")

if __name__ == "__main__":
    main()