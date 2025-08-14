#!/usr/bin/env python3
"""
Create averaged AUROC contour plots for BraTS18 benchmark.
Shows both validation and test metrics averaged across all modalities.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def create_averaged_auroc_visualization():
    """Create comprehensive averaged AUROC visualization for BraTS18."""
    
    # Load processed data
    data_dir = Path('/home/t-jiachentu/repos/benchmarking/misc/data/ablation/contour_analysis/data')
    output_dir = Path('/home/t-jiachentu/repos/benchmarking/misc/data/ablation/contour_analysis')
    
    print("Loading processed grids...")
    data = np.load(data_dir / 'processed_grids.npz', allow_pickle=True)
    grids = data['grids'].item()
    masking_ratios = data['masking_ratios']
    noise_levels = data['noise_levels']
    modalities = ['T1', 'T1CE', 'T2', 'FLAIR']
    
    # Calculate averaged AUROC across all modalities
    test_auroc_avg = np.zeros((len(noise_levels), len(masking_ratios)))
    val_auroc_avg = np.zeros((len(noise_levels), len(masking_ratios)))
    
    for modality in modalities:
        test_auroc_avg += grids[modality]['Test_AUROC']
        val_auroc_avg += grids[modality]['Val_AUROC']
    
    test_auroc_avg /= len(modalities)
    val_auroc_avg /= len(modalities)
    
    # Create figure with comprehensive visualization
    fig = plt.figure(figsize=(24, 20))
    fig.suptitle('BraTS18 LGG vs HGG - FlowMDAE Ablation Study\nAveraged AUROC Across All Modalities (T1, T1CE, T2, FLAIR)', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # Prepare labels
    masking_labels = [f'{m}%' for m in masking_ratios]
    noise_labels = [f'{n}%' for n in noise_levels]
    
    # Color maps for different metrics
    cmap_test = 'RdYlGn'  # Red to Yellow to Green for test
    cmap_val = 'PuBuGn'   # Purple to Blue to Green for validation
    
    # ========== ROW 1: TEST METRICS ==========
    
    # 1. Test AUROC Contour Plot
    ax1 = plt.subplot(3, 4, 1)
    X, Y = np.meshgrid(range(len(masking_ratios)), range(len(noise_levels)))
    
    levels = np.linspace(test_auroc_avg.min(), test_auroc_avg.max(), 15)
    contour_test = ax1.contourf(X, Y, test_auroc_avg, levels=levels, cmap=cmap_test, alpha=0.9)
    contour_lines = ax1.contour(X, Y, test_auroc_avg, levels=10, colors='black', 
                                alpha=0.4, linewidths=0.8)
    ax1.clabel(contour_lines, inline=True, fontsize=9, fmt='%.3f')
    
    # Add data points and values
    for i in range(len(noise_levels)):
        for j in range(len(masking_ratios)):
            ax1.plot(j, i, 'ko', markersize=8)
            color = 'white' if test_auroc_avg[i,j] < 0.55 else 'black'
            ax1.text(j, i, f'{test_auroc_avg[i,j]:.3f}', ha='center', va='center', 
                    color=color, fontweight='bold', fontsize=10)
    
    ax1.set_xticks(range(len(masking_ratios)))
    ax1.set_yticks(range(len(noise_levels)))
    ax1.set_xticklabels(masking_labels, fontsize=11)
    ax1.set_yticklabels(noise_labels, fontsize=11)
    ax1.set_xlabel('Masking Ratio', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Noise Corruption Level', fontweight='bold', fontsize=12)
    ax1.set_title('Test AUROC - Contour Plot', fontweight='bold', fontsize=14)
    plt.colorbar(contour_test, ax=ax1, label='AUROC', shrink=0.8)
    
    # 2. Test AUROC Heatmap
    ax2 = plt.subplot(3, 4, 2)
    sns.heatmap(test_auroc_avg, annot=True, fmt='.3f', cmap=cmap_test, 
                xticklabels=masking_labels, yticklabels=noise_labels,
                cbar_kws={'label': 'Test AUROC'}, ax=ax2, 
                annot_kws={'fontweight': 'bold', 'fontsize': 11},
                linewidths=1, linecolor='gray', vmin=0.45, vmax=0.65)
    ax2.set_xlabel('Masking Ratio', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Noise Corruption Level', fontweight='bold', fontsize=12)
    ax2.set_title('Test AUROC - Heatmap', fontweight='bold', fontsize=14)
    
    # 3. Test AUROC 3D Surface
    ax3 = fig.add_subplot(3, 4, 3, projection='3d')
    X_3d, Y_3d = np.meshgrid(range(len(masking_ratios)), range(len(noise_levels)))
    surface = ax3.plot_surface(X_3d, Y_3d, test_auroc_avg, cmap=cmap_test, 
                              alpha=0.9, edgecolor='none', antialiased=True)
    ax3.plot_wireframe(X_3d, Y_3d, test_auroc_avg, color='black', 
                      alpha=0.2, linewidth=0.5)
    
    ax3.set_xticks(range(len(masking_ratios)))
    ax3.set_yticks(range(len(noise_levels)))
    ax3.set_xticklabels(masking_labels, fontsize=10)
    ax3.set_yticklabels(noise_labels, fontsize=10)
    ax3.set_xlabel('Masking Ratio', fontweight='bold', labelpad=10)
    ax3.set_ylabel('Noise Level', fontweight='bold', labelpad=10)
    ax3.set_zlabel('Test AUROC', fontweight='bold', labelpad=10)
    ax3.set_title('Test AUROC - 3D Surface', fontweight='bold', fontsize=14)
    ax3.view_init(elev=25, azim=45)
    
    # 4. Test Trends
    ax4 = plt.subplot(3, 4, 4)
    colors = plt.cm.coolwarm(np.linspace(0.2, 0.8, len(noise_levels)))
    markers = ['o', 's', '^', 'D']
    
    for i, noise in enumerate(noise_labels):
        ax4.plot(range(len(masking_ratios)), test_auroc_avg[i, :], 
                color=colors[i], marker=markers[i], linewidth=2.5, markersize=10,
                label=f'Noise {noise}', markerfacecolor='white', markeredgewidth=2)
    
    ax4.set_xticks(range(len(masking_ratios)))
    ax4.set_xticklabels(masking_labels, fontsize=11)
    ax4.set_xlabel('Masking Ratio', fontweight='bold', fontsize=12)
    ax4.set_ylabel('Test AUROC', fontweight='bold', fontsize=12)
    ax4.set_title('Test AUROC Trends', fontweight='bold', fontsize=14)
    ax4.legend(title='Noise Level', loc='best', fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0.45, 0.65])
    
    # Find and annotate best test configuration
    best_test_idx = np.unravel_index(np.argmax(test_auroc_avg), test_auroc_avg.shape)
    best_test_value = test_auroc_avg[best_test_idx]
    ax4.annotate(f'Best: {best_test_value:.3f}\n(M{masking_ratios[best_test_idx[1]]}, N{noise_levels[best_test_idx[0]]})',
                xy=(best_test_idx[1], best_test_value), xytext=(best_test_idx[1]-0.5, best_test_value+0.02),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=11, fontweight='bold', color='red',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
    
    # ========== ROW 2: VALIDATION METRICS ==========
    
    # 5. Val AUROC Contour Plot
    ax5 = plt.subplot(3, 4, 5)
    contour_val = ax5.contourf(X, Y, val_auroc_avg, levels=levels, cmap=cmap_val, alpha=0.9)
    contour_lines_val = ax5.contour(X, Y, val_auroc_avg, levels=10, colors='black', 
                                    alpha=0.4, linewidths=0.8)
    ax5.clabel(contour_lines_val, inline=True, fontsize=9, fmt='%.3f')
    
    for i in range(len(noise_levels)):
        for j in range(len(masking_ratios)):
            ax5.plot(j, i, 'ko', markersize=8)
            color = 'white' if val_auroc_avg[i,j] < 0.50 else 'black'
            ax5.text(j, i, f'{val_auroc_avg[i,j]:.3f}', ha='center', va='center', 
                    color=color, fontweight='bold', fontsize=10)
    
    ax5.set_xticks(range(len(masking_ratios)))
    ax5.set_yticks(range(len(noise_levels)))
    ax5.set_xticklabels(masking_labels, fontsize=11)
    ax5.set_yticklabels(noise_labels, fontsize=11)
    ax5.set_xlabel('Masking Ratio', fontweight='bold', fontsize=12)
    ax5.set_ylabel('Noise Corruption Level', fontweight='bold', fontsize=12)
    ax5.set_title('Validation AUROC - Contour Plot', fontweight='bold', fontsize=14)
    plt.colorbar(contour_val, ax=ax5, label='AUROC', shrink=0.8)
    
    # 6. Val AUROC Heatmap
    ax6 = plt.subplot(3, 4, 6)
    sns.heatmap(val_auroc_avg, annot=True, fmt='.3f', cmap=cmap_val, 
                xticklabels=masking_labels, yticklabels=noise_labels,
                cbar_kws={'label': 'Val AUROC'}, ax=ax6, 
                annot_kws={'fontweight': 'bold', 'fontsize': 11},
                linewidths=1, linecolor='gray', vmin=0.40, vmax=0.60)
    ax6.set_xlabel('Masking Ratio', fontweight='bold', fontsize=12)
    ax6.set_ylabel('Noise Corruption Level', fontweight='bold', fontsize=12)
    ax6.set_title('Validation AUROC - Heatmap', fontweight='bold', fontsize=14)
    
    # 7. Val AUROC 3D Surface
    ax7 = fig.add_subplot(3, 4, 7, projection='3d')
    surface_val = ax7.plot_surface(X_3d, Y_3d, val_auroc_avg, cmap=cmap_val, 
                                   alpha=0.9, edgecolor='none', antialiased=True)
    ax7.plot_wireframe(X_3d, Y_3d, val_auroc_avg, color='black', 
                      alpha=0.2, linewidth=0.5)
    
    ax7.set_xticks(range(len(masking_ratios)))
    ax7.set_yticks(range(len(noise_levels)))
    ax7.set_xticklabels(masking_labels, fontsize=10)
    ax7.set_yticklabels(noise_labels, fontsize=10)
    ax7.set_xlabel('Masking Ratio', fontweight='bold', labelpad=10)
    ax7.set_ylabel('Noise Level', fontweight='bold', labelpad=10)
    ax7.set_zlabel('Val AUROC', fontweight='bold', labelpad=10)
    ax7.set_title('Validation AUROC - 3D Surface', fontweight='bold', fontsize=14)
    ax7.view_init(elev=25, azim=45)
    
    # 8. Val Trends
    ax8 = plt.subplot(3, 4, 8)
    
    for i, noise in enumerate(noise_labels):
        ax8.plot(range(len(masking_ratios)), val_auroc_avg[i, :], 
                color=colors[i], marker=markers[i], linewidth=2.5, markersize=10,
                label=f'Noise {noise}', markerfacecolor='white', markeredgewidth=2)
    
    ax8.set_xticks(range(len(masking_ratios)))
    ax8.set_xticklabels(masking_labels, fontsize=11)
    ax8.set_xlabel('Masking Ratio', fontweight='bold', fontsize=12)
    ax8.set_ylabel('Validation AUROC', fontweight='bold', fontsize=12)
    ax8.set_title('Validation AUROC Trends', fontweight='bold', fontsize=14)
    ax8.legend(title='Noise Level', loc='best', fontsize=10)
    ax8.grid(True, alpha=0.3)
    ax8.set_ylim([0.40, 0.60])
    
    # Find and annotate best val configuration
    best_val_idx = np.unravel_index(np.argmax(val_auroc_avg), val_auroc_avg.shape)
    best_val_value = val_auroc_avg[best_val_idx]
    ax8.annotate(f'Best: {best_val_value:.3f}\n(M{masking_ratios[best_val_idx[1]]}, N{noise_levels[best_val_idx[0]]})',
                xy=(best_val_idx[1], best_val_value), xytext=(best_val_idx[1]-0.5, best_val_value+0.02),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=11, fontweight='bold', color='red',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
    
    # ========== ROW 3: COMPARISON AND STATISTICS ==========
    
    # 9. Direct Comparison
    ax9 = plt.subplot(3, 4, 9)
    # Plot test vs val for each configuration
    test_flat = test_auroc_avg.flatten()
    val_flat = val_auroc_avg.flatten()
    
    ax9.scatter(val_flat, test_flat, s=100, alpha=0.7, c=range(len(test_flat)), cmap='viridis')
    
    # Add diagonal line
    min_val = min(val_flat.min(), test_flat.min())
    max_val = max(val_flat.max(), test_flat.max())
    ax9.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='y=x')
    
    # Add labels for best configs
    for i, (v, t) in enumerate(zip(val_flat, test_flat)):
        if t == test_flat.max() or v == val_flat.max():
            i_idx, j_idx = np.unravel_index(i, test_auroc_avg.shape)
            ax9.annotate(f'M{masking_ratios[j_idx]}\nN{noise_levels[i_idx]}',
                        xy=(v, t), xytext=(v+0.01, t+0.01),
                        fontsize=9, alpha=0.8)
    
    ax9.set_xlabel('Validation AUROC', fontweight='bold', fontsize=12)
    ax9.set_ylabel('Test AUROC', fontweight='bold', fontsize=12)
    ax9.set_title('Test vs Validation AUROC', fontweight='bold', fontsize=14)
    ax9.grid(True, alpha=0.3)
    ax9.legend()
    
    # 10. Difference Plot (Test - Val)
    ax10 = plt.subplot(3, 4, 10)
    diff = test_auroc_avg - val_auroc_avg
    sns.heatmap(diff, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                xticklabels=masking_labels, yticklabels=noise_labels,
                cbar_kws={'label': 'Test - Val AUROC'}, ax=ax10,
                annot_kws={'fontweight': 'bold', 'fontsize': 11},
                linewidths=1, linecolor='gray')
    ax10.set_xlabel('Masking Ratio', fontweight='bold', fontsize=12)
    ax10.set_ylabel('Noise Corruption Level', fontweight='bold', fontsize=12)
    ax10.set_title('Generalization Gap (Test - Val)', fontweight='bold', fontsize=14)
    
    # 11. Statistics Summary
    ax11 = plt.subplot(3, 4, 11)
    ax11.axis('off')
    
    stats_text = f"""
    COMPREHENSIVE STATISTICS
    {'='*45}
    
    TEST AUROC:
      Mean: {np.mean(test_auroc_avg):.4f} ± {np.std(test_auroc_avg):.4f}
      Range: [{np.min(test_auroc_avg):.4f}, {np.max(test_auroc_avg):.4f}]
      Best: M{masking_ratios[best_test_idx[1]]}_N{noise_levels[best_test_idx[0]]} = {best_test_value:.4f}
      
    VALIDATION AUROC:
      Mean: {np.mean(val_auroc_avg):.4f} ± {np.std(val_auroc_avg):.4f}
      Range: [{np.min(val_auroc_avg):.4f}, {np.max(val_auroc_avg):.4f}]
      Best: M{masking_ratios[best_val_idx[1]]}_N{noise_levels[best_val_idx[0]]} = {best_val_value:.4f}
    
    GENERALIZATION GAP:
      Mean Gap: {np.mean(diff):.4f}
      Max Overfit: {np.min(diff):.4f}
      Max Underfit: {np.max(diff):.4f}
    
    OPTIMAL SETTINGS (Test):
      Best Masking: M{masking_ratios[np.argmax(np.mean(test_auroc_avg, axis=0))]} 
                    ({np.max(np.mean(test_auroc_avg, axis=0)):.4f})
      Best Noise: N{noise_levels[np.argmax(np.mean(test_auroc_avg, axis=1))]} 
                  ({np.max(np.mean(test_auroc_avg, axis=1)):.4f})
    """
    
    ax11.text(0.05, 0.95, stats_text, transform=ax11.transAxes, 
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    ax11.set_title('Summary Statistics', fontweight='bold', fontsize=14)
    
    # 12. Key Insights
    ax12 = plt.subplot(3, 4, 12)
    ax12.axis('off')
    
    insights_text = f"""
    KEY INSIGHTS
    {'='*45}
    
    • Optimal test config: M{masking_ratios[best_test_idx[1]]}_N{noise_levels[best_test_idx[0]]}
      (AUROC = {best_test_value:.3f})
    
    • Performance generally {'decreases' if np.mean(test_auroc_avg[:, -1]) < np.mean(test_auroc_avg[:, 0]) else 'increases'} 
      with higher masking ratios
    
    • Noise level N{noise_levels[np.argmax(np.mean(test_auroc_avg, axis=1))]} shows 
      best average performance
    
    • {'Good generalization' if np.mean(diff) > 0 else 'Overfitting tendency'}
      (avg gap: {np.mean(diff):.3f})
    
    • Validation and test metrics are
      {'well correlated' if np.corrcoef(test_flat, val_flat)[0,1] > 0.7 else 'moderately correlated'}
      (r = {np.corrcoef(test_flat, val_flat)[0,1]:.3f})
    
    • Most stable config across val/test:
      M{masking_ratios[np.argmin(np.abs(diff)).item() % len(masking_ratios)]}_N{noise_levels[np.argmin(np.abs(diff)).item() // len(masking_ratios)]}
      (gap = {diff.flatten()[np.argmin(np.abs(diff))]:.3f})
    """
    
    ax12.text(0.05, 0.95, insights_text, transform=ax12.transAxes, 
             fontsize=11, verticalalignment='top', fontfamily='sans-serif',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
    ax12.set_title('Key Insights', fontweight='bold', fontsize=14)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.94, hspace=0.3, wspace=0.3)
    
    # Save figure
    output_path = output_dir / 'brats18_averaged_auroc_comprehensive.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.show()
    
    print(f"\nPlot saved to:")
    print(f"  - {output_path}")
    print(f"  - {output_path.with_suffix('.pdf')}")
    
    # Print summary to console
    print("\n" + "="*60)
    print("BRATS18 AVERAGED AUROC SUMMARY")
    print("="*60)
    print(f"\nTest AUROC:")
    print(f"  Best: M{masking_ratios[best_test_idx[1]]}_N{noise_levels[best_test_idx[0]]} = {best_test_value:.4f}")
    print(f"  Mean: {np.mean(test_auroc_avg):.4f} ± {np.std(test_auroc_avg):.4f}")
    print(f"\nValidation AUROC:")
    print(f"  Best: M{masking_ratios[best_val_idx[1]]}_N{noise_levels[best_val_idx[0]]} = {best_val_value:.4f}")
    print(f"  Mean: {np.mean(val_auroc_avg):.4f} ± {np.std(val_auroc_avg):.4f}")
    print(f"\nGeneralization Gap (Test - Val):")
    print(f"  Mean: {np.mean(diff):.4f}")
    print(f"  Range: [{np.min(diff):.4f}, {np.max(diff):.4f}]")

if __name__ == "__main__":
    create_averaged_auroc_visualization()