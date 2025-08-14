#!/usr/bin/env python3
"""
Create averaged AUROC contour plots for UCSF PDGM IDH classification benchmark.
Shows both validation and test metrics averaged across available modalities (SWI, ASL).
Handles sparse data with interpolation.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.interpolate import griddata
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.3)

def interpolate_sparse_grid(grid, masking_ratios, noise_levels):
    """Interpolate sparse grid data for smoother visualization."""
    # Create meshgrid for interpolation
    X, Y = np.meshgrid(range(len(masking_ratios)), range(len(noise_levels)))
    
    # Get valid data points
    valid_mask = ~np.isnan(grid)
    if np.sum(valid_mask) < 3:  # Not enough points for interpolation
        return grid
    
    points = np.column_stack((X[valid_mask], Y[valid_mask]))
    values = grid[valid_mask]
    
    # Create fine grid for interpolation
    grid_x, grid_y = np.meshgrid(np.linspace(0, len(masking_ratios)-1, 50),
                                 np.linspace(0, len(noise_levels)-1, 50))
    
    # Interpolate
    try:
        interpolated = griddata(points, values, (grid_x, grid_y), method='cubic')
        # Fill NaN values with nearest neighbor interpolation
        if np.any(np.isnan(interpolated)):
            interpolated_nn = griddata(points, values, (grid_x, grid_y), method='nearest')
            interpolated[np.isnan(interpolated)] = interpolated_nn[np.isnan(interpolated)]
    except:
        # Fallback to linear interpolation if cubic fails
        interpolated = griddata(points, values, (grid_x, grid_y), method='linear')
    
    return interpolated, grid_x, grid_y

def create_ucsf_pdgm_auroc_visualization():
    """Create comprehensive averaged AUROC visualization for UCSF PDGM."""
    
    # Load processed data
    data_dir = Path('/home/t-jiachentu/repos/benchmarking/misc/data/ablation/ucsf_pdgm_contour_analysis/data')
    output_dir = Path('/home/t-jiachentu/repos/benchmarking/misc/data/ablation/ucsf_pdgm_contour_analysis')
    
    print("Loading UCSF PDGM processed grids...")
    data = np.load(data_dir / 'processed_grids.npz', allow_pickle=True)
    grids = data['grids'].item()
    masking_ratios = data['masking_ratios']
    noise_levels = data['noise_levels']
    modalities = ['SWI', 'ASL']
    
    # Calculate averaged AUROC across available modalities
    test_auroc_avg = np.zeros((len(noise_levels), len(masking_ratios)))
    val_auroc_avg = np.zeros((len(noise_levels), len(masking_ratios)))
    count_matrix = np.zeros((len(noise_levels), len(masking_ratios)))
    
    # Average only where data exists
    for modality in modalities:
        test_grid = grids[modality]['Test_AUROC']
        val_grid = grids[modality]['Val_AUROC']
        
        # Add to average only where not NaN
        valid_mask = ~np.isnan(test_grid)
        test_auroc_avg[valid_mask] += test_grid[valid_mask]
        
        valid_mask_val = ~np.isnan(val_grid)
        val_auroc_avg[valid_mask_val] += val_grid[valid_mask_val]
        
        # Count valid entries
        count_matrix[valid_mask] += 1
    
    # Compute average
    test_auroc_avg[count_matrix > 0] /= count_matrix[count_matrix > 0]
    test_auroc_avg[count_matrix == 0] = np.nan
    
    # For validation, use same count matrix
    val_auroc_avg[count_matrix > 0] /= count_matrix[count_matrix > 0]
    val_auroc_avg[count_matrix == 0] = np.nan
    
    # Create figure
    fig = plt.figure(figsize=(22, 14))
    fig.suptitle('UCSF-PDGM IDH Classification - FlowMDAE Ablation Study\nAveraged AUROC Across Specialized MRI Sequences (SWI, ASL)', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Prepare labels
    masking_labels = [f'{m}%' for m in masking_ratios]
    noise_labels = [f'{n}%' for n in noise_levels]
    
    # Custom colormaps
    cmap_test = 'RdYlGn'
    cmap_val = 'PuBuGn'
    
    # ========== TEST METRICS VISUALIZATION ==========
    
    # 1. Test AUROC Contour (with interpolation for sparse data)
    ax1 = plt.subplot(2, 4, 1)
    X, Y = np.meshgrid(range(len(masking_ratios)), range(len(noise_levels)))
    
    # Interpolate for smoother visualization
    if np.sum(~np.isnan(test_auroc_avg)) > 3:
        interpolated_test, grid_x, grid_y = interpolate_sparse_grid(test_auroc_avg, masking_ratios, noise_levels)
        
        # Plot interpolated contour
        levels = np.linspace(np.nanmin(test_auroc_avg), np.nanmax(test_auroc_avg), 15)
        contour_test = ax1.contourf(grid_x, grid_y, interpolated_test, levels=levels, 
                                    cmap=cmap_test, alpha=0.8)
        
        # Add contour lines
        contour_lines = ax1.contour(grid_x, grid_y, interpolated_test, levels=8, 
                                   colors='white', alpha=0.5, linewidths=1.0)
    else:
        # Fallback to simple plotting if not enough data
        contour_test = ax1.contourf(X, Y, np.nan_to_num(test_auroc_avg, nan=np.nanmean(test_auroc_avg)), 
                                    levels=15, cmap=cmap_test, alpha=0.8)
    
    # Add actual data points
    for i in range(len(noise_levels)):
        for j in range(len(masking_ratios)):
            if not np.isnan(test_auroc_avg[i, j]):
                ax1.plot(j, i, 'ko', markersize=10, markeredgecolor='white', markeredgewidth=1.5)
                color = 'white' if test_auroc_avg[i,j] < 0.7 else 'black'
                ax1.text(j, i-0.15, f'{test_auroc_avg[i,j]:.3f}', ha='center', va='center', 
                        color=color, fontweight='bold', fontsize=10,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                                 edgecolor='gray', alpha=0.8))
            else:
                # Mark missing data points
                ax1.plot(j, i, 'x', color='red', markersize=8, alpha=0.5)
    
    ax1.set_xticks(range(len(masking_ratios)))
    ax1.set_yticks(range(len(noise_levels)))
    ax1.set_xticklabels(masking_labels, fontsize=11)
    ax1.set_yticklabels(noise_labels, fontsize=11)
    ax1.set_xlabel('Masking Ratio', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Noise Corruption Level', fontweight='bold', fontsize=12)
    ax1.set_title('Test AUROC - Contour Plot', fontweight='bold', fontsize=13)
    plt.colorbar(contour_test, ax=ax1, label='AUROC', shrink=0.8)
    
    # 2. Test AUROC Heatmap
    ax2 = plt.subplot(2, 4, 2)
    # Create mask for missing values
    mask = np.isnan(test_auroc_avg)
    sns.heatmap(test_auroc_avg, annot=True, fmt='.3f', cmap=cmap_test, 
                mask=mask, xticklabels=masking_labels, yticklabels=noise_labels,
                cbar_kws={'label': 'Test AUROC'}, ax=ax2, 
                annot_kws={'fontweight': 'bold', 'fontsize': 11},
                linewidths=1.5, linecolor='gray', vmin=0.65, vmax=0.85,
                square=True)
    ax2.set_xlabel('Masking Ratio', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Noise Corruption Level', fontweight='bold', fontsize=12)
    ax2.set_title('Test AUROC - Heatmap', fontweight='bold', fontsize=13)
    
    # 3. Test Trends by Modality
    ax3 = plt.subplot(2, 4, 3)
    colors_mod = ['#e41a1c', '#377eb8']
    markers_mod = ['o', 's']
    
    for idx, modality in enumerate(modalities):
        test_grid = grids[modality]['Test_AUROC']
        # Plot mean across noise levels for each masking ratio
        mean_vals = np.nanmean(test_grid, axis=0)
        ax3.plot(range(len(masking_ratios)), mean_vals, 
                color=colors_mod[idx], marker=markers_mod[idx], 
                linewidth=2.5, markersize=10, label=modality,
                markerfacecolor='white', markeredgewidth=2)
    
    ax3.set_xticks(range(len(masking_ratios)))
    ax3.set_xticklabels(masking_labels, fontsize=11)
    ax3.set_xlabel('Masking Ratio', fontweight='bold', fontsize=12)
    ax3.set_ylabel('Test AUROC (Mean)', fontweight='bold', fontsize=12)
    ax3.set_title('Test Performance by Modality', fontweight='bold', fontsize=13)
    ax3.legend(fontsize=11, loc='best')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0.65, 0.85])
    
    # 4. Coverage Map
    ax4 = plt.subplot(2, 4, 4)
    coverage = (~np.isnan(test_auroc_avg)).astype(int)
    sns.heatmap(coverage, annot=False, cmap=['lightcoral', 'lightgreen'], 
                xticklabels=masking_labels, yticklabels=noise_labels,
                cbar_kws={'label': 'Data Available', 'ticks': [0, 1]}, ax=ax4,
                linewidths=1.5, linecolor='black', square=True)
    ax4.set_xlabel('Masking Ratio', fontweight='bold', fontsize=12)
    ax4.set_ylabel('Noise Corruption Level', fontweight='bold', fontsize=12)
    ax4.set_title('Configuration Coverage', fontweight='bold', fontsize=13)
    
    # Add text annotations for coverage
    for i in range(len(noise_levels)):
        for j in range(len(masking_ratios)):
            text = '✓' if not np.isnan(test_auroc_avg[i, j]) else '✗'
            color = 'darkgreen' if text == '✓' else 'darkred'
            ax4.text(j+0.5, i+0.5, text, ha='center', va='center',
                    color=color, fontsize=14, fontweight='bold')
    
    # ========== VALIDATION METRICS VISUALIZATION ==========
    
    # 5. Val AUROC Contour
    ax5 = plt.subplot(2, 4, 5)
    
    if np.sum(~np.isnan(val_auroc_avg)) > 3:
        interpolated_val, grid_x_val, grid_y_val = interpolate_sparse_grid(val_auroc_avg, masking_ratios, noise_levels)
        levels_val = np.linspace(np.nanmin(val_auroc_avg), np.nanmax(val_auroc_avg), 15)
        contour_val = ax5.contourf(grid_x_val, grid_y_val, interpolated_val, 
                                   levels=levels_val, cmap=cmap_val, alpha=0.8)
        contour_lines_val = ax5.contour(grid_x_val, grid_y_val, interpolated_val, 
                                        levels=8, colors='white', alpha=0.5, linewidths=1.0)
    else:
        contour_val = ax5.contourf(X, Y, np.nan_to_num(val_auroc_avg, nan=np.nanmean(val_auroc_avg)), 
                                   levels=15, cmap=cmap_val, alpha=0.8)
    
    # Add actual data points for validation
    for i in range(len(noise_levels)):
        for j in range(len(masking_ratios)):
            if not np.isnan(val_auroc_avg[i, j]):
                ax5.plot(j, i, 'ko', markersize=10, markeredgecolor='white', markeredgewidth=1.5)
                color = 'white' if val_auroc_avg[i,j] < 0.65 else 'black'
                ax5.text(j, i-0.15, f'{val_auroc_avg[i,j]:.3f}', ha='center', va='center', 
                        color=color, fontweight='bold', fontsize=10,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                                 edgecolor='gray', alpha=0.8))
            else:
                ax5.plot(j, i, 'x', color='red', markersize=8, alpha=0.5)
    
    ax5.set_xticks(range(len(masking_ratios)))
    ax5.set_yticks(range(len(noise_levels)))
    ax5.set_xticklabels(masking_labels, fontsize=11)
    ax5.set_yticklabels(noise_labels, fontsize=11)
    ax5.set_xlabel('Masking Ratio', fontweight='bold', fontsize=12)
    ax5.set_ylabel('Noise Corruption Level', fontweight='bold', fontsize=12)
    ax5.set_title('Validation AUROC - Contour Plot', fontweight='bold', fontsize=13)
    plt.colorbar(contour_val, ax=ax5, label='AUROC', shrink=0.8)
    
    # 6. Val AUROC Heatmap
    ax6 = plt.subplot(2, 4, 6)
    mask_val = np.isnan(val_auroc_avg)
    sns.heatmap(val_auroc_avg, annot=True, fmt='.3f', cmap=cmap_val, 
                mask=mask_val, xticklabels=masking_labels, yticklabels=noise_labels,
                cbar_kws={'label': 'Val AUROC'}, ax=ax6, 
                annot_kws={'fontweight': 'bold', 'fontsize': 11},
                linewidths=1.5, linecolor='gray', vmin=0.60, vmax=0.80,
                square=True)
    ax6.set_xlabel('Masking Ratio', fontweight='bold', fontsize=12)
    ax6.set_ylabel('Noise Corruption Level', fontweight='bold', fontsize=12)
    ax6.set_title('Validation AUROC - Heatmap', fontweight='bold', fontsize=13)
    
    # 7. Comparison Plot
    ax7 = plt.subplot(2, 4, 7)
    test_flat = test_auroc_avg[~np.isnan(test_auroc_avg)]
    val_flat = val_auroc_avg[~np.isnan(val_auroc_avg)]
    
    if len(test_flat) > 0 and len(val_flat) > 0:
        ax7.scatter(val_flat, test_flat, s=100, alpha=0.7, c=range(len(test_flat)), cmap='viridis')
        
        # Add diagonal line
        min_val = min(val_flat.min(), test_flat.min())
        max_val = max(val_flat.max(), test_flat.max())
        ax7.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='y=x')
        
        ax7.set_xlabel('Validation AUROC', fontweight='bold', fontsize=12)
        ax7.set_ylabel('Test AUROC', fontweight='bold', fontsize=12)
        ax7.set_title('Test vs Validation', fontweight='bold', fontsize=13)
        ax7.grid(True, alpha=0.3)
        ax7.legend()
    
    # 8. Summary Statistics
    ax8 = plt.subplot(2, 4, 8)
    ax8.axis('off')
    
    # Calculate statistics
    test_valid = test_auroc_avg[~np.isnan(test_auroc_avg)]
    val_valid = val_auroc_avg[~np.isnan(val_auroc_avg)]
    
    if len(test_valid) > 0:
        best_test_idx = np.nanargmax(test_auroc_avg)
        best_test_i, best_test_j = np.unravel_index(best_test_idx, test_auroc_avg.shape)
        best_test_value = test_auroc_avg[best_test_i, best_test_j]
        
        stats_text = f"""
SUMMARY STATISTICS
{'='*35}

TEST AUROC:
  Mean: {np.mean(test_valid):.3f} ± {np.std(test_valid):.3f}
  Range: [{np.min(test_valid):.3f}, {np.max(test_valid):.3f}]
  Best: M{masking_ratios[best_test_j]}_N{noise_levels[best_test_i]}
        = {best_test_value:.3f}

VALIDATION AUROC:
  Mean: {np.mean(val_valid):.3f} ± {np.std(val_valid):.3f}
  Range: [{np.min(val_valid):.3f}, {np.max(val_valid):.3f}]

DATA COVERAGE:
  Configurations: {len(test_valid)}/16
  Coverage: {len(test_valid)/16*100:.1f}%
  
MODALITY PERFORMANCE:
  SWI Mean: {np.nanmean(grids['SWI']['Test_AUROC']):.3f}
  ASL Mean: {np.nanmean(grids['ASL']['Test_AUROC']):.3f}

Note: UCSF-PDGM uses specialized
MRI sequences for generalization
testing. Sparse coverage is expected.
        """
        
        ax8.text(0.05, 0.95, stats_text, transform=ax8.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, hspace=0.25, wspace=0.25)
    
    # Save figure
    output_path = output_dir / 'ucsf_pdgm_averaged_auroc_comprehensive.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"\nPlot saved to:")
    print(f"  - {output_path}")
    print(f"  - {output_path.with_suffix('.pdf')}")
    
    # Print summary
    print("\n" + "="*60)
    print("UCSF-PDGM IDH CLASSIFICATION - AUROC SUMMARY")
    print("="*60)
    
    if len(test_valid) > 0:
        print(f"\nTest AUROC (Averaged across SWI & ASL):")
        print(f"  Best: M{masking_ratios[best_test_j]}_N{noise_levels[best_test_i]} = {best_test_value:.3f}")
        print(f"  Mean: {np.mean(test_valid):.3f} ± {np.std(test_valid):.3f}")
        print(f"  Range: [{np.min(test_valid):.3f}, {np.max(test_valid):.3f}]")
        
        print(f"\nValidation AUROC:")
        print(f"  Mean: {np.mean(val_valid):.3f} ± {np.std(val_valid):.3f}")
        print(f"  Range: [{np.min(val_valid):.3f}, {np.max(val_valid):.3f}]")
        
        print(f"\nData Coverage:")
        print(f"  {len(test_valid)}/16 configurations ({len(test_valid)/16*100:.1f}%)")
        print(f"\nNote: UCSF-PDGM is a generalization test benchmark with specialized sequences")

if __name__ == "__main__":
    create_ucsf_pdgm_auroc_visualization()