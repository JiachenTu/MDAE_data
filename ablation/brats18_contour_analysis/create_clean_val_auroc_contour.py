#!/usr/bin/env python3
"""
Create a clean, publication-ready contour plot for validation AUROC metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# Set style for clean visualization
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.4)

def create_clean_contour_plot():
    """Create a clean contour visualization for validation AUROC."""
    
    # Load the data
    data_dir = Path('/home/t-jiachentu/repos/benchmarking/misc/data/ablation/contour_analysis/data')
    output_dir = Path('/home/t-jiachentu/repos/benchmarking/misc/data/ablation/contour_analysis')
    
    # Load processed grids
    data = np.load(data_dir / 'processed_grids.npz', allow_pickle=True)
    grids = data['grids'].item()
    masking_ratios = [25, 50, 75, 95]
    noise_levels = [25, 50, 75, 100]
    modalities = ['T1', 'T1CE', 'T2', 'FLAIR']
    
    # Calculate averaged validation AUROC
    val_auroc_avg = np.zeros((len(noise_levels), len(masking_ratios)))
    for modality in modalities:
        val_auroc_avg += grids[modality]['Val_AUROC']
    val_auroc_avg /= len(modalities)
    
    # Create figure with optimal size for clarity
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('BraTS18 LGG vs HGG - Validation AUROC Analysis\nAveraged Across All Modalities (T1, T1CE, T2, FLAIR)', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    # Create custom colormap for better visibility
    colors = ['#440154', '#31688e', '#35b779', '#fde725']  # Viridis colors
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
    
    # ============ SUBPLOT 1: CONTOUR PLOT ============
    ax1 = axes[0]
    X, Y = np.meshgrid(masking_ratios, noise_levels)
    
    # Create smooth contour plot
    levels = np.linspace(0.52, 0.60, 17)  # Focused range for better visibility
    cs = ax1.contourf(X, Y, val_auroc_avg, levels=levels, cmap=cmap, alpha=0.9, extend='both')
    
    # Add contour lines
    cs2 = ax1.contour(X, Y, val_auroc_avg, levels=8, colors='white', alpha=0.5, linewidths=1.0)
    ax1.clabel(cs2, inline=True, fontsize=10, fmt='%.3f', colors='white')
    
    # Add data points with values
    for i, n in enumerate(noise_levels):
        for j, m in enumerate(masking_ratios):
            # Determine text color based on background
            val = val_auroc_avg[i, j]
            text_color = 'white' if val < 0.56 else 'black'
            
            # Add marker
            ax1.plot(m, n, 'o', color='black', markersize=8, 
                    markeredgecolor='white', markeredgewidth=1)
            
            # Add value text
            ax1.text(m, n-3, f'{val:.3f}', ha='center', va='center',
                    fontsize=11, fontweight='bold', color=text_color,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                             edgecolor='none', alpha=0.7))
    
    # Formatting
    ax1.set_xlabel('Masking Ratio (%)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Noise Corruption Level (%)', fontsize=13, fontweight='bold')
    ax1.set_title('Contour Plot with Values', fontsize=14, fontweight='bold')
    ax1.set_xticks(masking_ratios)
    ax1.set_yticks(noise_levels)
    ax1.grid(True, alpha=0.2, linestyle='--')
    
    # Add colorbar
    cbar1 = plt.colorbar(cs, ax=ax1, label='Validation AUROC', shrink=0.9)
    cbar1.ax.tick_params(labelsize=11)
    
    # Highlight best configuration
    best_idx = np.unravel_index(np.argmax(val_auroc_avg), val_auroc_avg.shape)
    best_val = val_auroc_avg[best_idx]
    ax1.scatter(masking_ratios[best_idx[1]], noise_levels[best_idx[0]], 
               s=300, marker='*', color='red', edgecolor='darkred', linewidth=2,
               label=f'Best: M{masking_ratios[best_idx[1]]}_N{noise_levels[best_idx[0]]} = {best_val:.3f}')
    ax1.legend(loc='lower left', fontsize=11, framealpha=0.9)
    
    # ============ SUBPLOT 2: HEATMAP ============
    ax2 = axes[1]
    
    # Create annotated heatmap
    sns.heatmap(val_auroc_avg, annot=True, fmt='.3f', cmap=cmap,
                xticklabels=[f'{m}%' for m in masking_ratios],
                yticklabels=[f'{n}%' for n in noise_levels],
                cbar_kws={'label': 'Validation AUROC', 'shrink': 0.9},
                ax=ax2, linewidths=2, linecolor='white',
                annot_kws={'fontsize': 12, 'fontweight': 'bold'},
                vmin=0.52, vmax=0.60, square=True)
    
    ax2.set_xlabel('Masking Ratio', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Noise Corruption Level', fontsize=13, fontweight='bold')
    ax2.set_title('Heatmap Representation', fontsize=14, fontweight='bold')
    
    # Rotate labels for better readability
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)
    ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)
    
    # ============ SUBPLOT 3: LINE PLOT TRENDS ============
    ax3 = axes[2]
    
    # Define distinct colors and markers for each noise level
    colors_lines = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
    markers = ['o', 's', '^', 'D']
    linestyles = ['-', '--', '-.', ':']
    
    # Plot trends for each noise level
    for i, n in enumerate(noise_levels):
        ax3.plot(masking_ratios, val_auroc_avg[i, :], 
                color=colors_lines[i], marker=markers[i], 
                markersize=10, linewidth=2.5, linestyle=linestyles[i],
                label=f'Noise {n}%', markerfacecolor='white', 
                markeredgewidth=2, markeredgecolor=colors_lines[i])
    
    # Add grid and formatting
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_xlabel('Masking Ratio (%)', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Validation AUROC', fontsize=13, fontweight='bold')
    ax3.set_title('Performance Trends', fontsize=14, fontweight='bold')
    ax3.set_xticks(masking_ratios)
    ax3.set_ylim([0.51, 0.61])
    
    # Add legend with better positioning
    ax3.legend(loc='best', fontsize=11, framealpha=0.9, ncol=2)
    
    # Highlight best point
    ax3.scatter(masking_ratios[best_idx[1]], best_val, 
               s=200, marker='*', color='gold', edgecolor='darkred', 
               linewidth=2, zorder=5)
    
    # Add annotation for best point
    ax3.annotate(f'Best: {best_val:.3f}',
                xy=(masking_ratios[best_idx[1]], best_val),
                xytext=(masking_ratios[best_idx[1]]+10, best_val+0.005),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                fontsize=11, fontweight='bold', color='red',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', 
                         edgecolor='red', alpha=0.8))
    
    # Add horizontal line for mean
    mean_val = np.mean(val_auroc_avg)
    ax3.axhline(y=mean_val, color='gray', linestyle='--', alpha=0.5, 
               label=f'Mean: {mean_val:.3f}')
    
    # Update legend to include mean line
    ax3.legend(loc='best', fontsize=11, framealpha=0.9, ncol=2)
    
    # ============ ADD SUMMARY STATISTICS BOX ============
    # Create text box with key statistics
    stats_text = (
        f"Summary Statistics\n"
        f"{'─'*20}\n"
        f"Best: M{masking_ratios[best_idx[1]]}_N{noise_levels[best_idx[0]]} = {best_val:.3f}\n"
        f"Mean: {np.mean(val_auroc_avg):.3f} ± {np.std(val_auroc_avg):.3f}\n"
        f"Range: [{np.min(val_auroc_avg):.3f}, {np.max(val_auroc_avg):.3f}]\n"
    )
    
    # Add text box below the plots
    fig.text(0.5, -0.05, stats_text, ha='center', fontsize=12,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', 
                     edgecolor='black', alpha=0.8))
    
    # Adjust layout for optimal spacing
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, top=0.92, wspace=0.3)
    
    # Save the figure
    output_path = output_dir / 'brats18_val_auroc_clean_contour.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"\nClean contour plot saved to:")
    print(f"  - {output_path}")
    print(f"  - {output_path.with_suffix('.pdf')}")
    
    # Print the validation AUROC grid for reference
    print("\n" + "="*60)
    print("VALIDATION AUROC GRID")
    print("="*60)
    print("\nMasking Ratios: M25, M50, M75, M95")
    print("Noise Levels: N25, N50, N75, N100\n")
    
    print("         M25    M50    M75    M95")
    print("    " + "─"*35)
    for i, n in enumerate(noise_levels):
        row_str = f"N{n:3}: "
        for j in range(len(masking_ratios)):
            row_str += f"  {val_auroc_avg[i,j]:.3f}"
        print(row_str)
    
    print("\n" + "="*60)
    print(f"Best Configuration: M{masking_ratios[best_idx[1]]}_N{noise_levels[best_idx[0]]} = {best_val:.3f}")
    print("="*60)

if __name__ == "__main__":
    create_clean_contour_plot()