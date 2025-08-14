#!/usr/bin/env python3
"""
Analyze FlowMDAE ablation results and generate summary insights.
"""

import numpy as np
import json
from pathlib import Path

def main():
    # Load data
    data_dir = Path('/home/t-jiachentu/repos/benchmarking/misc/data/ablation/contour_analysis/data')
    
    # Load grids
    data = np.load(data_dir / 'processed_grids.npz', allow_pickle=True)
    grids = data['grids'].item()
    masking_ratios = data['masking_ratios']
    noise_levels = data['noise_levels']
    modalities = data['modalities']
    
    print("\n" + "="*80)
    print("FLOWMDAE ABLATION ANALYSIS: MASKING RATIO & NOISE CORRUPTION LEVEL")
    print("="*80)
    
    # Analyze Test AUROC for each modality
    print("\n" + "-"*80)
    print("TEST AUROC ANALYSIS BY MODALITY")
    print("-"*80)
    
    best_configs = {}
    
    for modality in modalities:
        test_auroc = grids[modality]['Test_AUROC']
        
        print(f"\n{modality}:")
        print(f"  Range: {np.nanmin(test_auroc):.3f} - {np.nanmax(test_auroc):.3f}")
        print(f"  Mean: {np.nanmean(test_auroc):.3f} ± {np.nanstd(test_auroc):.3f}")
        
        # Find best configuration
        best_idx = np.nanargmax(test_auroc)
        best_i, best_j = np.unravel_index(best_idx, test_auroc.shape)
        best_value = test_auroc[best_i, best_j]
        best_configs[modality] = (masking_ratios[best_j], noise_levels[best_i], best_value)
        
        print(f"  Best: M{masking_ratios[best_j]}_N{noise_levels[best_i]} = {best_value:.3f}")
        
        # Find worst configuration
        worst_idx = np.nanargmin(test_auroc)
        worst_i, worst_j = np.unravel_index(worst_idx, test_auroc.shape)
        worst_value = test_auroc[worst_i, worst_j]
        print(f"  Worst: M{masking_ratios[worst_j]}_N{noise_levels[worst_i]} = {worst_value:.3f}")
        
        # Analyze by masking ratio
        print(f"\n  By Masking Ratio (mean AUROC):")
        for j, m in enumerate(masking_ratios):
            mean_val = np.nanmean(test_auroc[:, j])
            print(f"    M{m}: {mean_val:.3f}")
        
        # Analyze by noise level
        print(f"\n  By Noise Level (mean AUROC):")
        for i, n in enumerate(noise_levels):
            mean_val = np.nanmean(test_auroc[i, :])
            print(f"    N{n}: {mean_val:.3f}")
    
    # Overall analysis across all modalities
    print("\n" + "-"*80)
    print("CROSS-MODALITY ANALYSIS")
    print("-"*80)
    
    # Combine all test AUROC values
    all_aurocs = []
    for modality in modalities:
        all_aurocs.append(grids[modality]['Test_AUROC'].flatten())
    all_aurocs = np.concatenate(all_aurocs)
    
    print(f"\nOverall Test AUROC Statistics:")
    print(f"  Mean: {np.nanmean(all_aurocs):.3f}")
    print(f"  Std: {np.nanstd(all_aurocs):.3f}")
    print(f"  Median: {np.nanmedian(all_aurocs):.3f}")
    print(f"  Range: {np.nanmin(all_aurocs):.3f} - {np.nanmax(all_aurocs):.3f}")
    
    # Best configuration overall
    print(f"\nBest Configurations by Modality:")
    for modality, (m, n, val) in best_configs.items():
        print(f"  {modality}: M{m}_N{n} = {val:.3f}")
    
    # Average performance by masking ratio across all modalities
    print(f"\nAverage Performance by Masking Ratio (across all modalities):")
    for j, m in enumerate(masking_ratios):
        values = []
        for modality in modalities:
            values.extend(grids[modality]['Test_AUROC'][:, j])
        mean_val = np.nanmean(values)
        std_val = np.nanstd(values)
        print(f"  M{m}: {mean_val:.3f} ± {std_val:.3f}")
    
    # Average performance by noise level across all modalities
    print(f"\nAverage Performance by Noise Level (across all modalities):")
    for i, n in enumerate(noise_levels):
        values = []
        for modality in modalities:
            values.extend(grids[modality]['Test_AUROC'][i, :])
        mean_val = np.nanmean(values)
        std_val = np.nanstd(values)
        print(f"  N{n}: {mean_val:.3f} ± {std_val:.3f}")
    
    # Key insights
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    
    insights = []
    
    # Best overall configuration
    best_overall = max(best_configs.items(), key=lambda x: x[1][2])
    insights.append(f"• Best overall: {best_overall[0]} with M{best_overall[1][0]}_N{best_overall[1][1]} (AUROC={best_overall[1][2]:.3f})")
    
    # Modality performance ranking
    modality_means = [(mod, np.nanmean(grids[mod]['Test_AUROC'])) for mod in modalities]
    modality_means.sort(key=lambda x: x[1], reverse=True)
    insights.append(f"• Modality ranking: {' > '.join([f'{m[0]} ({m[1]:.3f})' for m in modality_means])}")
    
    # Optimal masking ratio
    masking_means = []
    for j, m in enumerate(masking_ratios):
        values = []
        for modality in modalities:
            values.extend(grids[modality]['Test_AUROC'][:, j])
        masking_means.append((m, np.nanmean(values)))
    best_masking = max(masking_means, key=lambda x: x[1])
    insights.append(f"• Optimal masking ratio: M{best_masking[0]} (mean AUROC={best_masking[1]:.3f})")
    
    # Optimal noise level
    noise_means = []
    for i, n in enumerate(noise_levels):
        values = []
        for modality in modalities:
            values.extend(grids[modality]['Test_AUROC'][i, :])
        noise_means.append((n, np.nanmean(values)))
    best_noise = max(noise_means, key=lambda x: x[1])
    insights.append(f"• Optimal noise level: N{best_noise[0]} (mean AUROC={best_noise[1]:.3f})")
    
    # Performance variation
    if masking_means[-1][1] < masking_means[0][1]:
        insights.append("• Performance generally decreases with higher masking ratios")
    
    # Modality-specific patterns
    t1ce_mean = np.nanmean(grids['T1CE']['Test_AUROC'])
    flair_mean = np.nanmean(grids['FLAIR']['Test_AUROC'])
    if t1ce_mean > flair_mean * 1.5:
        insights.append(f"• T1CE shows significantly better performance ({t1ce_mean:.3f}) than FLAIR ({flair_mean:.3f})")
    
    for insight in insights:
        print(insight)
    
    # Save summary to file
    summary_path = data_dir / 'ablation_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("FLOWMDAE ABLATION ANALYSIS SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write("Best Configurations by Modality:\n")
        for modality, (m, n, val) in best_configs.items():
            f.write(f"  {modality}: M{m}_N{n} = {val:.3f}\n")
        f.write("\nKey Insights:\n")
        for insight in insights:
            f.write(f"{insight}\n")
    
    print(f"\nSummary saved to: {summary_path}")

if __name__ == "__main__":
    main()