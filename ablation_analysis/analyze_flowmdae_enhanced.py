#!/usr/bin/env python3
"""
Enhanced FlowMDAE Analysis with BraTS18 Focus and Contour Maps
Comprehensive comparison of Fixed vs Random masking strategies
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats, interpolate
from matplotlib import cm
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class FlowMDAEEnhancedAnalyzer:
    """Enhanced analysis with BraTS18 focus and contour visualizations"""
    
    def __init__(self):
        self.base_path = Path(__file__).parent
        self.raw_data_path = self.base_path / 'raw_data_extracted'
        self.output_dir = self.base_path / 'visualizations' / 'flowmdae_enhanced'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Grid parameters
        self.masking_ratios = [25, 50, 75, 95]
        self.noise_levels = [25, 50, 75, 100]
        
    def load_data(self):
        """Load extracted FlowMDAE category data"""
        fixed_df = pd.read_csv(self.raw_data_path / 'flowmdae_fixed_masking.csv')
        random_df = pd.read_csv(self.raw_data_path / 'flowmdae_random_sampling.csv')
        brats18_df = pd.read_csv(self.raw_data_path / 'flowmdae_brats18_comparison.csv')
        
        return fixed_df, random_df, brats18_df
    
    def create_contour_maps(self, fixed_df, random_df, dataset_filter=None):
        """Create contour map visualizations for performance landscape"""
        
        # Filter by dataset if specified
        if dataset_filter:
            fixed_df = fixed_df[fixed_df['benchmark'] == dataset_filter]
            random_df = random_df[random_df['benchmark'] == dataset_filter]
            title_suffix = f' ({dataset_filter})'
        else:
            title_suffix = ' (All Datasets)'
        
        # Create pivot tables for performance data
        fixed_pivot = fixed_df.pivot_table(
            values='test_auroc',
            index='masking_ratio',
            columns='noise_level',
            aggfunc='mean'
        )
        
        random_pivot = random_df.pivot_table(
            values='test_auroc',
            index='masking_ratio',
            columns='noise_level',
            aggfunc='mean'
        )
        
        # Create meshgrid for interpolation
        M_vals = np.array(self.masking_ratios)
        N_vals = np.array(self.noise_levels)
        M_grid, N_grid = np.meshgrid(
            np.linspace(M_vals.min(), M_vals.max(), 100),
            np.linspace(N_vals.min(), N_vals.max(), 100)
        )
        
        # Interpolate for smoother contours
        fixed_interp = self._interpolate_surface(fixed_pivot, M_grid, N_grid)
        random_interp = self._interpolate_surface(random_pivot, M_grid, N_grid)
        diff_interp = fixed_interp - random_interp
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # Common contour levels
        levels = np.linspace(0.4, 0.75, 15)
        diff_levels = np.linspace(-0.15, 0.15, 15)
        
        # 1. Fixed Masking Contour
        cs1 = axes[0].contourf(M_grid, N_grid, fixed_interp, levels=levels, cmap='YlOrRd', alpha=0.8)
        axes[0].contour(M_grid, N_grid, fixed_interp, levels=levels, colors='black', linewidths=0.5, alpha=0.4)
        # Plot actual data points
        for m in fixed_pivot.index:
            for n in fixed_pivot.columns:
                if not np.isnan(fixed_pivot.loc[m, n]):
                    axes[0].scatter(m, n, s=50, c='blue', marker='o', zorder=5)
        axes[0].set_xlabel('Masking Ratio M (%)', fontsize=11)
        axes[0].set_ylabel('Noise Level N', fontsize=11)
        axes[0].set_title('Fixed Masking at M%', fontsize=12, fontweight='bold')
        fig.colorbar(cs1, ax=axes[0], label='AUROC')
        
        # 2. Random Sampling Contour
        cs2 = axes[1].contourf(M_grid, N_grid, random_interp, levels=levels, cmap='YlOrRd', alpha=0.8)
        axes[1].contour(M_grid, N_grid, random_interp, levels=levels, colors='black', linewidths=0.5, alpha=0.4)
        for m in random_pivot.index:
            for n in random_pivot.columns:
                if not np.isnan(random_pivot.loc[m, n]):
                    axes[1].scatter(m, n, s=50, c='blue', marker='o', zorder=5)
        axes[1].set_xlabel('Masking Ratio M (%)', fontsize=11)
        axes[1].set_ylabel('Noise Level N', fontsize=11)
        axes[1].set_title('Random Sampling [1%, M%]', fontsize=12, fontweight='bold')
        fig.colorbar(cs2, ax=axes[1], label='AUROC')
        
        # 3. Difference Contour (Fixed - Random)
        cs3 = axes[2].contourf(M_grid, N_grid, diff_interp, levels=diff_levels, cmap='RdBu_r', alpha=0.8)
        axes[2].contour(M_grid, N_grid, diff_interp, levels=[0], colors='black', linewidths=2)
        axes[2].set_xlabel('Masking Ratio M (%)', fontsize=11)
        axes[2].set_ylabel('Noise Level N', fontsize=11)
        axes[2].set_title('Difference (Fixed - Random)', fontsize=12, fontweight='bold')
        cbar3 = fig.colorbar(cs3, ax=axes[2], label='AUROC Difference')
        
        # 4. Combined overlay
        axes[3].contour(M_grid, N_grid, fixed_interp, levels=levels, colors='red', linewidths=1.5, alpha=0.7)
        axes[3].contour(M_grid, N_grid, random_interp, levels=levels, colors='blue', linewidths=1.5, alpha=0.7)
        axes[3].set_xlabel('Masking Ratio M (%)', fontsize=11)
        axes[3].set_ylabel('Noise Level N', fontsize=11)
        axes[3].set_title('Overlay (Red=Fixed, Blue=Random)', fontsize=12, fontweight='bold')
        axes[3].legend(['Fixed Masking', 'Random Sampling'], loc='upper right')
        
        # Add grid
        for ax in axes:
            ax.grid(True, alpha=0.3)
            ax.set_xticks(self.masking_ratios)
            ax.set_yticks(self.noise_levels)
        
        plt.suptitle(f'FlowMDAE Performance Landscape{title_suffix}', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Save figure
        filename = 'contour_comparison_brats18.png' if dataset_filter else 'contour_comparison_all.png'
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return fixed_interp, random_interp, diff_interp
    
    def _interpolate_surface(self, pivot_data, M_grid, N_grid):
        """Interpolate surface for smooth contours"""
        # Get valid data points
        points = []
        values = []
        for m in pivot_data.index:
            for n in pivot_data.columns:
                if not np.isnan(pivot_data.loc[m, n]):
                    points.append([m, n])
                    values.append(pivot_data.loc[m, n])
        
        if len(points) > 3:  # Need at least 4 points for 2D interpolation
            points = np.array(points)
            values = np.array(values)
            
            # Use cubic interpolation for smooth surface
            interp_func = interpolate.griddata(
                points, values, (M_grid, N_grid), 
                method='cubic', fill_value=np.nanmean(values)
            )
            return interp_func
        else:
            return np.full_like(M_grid, np.nan)
    
    def analyze_brats18_modalities(self, brats18_df):
        """Analyze performance across modalities for BraTS18"""
        
        # Separate by category
        fixed_df = brats18_df[brats18_df['category'] == 'Fixed Masking']
        random_df = brats18_df[brats18_df['category'] == 'Random Sampling']
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
        
        modalities = ['flair', 't1', 't1ce', 't2']
        
        for idx, modality in enumerate(modalities):
            ax = axes[idx]
            
            # Filter by modality
            fixed_mod = fixed_df[fixed_df['modality'] == modality]
            random_mod = random_df[random_df['modality'] == modality]
            
            # Aggregate by parameter combination
            fixed_perf = fixed_mod.groupby('param_combo')['test_auroc'].agg(['mean', 'std'])
            random_perf = random_mod.groupby('param_combo')['test_auroc'].agg(['mean', 'std'])
            
            # Ensure same order
            params = sorted(set(fixed_perf.index) | set(random_perf.index))
            
            x = np.arange(len(params))
            width = 0.35
            
            # Get values
            fixed_means = [fixed_perf.loc[p, 'mean'] if p in fixed_perf.index else 0 for p in params]
            fixed_stds = [fixed_perf.loc[p, 'std'] if p in fixed_perf.index else 0 for p in params]
            
            random_means = [random_perf.loc[p, 'mean'] if p in random_perf.index else 0 for p in params]
            random_stds = [random_perf.loc[p, 'std'] if p in random_perf.index else 0 for p in params]
            
            # Plot bars
            bars1 = ax.bar(x - width/2, fixed_means, width, yerr=fixed_stds,
                          label='Fixed Masking', color='#66B2FF', capsize=3, alpha=0.8)
            bars2 = ax.bar(x + width/2, random_means, width, yerr=random_stds,
                          label='Random Sampling', color='#FF9999', capsize=3, alpha=0.8)
            
            # Customization
            ax.set_xlabel('Parameter Combination', fontsize=10)
            ax.set_ylabel('AUROC', fontsize=10)
            ax.set_title(f'Modality: {modality.upper()}', fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(params, rotation=45, ha='right', fontsize=8)
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim([0, 1])
            
            # Add mean lines
            if fixed_means:
                ax.axhline(y=np.mean([m for m in fixed_means if m > 0]), 
                          color='blue', linestyle='--', alpha=0.5, linewidth=1)
            if random_means:
                ax.axhline(y=np.mean([m for m in random_means if m > 0]), 
                          color='red', linestyle='--', alpha=0.5, linewidth=1)
        
        plt.suptitle('BraTS18: Fixed vs Random Masking Across Modalities', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'brats18_modality_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create summary statistics table
        modality_stats = []
        for modality in modalities:
            fixed_mod = fixed_df[fixed_df['modality'] == modality]
            random_mod = random_df[random_df['modality'] == modality]
            
            stats_row = {
                'Modality': modality.upper(),
                'Fixed_Mean': fixed_mod['test_auroc'].mean(),
                'Fixed_Std': fixed_mod['test_auroc'].std(),
                'Random_Mean': random_mod['test_auroc'].mean(),
                'Random_Std': random_mod['test_auroc'].std(),
                'Difference': fixed_mod['test_auroc'].mean() - random_mod['test_auroc'].mean()
            }
            modality_stats.append(stats_row)
        
        stats_df = pd.DataFrame(modality_stats)
        return stats_df
    
    def create_performance_trends(self, fixed_df, random_df):
        """Create line plots showing trends across M and N"""
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Trend across Masking Ratio (averaging over N)
        fixed_m_trend = fixed_df.groupby('masking_ratio')['test_auroc'].agg(['mean', 'std'])
        random_m_trend = random_df.groupby('masking_ratio')['test_auroc'].agg(['mean', 'std'])
        
        axes[0].errorbar(fixed_m_trend.index, fixed_m_trend['mean'], yerr=fixed_m_trend['std'],
                        marker='o', markersize=8, linewidth=2, capsize=5, 
                        label='Fixed Masking', color='#66B2FF')
        axes[0].errorbar(random_m_trend.index, random_m_trend['mean'], yerr=random_m_trend['std'],
                        marker='s', markersize=8, linewidth=2, capsize=5,
                        label='Random Sampling', color='#FF9999')
        axes[0].set_xlabel('Masking Ratio M (%)', fontsize=12)
        axes[0].set_ylabel('Mean AUROC', fontsize=12)
        axes[0].set_title('Performance Trend Across Masking Ratio', fontsize=12, fontweight='bold')
        axes[0].legend(loc='best', fontsize=11)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xticks(self.masking_ratios)
        
        # Trend across Noise Level (averaging over M)
        fixed_n_trend = fixed_df.groupby('noise_level')['test_auroc'].agg(['mean', 'std'])
        random_n_trend = random_df.groupby('noise_level')['test_auroc'].agg(['mean', 'std'])
        
        axes[1].errorbar(fixed_n_trend.index, fixed_n_trend['mean'], yerr=fixed_n_trend['std'],
                        marker='o', markersize=8, linewidth=2, capsize=5,
                        label='Fixed Masking', color='#66B2FF')
        axes[1].errorbar(random_n_trend.index, random_n_trend['mean'], yerr=random_n_trend['std'],
                        marker='s', markersize=8, linewidth=2, capsize=5,
                        label='Random Sampling', color='#FF9999')
        axes[1].set_xlabel('Noise Level N', fontsize=12)
        axes[1].set_ylabel('Mean AUROC', fontsize=12)
        axes[1].set_title('Performance Trend Across Noise Level', fontsize=12, fontweight='bold')
        axes[1].legend(loc='best', fontsize=11)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xticks(self.noise_levels)
        
        plt.suptitle('FlowMDAE Performance Trends', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_trends.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_analysis(self):
        """Run complete enhanced analysis"""
        print("Enhanced FlowMDAE Analysis")
        print("="*60)
        
        # Load data
        print("\nLoading data...")
        fixed_df, random_df, brats18_df = self.load_data()
        print(f"  Fixed Masking: {len(fixed_df)} runs")
        print(f"  Random Sampling: {len(random_df)} runs")
        print(f"  BraTS18 Combined: {len(brats18_df)} runs")
        
        # Create contour maps for all data
        print("\nCreating contour maps (all datasets)...")
        self.create_contour_maps(fixed_df, random_df)
        
        # Create contour maps for BraTS18
        print("Creating contour maps (BraTS18 only)...")
        self.create_contour_maps(fixed_df, random_df, dataset_filter='brats18_lgg_vs_hgg')
        
        # Analyze BraTS18 modalities
        print("\nAnalyzing BraTS18 modalities...")
        modality_stats = self.analyze_brats18_modalities(brats18_df)
        
        # Create performance trend plots
        print("Creating performance trend plots...")
        self.create_performance_trends(fixed_df, random_df)
        
        # Save modality statistics
        tables_dir = self.base_path / 'tables'
        tables_dir.mkdir(exist_ok=True)
        modality_stats.to_csv(tables_dir / 'brats18_modality_stats.csv', index=False)
        
        # Print summary
        print("\n" + "="*60)
        print("ANALYSIS SUMMARY")
        print("="*60)
        
        print("\nBraTS18 Modality Performance:")
        print(modality_stats.to_string(index=False))
        
        # Overall BraTS18 comparison
        brats18_fixed = brats18_df[brats18_df['category'] == 'Fixed Masking']
        brats18_random = brats18_df[brats18_df['category'] == 'Random Sampling']
        
        print(f"\nBraTS18 Overall:")
        print(f"  Fixed Masking:   {brats18_fixed['test_auroc'].mean():.4f} ± {brats18_fixed['test_auroc'].std():.4f}")
        print(f"  Random Sampling: {brats18_random['test_auroc'].mean():.4f} ± {brats18_random['test_auroc'].std():.4f}")
        print(f"  Difference:      {brats18_fixed['test_auroc'].mean() - brats18_random['test_auroc'].mean():+.4f}")
        
        # Statistical test
        t_stat, p_value = stats.ttest_ind(
            brats18_fixed['test_auroc'].dropna(),
            brats18_random['test_auroc'].dropna()
        )
        print(f"  P-value:         {p_value:.4f}")
        print(f"  Significant:     {'Yes' if p_value < 0.05 else 'No'}")
        
        print(f"\nOutputs saved to:")
        print(f"  Visualizations: {self.output_dir}")
        print(f"  Tables: {tables_dir}")
        
        return modality_stats

def main():
    analyzer = FlowMDAEEnhancedAnalyzer()
    analyzer.run_analysis()

if __name__ == "__main__":
    main()