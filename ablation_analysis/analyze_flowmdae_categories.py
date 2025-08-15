#!/usr/bin/env python3
"""
FlowMDAE Categories Analysis
Compares Fixed Masking vs Random Sampling FlowMDAE strategies
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class FlowMDAECategoryAnalyzer:
    """Analyze and compare Fixed vs Random FlowMDAE categories"""
    
    def __init__(self):
        self.base_path = Path(__file__).parent
        self.output_dir = self.base_path / 'visualizations' / 'flowmdae_categories'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Grid parameters
        self.masking_ratios = [25, 50, 75, 95]
        self.noise_levels = [25, 50, 75, 100]
        
    def load_data(self):
        """Load both FlowMDAE categories"""
        # Random sampling (from flowmdae_raw.csv)
        random_df = pd.read_csv(self.base_path / 'raw_data_extracted' / 'flowmdae_raw.csv')
        random_df['category'] = 'Random Sampling'
        
        # Fixed masking (from other_ablations_raw.csv)
        other_df = pd.read_csv(self.base_path / 'raw_data_extracted' / 'other_ablations_raw.csv')
        fixed_mask = other_df['notes'].str.contains('FIXED.*Masking.*FLOWMDAE', case=False, na=False)
        fixed_df = other_df[fixed_mask].copy()
        
        # Extract parameters for fixed masking
        def extract_params(notes):
            match = re.search(r'RESENC_FLOWMDAE_M(\d+)_N(\d+)', notes)
            if match:
                return f'M{match.group(1)}_N{match.group(2)}', int(match.group(1)), int(match.group(2))
            return None, None, None
        
        params_data = fixed_df['notes'].apply(lambda x: extract_params(x))
        fixed_df['param_combo'] = params_data.apply(lambda x: x[0])
        fixed_df['masking_ratio'] = params_data.apply(lambda x: x[1])
        fixed_df['noise_level'] = params_data.apply(lambda x: x[2])
        fixed_df['category'] = 'Fixed Masking'
        
        return random_df, fixed_df
    
    def create_comparison_heatmaps(self, random_df, fixed_df):
        """Create side-by-side heatmaps for both categories"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Prepare pivot tables
        random_pivot = random_df.pivot_table(
            values='test_auroc',
            index='masking_ratio',
            columns='noise_level',
            aggfunc='mean'
        )
        
        fixed_pivot = fixed_df.pivot_table(
            values='test_auroc',
            index='masking_ratio',
            columns='noise_level',
            aggfunc='mean'
        )
        
        # Calculate difference
        diff_pivot = random_pivot - fixed_pivot
        
        # Plot Random Sampling heatmap
        sns.heatmap(random_pivot, annot=True, fmt='.3f', cmap='YlOrRd',
                   vmin=0.4, vmax=0.7, ax=axes[0], cbar_kws={'label': 'AUROC'})
        axes[0].set_title('Random Sampling [1%, M%]', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Noise Level (N)', fontsize=12)
        axes[0].set_ylabel('Max Masking Ratio (M%)', fontsize=12)
        
        # Plot Fixed Masking heatmap
        sns.heatmap(fixed_pivot, annot=True, fmt='.3f', cmap='YlOrRd',
                   vmin=0.4, vmax=0.7, ax=axes[1], cbar_kws={'label': 'AUROC'})
        axes[1].set_title('Fixed Masking at M%', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Noise Level (N)', fontsize=12)
        axes[1].set_ylabel('Masking Ratio (M%)', fontsize=12)
        
        # Plot Difference heatmap
        sns.heatmap(diff_pivot, annot=True, fmt='.3f', cmap='RdBu_r',
                   center=0, ax=axes[2], cbar_kws={'label': 'AUROC Difference'})
        axes[2].set_title('Difference (Random - Fixed)', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Noise Level (N)', fontsize=12)
        axes[2].set_ylabel('Masking Ratio (M%)', fontsize=12)
        
        plt.suptitle('FlowMDAE: Fixed vs Random Masking Comparison', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'flowmdae_category_heatmaps.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return random_pivot, fixed_pivot, diff_pivot
    
    def create_performance_comparison_plot(self, random_df, fixed_df):
        """Create bar plot comparing performance across parameter combinations"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Aggregate by parameter combination
        random_perf = random_df.groupby('param_combo')['test_auroc'].agg(['mean', 'std'])
        fixed_perf = fixed_df.groupby('param_combo')['test_auroc'].agg(['mean', 'std'])
        
        # Ensure same order
        params = sorted(set(random_perf.index) | set(fixed_perf.index))
        
        x = np.arange(len(params))
        width = 0.35
        
        # Plot bars
        random_means = [random_perf.loc[p, 'mean'] if p in random_perf.index else 0 for p in params]
        random_stds = [random_perf.loc[p, 'std'] if p in random_perf.index else 0 for p in params]
        
        fixed_means = [fixed_perf.loc[p, 'mean'] if p in fixed_perf.index else 0 for p in params]
        fixed_stds = [fixed_perf.loc[p, 'std'] if p in fixed_perf.index else 0 for p in params]
        
        bars1 = ax.bar(x - width/2, random_means, width, yerr=random_stds,
                      label='Random Sampling [1%, M%]', color='#FF9999', capsize=5)
        bars2 = ax.bar(x + width/2, fixed_means, width, yerr=fixed_stds,
                      label='Fixed Masking at M%', color='#66B2FF', capsize=5)
        
        # Customization
        ax.set_xlabel('Parameter Combination', fontsize=12, fontweight='bold')
        ax.set_ylabel('AUROC', fontsize=12, fontweight='bold')
        ax.set_title('FlowMDAE Performance: Fixed vs Random Masking', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(params, rotation=45, ha='right')
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1])
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'flowmdae_category_comparison_bars.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def perform_statistical_tests(self, random_df, fixed_df):
        """Perform statistical comparisons between categories"""
        results = {}
        
        # Overall comparison
        random_aurocs = random_df['test_auroc'].dropna()
        fixed_aurocs = fixed_df['test_auroc'].dropna()
        
        # Independent t-test (since different runs)
        t_stat, p_value = stats.ttest_ind(fixed_aurocs, random_aurocs)
        
        results['overall'] = {
            'fixed_mean': fixed_aurocs.mean(),
            'fixed_std': fixed_aurocs.std(),
            'random_mean': random_aurocs.mean(),
            'random_std': random_aurocs.std(),
            'difference': fixed_aurocs.mean() - random_aurocs.mean(),
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
        
        # Per-parameter comparison
        results['per_parameter'] = {}
        common_params = set(random_df['param_combo'].dropna()) & set(fixed_df['param_combo'].dropna())
        
        for param in sorted(common_params):
            random_param = random_df[random_df['param_combo'] == param]['test_auroc'].dropna()
            fixed_param = fixed_df[fixed_df['param_combo'] == param]['test_auroc'].dropna()
            
            if len(random_param) > 0 and len(fixed_param) > 0:
                t_stat, p_value = stats.ttest_ind(fixed_param, random_param)
                
                results['per_parameter'][param] = {
                    'fixed_mean': fixed_param.mean(),
                    'random_mean': random_param.mean(),
                    'difference': fixed_param.mean() - random_param.mean(),
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
        
        return results
    
    def create_summary_table(self, stats_results, random_pivot, fixed_pivot):
        """Create LaTeX and CSV summary tables"""
        tables_dir = self.base_path / 'tables'
        tables_dir.mkdir(exist_ok=True)
        
        # Create comparison DataFrame
        comparison_data = []
        
        for m in self.masking_ratios:
            for n in self.noise_levels:
                param = f'M{m}_N{n}'
                row = {
                    'Parameter': param,
                    'M (Masking)': m,
                    'N (Noise Level)': n
                }
                
                # Get performance values
                if m in fixed_pivot.index and n in fixed_pivot.columns:
                    row['Fixed_AUROC'] = fixed_pivot.loc[m, n]
                else:
                    row['Fixed_AUROC'] = np.nan
                    
                if m in random_pivot.index and n in random_pivot.columns:
                    row['Random_AUROC'] = random_pivot.loc[m, n]
                else:
                    row['Random_AUROC'] = np.nan
                
                # Calculate difference
                if not np.isnan(row['Fixed_AUROC']) and not np.isnan(row['Random_AUROC']):
                    row['Difference'] = row['Fixed_AUROC'] - row['Random_AUROC']
                    row['Winner'] = 'Fixed' if row['Difference'] > 0 else 'Random'
                else:
                    row['Difference'] = np.nan
                    row['Winner'] = '-'
                
                comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Format for display
        display_df = comparison_df.copy()
        display_df['Fixed_AUROC'] = display_df['Fixed_AUROC'].apply(lambda x: f'{x:.4f}' if not np.isnan(x) else '-')
        display_df['Random_AUROC'] = display_df['Random_AUROC'].apply(lambda x: f'{x:.4f}' if not np.isnan(x) else '-')
        display_df['Difference'] = display_df['Difference'].apply(lambda x: f'{x:+.4f}' if not np.isnan(x) else '-')
        
        # Save CSV
        display_df.to_csv(tables_dir / 'flowmdae_category_comparison.csv', index=False)
        
        # Create LaTeX table
        latex_table = display_df.to_latex(
            index=False,
            caption='FlowMDAE: Fixed vs Random Masking Performance Comparison',
            label='tab:flowmdae_comparison',
            column_format='lcccccc',
            escape=False
        )
        
        with open(tables_dir / 'flowmdae_category_comparison.tex', 'w') as f:
            f.write(latex_table)
        
        # Create summary statistics table
        overall = stats_results['overall']
        summary_data = {
            'Metric': ['Mean AUROC', 'Std Dev', 'Difference', 'T-statistic', 'P-value', 'Significant'],
            'Fixed Masking': [
                f"{overall['fixed_mean']:.4f}",
                f"{overall['fixed_std']:.4f}",
                '-',
                '-',
                '-',
                '-'
            ],
            'Random Sampling': [
                f"{overall['random_mean']:.4f}",
                f"{overall['random_std']:.4f}",
                f"{overall['difference']:+.4f}",
                f"{overall['t_statistic']:.3f}",
                f"{overall['p_value']:.4f}",
                'Yes' if overall['significant'] else 'No'
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(tables_dir / 'flowmdae_statistical_summary.csv', index=False)
        
        return comparison_df, summary_df
    
    def run_analysis(self):
        """Run complete analysis pipeline"""
        print("FlowMDAE Categories Analysis")
        print("="*60)
        
        # Load data
        print("\nLoading data...")
        random_df, fixed_df = self.load_data()
        print(f"  Random Sampling: {len(random_df)} runs")
        print(f"  Fixed Masking: {len(fixed_df)} runs")
        
        # Create visualizations
        print("\nCreating visualizations...")
        random_pivot, fixed_pivot, diff_pivot = self.create_comparison_heatmaps(random_df, fixed_df)
        self.create_performance_comparison_plot(random_df, fixed_df)
        
        # Statistical tests
        print("\nPerforming statistical tests...")
        stats_results = self.perform_statistical_tests(random_df, fixed_df)
        
        # Create tables
        print("\nGenerating tables...")
        comparison_df, summary_df = self.create_summary_table(stats_results, random_pivot, fixed_pivot)
        
        # Print summary
        print("\n" + "="*60)
        print("ANALYSIS SUMMARY")
        print("="*60)
        
        overall = stats_results['overall']
        print(f"\nOverall Performance:")
        print(f"  Fixed Masking:    {overall['fixed_mean']:.4f} ± {overall['fixed_std']:.4f}")
        print(f"  Random Sampling:  {overall['random_mean']:.4f} ± {overall['random_std']:.4f}")
        print(f"  Difference:       {overall['difference']:+.4f}")
        print(f"  P-value:          {overall['p_value']:.4f}")
        print(f"  Significant:      {'Yes' if overall['significant'] else 'No'}")
        
        # Find best configuration for each category
        best_fixed = fixed_df.groupby('param_combo')['test_auroc'].mean().idxmax()
        best_fixed_score = fixed_df.groupby('param_combo')['test_auroc'].mean().max()
        
        best_random = random_df.groupby('param_combo')['test_auroc'].mean().idxmax()
        best_random_score = random_df.groupby('param_combo')['test_auroc'].mean().max()
        
        print(f"\nBest Configurations:")
        print(f"  Fixed Masking:    {best_fixed} (AUROC: {best_fixed_score:.4f})")
        print(f"  Random Sampling:  {best_random} (AUROC: {best_random_score:.4f})")
        
        # Count where each strategy wins
        wins = {'Fixed': 0, 'Random': 0, 'Tie': 0}
        for param, param_stats in stats_results['per_parameter'].items():
            diff = param_stats['difference']
            if abs(diff) < 0.001:
                wins['Tie'] += 1
            elif diff > 0:
                wins['Fixed'] += 1
            else:
                wins['Random'] += 1
        
        print(f"\nHead-to-Head Wins (out of {len(stats_results['per_parameter'])} comparisons):")
        print(f"  Fixed Masking:    {wins['Fixed']}")
        print(f"  Random Sampling:  {wins['Random']}")
        print(f"  Ties:            {wins['Tie']}")
        
        print(f"\nOutputs saved to:")
        print(f"  Visualizations: {self.output_dir}")
        print(f"  Tables: {self.base_path / 'tables'}")
        
        return stats_results

def main():
    analyzer = FlowMDAECategoryAnalyzer()
    analyzer.run_analysis()

if __name__ == "__main__":
    main()