#!/usr/bin/env python3
"""
Pair Continue Pretrained and Vanilla Runs for Analysis

This script pairs CONT_PRETRAINED runs with their vanilla counterparts
and calculates performance improvements.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


class ContPretrainedAnalyzer:
    """Analyze continue pretrained vs vanilla performance."""
    
    def __init__(self):
        """Initialize the analyzer."""
        self.base_path = Path(__file__).parent.parent
        self.raw_data_path = self.base_path / 'raw_data'
        self.processed_path = self.base_path / 'processed_data'
        self.viz_path = self.base_path / 'visualizations'
        self.tables_path = self.base_path / 'tables'
        
        # Create output directories
        self.processed_path.mkdir(exist_ok=True)
        self.viz_path.mkdir(exist_ok=True)
        self.tables_path.mkdir(exist_ok=True)
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load continue pretrained and vanilla runs."""
        # Load categorized data
        with open(self.raw_data_path / 'categorized_runs.json', 'r') as f:
            categorized = json.load(f)
        
        # Convert to DataFrames
        cont_df = pd.json_normalize(categorized['cont_pretrained'])
        vanilla_df = pd.json_normalize(categorized['vanilla'])
        
        return cont_df, vanilla_df
    
    def extract_method_from_notes(self, notes: str, is_cont_pretrained: bool) -> str:
        """Extract method name from Notes field."""
        if is_cont_pretrained:
            if 'CONT_PRETRAINED_RESENC_' in notes:
                return notes.split('CONT_PRETRAINED_RESENC_')[1].split()[0]
        else:
            if 'RESENC_' in notes:
                method = notes.split('RESENC_')[1].split()[0]
                # Handle special cases - MDAE is the vanilla MAE
                if method == 'MDAE':
                    return 'MAE'  # RESENC_MDAE is vanilla MAE
                elif method == 'PRETRAINED':
                    return 'MAE'  # resenc_pretrained is also vanilla MAE
                return method
            elif 'resenc_pretrained' in notes.lower():
                return 'MAE'
        return 'UNKNOWN'
    
    def pair_runs(self, cont_df: pd.DataFrame, vanilla_df: pd.DataFrame) -> pd.DataFrame:
        """
        Pair continue pretrained runs with vanilla counterparts.
        
        Matching criteria:
        - Same benchmark
        - Same modality
        - Same method (MAE, VOCO, etc.)
        """
        pairs = []
        
        # Extract methods for all runs
        cont_df['method'] = cont_df['Notes'].apply(
            lambda x: self.extract_method_from_notes(x, True)
        )
        vanilla_df['method'] = vanilla_df['Notes'].apply(
            lambda x: self.extract_method_from_notes(x, False)
        )
        
        # Group continue pretrained runs
        for _, cont_run in cont_df.iterrows():
            benchmark = cont_run['benchmark']
            modality = cont_run['modality']
            method = cont_run['method']
            
            # Find matching vanilla runs
            matches = vanilla_df[
                (vanilla_df['benchmark'] == benchmark) &
                (vanilla_df['modality'] == modality) &
                (vanilla_df['method'] == method)
            ]
            
            if len(matches) > 0:
                # Take the best performing vanilla run for fair comparison
                best_vanilla = matches.loc[
                    matches['metrics.Test_AUROC'].idxmax()
                ]
                
                pairs.append({
                    'benchmark': benchmark,
                    'modality': modality,
                    'method': method,
                    'cont_pretrained_id': cont_run['id'],
                    'cont_pretrained_name': cont_run['name'],
                    'cont_pretrained_auroc': cont_run.get('metrics.Test_AUROC', np.nan),
                    'cont_pretrained_ap': cont_run.get('metrics.Test_AP', np.nan),
                    'cont_pretrained_f1': cont_run.get('metrics.Test_F1', np.nan),
                    'vanilla_id': best_vanilla['id'],
                    'vanilla_name': best_vanilla['name'],
                    'vanilla_auroc': best_vanilla.get('metrics.Test_AUROC', np.nan),
                    'vanilla_ap': best_vanilla.get('metrics.Test_AP', np.nan),
                    'vanilla_f1': best_vanilla.get('metrics.Test_F1', np.nan),
                })
        
        pairs_df = pd.DataFrame(pairs)
        
        # Calculate improvements
        pairs_df['auroc_improvement'] = pairs_df['cont_pretrained_auroc'] - pairs_df['vanilla_auroc']
        pairs_df['ap_improvement'] = pairs_df['cont_pretrained_ap'] - pairs_df['vanilla_ap']
        pairs_df['f1_improvement'] = pairs_df['cont_pretrained_f1'] - pairs_df['vanilla_f1']
        
        # Calculate percentage improvements
        pairs_df['auroc_improvement_pct'] = (pairs_df['auroc_improvement'] / pairs_df['vanilla_auroc']) * 100
        pairs_df['ap_improvement_pct'] = (pairs_df['ap_improvement'] / pairs_df['vanilla_ap']) * 100
        pairs_df['f1_improvement_pct'] = (pairs_df['f1_improvement'] / pairs_df['vanilla_f1']) * 100
        
        return pairs_df
    
    def calculate_statistics(self, pairs_df: pd.DataFrame) -> Dict:
        """Calculate statistical comparisons."""
        stats_results = {}
        
        # Overall statistics
        stats_results['overall'] = {
            'n_pairs': len(pairs_df),
            'mean_auroc_improvement': pairs_df['auroc_improvement'].mean(),
            'std_auroc_improvement': pairs_df['auroc_improvement'].std(),
            'median_auroc_improvement': pairs_df['auroc_improvement'].median(),
            'positive_improvement_ratio': (pairs_df['auroc_improvement'] > 0).mean(),
        }
        
        # Paired t-test
        cont_aurocs = pairs_df['cont_pretrained_auroc'].dropna()
        vanilla_aurocs = pairs_df['vanilla_auroc'].dropna()
        if len(cont_aurocs) > 0 and len(vanilla_aurocs) > 0:
            t_stat, p_value = stats.ttest_rel(cont_aurocs, vanilla_aurocs)
            stats_results['overall']['t_statistic'] = t_stat
            stats_results['overall']['p_value'] = p_value
        
        # Per-method statistics
        stats_results['by_method'] = {}
        for method in pairs_df['method'].unique():
            method_df = pairs_df[pairs_df['method'] == method]
            stats_results['by_method'][method] = {
                'n_pairs': len(method_df),
                'mean_auroc_improvement': method_df['auroc_improvement'].mean(),
                'std_auroc_improvement': method_df['auroc_improvement'].std(),
                'median_auroc_improvement': method_df['auroc_improvement'].median(),
                'positive_improvement_ratio': (method_df['auroc_improvement'] > 0).mean(),
            }
            
            # Paired t-test for this method
            if len(method_df) > 1:
                cont = method_df['cont_pretrained_auroc'].dropna()
                vanilla = method_df['vanilla_auroc'].dropna()
                if len(cont) > 0 and len(vanilla) > 0:
                    t_stat, p_value = stats.ttest_rel(cont, vanilla)
                    stats_results['by_method'][method]['t_statistic'] = t_stat
                    stats_results['by_method'][method]['p_value'] = p_value
        
        # Per-benchmark statistics
        stats_results['by_benchmark'] = {}
        for benchmark in pairs_df['benchmark'].unique():
            bench_df = pairs_df[pairs_df['benchmark'] == benchmark]
            stats_results['by_benchmark'][benchmark] = {
                'n_pairs': len(bench_df),
                'mean_auroc_improvement': bench_df['auroc_improvement'].mean(),
                'std_auroc_improvement': bench_df['auroc_improvement'].std(),
                'positive_improvement_ratio': (bench_df['auroc_improvement'] > 0).mean(),
            }
        
        return stats_results
    
    def create_visualizations(self, pairs_df: pd.DataFrame, stats_results: Dict):
        """Create visualizations for the analysis."""
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.dpi'] = 100
        
        # 1. Method-wise improvement bar plot
        fig, ax = plt.subplots(figsize=(10, 6))
        method_improvements = pairs_df.groupby('method')['auroc_improvement'].agg(['mean', 'std'])
        method_improvements = method_improvements.sort_values('mean', ascending=False)
        
        bars = ax.bar(method_improvements.index, method_improvements['mean'], 
                      yerr=method_improvements['std'], capsize=5)
        
        # Color bars based on positive/negative
        colors = ['green' if x > 0 else 'red' for x in method_improvements['mean']]
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Method')
        ax.set_ylabel('AUROC Improvement')
        ax.set_title('Continue Pretraining Effect by Method\n(CONT_PRETRAINED - Vanilla)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.viz_path / 'method_improvement_bars.png')
        plt.close()
        
        # 2. Benchmark-wise improvement heatmap
        pivot_data = pairs_df.pivot_table(
            values='auroc_improvement',
            index='benchmark',
            columns='method',
            aggfunc='mean'
        )
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlGn', 
                   center=0, cbar_kws={'label': 'AUROC Improvement'})
        ax.set_title('Continue Pretraining Effect Heatmap\n(AUROC Improvement by Benchmark and Method)')
        plt.tight_layout()
        plt.savefig(self.viz_path / 'benchmark_method_heatmap.png')
        plt.close()
        
        # 3. Distribution of improvements
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        metrics = ['auroc_improvement', 'ap_improvement', 'f1_improvement']
        titles = ['AUROC Improvement', 'AP Improvement', 'F1 Improvement']
        
        for ax, metric, title in zip(axes, metrics, titles):
            data = pairs_df[metric].dropna()
            ax.hist(data, bins=20, edgecolor='black', alpha=0.7)
            ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
            ax.axvline(x=data.mean(), color='blue', linestyle='-', linewidth=2, 
                      label=f'Mean: {data.mean():.3f}')
            ax.set_xlabel('Improvement')
            ax.set_ylabel('Count')
            ax.set_title(title)
            ax.legend()
        
        plt.suptitle('Distribution of Performance Improvements\n(Continue Pretrained - Vanilla)')
        plt.tight_layout()
        plt.savefig(self.viz_path / 'improvement_distributions.png')
        plt.close()
        
        # 4. Scatter plot: Vanilla vs Continue Pretrained
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Color by method
        methods = pairs_df['method'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
        
        for method, color in zip(methods, colors):
            method_data = pairs_df[pairs_df['method'] == method]
            ax.scatter(method_data['vanilla_auroc'], 
                      method_data['cont_pretrained_auroc'],
                      label=method, alpha=0.6, s=50, color=color)
        
        # Add diagonal line
        min_val = min(pairs_df['vanilla_auroc'].min(), pairs_df['cont_pretrained_auroc'].min())
        max_val = max(pairs_df['vanilla_auroc'].max(), pairs_df['cont_pretrained_auroc'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3)
        
        ax.set_xlabel('Vanilla AUROC')
        ax.set_ylabel('Continue Pretrained AUROC')
        ax.set_title('Continue Pretrained vs Vanilla Performance')
        ax.legend(loc='lower right')
        ax.set_aspect('equal')
        plt.tight_layout()
        plt.savefig(self.viz_path / 'vanilla_vs_cont_pretrained_scatter.png')
        plt.close()
        
        print(f"Visualizations saved to {self.viz_path}")
    
    def create_tables(self, pairs_df: pd.DataFrame, stats_results: Dict):
        """Create summary tables."""
        # Method summary table
        method_summary = []
        for method, stats in stats_results['by_method'].items():
            method_summary.append({
                'Method': method,
                'N Pairs': stats['n_pairs'],
                'Mean Improvement': f"{stats['mean_auroc_improvement']:.4f}",
                'Std Dev': f"{stats['std_auroc_improvement']:.4f}",
                'Positive Ratio': f"{stats['positive_improvement_ratio']:.2%}",
                'P-value': f"{stats.get('p_value', np.nan):.4f}" if 'p_value' in stats else 'N/A'
            })
        
        method_df = pd.DataFrame(method_summary)
        method_df = method_df.sort_values('Mean Improvement', ascending=False)
        
        # Save as CSV
        method_df.to_csv(self.tables_path / 'method_summary.csv', index=False)
        
        # Save as LaTeX
        with open(self.tables_path / 'method_summary.tex', 'w') as f:
            f.write(method_df.to_latex(index=False, escape=False))
        
        # Top improvements table
        top_improvements = pairs_df.nlargest(10, 'auroc_improvement')[
            ['benchmark', 'method', 'modality', 'vanilla_auroc', 
             'cont_pretrained_auroc', 'auroc_improvement']
        ].copy()
        top_improvements.columns = ['Benchmark', 'Method', 'Modality', 
                                   'Vanilla AUROC', 'Cont. Pretrained AUROC', 'Improvement']
        
        # Format numbers
        for col in ['Vanilla AUROC', 'Cont. Pretrained AUROC', 'Improvement']:
            top_improvements[col] = top_improvements[col].apply(lambda x: f"{x:.4f}")
        
        top_improvements.to_csv(self.tables_path / 'top_improvements.csv', index=False)
        
        # Overall summary
        overall = stats_results['overall']
        summary_text = f"""
Continue Pretraining Analysis Summary
=====================================

Total Paired Comparisons: {overall['n_pairs']}

AUROC Improvements:
- Mean: {overall['mean_auroc_improvement']:.4f}
- Std Dev: {overall['std_auroc_improvement']:.4f}
- Median: {overall['median_auroc_improvement']:.4f}
- Positive Improvement Ratio: {overall['positive_improvement_ratio']:.2%}

Statistical Test (Paired t-test):
- T-statistic: {overall.get('t_statistic', 'N/A')}
- P-value: {overall.get('p_value', 'N/A')}

Interpretation:
{self._interpret_results(overall)}
"""
        
        with open(self.tables_path / 'summary.txt', 'w') as f:
            f.write(summary_text)
        
        print(f"Tables saved to {self.tables_path}")
    
    def _interpret_results(self, overall_stats: Dict) -> str:
        """Interpret the statistical results."""
        mean_imp = overall_stats['mean_auroc_improvement']
        p_value = overall_stats.get('p_value', 1.0)
        positive_ratio = overall_stats['positive_improvement_ratio']
        
        interpretation = []
        
        if mean_imp > 0:
            interpretation.append(f"Continue pretraining from MDAE shows an average AUROC improvement of {mean_imp:.4f}.")
        else:
            interpretation.append(f"Continue pretraining from MDAE shows an average AUROC decrease of {abs(mean_imp):.4f}.")
        
        if p_value < 0.05:
            interpretation.append("This difference is statistically significant (p < 0.05).")
        else:
            interpretation.append("This difference is not statistically significant (p >= 0.05).")
        
        interpretation.append(f"{positive_ratio:.1%} of continue pretrained models show improvement over vanilla.")
        
        return '\n'.join(interpretation)
    
    def save_processed_data(self, pairs_df: pd.DataFrame, stats_results: Dict):
        """Save processed data for future use."""
        # Save paired data
        pairs_df.to_csv(self.processed_path / 'paired_runs.csv', index=False)
        
        # Save statistics
        with open(self.processed_path / 'statistics.json', 'w') as f:
            json.dump(stats_results, f, indent=2, default=str)
        
        print(f"Processed data saved to {self.processed_path}")


def main():
    """Main execution function."""
    print("Continue Pretraining Analysis")
    print("="*60)
    
    analyzer = ContPretrainedAnalyzer()
    
    # Load data
    print("\nLoading data...")
    cont_df, vanilla_df = analyzer.load_data()
    print(f"Loaded {len(cont_df)} continue pretrained runs")
    print(f"Loaded {len(vanilla_df)} vanilla runs")
    
    # Pair runs
    print("\nPairing runs...")
    pairs_df = analyzer.pair_runs(cont_df, vanilla_df)
    print(f"Created {len(pairs_df)} pairs")
    
    # Calculate statistics
    print("\nCalculating statistics...")
    stats_results = analyzer.calculate_statistics(pairs_df)
    
    # Create visualizations
    print("\nCreating visualizations...")
    analyzer.create_visualizations(pairs_df, stats_results)
    
    # Create tables
    print("\nCreating tables...")
    analyzer.create_tables(pairs_df, stats_results)
    
    # Save processed data
    print("\nSaving processed data...")
    analyzer.save_processed_data(pairs_df, stats_results)
    
    print("\nAnalysis complete!")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    overall = stats_results['overall']
    print(f"Mean AUROC Improvement: {overall['mean_auroc_improvement']:.4f}")
    print(f"Positive Improvement Ratio: {overall['positive_improvement_ratio']:.2%}")
    if 'p_value' in overall:
        print(f"P-value: {overall['p_value']:.4f}")
    
    print("\nTop performing methods:")
    for method, stats in sorted(stats_results['by_method'].items(), 
                               key=lambda x: x[1]['mean_auroc_improvement'], 
                               reverse=True)[:3]:
        print(f"  - {method}: {stats['mean_auroc_improvement']:.4f} mean improvement")


if __name__ == "__main__":
    main()