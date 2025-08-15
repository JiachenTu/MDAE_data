#!/usr/bin/env python3
"""
Multi-Modality Results Analysis Script
Analyzes multi-modality experiment results and compares with single-modality baselines
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re

class MultiModalityAnalyzer:
    """Analyze multi-modality experiment results."""
    
    # Method patterns from README_METHODS.md
    METHOD_PATTERNS = {
        'MDAE': r'^(resenc_MDAETrainer_RandomMask_Flow_BS48_2000ep|resenc_MDAE_pretrained|resenc_MDAE_scratch)',
        'MDAE (TC)': r'^(resenc_time_conditioned|resenc_multimodal_mm_mdae)',
        'MAE': r'^resenc_pretrained_',
        'SimCLR': r'^resenc_SimCLR',
        'VoCo': r'^resenc_VoCo',
        'MG': r'^resenc_MG',
        'SwinUNETR': r'^resenc_SwinUNETR',
        'VF': r'^resenc_VF',
        'S3D': r'^resenc_S3D',
        'BrainIAC': r'^brainiac_pretrained',
        'MRI-Core': r'^mri_core',
        'BrainMVP': r'^brainmvp',
        'DinoV2': r'^dinov2',
        'ResNet-50 (Scratch)': r'^brainiac_scratch'
    }
    
    def __init__(self, base_path: Path = None):
        """Initialize the analyzer."""
        self.base_path = base_path or Path(__file__).parent
        self.multi_data_path = self.base_path / "raw_data" / "20250814_multi"
        self.single_data_path = self.base_path / "raw_data" / "20250811"
        self.output_path = self.base_path / "processed_data" / "multi_modality_analysis"
        self.output_path.mkdir(parents=True, exist_ok=True)
        
    def identify_method(self, run_name: str) -> str:
        """
        Identify method from run name using pattern matching.
        
        Args:
            run_name: Run name from WandB
            
        Returns:
            Method name
        """
        # Check MDAE patterns first (more specific)
        if re.match(self.METHOD_PATTERNS['MDAE'], run_name):
            return 'MDAE'
        if re.match(self.METHOD_PATTERNS['MDAE (TC)'], run_name):
            return 'MDAE (TC)'
            
        # Check other methods
        for method, pattern in self.METHOD_PATTERNS.items():
            if method not in ['MDAE', 'MDAE (TC)', 'MAE']:  # Skip already checked
                if re.match(pattern, run_name):
                    return method
        
        # MAE pattern is most generic, check last
        if re.match(self.METHOD_PATTERNS['MAE'], run_name):
            return 'MAE'
            
        return 'Unknown'
    
    def load_multi_modality_data(self, benchmark: str) -> pd.DataFrame:
        """
        Load multi-modality data for a benchmark.
        
        Args:
            benchmark: Benchmark name
            
        Returns:
            DataFrame with run results
        """
        csv_path = self.multi_data_path / benchmark / "runs_summary.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            # Add method identification
            df['method'] = df['run_name'].apply(self.identify_method)
            return df
        return pd.DataFrame()
    
    def load_single_modality_data(self, benchmark: str) -> pd.DataFrame:
        """
        Load single-modality data for comparison.
        
        Args:
            benchmark: Benchmark name
            
        Returns:
            DataFrame with combined single-modality results
        """
        # Map benchmark names to single-modality paths
        benchmark_map = {
            'brats18_lgg_vs_hgg': 'brats18_lgg_vs_hgg',
            'brats23_gli_vs_men': 'brats23_gli_vs_men',
            'brats23_gli_vs_met': 'brats23_gli_vs_met',
            'brats23_men_vs_met': 'brats23_men_vs_met',
            'rsna_miccai_mgmt': 'rsna_miccai_mgmt_methylation',
            'ucsf_pdgm_idh': 'ucsf_pdgm_idh_classification',
            'tcga_gbm_dss_1year': 'tcga_gbm_dss_1year',
            'tcga_gbm_pfi_1year': 'tcga_gbm_pfi_1year',
            'upenn_gbm_age_group': 'upenn_gbm_age_group',
            'upenn_gbm_gender': 'upenn_gbm_gender',
            'upenn_gbm_gtr_status': 'upenn_gbm_gtr_status',
            'upenn_gbm_idh1_status': 'upenn_gbm_idh1_status',
            'upenn_gbm_survival_18month': 'upenn_gbm_survival_18month',
            'upenn_gbm_survival_1year': 'upenn_gbm_survival_1year',
            'upenn_gbm_survival_2year': 'upenn_gbm_survival_2year'
        }
        
        single_benchmark = benchmark_map.get(benchmark, benchmark)
        csv_path = self.single_data_path / single_benchmark / "runs_summary.csv"
        
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            # Add method identification
            df['method'] = df['run_name'].apply(self.identify_method)
            return df
        return pd.DataFrame()
    
    def extract_best_metrics(self, df: pd.DataFrame, metric_col: str = 'metric_Test_AUROC') -> Dict:
        """
        Extract best metrics per method.
        
        Args:
            df: DataFrame with run results
            metric_col: Metric column to analyze
            
        Returns:
            Dictionary with best metrics per method
        """
        if df.empty or metric_col not in df.columns:
            return {}
            
        results = {}
        for method in df['method'].unique():
            method_df = df[df['method'] == method]
            valid_metrics = method_df[metric_col].dropna()
            
            if not valid_metrics.empty:
                results[method] = {
                    'best': valid_metrics.max(),
                    'mean': valid_metrics.mean(),
                    'std': valid_metrics.std(),
                    'count': len(valid_metrics)
                }
        
        return results
    
    def compare_multi_vs_single(self, benchmark: str) -> Dict:
        """
        Compare multi-modality vs single-modality performance.
        
        Args:
            benchmark: Benchmark name
            
        Returns:
            Comparison results
        """
        print(f"\nAnalyzing {benchmark}...")
        
        # Load data
        multi_df = self.load_multi_modality_data(benchmark)
        single_df = self.load_single_modality_data(benchmark)
        
        comparison = {
            'benchmark': benchmark,
            'multi_modality': {},
            'single_modality': {},
            'improvement': {}
        }
        
        # Try different metric naming conventions
        # Multi-modality uses metric_Test/AUROC or metric_Test_AUROC
        if 'metric_Test/AUROC' in multi_df.columns:
            auroc_col_multi = 'metric_Test/AUROC'
        elif 'metric_Test_AUROC' in multi_df.columns:
            auroc_col_multi = 'metric_Test_AUROC'
        else:
            auroc_col_multi = 'Test_AUROC'  # Fallback
            
        # Single-modality uses Test_AUROC (no prefix)
        if 'Test_AUROC' in single_df.columns:
            auroc_col_single = 'Test_AUROC'
        elif 'metric_Test_AUROC' in single_df.columns:
            auroc_col_single = 'metric_Test_AUROC'
        else:
            auroc_col_single = 'metric_Test/AUROC'  # Fallback
        
        # Same for AP
        if 'metric_Test/AP' in multi_df.columns:
            ap_col_multi = 'metric_Test/AP'
        elif 'metric_Test_AP' in multi_df.columns:
            ap_col_multi = 'metric_Test_AP'
        else:
            ap_col_multi = 'Test_AP'  # Fallback
            
        if 'Test_AP' in single_df.columns:
            ap_col_single = 'Test_AP'
        elif 'metric_Test_AP' in single_df.columns:
            ap_col_single = 'metric_Test_AP'
        else:
            ap_col_single = 'metric_Test/AP'  # Fallback
        
        # Analyze AUROC
        if not multi_df.empty:
            multi_auroc = self.extract_best_metrics(multi_df, auroc_col_multi)
            comparison['multi_modality']['auroc'] = multi_auroc
            
        if not single_df.empty:
            single_auroc = self.extract_best_metrics(single_df, auroc_col_single)
            comparison['single_modality']['auroc'] = single_auroc
        
        # Analyze AP
        if not multi_df.empty:
            multi_ap = self.extract_best_metrics(multi_df, ap_col_multi)
            comparison['multi_modality']['ap'] = multi_ap
            
        if not single_df.empty:
            single_ap = self.extract_best_metrics(single_df, ap_col_single)
            comparison['single_modality']['ap'] = single_ap
        
        # Calculate improvements for MDAE
        if 'MDAE' in comparison['multi_modality'].get('auroc', {}) and \
           'MDAE' in comparison['single_modality'].get('auroc', {}):
            multi_best = comparison['multi_modality']['auroc']['MDAE']['best']
            single_best = comparison['single_modality']['auroc']['MDAE']['best']
            improvement = ((multi_best - single_best) / single_best) * 100
            comparison['improvement']['mdae_auroc'] = {
                'absolute': multi_best - single_best,
                'relative_percent': improvement
            }
        
        return comparison
    
    def analyze_all_benchmarks(self) -> List[Dict]:
        """
        Analyze all multi-modality benchmarks.
        
        Returns:
            List of comparison results
        """
        results = []
        
        # Get all benchmark directories
        benchmarks = [d.name for d in self.multi_data_path.iterdir() if d.is_dir()]
        
        for benchmark in sorted(benchmarks):
            comparison = self.compare_multi_vs_single(benchmark)
            results.append(comparison)
        
        return results
    
    def create_summary_table(self, results: List[Dict]) -> pd.DataFrame:
        """
        Create summary table of multi vs single modality performance.
        
        Args:
            results: List of comparison results
            
        Returns:
            Summary DataFrame
        """
        summary_data = []
        
        for result in results:
            benchmark = result['benchmark']
            
            # Extract MDAE performance
            multi_mdae = result['multi_modality'].get('auroc', {}).get('MDAE', {})
            single_mdae = result['single_modality'].get('auroc', {}).get('MDAE', {})
            
            if multi_mdae and single_mdae:
                row = {
                    'Benchmark': benchmark,
                    'Single_MDAE_AUROC': single_mdae.get('best', np.nan),
                    'Multi_MDAE_AUROC': multi_mdae.get('best', np.nan),
                    'Improvement': result['improvement'].get('mdae_auroc', {}).get('absolute', 0),
                    'Improvement_%': result['improvement'].get('mdae_auroc', {}).get('relative_percent', 0)
                }
                
                # Add other top methods
                for method in ['MAE', 'SimCLR', 'VoCo', 'BrainMVP']:
                    multi_method = result['multi_modality'].get('auroc', {}).get(method, {})
                    if multi_method:
                        row[f'Multi_{method}_AUROC'] = multi_method.get('best', np.nan)
                
                summary_data.append(row)
        
        df = pd.DataFrame(summary_data)
        if not df.empty and 'Improvement_%' in df.columns:
            return df.sort_values('Improvement_%', ascending=False)
        return df
    
    def create_visualization(self, results: List[Dict]):
        """
        Create visualizations comparing multi vs single modality.
        
        Args:
            results: List of comparison results
        """
        # Prepare data for visualization
        benchmarks = []
        single_scores = []
        multi_scores = []
        improvements = []
        
        for result in results:
            multi_mdae = result['multi_modality'].get('auroc', {}).get('MDAE', {})
            single_mdae = result['single_modality'].get('auroc', {}).get('MDAE', {})
            
            if multi_mdae and single_mdae:
                benchmarks.append(result['benchmark'].replace('_', ' ').title())
                single_scores.append(single_mdae.get('best', 0))
                multi_scores.append(multi_mdae.get('best', 0))
                improvements.append(result['improvement'].get('mdae_auroc', {}).get('relative_percent', 0))
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Direct comparison
        x = np.arange(len(benchmarks))
        width = 0.35
        
        axes[0].bar(x - width/2, single_scores, width, label='Single-Modality', color='#1f77b4')
        axes[0].bar(x + width/2, multi_scores, width, label='Multi-Modality', color='#ff7f0e')
        axes[0].set_xlabel('Benchmark')
        axes[0].set_ylabel('AUROC')
        axes[0].set_title('MDAE Performance: Single vs Multi-Modality')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(benchmarks, rotation=45, ha='right')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Improvement percentages
        colors = ['green' if i > 0 else 'red' for i in improvements]
        axes[1].barh(benchmarks, improvements, color=colors)
        axes[1].set_xlabel('Improvement (%)')
        axes[1].set_title('Multi-Modality Improvement over Single-Modality')
        axes[1].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_path / 'multi_vs_single_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved to {self.output_path / 'multi_vs_single_comparison.png'}")
    
    def generate_report(self, results: List[Dict], summary_df: pd.DataFrame):
        """
        Generate comprehensive analysis report.
        
        Args:
            results: List of comparison results
            summary_df: Summary DataFrame
        """
        report = []
        report.append("# Multi-Modality Analysis Report")
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("\n## Executive Summary")
        
        # Calculate overall statistics
        if not summary_df.empty and 'Improvement_%' in summary_df.columns:
            avg_improvement = summary_df['Improvement_%'].mean()
            best_improvement = summary_df.loc[summary_df['Improvement_%'].idxmax()]
            worst_improvement = summary_df.loc[summary_df['Improvement_%'].idxmin()]
        else:
            avg_improvement = 0
            best_improvement = {'Benchmark': 'N/A', 'Improvement_%': 0}
            worst_improvement = {'Benchmark': 'N/A', 'Improvement_%': 0}
        
        report.append(f"\n- **Average Improvement**: {avg_improvement:.2f}%")
        report.append(f"- **Best Improvement**: {best_improvement['Benchmark']} ({best_improvement['Improvement_%']:.2f}%)")
        report.append(f"- **Worst Performance**: {worst_improvement['Benchmark']} ({worst_improvement['Improvement_%']:.2f}%)")
        
        # Benchmarks with positive improvement
        if not summary_df.empty and 'Improvement_%' in summary_df.columns:
            positive_count = (summary_df['Improvement_%'] > 0).sum()
            total_count = len(summary_df)
        else:
            positive_count = 0
            total_count = 0
        report.append(f"- **Positive Improvements**: {positive_count}/{total_count} benchmarks")
        
        report.append("\n## Detailed Results\n")
        report.append(summary_df.to_markdown(index=False))
        
        report.append("\n## Per-Benchmark Analysis")
        
        for result in results:
            benchmark = result['benchmark']
            report.append(f"\n### {benchmark.replace('_', ' ').title()}")
            
            # Multi-modality top methods
            multi_methods = result['multi_modality'].get('auroc', {})
            if multi_methods:
                report.append("\n**Multi-Modality Top 5 Methods:**")
                sorted_methods = sorted(multi_methods.items(), 
                                      key=lambda x: x[1].get('best', 0), 
                                      reverse=True)[:5]
                for method, scores in sorted_methods:
                    report.append(f"- {method}: {scores.get('best', 0):.4f}")
            
            # Improvement analysis
            improvement = result['improvement'].get('mdae_auroc', {})
            if improvement:
                report.append(f"\n**MDAE Improvement:** {improvement.get('absolute', 0):.4f} "
                            f"({improvement.get('relative_percent', 0):.2f}%)")
        
        # Save report
        report_path = self.output_path / 'multi_modality_analysis_report.md'
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"Report saved to {report_path}")
    
    def run_analysis(self):
        """Run complete multi-modality analysis."""
        print("=" * 60)
        print("Multi-Modality Analysis")
        print("=" * 60)
        
        # Analyze all benchmarks
        results = self.analyze_all_benchmarks()
        
        # Create summary table
        summary_df = self.create_summary_table(results)
        
        # Save summary as CSV
        summary_path = self.output_path / 'multi_modality_summary.csv'
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSummary saved to {summary_path}")
        
        # Create visualizations
        if not summary_df.empty:
            self.create_visualization(results)
        
        # Generate report
        self.generate_report(results, summary_df)
        
        # Print summary
        print("\n" + "=" * 60)
        print("Analysis Complete!")
        print("=" * 60)
        print(f"\nKey Findings:")
        if not summary_df.empty and 'Improvement_%' in summary_df.columns:
            print(f"- Average improvement: {summary_df['Improvement_%'].mean():.2f}%")
            print(f"- Positive improvements: {(summary_df['Improvement_%'] > 0).sum()}/{len(summary_df)}")
        else:
            print(f"- No MDAE comparison data available")
        print(f"- Benchmarks analyzed: {len(results)}")
        
        return results, summary_df


def main():
    """Main function to run the analysis."""
    analyzer = MultiModalityAnalyzer()
    results, summary = analyzer.run_analysis()
    
    # Display top improvements
    if not summary.empty and all(col in summary.columns for col in ['Benchmark', 'Single_MDAE_AUROC', 'Multi_MDAE_AUROC', 'Improvement_%']):
        print("\nTop 5 Improvements:")
        print(summary.head()[['Benchmark', 'Single_MDAE_AUROC', 'Multi_MDAE_AUROC', 'Improvement_%']])


if __name__ == "__main__":
    main()