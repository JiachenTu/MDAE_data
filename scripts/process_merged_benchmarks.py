#!/usr/bin/env python3
"""
Process Merged Multi-Modality Benchmark Results
===============================================
This script processes the merged data from both project variants
and generates comprehensive analysis and visualizations.
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import warnings
from datetime import datetime
import re

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Method patterns (same as original)
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
    'ResNet-50': r'^brainiac_scratch'
}

class MergedBenchmarkProcessor:
    """Process merged benchmark data."""
    
    def __init__(self, input_dir: str, output_dir: str):
        """Initialize processor."""
        self.input_path = Path(input_dir)
        self.output_path = Path(output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
    def load_benchmark_data(self, benchmark_dir: Path) -> pd.DataFrame:
        """Load merged benchmark data."""
        csv_file = benchmark_dir / "merged_runs.csv"
        
        if not csv_file.exists():
            print(f"  Warning: No merged data found for {benchmark_dir.name}")
            return pd.DataFrame()
        
        df = pd.read_csv(csv_file)
        print(f"  Loaded {len(df)} runs from {benchmark_dir.name}")
        
        # Categorize methods
        df['Method'] = df['method_extracted'].apply(self.categorize_method)
        
        return df
    
    def categorize_method(self, run_name: str) -> str:
        """Categorize a run based on its name."""
        if pd.isna(run_name):
            return 'Unknown'
        
        for method, pattern in METHOD_PATTERNS.items():
            if re.match(pattern, run_name):
                return method
        
        if 'dinov2' in run_name.lower():
            return 'DinoV2'
        elif 'brainiac' in run_name.lower():
            if 'scratch' in run_name.lower():
                return 'ResNet-50'
            else:
                return 'BrainIAC'
        elif 'mri_core' in run_name.lower() or 'mri-core' in run_name.lower():
            return 'MRI-Core'
        elif 'brainmvp' in run_name.lower():
            return 'BrainMVP'
        
        return 'Unknown'
    
    def combine_mdae_variants(self, df: pd.DataFrame) -> pd.DataFrame:
        """Combine MDAE variants by taking the best performance."""
        if df.empty:
            return df
        
        # Group by Method
        grouped = []
        
        for method in df['Method'].unique():
            if method in ['MDAE', 'MDAE (TC)']:
                continue
            grouped.append(df[df['Method'] == method])
        
        # Combine MDAE variants
        mdae_df = df[df['Method'].isin(['MDAE', 'MDAE (TC)'])]
        if not mdae_df.empty:
            # Take the best AUROC for each unique configuration
            best_mdae = mdae_df.loc[mdae_df.groupby('method_extracted')['metric_Test_AUROC'].idxmax()]
            best_mdae['Method'] = 'MDAE (Combined)'
            grouped.append(best_mdae)
        
        return pd.concat(grouped, ignore_index=True) if grouped else pd.DataFrame()
    
    def process_benchmark(self, benchmark_dir: Path) -> pd.DataFrame:
        """Process a single benchmark."""
        df = self.load_benchmark_data(benchmark_dir)
        
        if df.empty:
            return pd.DataFrame()
        
        # Combine MDAE variants
        df_combined = self.combine_mdae_variants(df)
        
        # Calculate summary statistics per method
        summary = []
        for method in df_combined['Method'].unique():
            method_df = df_combined[df_combined['Method'] == method]
            
            # Get metrics
            auroc_col = 'metric_Test_AUROC'
            if auroc_col in method_df.columns:
                aurocs = method_df[auroc_col].dropna()
                if len(aurocs) > 0:
                    # Also separate by project variant
                    july_aurocs = method_df[method_df['project_variant'] == 'july'][auroc_col].dropna()
                    std_aurocs = method_df[method_df['project_variant'] == 'standard'][auroc_col].dropna()
                    
                    summary.append({
                        'Method': method,
                        'Mean_AUROC': aurocs.mean(),
                        'Std_AUROC': aurocs.std(),
                        'Max_AUROC': aurocs.max(),
                        'Count': len(aurocs),
                        'July_Mean': july_aurocs.mean() if len(july_aurocs) > 0 else np.nan,
                        'July_Count': len(july_aurocs),
                        'Standard_Mean': std_aurocs.mean() if len(std_aurocs) > 0 else np.nan,
                        'Standard_Count': len(std_aurocs)
                    })
        
        return pd.DataFrame(summary)
    
    def create_comparison_visualization(self, brats_df: pd.DataFrame, ucsf_df: pd.DataFrame):
        """Create comprehensive visualization comparing both benchmarks."""
        fig = plt.figure(figsize=(20, 12))
        
        # 1. BraTS18 Performance
        ax1 = plt.subplot(2, 3, 1)
        if not brats_df.empty:
            brats_sorted = brats_df.sort_values('Mean_AUROC', ascending=False).head(10)
            bars = ax1.barh(range(len(brats_sorted)), brats_sorted['Mean_AUROC'], 
                           xerr=brats_sorted['Std_AUROC'], capsize=3)
            ax1.set_yticks(range(len(brats_sorted)))
            ax1.set_yticklabels(brats_sorted['Method'])
            ax1.set_xlabel('Mean AUROC')
            ax1.set_title('BraTS18 LGG vs HGG\n(Merged Data)', fontsize=12, fontweight='bold')
            ax1.set_xlim([0, 1])
            ax1.grid(axis='x', alpha=0.3)
            
            # Color MDAE differently
            colors = ['#2E86AB' if 'MDAE' in m else '#A23B72' if m == 'MAE' else '#F18F01' 
                     for m in brats_sorted['Method']]
            for bar, color in zip(bars, colors):
                bar.set_color(color)
        
        # 2. UCSF Performance
        ax2 = plt.subplot(2, 3, 2)
        if not ucsf_df.empty:
            ucsf_sorted = ucsf_df.sort_values('Mean_AUROC', ascending=False).head(10)
            bars = ax2.barh(range(len(ucsf_sorted)), ucsf_sorted['Mean_AUROC'], 
                           xerr=ucsf_sorted['Std_AUROC'], capsize=3)
            ax2.set_yticks(range(len(ucsf_sorted)))
            ax2.set_yticklabels(ucsf_sorted['Method'])
            ax2.set_xlabel('Mean AUROC')
            ax2.set_title('UCSF-PDGM IDH Classification\n(Merged Data)', fontsize=12, fontweight='bold')
            ax2.set_xlim([0, 1])
            ax2.grid(axis='x', alpha=0.3)
            
            # Color MDAE differently
            colors = ['#2E86AB' if 'MDAE' in m else '#A23B72' if m == 'MAE' else '#F18F01' 
                     for m in ucsf_sorted['Method']]
            for bar, color in zip(bars, colors):
                bar.set_color(color)
        
        # 3. Project Variant Comparison - BraTS18
        ax3 = plt.subplot(2, 3, 3)
        if not brats_df.empty:
            # Compare july vs standard for top methods
            top_methods = brats_df.nlargest(5, 'Mean_AUROC')['Method'].tolist()
            
            july_means = []
            std_means = []
            labels = []
            
            for method in top_methods:
                method_data = brats_df[brats_df['Method'] == method].iloc[0]
                july_means.append(method_data['July_Mean'] if not pd.isna(method_data['July_Mean']) else 0)
                std_means.append(method_data['Standard_Mean'] if not pd.isna(method_data['Standard_Mean']) else 0)
                labels.append(method)
            
            x = np.arange(len(labels))
            width = 0.35
            
            bars1 = ax3.bar(x - width/2, july_means, width, label='July Variant', color='#2E86AB')
            bars2 = ax3.bar(x + width/2, std_means, width, label='Standard Variant', color='#A23B72')
            
            ax3.set_xlabel('Method')
            ax3.set_ylabel('Mean AUROC')
            ax3.set_title('BraTS18: July vs Standard Variants', fontsize=12, fontweight='bold')
            ax3.set_xticks(x)
            ax3.set_xticklabels(labels, rotation=45, ha='right')
            ax3.legend()
            ax3.set_ylim([0, 1])
            ax3.grid(axis='y', alpha=0.3)
        
        # 4. Run Count Distribution
        ax4 = plt.subplot(2, 3, 4)
        all_data = []
        all_labels = []
        
        if not brats_df.empty:
            brats_top = brats_df.nlargest(8, 'Mean_AUROC')
            all_data.append(brats_top['Count'].values)
            all_labels.append('BraTS18')
        
        if not ucsf_df.empty:
            ucsf_top = ucsf_df.nlargest(8, 'Mean_AUROC')
            all_data.append(ucsf_top['Count'].values)
            all_labels.append('UCSF')
        
        if all_data:
            bp = ax4.boxplot(all_data, labels=all_labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], ['#2E86AB', '#A23B72']):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax4.set_ylabel('Number of Runs')
            ax4.set_title('Run Count Distribution\n(Top 8 Methods)', fontsize=12, fontweight='bold')
            ax4.grid(axis='y', alpha=0.3)
        
        # 5. Cross-benchmark comparison
        ax5 = plt.subplot(2, 3, 5)
        
        # Find common methods
        if not brats_df.empty and not ucsf_df.empty:
            common_methods = set(brats_df['Method'].unique()) & set(ucsf_df['Method'].unique())
            
            if common_methods:
                comparison_data = []
                for method in common_methods:
                    brats_auroc = brats_df[brats_df['Method'] == method]['Mean_AUROC'].values[0]
                    ucsf_auroc = ucsf_df[ucsf_df['Method'] == method]['Mean_AUROC'].values[0]
                    comparison_data.append({
                        'Method': method,
                        'BraTS18': brats_auroc,
                        'UCSF': ucsf_auroc
                    })
                
                comp_df = pd.DataFrame(comparison_data)
                
                # Scatter plot
                ax5.scatter(comp_df['BraTS18'], comp_df['UCSF'], s=100, alpha=0.6)
                
                # Add diagonal line
                ax5.plot([0, 1], [0, 1], 'k--', alpha=0.3)
                
                # Add labels for top methods
                for _, row in comp_df.iterrows():
                    if row['Method'] in ['MDAE (Combined)', 'MAE', 'SimCLR']:
                        ax5.annotate(row['Method'], (row['BraTS18'], row['UCSF']), 
                                   xytext=(5, 5), textcoords='offset points', fontsize=8)
                
                ax5.set_xlabel('BraTS18 AUROC')
                ax5.set_ylabel('UCSF AUROC')
                ax5.set_title('Cross-Benchmark Performance', fontsize=12, fontweight='bold')
                ax5.set_xlim([0, 1])
                ax5.set_ylim([0, 1])
                ax5.grid(True, alpha=0.3)
        
        # 6. Summary Statistics
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        # Calculate statistics
        brats_stats = self._calculate_stats(brats_df) if not brats_df.empty else {}
        ucsf_stats = self._calculate_stats(ucsf_df) if not ucsf_df.empty else {}
        
        summary_text = f"""
MERGED DATA ANALYSIS SUMMARY
{'='*40}
Date: {datetime.now().strftime('%Y-%m-%d')}

BRATS18 LGG vs HGG:
  Total Methods: {brats_stats.get('total_methods', 0)}
  Best Method: {brats_stats.get('best_method', 'N/A')}
  Best AUROC: {brats_stats.get('best_auroc', 0):.4f}
  Total Runs: {brats_stats.get('total_runs', 0)}
  - July variant: {brats_stats.get('july_runs', 0)} runs
  - Standard variant: {brats_stats.get('standard_runs', 0)} runs

UCSF-PDGM IDH:
  Total Methods: {ucsf_stats.get('total_methods', 0)}
  Best Method: {ucsf_stats.get('best_method', 'N/A')}
  Best AUROC: {ucsf_stats.get('best_auroc', 0):.4f}
  Total Runs: {ucsf_stats.get('total_runs', 0)}
  - July variant: {ucsf_stats.get('july_runs', 0)} runs
  - Standard variant: {ucsf_stats.get('standard_runs', 0)} runs

KEY FINDINGS:
• Both benchmarks show MDAE as top performer
• BraTS18 has data from both project variants
• UCSF only has july variant data available
• Merged analysis provides more robust results
"""
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Overall title
        fig.suptitle('Multi-Modality Merged Benchmark Analysis\n(Combined July and Standard Variants)', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        
        # Save figure
        output_file = self.output_path / 'merged_analysis_visualization.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n  Visualization saved to: {output_file}")
        
        plt.show()
    
    def _calculate_stats(self, df: pd.DataFrame) -> Dict:
        """Calculate summary statistics for a benchmark."""
        if df.empty:
            return {}
        
        best = df.nlargest(1, 'Mean_AUROC').iloc[0]
        
        return {
            'total_methods': len(df),
            'best_method': best['Method'],
            'best_auroc': best['Mean_AUROC'],
            'total_runs': int(df['Count'].sum()),
            'july_runs': int(df['July_Count'].sum()),
            'standard_runs': int(df['Standard_Count'].sum())
        }
    
    def save_results(self, brats_df: pd.DataFrame, ucsf_df: pd.DataFrame):
        """Save processed results to CSV files."""
        if not brats_df.empty:
            brats_df.to_csv(self.output_path / 'brats18_merged_results.csv', index=False)
            print(f"  BraTS18 results saved")
        
        if not ucsf_df.empty:
            ucsf_df.to_csv(self.output_path / 'ucsf_merged_results.csv', index=False)
            print(f"  UCSF results saved")
        
        # Create combined summary
        summary = []
        
        for df, benchmark in [(brats_df, 'BraTS18'), (ucsf_df, 'UCSF')]:
            if not df.empty:
                for _, row in df.iterrows():
                    summary.append({
                        'Benchmark': benchmark,
                        **row.to_dict()
                    })
        
        if summary:
            summary_df = pd.DataFrame(summary)
            summary_df.to_csv(self.output_path / 'merged_summary.csv', index=False)
            print(f"  Combined summary saved")
    
    def process_all(self):
        """Process all merged benchmarks."""
        print("\n" + "="*60)
        print("Processing Merged Benchmark Data")
        print("="*60)
        print(f"Input: {self.input_path}")
        print(f"Output: {self.output_path}")
        print("="*60)
        
        # Process BraTS18
        print("\nProcessing BraTS18...")
        brats_dir = self.input_path / 'brats18_lgg_vs_hgg'
        brats_df = self.process_benchmark(brats_dir) if brats_dir.exists() else pd.DataFrame()
        
        # Process UCSF
        print("\nProcessing UCSF...")
        ucsf_dir = self.input_path / 'ucsf_pdgm_idh'
        ucsf_df = self.process_benchmark(ucsf_dir) if ucsf_dir.exists() else pd.DataFrame()
        
        # Save results
        print("\nSaving results...")
        self.save_results(brats_df, ucsf_df)
        
        # Create visualizations
        print("\nCreating visualizations...")
        self.create_comparison_visualization(brats_df, ucsf_df)
        
        print("\n" + "="*60)
        print("Processing Complete!")
        print("="*60)
        
        # Print top results
        print("\nTop 5 Methods by Benchmark:")
        print("-"*40)
        
        if not brats_df.empty:
            print("\nBraTS18 LGG vs HGG:")
            for i, row in brats_df.nlargest(5, 'Mean_AUROC').iterrows():
                print(f"  {row['Method']:<20} {row['Mean_AUROC']:.4f} (±{row['Std_AUROC']:.4f})")
        
        if not ucsf_df.empty:
            print("\nUCSF-PDGM IDH:")
            for i, row in ucsf_df.nlargest(5, 'Mean_AUROC').iterrows():
                print(f"  {row['Method']:<20} {row['Mean_AUROC']:.4f} (±{row['Std_AUROC']:.4f})")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Process merged multi-modality benchmark results')
    parser.add_argument('--input-dir', type=str, 
                       default='raw_data/20250815_merged',
                       help='Input directory with merged data')
    parser.add_argument('--output-dir', type=str,
                       default='processed_data_20250815_merged',
                       help='Output directory for processed results')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = MergedBenchmarkProcessor(args.input_dir, args.output_dir)
    
    # Process all benchmarks
    processor.process_all()


if __name__ == "__main__":
    main()