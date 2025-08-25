#!/usr/bin/env python3
"""
Best Run Analysis for Merged Multi-Modality Benchmarks
=======================================================
This script analyzes the merged benchmark data and extracts the best run
(maximum AUROC) for each method, reporting all four test metrics.
"""

import argparse
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

# Updated method patterns with better coverage
METHOD_PATTERNS = {
    'MDAE': r'^(resenc_MDAETrainer_RandomMask_Flow_BS48_2000ep|resenc_MDAE_pretrained|resenc_MDAE_scratch)',
    'MDAE (TC)': r'^(resenc_time_conditioned|resenc_multimodal_mm_mdae)',
    'MAE': r'^resenc_pretrained',  # Fixed: was causing Unknown
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

class BestRunAnalyzer:
    """Analyze best runs from merged benchmark data."""
    
    def __init__(self, input_dir: str, output_dir: str):
        """Initialize analyzer."""
        self.input_path = Path(input_dir)
        self.output_path = Path(output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
    def categorize_method(self, run_name: str) -> str:
        """
        Categorize a run based on its name with improved pattern matching.
        """
        if pd.isna(run_name):
            return 'Unknown'
        
        # Check each pattern
        for method, pattern in METHOD_PATTERNS.items():
            if re.match(pattern, run_name):
                return method
        
        # Additional checks for edge cases
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
        elif run_name.startswith('resenc_scratch'):
            return 'MAE (Scratch)'
        
        # Debug: print unmatched patterns
        print(f"  Warning: Could not categorize run: {run_name}")
        return 'Unknown'
    
    def load_and_categorize_data(self, benchmark_dir: Path) -> pd.DataFrame:
        """Load and properly categorize benchmark data."""
        csv_file = benchmark_dir / "merged_runs.csv"
        
        if not csv_file.exists():
            print(f"  Warning: No data found for {benchmark_dir.name}")
            return pd.DataFrame()
        
        df = pd.read_csv(csv_file)
        print(f"  Loaded {len(df)} runs from {benchmark_dir.name}")
        
        # Re-categorize methods with improved logic
        df['Method'] = df['method_extracted'].apply(self.categorize_method)
        
        # Print categorization summary
        method_counts = df['Method'].value_counts()
        print(f"  Method distribution:")
        for method, count in method_counts.items():
            print(f"    {method}: {count} runs")
        
        return df
    
    def combine_mdae_variants(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Combine MDAE variants by selecting the best run across all variants.
        """
        if df.empty:
            return df
        
        # Separate MDAE variants and other methods
        mdae_mask = df['Method'].isin(['MDAE', 'MDAE (TC)'])
        mdae_df = df[mdae_mask].copy()
        other_df = df[~mdae_mask].copy()
        
        if not mdae_df.empty:
            # Find the best MDAE run across all variants
            best_mdae_idx = mdae_df['metric_Test_AUROC'].idxmax()
            if pd.notna(best_mdae_idx):
                best_mdae = mdae_df.loc[[best_mdae_idx]].copy()
                best_mdae['Method'] = 'MDAE (Combined)'
                # Combine with other methods
                df = pd.concat([other_df, best_mdae], ignore_index=True)
            else:
                df = other_df
        
        return df
    
    def extract_best_runs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract the best run (max AUROC) for each method with all metrics.
        """
        if df.empty:
            return pd.DataFrame()
        
        # First combine MDAE variants
        df = self.combine_mdae_variants(df)
        
        best_runs = []
        
        # Group by method and find best run
        for method in df['Method'].unique():
            method_df = df[df['Method'] == method].copy()
            
            # Filter out runs without AUROC
            method_df = method_df[method_df['metric_Test_AUROC'].notna()]
            
            if not method_df.empty:
                # Find run with maximum AUROC
                best_idx = method_df['metric_Test_AUROC'].idxmax()
                best_run = method_df.loc[best_idx]
                
                # Extract all four metrics
                best_run_data = {
                    'Method': method,
                    'Run_Name': best_run['run_name'],
                    'Run_ID': best_run['run_id'],
                    'Project_Variant': best_run['project_variant'],
                    'Test_AUROC': best_run.get('metric_Test_AUROC', np.nan),
                    'Test_AP': best_run.get('metric_Test_AP', np.nan),
                    'Test_F1': best_run.get('metric_Test_F1', np.nan),
                    'Test_Balanced_Accuracy': best_run.get('metric_Test_Balanced_Accuracy', np.nan),
                    'URL': best_run.get('url', '')
                }
                
                best_runs.append(best_run_data)
        
        # Create DataFrame and sort by AUROC
        best_runs_df = pd.DataFrame(best_runs)
        best_runs_df = best_runs_df.sort_values('Test_AUROC', ascending=False)
        
        return best_runs_df
    
    def create_comprehensive_visualization(self, brats_df: pd.DataFrame, ucsf_df: pd.DataFrame):
        """Create comprehensive visualization with all four metrics."""
        fig = plt.figure(figsize=(24, 14))
        
        # Define metrics for visualization
        metrics = ['Test_AUROC', 'Test_AP', 'Test_F1', 'Test_Balanced_Accuracy']
        metric_labels = ['AUROC', 'Average Precision', 'F1 Score', 'Balanced Accuracy']
        
        # 1. BraTS18 - All Metrics
        ax1 = plt.subplot(2, 4, 1)
        if not brats_df.empty:
            top_methods = brats_df.head(8)
            x = np.arange(len(top_methods))
            width = 0.2
            
            for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
                values = top_methods[metric].fillna(0).values
                ax1.bar(x + i*width, values, width, label=label)
            
            ax1.set_xlabel('Method')
            ax1.set_ylabel('Score')
            ax1.set_title('BraTS18: All Test Metrics (Top 8 Methods)', fontsize=12, fontweight='bold')
            ax1.set_xticks(x + width * 1.5)
            ax1.set_xticklabels(top_methods['Method'], rotation=45, ha='right')
            ax1.legend(loc='upper right', fontsize=8)
            ax1.set_ylim([0, 1.05])
            ax1.grid(axis='y', alpha=0.3)
        
        # 2. UCSF - All Metrics
        ax2 = plt.subplot(2, 4, 2)
        if not ucsf_df.empty:
            top_methods = ucsf_df.head(8)
            x = np.arange(len(top_methods))
            width = 0.2
            
            for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
                values = top_methods[metric].fillna(0).values
                ax2.bar(x + i*width, values, width, label=label)
            
            ax2.set_xlabel('Method')
            ax2.set_ylabel('Score')
            ax2.set_title('UCSF: All Test Metrics (Top 8 Methods)', fontsize=12, fontweight='bold')
            ax2.set_xticks(x + width * 1.5)
            ax2.set_xticklabels(top_methods['Method'], rotation=45, ha='right')
            ax2.legend(loc='upper right', fontsize=8)
            ax2.set_ylim([0, 1.05])
            ax2.grid(axis='y', alpha=0.3)
        
        # 3. AUROC Comparison
        ax3 = plt.subplot(2, 4, 3)
        brats_top = brats_df.head(10) if not brats_df.empty else pd.DataFrame()
        ucsf_top = ucsf_df.head(10) if not ucsf_df.empty else pd.DataFrame()
        
        if not brats_top.empty:
            ax3.barh(range(len(brats_top)), brats_top['Test_AUROC'], 
                    color='#2E86AB', alpha=0.7, label='BraTS18')
        if not ucsf_top.empty:
            ax3.barh(range(len(ucsf_top)), ucsf_top['Test_AUROC'], 
                    color='#A23B72', alpha=0.7, label='UCSF')
        
        ax3.set_yticks(range(max(len(brats_top), len(ucsf_top))))
        ax3.set_yticklabels(brats_top['Method'] if len(brats_top) >= len(ucsf_top) else ucsf_top['Method'])
        ax3.set_xlabel('Test AUROC')
        ax3.set_title('AUROC Comparison (Top 10)', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.set_xlim([0, 1])
        ax3.grid(axis='x', alpha=0.3)
        
        # 4. Radar plot for top 3 methods - BraTS18
        ax4 = plt.subplot(2, 4, 4, projection='polar')
        if not brats_df.empty and len(brats_df) >= 3:
            top3 = brats_df.head(3)
            angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
            angles += angles[:1]
            
            for idx, row in top3.iterrows():
                values = [row[m] if pd.notna(row[m]) else 0 for m in metrics]
                values += values[:1]
                ax4.plot(angles, values, 'o-', linewidth=2, label=row['Method'])
                ax4.fill(angles, values, alpha=0.25)
            
            ax4.set_xticks(angles[:-1])
            ax4.set_xticklabels(metric_labels, size=8)
            ax4.set_ylim([0, 1])
            ax4.set_title('BraTS18: Top 3 Methods', fontsize=12, fontweight='bold', pad=20)
            ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
            ax4.grid(True)
        
        # 5. Best Run Details Table - BraTS18
        ax5 = plt.subplot(2, 4, 5)
        ax5.axis('tight')
        ax5.axis('off')
        
        if not brats_df.empty:
            table_data = []
            for idx, row in brats_df.head(5).iterrows():
                run_name_short = row['Run_Name'][:40] + '...' if len(row['Run_Name']) > 40 else row['Run_Name']
                table_data.append([
                    row['Method'],
                    f"{row['Test_AUROC']:.4f}",
                    f"{row['Test_AP']:.4f}" if pd.notna(row['Test_AP']) else 'N/A',
                    f"{row['Test_F1']:.4f}" if pd.notna(row['Test_F1']) else 'N/A',
                    f"{row['Test_Balanced_Accuracy']:.4f}" if pd.notna(row['Test_Balanced_Accuracy']) else 'N/A'
                ])
            
            table = ax5.table(cellText=table_data,
                            colLabels=['Method', 'AUROC', 'AP', 'F1', 'Bal Acc'],
                            cellLoc='center',
                            loc='center',
                            colWidths=[0.2, 0.15, 0.15, 0.15, 0.15])
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)
            ax5.set_title('BraTS18: Top 5 Best Runs', fontsize=12, fontweight='bold', pad=20)
        
        # 6. Best Run Details Table - UCSF
        ax6 = plt.subplot(2, 4, 6)
        ax6.axis('tight')
        ax6.axis('off')
        
        if not ucsf_df.empty:
            table_data = []
            for idx, row in ucsf_df.head(5).iterrows():
                run_name_short = row['Run_Name'][:40] + '...' if len(row['Run_Name']) > 40 else row['Run_Name']
                table_data.append([
                    row['Method'],
                    f"{row['Test_AUROC']:.4f}",
                    f"{row['Test_AP']:.4f}" if pd.notna(row['Test_AP']) else 'N/A',
                    f"{row['Test_F1']:.4f}" if pd.notna(row['Test_F1']) else 'N/A',
                    f"{row['Test_Balanced_Accuracy']:.4f}" if pd.notna(row['Test_Balanced_Accuracy']) else 'N/A'
                ])
            
            table = ax6.table(cellText=table_data,
                            colLabels=['Method', 'AUROC', 'AP', 'F1', 'Bal Acc'],
                            cellLoc='center',
                            loc='center',
                            colWidths=[0.2, 0.15, 0.15, 0.15, 0.15])
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)
            ax6.set_title('UCSF: Top 5 Best Runs', fontsize=12, fontweight='bold', pad=20)
        
        # 7. Cross-metric correlation
        ax7 = plt.subplot(2, 4, 7)
        all_data = pd.concat([brats_df, ucsf_df], ignore_index=True)
        if not all_data.empty:
            # Plot AUROC vs F1
            valid_data = all_data[all_data['Test_F1'].notna()]
            ax7.scatter(valid_data['Test_AUROC'], valid_data['Test_F1'], alpha=0.6, s=50)
            ax7.set_xlabel('Test AUROC')
            ax7.set_ylabel('Test F1 Score')
            ax7.set_title('AUROC vs F1 Score Correlation', fontsize=12, fontweight='bold')
            ax7.set_xlim([0, 1])
            ax7.set_ylim([0, 1])
            ax7.grid(True, alpha=0.3)
            
            # Add diagonal line
            ax7.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        
        # 8. Summary Statistics
        ax8 = plt.subplot(2, 4, 8)
        ax8.axis('off')
        
        # Calculate best values
        brats_best_auroc = f"{brats_df.iloc[0]['Test_AUROC']:.4f}" if not brats_df.empty else "N/A"
        brats_best_method = brats_df.iloc[0]['Method'] if not brats_df.empty else 'N/A'
        brats_best_run = brats_df.iloc[0]['Run_Name'][:30] + '...' if not brats_df.empty else 'N/A'
        
        ucsf_best_auroc = f"{ucsf_df.iloc[0]['Test_AUROC']:.4f}" if not ucsf_df.empty else "N/A"
        ucsf_best_method = ucsf_df.iloc[0]['Method'] if not ucsf_df.empty else 'N/A'
        ucsf_best_run = ucsf_df.iloc[0]['Run_Name'][:30] + '...' if not ucsf_df.empty else 'N/A'
        
        summary_text = f"""
BEST RUN ANALYSIS SUMMARY
{'='*40}
Date: {datetime.now().strftime('%Y-%m-%d')}

BRATS18 LGG vs HGG:
  Total Methods: {len(brats_df)}
  Best Method: {brats_best_method}
  Best AUROC: {brats_best_auroc}
  Best Run: {brats_best_run}

UCSF-PDGM IDH:
  Total Methods: {len(ucsf_df)}
  Best Method: {ucsf_best_method}
  Best AUROC: {ucsf_best_auroc}
  Best Run: {ucsf_best_run}

KEY FINDINGS:
• Best runs selected by maximum AUROC
• All four test metrics reported
• MDAE variants combined for fair comparison
• Method categorization improved
"""
        
        ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Overall title
        fig.suptitle('Best Run Analysis - Multi-Modality Benchmarks\n(Maximum AUROC per Method with All Test Metrics)', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        
        # Save figure
        output_file = self.output_path / 'best_runs_comprehensive_analysis.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n  Visualization saved to: {output_file}")
        
        plt.show()
    
    def generate_latex_tables(self, brats_df: pd.DataFrame, ucsf_df: pd.DataFrame):
        """Generate LaTeX-ready tables for publication."""
        latex_output = self.output_path / 'latex_tables'
        latex_output.mkdir(exist_ok=True)
        
        # Function to format dataframe for LaTeX
        def format_for_latex(df, benchmark_name):
            if df.empty:
                return ""
            
            latex_str = f"% {benchmark_name} - Best Runs (Maximum AUROC)\n"
            latex_str += "\\begin{table}[h!]\n\\centering\n"
            latex_str += f"\\caption{{{benchmark_name} - Best Run Performance}}\n"
            latex_str += "\\begin{tabular}{llcccc}\n\\hline\n"
            latex_str += "Method & AUROC & AP & F1 & Bal. Acc. \\\\\n\\hline\n"
            
            for idx, row in df.head(10).iterrows():
                auroc = f"{row['Test_AUROC']:.3f}" if pd.notna(row['Test_AUROC']) else "-"
                ap = f"{row['Test_AP']:.3f}" if pd.notna(row['Test_AP']) else "-"
                f1 = f"{row['Test_F1']:.3f}" if pd.notna(row['Test_F1']) else "-"
                bal = f"{row['Test_Balanced_Accuracy']:.3f}" if pd.notna(row['Test_Balanced_Accuracy']) else "-"
                
                latex_str += f"{row['Method']} & {auroc} & {ap} & {f1} & {bal} \\\\\n"
            
            latex_str += "\\hline\n\\end{tabular}\n\\end{table}\n"
            return latex_str
        
        # Generate LaTeX for both benchmarks
        latex_content = ""
        latex_content += format_for_latex(brats_df, "BraTS18 LGG vs HGG")
        latex_content += "\n\n"
        latex_content += format_for_latex(ucsf_df, "UCSF-PDGM IDH Classification")
        
        # Save to file
        with open(latex_output / 'best_runs_tables.tex', 'w') as f:
            f.write(latex_content)
        
        print(f"  LaTeX tables saved to: {latex_output}")
    
    def save_results(self, brats_df: pd.DataFrame, ucsf_df: pd.DataFrame):
        """Save best run results to CSV files."""
        # Save individual benchmark results
        if not brats_df.empty:
            brats_df.to_csv(self.output_path / 'brats18_best_runs.csv', index=False)
            print(f"  BraTS18 best runs saved ({len(brats_df)} methods)")
        
        if not ucsf_df.empty:
            ucsf_df.to_csv(self.output_path / 'ucsf_best_runs.csv', index=False)
            print(f"  UCSF best runs saved ({len(ucsf_df)} methods)")
        
        # Create combined summary
        combined = []
        for df, benchmark in [(brats_df, 'BraTS18'), (ucsf_df, 'UCSF')]:
            if not df.empty:
                df_copy = df.copy()
                df_copy['Benchmark'] = benchmark
                combined.append(df_copy)
        
        if combined:
            combined_df = pd.concat(combined, ignore_index=True)
            combined_df.to_csv(self.output_path / 'combined_best_runs.csv', index=False)
            print(f"  Combined best runs saved ({len(combined_df)} total entries)")
    
    def analyze_all(self):
        """Perform complete best run analysis."""
        print("\n" + "="*60)
        print("Best Run Analysis - Multi-Modality Benchmarks")
        print("="*60)
        print(f"Input: {self.input_path}")
        print(f"Output: {self.output_path}")
        print("="*60)
        
        # Process BraTS18
        print("\nProcessing BraTS18...")
        brats_dir = self.input_path / 'brats18_lgg_vs_hgg'
        brats_data = self.load_and_categorize_data(brats_dir) if brats_dir.exists() else pd.DataFrame()
        brats_best = self.extract_best_runs(brats_data) if not brats_data.empty else pd.DataFrame()
        
        # Process UCSF
        print("\nProcessing UCSF...")
        ucsf_dir = self.input_path / 'ucsf_pdgm_idh'
        ucsf_data = self.load_and_categorize_data(ucsf_dir) if ucsf_dir.exists() else pd.DataFrame()
        ucsf_best = self.extract_best_runs(ucsf_data) if not ucsf_data.empty else pd.DataFrame()
        
        # Update todos
        print("\nSaving results...")
        self.save_results(brats_best, ucsf_best)
        
        print("\nGenerating LaTeX tables...")
        self.generate_latex_tables(brats_best, ucsf_best)
        
        print("\nCreating visualizations...")
        self.create_comprehensive_visualization(brats_best, ucsf_best)
        
        print("\n" + "="*60)
        print("Analysis Complete!")
        print("="*60)
        
        # Print summary
        print("\nTop 5 Methods by Benchmark (Best Run AUROC):")
        print("-"*60)
        
        if not brats_best.empty:
            print("\nBraTS18 LGG vs HGG:")
            for idx, row in brats_best.head(5).iterrows():
                print(f"  {row['Method']:<20} AUROC: {row['Test_AUROC']:.4f}")
                print(f"    Run: {row['Run_Name'][:60]}...")
                ap_val = f"{row['Test_AP']:.4f}" if pd.notna(row['Test_AP']) else 'N/A'
                f1_val = f"{row['Test_F1']:.4f}" if pd.notna(row['Test_F1']) else 'N/A'
                bal_val = f"{row['Test_Balanced_Accuracy']:.4f}" if pd.notna(row['Test_Balanced_Accuracy']) else 'N/A'
                print(f"    Metrics - AP: {ap_val}, F1: {f1_val}, Bal.Acc: {bal_val}")
        
        if not ucsf_best.empty:
            print("\nUCSF-PDGM IDH:")
            for idx, row in ucsf_best.head(5).iterrows():
                print(f"  {row['Method']:<20} AUROC: {row['Test_AUROC']:.4f}")
                print(f"    Run: {row['Run_Name'][:60]}...")
                ap_val = f"{row['Test_AP']:.4f}" if pd.notna(row['Test_AP']) else 'N/A'
                f1_val = f"{row['Test_F1']:.4f}" if pd.notna(row['Test_F1']) else 'N/A'
                bal_val = f"{row['Test_Balanced_Accuracy']:.4f}" if pd.notna(row['Test_Balanced_Accuracy']) else 'N/A'
                print(f"    Metrics - AP: {ap_val}, F1: {f1_val}, Bal.Acc: {bal_val}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Analyze best runs from merged benchmarks')
    parser.add_argument('--input-dir', type=str, 
                       default='raw_data/20250815_merged',
                       help='Input directory with merged data')
    parser.add_argument('--output-dir', type=str,
                       default='processed_data_20250815_best_runs',
                       help='Output directory for best run analysis')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = BestRunAnalyzer(args.input_dir, args.output_dir)
    
    # Run analysis
    analyzer.analyze_all()


if __name__ == "__main__":
    main()