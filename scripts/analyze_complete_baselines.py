#!/usr/bin/env python3
"""
Complete Baseline Analysis for Merged Multi-Modality Benchmarks
================================================================
This script analyzes ALL baselines, including those without successful runs,
and generates comprehensive tables showing the complete picture.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Set
import warnings
from datetime import datetime
import re

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Define ALL expected baselines
EXPECTED_BASELINES = [
    'MDAE',
    'MDAE (TC)', 
    'MAE',
    'SimCLR',
    'VoCo',
    'MG',
    'SwinUNETR',
    'VF',
    'S3D',
    'BrainIAC',
    'MRI-Core',
    'BrainMVP',
    'DinoV2',
    'ResNet-50'
]

# Updated method patterns
METHOD_PATTERNS = {
    'MDAE': r'^(resenc_MDAETrainer_RandomMask_Flow_BS48_2000ep|resenc_MDAE_pretrained|resenc_MDAE_scratch)',
    'MDAE (TC)': r'^(resenc_time_conditioned|resenc_multimodal_mm_mdae)',
    'MAE': r'^resenc_pretrained',
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

class CompleteBaselineAnalyzer:
    """Analyze all baselines from merged benchmark data."""
    
    def __init__(self, input_dir: str, output_dir: str):
        """Initialize analyzer."""
        self.input_path = Path(input_dir)
        self.output_path = Path(output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
    def categorize_method(self, run_name: str) -> str:
        """Categorize a run based on its name."""
        if pd.isna(run_name):
            return 'Unknown'
        
        for method, pattern in METHOD_PATTERNS.items():
            if re.match(pattern, run_name):
                return method
        
        # Additional checks
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
        
        print(f"  Warning: Could not categorize run: {run_name}")
        return 'Unknown'
    
    def load_and_categorize_data(self, benchmark_dir: Path) -> pd.DataFrame:
        """Load and properly categorize benchmark data."""
        csv_file = benchmark_dir / "merged_runs.csv"
        
        if not csv_file.exists():
            print(f"  Warning: No data found for {benchmark_dir.name}")
            return pd.DataFrame()
        
        df = pd.read_csv(csv_file)
        print(f"  Loaded {len(df)} total runs from {benchmark_dir.name}")
        
        # Re-categorize methods
        df['Method'] = df['method_extracted'].apply(self.categorize_method)
        
        # Print run statistics
        total_runs = len(df)
        successful_runs = df['metric_Test_AUROC'].notna().sum()
        failed_runs = total_runs - successful_runs
        
        print(f"  Run statistics:")
        print(f"    - Total runs: {total_runs}")
        print(f"    - Successful runs (with AUROC): {successful_runs}")
        print(f"    - Failed/crashed runs: {failed_runs}")
        
        # Print method distribution
        method_counts = df.groupby('Method').agg({
            'run_id': 'count',
            'metric_Test_AUROC': lambda x: x.notna().sum()
        }).rename(columns={'run_id': 'total_runs', 'metric_Test_AUROC': 'successful_runs'})
        
        print(f"  Method distribution:")
        for method, counts in method_counts.iterrows():
            print(f"    {method}: {counts['total_runs']} runs ({counts['successful_runs']} successful)")
        
        return df
    
    def combine_mdae_variants(self, df: pd.DataFrame) -> pd.DataFrame:
        """Combine MDAE variants by selecting the best run."""
        if df.empty:
            return df
        
        # Separate MDAE variants and other methods
        mdae_mask = df['Method'].isin(['MDAE', 'MDAE (TC)'])
        mdae_df = df[mdae_mask].copy()
        other_df = df[~mdae_mask].copy()
        
        if not mdae_df.empty:
            # Filter for successful runs only
            mdae_successful = mdae_df[mdae_df['metric_Test_AUROC'].notna()]
            if not mdae_successful.empty:
                # Find the best MDAE run
                best_mdae_idx = mdae_successful['metric_Test_AUROC'].idxmax()
                best_mdae = mdae_successful.loc[[best_mdae_idx]].copy()
                best_mdae['Method'] = 'MDAE (Combined)'
                # Combine with other methods
                df = pd.concat([other_df, best_mdae], ignore_index=True)
            else:
                # No successful MDAE runs
                df = other_df
        
        return df
    
    def extract_all_baselines(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract best runs for ALL expected baselines, including those without data.
        """
        if df.empty:
            # Return empty results for all baselines
            return self._create_empty_baseline_df()
        
        # First combine MDAE variants
        df = self.combine_mdae_variants(df)
        
        # Get all methods that should be included
        all_methods = set(EXPECTED_BASELINES)
        all_methods.discard('MDAE')  # Remove individual MDAE
        all_methods.discard('MDAE (TC)')  # Remove individual MDAE (TC)
        all_methods.add('MDAE (Combined)')  # Add combined version
        
        best_runs = []
        
        for method in all_methods:
            method_df = df[df['Method'] == method].copy()
            
            if method_df.empty:
                # No runs for this method
                best_runs.append({
                    'Method': method,
                    'Run_Name': 'No runs available',
                    'Run_ID': '-',
                    'Project_Variant': '-',
                    'Test_AUROC': np.nan,
                    'Test_AP': np.nan,
                    'Test_F1': np.nan,
                    'Test_Balanced_Accuracy': np.nan,
                    'Total_Runs': 0,
                    'Successful_Runs': 0,
                    'Status': 'No data'
                })
            else:
                # Count total and successful runs
                total_runs = len(method_df)
                successful_runs = method_df['metric_Test_AUROC'].notna().sum()
                
                # Filter for successful runs
                method_successful = method_df[method_df['metric_Test_AUROC'].notna()]
                
                if method_successful.empty:
                    # All runs failed
                    best_runs.append({
                        'Method': method,
                        'Run_Name': 'All runs failed/crashed',
                        'Run_ID': '-',
                        'Project_Variant': '-',
                        'Test_AUROC': np.nan,
                        'Test_AP': np.nan,
                        'Test_F1': np.nan,
                        'Test_Balanced_Accuracy': np.nan,
                        'Total_Runs': total_runs,
                        'Successful_Runs': 0,
                        'Status': 'Failed'
                    })
                else:
                    # Find best run
                    best_idx = method_successful['metric_Test_AUROC'].idxmax()
                    best_run = method_successful.loc[best_idx]
                    
                    best_runs.append({
                        'Method': method,
                        'Run_Name': best_run['run_name'],
                        'Run_ID': best_run['run_id'],
                        'Project_Variant': best_run['project_variant'],
                        'Test_AUROC': best_run.get('metric_Test_AUROC', np.nan),
                        'Test_AP': best_run.get('metric_Test_AP', np.nan),
                        'Test_F1': best_run.get('metric_Test_F1', np.nan),
                        'Test_Balanced_Accuracy': best_run.get('metric_Test_Balanced_Accuracy', np.nan),
                        'Total_Runs': total_runs,
                        'Successful_Runs': successful_runs,
                        'Status': 'Success'
                    })
        
        # Create DataFrame and sort
        best_runs_df = pd.DataFrame(best_runs)
        
        # Sort by AUROC (NaN values at the end)
        best_runs_df['sort_key'] = best_runs_df['Test_AUROC'].fillna(-1)
        best_runs_df = best_runs_df.sort_values('sort_key', ascending=False)
        best_runs_df = best_runs_df.drop('sort_key', axis=1)
        
        return best_runs_df
    
    def _create_empty_baseline_df(self) -> pd.DataFrame:
        """Create empty DataFrame with all baselines."""
        all_methods = set(EXPECTED_BASELINES)
        all_methods.discard('MDAE')
        all_methods.discard('MDAE (TC)')
        all_methods.add('MDAE (Combined)')
        
        empty_data = []
        for method in all_methods:
            empty_data.append({
                'Method': method,
                'Run_Name': 'No data',
                'Run_ID': '-',
                'Project_Variant': '-',
                'Test_AUROC': np.nan,
                'Test_AP': np.nan,
                'Test_F1': np.nan,
                'Test_Balanced_Accuracy': np.nan,
                'Total_Runs': 0,
                'Successful_Runs': 0,
                'Status': 'No data'
            })
        
        return pd.DataFrame(empty_data)
    
    def create_complete_table_visualization(self, brats_df: pd.DataFrame, ucsf_df: pd.DataFrame):
        """Create comprehensive table visualization with all baselines."""
        fig = plt.figure(figsize=(24, 16))
        
        # Create two large subplots for tables
        ax1 = plt.subplot(1, 2, 1)
        ax1.axis('tight')
        ax1.axis('off')
        
        # BraTS18 Table
        if not brats_df.empty:
            table_data = []
            colors = []
            
            for idx, row in brats_df.iterrows():
                # Format values
                if row['Status'] == 'Success':
                    auroc = f"{row['Test_AUROC']:.4f}" if pd.notna(row['Test_AUROC']) else '-'
                    ap = f"{row['Test_AP']:.4f}" if pd.notna(row['Test_AP']) else '-'
                    f1 = f"{row['Test_F1']:.4f}" if pd.notna(row['Test_F1']) else '-'
                    bal = f"{row['Test_Balanced_Accuracy']:.4f}" if pd.notna(row['Test_Balanced_Accuracy']) else '-'
                    status_color = 'lightgreen'
                elif row['Status'] == 'Failed':
                    auroc = ap = f1 = bal = 'Failed'
                    status_color = 'lightcoral'
                else:
                    auroc = ap = f1 = bal = 'No data'
                    status_color = 'lightgray'
                
                table_data.append([
                    row['Method'],
                    auroc,
                    ap,
                    f1,
                    bal,
                    f"{row['Successful_Runs']}/{row['Total_Runs']}"
                ])
                
                # Color based on status
                colors.append([status_color if i == 0 else 'white' for i in range(6)])
            
            table = ax1.table(cellText=table_data,
                            colLabels=['Method', 'AUROC', 'AP', 'F1', 'Bal.Acc', 'Runs'],
                            cellLoc='center',
                            loc='center',
                            cellColours=colors,
                            colWidths=[0.25, 0.15, 0.15, 0.15, 0.15, 0.15])
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.8)
            
            # Add header colors
            for i in range(6):
                table[(0, i)].set_facecolor('#4CAF50')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            ax1.set_title('BraTS18 LGG vs HGG - Complete Baselines', 
                         fontsize=14, fontweight='bold', pad=20)
        
        # UCSF Table
        ax2 = plt.subplot(1, 2, 2)
        ax2.axis('tight')
        ax2.axis('off')
        
        if not ucsf_df.empty:
            table_data = []
            colors = []
            
            for idx, row in ucsf_df.iterrows():
                # Format values
                if row['Status'] == 'Success':
                    auroc = f"{row['Test_AUROC']:.4f}" if pd.notna(row['Test_AUROC']) else '-'
                    ap = f"{row['Test_AP']:.4f}" if pd.notna(row['Test_AP']) else '-'
                    f1 = f"{row['Test_F1']:.4f}" if pd.notna(row['Test_F1']) else '-'
                    bal = f"{row['Test_Balanced_Accuracy']:.4f}" if pd.notna(row['Test_Balanced_Accuracy']) else '-'
                    status_color = 'lightgreen'
                elif row['Status'] == 'Failed':
                    auroc = ap = f1 = bal = 'Failed'
                    status_color = 'lightcoral'
                else:
                    auroc = ap = f1 = bal = 'No data'
                    status_color = 'lightgray'
                
                table_data.append([
                    row['Method'],
                    auroc,
                    ap,
                    f1,
                    bal,
                    f"{row['Successful_Runs']}/{row['Total_Runs']}"
                ])
                
                # Color based on status
                colors.append([status_color if i == 0 else 'white' for i in range(6)])
            
            table = ax2.table(cellText=table_data,
                            colLabels=['Method', 'AUROC', 'AP', 'F1', 'Bal.Acc', 'Runs'],
                            cellLoc='center',
                            loc='center',
                            cellColours=colors,
                            colWidths=[0.25, 0.15, 0.15, 0.15, 0.15, 0.15])
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.8)
            
            # Add header colors
            for i in range(6):
                table[(0, i)].set_facecolor('#2196F3')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            ax2.set_title('UCSF-PDGM IDH - Complete Baselines', 
                         fontsize=14, fontweight='bold', pad=20)
        
        # Overall title
        fig.suptitle('Complete Baseline Analysis - All Methods\n(Including Failed and Missing Runs)', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Add legend
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, fc='lightgreen', label='Successful runs'),
            plt.Rectangle((0, 0), 1, 1, fc='lightcoral', label='Failed/crashed runs'),
            plt.Rectangle((0, 0), 1, 1, fc='lightgray', label='No data available')
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=3, 
                  bbox_to_anchor=(0.5, -0.02))
        
        plt.tight_layout()
        
        # Save figure
        output_file = self.output_path / 'complete_baselines_table.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n  Table visualization saved to: {output_file}")
        
        plt.show()
    
    def generate_latex_tables(self, brats_df: pd.DataFrame, ucsf_df: pd.DataFrame):
        """Generate LaTeX tables with all baselines."""
        latex_output = self.output_path / 'latex_tables'
        latex_output.mkdir(exist_ok=True)
        
        def format_for_latex(df, benchmark_name):
            if df.empty:
                return ""
            
            latex_str = f"% {benchmark_name} - Complete Baselines\n"
            latex_str += "\\begin{table}[h!]\n\\centering\n"
            latex_str += f"\\caption{{{benchmark_name} - All Baselines}}\n"
            latex_str += "\\begin{tabular}{lcccc}\n\\hline\n"
            latex_str += "Method & AUROC & AP & F1 & Bal. Acc. \\\\\n\\hline\n"
            
            for idx, row in df.iterrows():
                if row['Status'] == 'Success':
                    auroc = f"{row['Test_AUROC']:.3f}" if pd.notna(row['Test_AUROC']) else "-"
                    ap = f"{row['Test_AP']:.3f}" if pd.notna(row['Test_AP']) else "-"
                    f1 = f"{row['Test_F1']:.3f}" if pd.notna(row['Test_F1']) else "-"
                    bal = f"{row['Test_Balanced_Accuracy']:.3f}" if pd.notna(row['Test_Balanced_Accuracy']) else "-"
                elif row['Status'] == 'Failed':
                    auroc = ap = f1 = bal = "-"
                else:
                    auroc = ap = f1 = bal = "-"
                
                # Highlight MDAE
                if 'MDAE' in row['Method']:
                    latex_str += f"\\textbf{{{row['Method']}}} & \\textbf{{{auroc}}} & {ap} & {f1} & {bal} \\\\\n"
                else:
                    latex_str += f"{row['Method']} & {auroc} & {ap} & {f1} & {bal} \\\\\n"
            
            latex_str += "\\hline\n\\end{tabular}\n\\end{table}\n"
            return latex_str
        
        # Generate LaTeX
        latex_content = ""
        latex_content += format_for_latex(brats_df, "BraTS18 LGG vs HGG")
        latex_content += "\n\n"
        latex_content += format_for_latex(ucsf_df, "UCSF-PDGM IDH Classification")
        
        # Save
        with open(latex_output / 'complete_baselines.tex', 'w') as f:
            f.write(latex_content)
        
        print(f"  LaTeX tables saved to: {latex_output}")
    
    def save_results(self, brats_df: pd.DataFrame, ucsf_df: pd.DataFrame):
        """Save complete baseline results."""
        if not brats_df.empty:
            brats_df.to_csv(self.output_path / 'brats18_complete_baselines.csv', index=False)
            print(f"  BraTS18 complete baselines saved")
        
        if not ucsf_df.empty:
            ucsf_df.to_csv(self.output_path / 'ucsf_complete_baselines.csv', index=False)
            print(f"  UCSF complete baselines saved")
        
        # Combined summary
        combined = []
        for df, benchmark in [(brats_df, 'BraTS18'), (ucsf_df, 'UCSF')]:
            if not df.empty:
                df_copy = df.copy()
                df_copy['Benchmark'] = benchmark
                combined.append(df_copy)
        
        if combined:
            combined_df = pd.concat(combined, ignore_index=True)
            combined_df.to_csv(self.output_path / 'all_baselines_complete.csv', index=False)
            print(f"  Combined complete baselines saved")
    
    def analyze_all(self):
        """Perform complete baseline analysis."""
        print("\n" + "="*60)
        print("Complete Baseline Analysis - All Methods")
        print("="*60)
        print(f"Input: {self.input_path}")
        print(f"Output: {self.output_path}")
        print("="*60)
        
        # Process BraTS18
        print("\nProcessing BraTS18...")
        brats_dir = self.input_path / 'brats18_lgg_vs_hgg'
        brats_data = self.load_and_categorize_data(brats_dir) if brats_dir.exists() else pd.DataFrame()
        brats_complete = self.extract_all_baselines(brats_data)
        
        # Process UCSF
        print("\nProcessing UCSF...")
        ucsf_dir = self.input_path / 'ucsf_pdgm_idh'
        ucsf_data = self.load_and_categorize_data(ucsf_dir) if ucsf_dir.exists() else pd.DataFrame()
        ucsf_complete = self.extract_all_baselines(ucsf_data)
        
        # Save results
        print("\nSaving results...")
        self.save_results(brats_complete, ucsf_complete)
        
        print("\nGenerating LaTeX tables...")
        self.generate_latex_tables(brats_complete, ucsf_complete)
        
        print("\nCreating visualizations...")
        self.create_complete_table_visualization(brats_complete, ucsf_complete)
        
        print("\n" + "="*60)
        print("Analysis Complete!")
        print("="*60)
        
        # Print summary
        print("\nSummary Statistics:")
        print("-"*60)
        
        for df, name in [(brats_complete, 'BraTS18'), (ucsf_complete, 'UCSF')]:
            if not df.empty:
                total_methods = len(df)
                successful_methods = (df['Status'] == 'Success').sum()
                failed_methods = (df['Status'] == 'Failed').sum()
                no_data_methods = (df['Status'] == 'No data').sum()
                
                print(f"\n{name}:")
                print(f"  Total baselines: {total_methods}")
                print(f"  With successful runs: {successful_methods}")
                print(f"  All runs failed: {failed_methods}")
                print(f"  No data available: {no_data_methods}")
                
                if successful_methods > 0:
                    print(f"\n  Top 3 methods:")
                    top3 = df[df['Status'] == 'Success'].head(3)
                    for idx, row in top3.iterrows():
                        auroc_str = f"{row['Test_AUROC']:.4f}" if pd.notna(row['Test_AUROC']) else 'N/A'
                        print(f"    {row['Method']}: {auroc_str}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Complete baseline analysis for all methods')
    parser.add_argument('--input-dir', type=str, 
                       default='raw_data/20250815_complete_merged',
                       help='Input directory with merged data')
    parser.add_argument('--output-dir', type=str,
                       default='processed_data_20250815_complete',
                       help='Output directory for complete analysis')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = CompleteBaselineAnalyzer(args.input_dir, args.output_dir)
    
    # Run analysis
    analyzer.analyze_all()


if __name__ == "__main__":
    main()