#!/usr/bin/env python3
"""
Extract Continue Pretrained and Vanilla RESENC Runs from WandB

This script extracts runs with CONT_PRETRAINED_RESENC_* patterns (continue pretrained from MDAE)
and their vanilla RESENC_* counterparts for comparison.
"""

import os
import json
import pandas as pd
import wandb
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
from collections import defaultdict


class ContPretrainedExtractor:
    """Extract continue pretrained and vanilla runs from WandB."""
    
    def __init__(self, entity: str = "t-jiachentu"):
        """Initialize the extractor."""
        self.entity = entity
        self.api = wandb.Api()
        self.base_path = Path(__file__).parent.parent
        self.benchmarks = [
            'brats18_lgg_vs_hgg',
            'brats23_gli_vs_men',
            'brats23_gli_vs_met', 
            'brats23_men_vs_met',
            'rsna_miccai_mgmt_methylation',
            'tcga_gbm_dss_1year',
            'tcga_gbm_pfi_1year',
            'ucsf_pdgm_idh_classification',
            'upenn_gbm_age_group',
            'upenn_gbm_gender',
            'upenn_gbm_gtr_status',
            'upenn_gbm_idh1_status',
            'upenn_gbm_survival_18month',
            'upenn_gbm_survival_1year',
            'upenn_gbm_survival_2year'
        ]
        
    def extract_runs_with_notes(self, project: str) -> List[Dict]:
        """
        Extract runs from a project, including the Notes field.
        
        Args:
            project: WandB project name
            
        Returns:
            List of run dictionaries with Notes field
        """
        runs_data = []
        project_path = f"{self.entity}/{project}"
        
        try:
            runs = self.api.runs(project_path)
            
            for run in runs:
                # Only include finished runs
                if run.state != 'finished':
                    continue
                    
                run_data = {
                    'id': run.id,
                    'name': run.name,
                    'state': run.state,
                    'created_at': run.created_at,
                    'project': project,
                    'Notes': run.notes if run.notes else '',  # Extract Notes field
                    'config': dict(run.config),
                    'summary': dict(run.summary),
                    'tags': run.tags,
                    'url': run.url
                }
                
                # Extract key metrics
                metrics = self._extract_metrics(run)
                run_data['metrics'] = metrics
                
                runs_data.append(run_data)
                
        except Exception as e:
            print(f"Error extracting from {project_path}: {e}")
            
        return runs_data
    
    def _extract_metrics(self, run) -> Dict:
        """Extract relevant metrics from a run."""
        metrics = {}
        summary = run.summary
        
        # Extract test metrics
        metric_keys = ['AUROC', 'AP', 'F1', 'Balanced_Accuracy']
        for key in metric_keys:
            test_key = f'Test/{key}'
            val_key = f'Val/{key}'
            
            if test_key in summary:
                metrics[f'Test_{key}'] = summary[test_key]
            if val_key in summary:
                metrics[f'Val_{key}'] = summary[val_key]
                
        # Extract epoch
        if 'epoch' in summary:
            metrics['epoch'] = summary['epoch']
            
        return metrics
    
    def extract_all_benchmarks(self) -> Dict:
        """
        Extract runs from all benchmarks.
        
        Returns:
            Dictionary with benchmark data
        """
        all_data = {
            'extraction_date': datetime.now().isoformat(),
            'benchmarks': {}
        }
        
        # Process each benchmark
        for benchmark in tqdm(self.benchmarks, desc="Processing benchmarks"):
            print(f"\nProcessing {benchmark}...")
            benchmark_data = {
                'runs': [],
                'modalities': ['t1', 't1gd', 't2', 'flair']
            }
            
            # Try different project patterns
            project_patterns = [
                f"july_stratified_{benchmark}_single_{{modality}}",
                f"july_{benchmark}_single_{{modality}}",
                f"stratified_{benchmark}_single_{{modality}}"
            ]
            
            for modality in benchmark_data['modalities']:
                for pattern in project_patterns:
                    project = pattern.format(modality=modality)
                    runs = self.extract_runs_with_notes(project)
                    
                    if runs:
                        print(f"  Found {len(runs)} runs in {project}")
                        # Add modality info to each run
                        for run in runs:
                            run['modality'] = modality
                            run['benchmark'] = benchmark
                        benchmark_data['runs'].extend(runs)
                        break  # Found runs, no need to try other patterns
            
            all_data['benchmarks'][benchmark] = benchmark_data
            
        return all_data
    
    def categorize_runs(self, all_data: Dict) -> Dict:
        """
        Categorize runs into continue pretrained and vanilla.
        
        Args:
            all_data: All extracted data
            
        Returns:
            Dictionary with categorized runs
        """
        categorized = {
            'cont_pretrained': [],
            'vanilla': [],
            'other': [],
            'statistics': {}
        }
        
        # Process each benchmark
        for benchmark, benchmark_data in all_data['benchmarks'].items():
            for run in benchmark_data['runs']:
                notes = run.get('Notes', '')
                
                # Check for CONT_PRETRAINED_RESENC pattern
                if 'CONT_PRETRAINED_RESENC' in notes:
                    categorized['cont_pretrained'].append(run)
                # Check for vanilla RESENC (without CONT_PRETRAINED)
                elif 'RESENC' in notes and 'CONT_PRETRAINED' not in notes:
                    categorized['vanilla'].append(run)
                # For vanilla MAE, check for resenc_pretrained pattern
                elif 'resenc_pretrained' in notes.lower():
                    categorized['vanilla'].append(run)
                else:
                    categorized['other'].append(run)
        
        # Calculate statistics
        cont_methods = defaultdict(int)
        vanilla_methods = defaultdict(int)
        
        for run in categorized['cont_pretrained']:
            notes = run.get('Notes', '')
            if 'CONT_PRETRAINED_RESENC_' in notes:
                method = notes.split('CONT_PRETRAINED_RESENC_')[1].split()[0]
                cont_methods[method] += 1
        
        for run in categorized['vanilla']:
            notes = run.get('Notes', '')
            if 'RESENC_' in notes:
                method = notes.split('RESENC_')[1].split()[0]
                vanilla_methods[method] += 1
            elif 'resenc_pretrained' in notes.lower():
                vanilla_methods['MAE'] += 1
        
        categorized['statistics'] = {
            'total_cont_pretrained': len(categorized['cont_pretrained']),
            'total_vanilla': len(categorized['vanilla']),
            'total_other': len(categorized['other']),
            'cont_pretrained_methods': dict(cont_methods),
            'vanilla_methods': dict(vanilla_methods)
        }
        
        return categorized
    
    def save_results(self, all_data: Dict, categorized: Dict):
        """Save extraction results."""
        output_dir = self.base_path / 'raw_data'
        output_dir.mkdir(exist_ok=True)
        
        # Save all data
        with open(output_dir / 'all_runs_with_notes.json', 'w') as f:
            json.dump(all_data, f, indent=2, default=str)
        
        # Save categorized data
        with open(output_dir / 'categorized_runs.json', 'w') as f:
            json.dump(categorized, f, indent=2, default=str)
        
        # Create DataFrames for analysis
        if categorized['cont_pretrained']:
            cont_df = pd.json_normalize(categorized['cont_pretrained'])
            cont_df.to_csv(output_dir / 'cont_pretrained_runs.csv', index=False)
            
        if categorized['vanilla']:
            vanilla_df = pd.json_normalize(categorized['vanilla'])
            vanilla_df.to_csv(output_dir / 'vanilla_resenc_runs.csv', index=False)
        
        # Print summary
        print("\n" + "="*60)
        print("EXTRACTION SUMMARY")
        print("="*60)
        stats = categorized['statistics']
        print(f"Total continue pretrained runs: {stats['total_cont_pretrained']}")
        print(f"Total vanilla RESENC runs: {stats['total_vanilla']}")
        print(f"Total other runs: {stats['total_other']}")
        
        print("\nContinue Pretrained Methods:")
        for method, count in sorted(stats['cont_pretrained_methods'].items()):
            print(f"  - {method}: {count} runs")
        
        print("\nVanilla Methods:")
        for method, count in sorted(stats['vanilla_methods'].items()):
            print(f"  - {method}: {count} runs")
        
        print(f"\nResults saved to: {output_dir}")


def main():
    """Main execution function."""
    print("Continue Pretraining Data Extraction")
    print("="*60)
    
    # Initialize extractor
    extractor = ContPretrainedExtractor()
    
    # Extract all data
    print("\nExtracting data from WandB...")
    all_data = extractor.extract_all_benchmarks()
    
    # Categorize runs
    print("\nCategorizing runs...")
    categorized = extractor.categorize_runs(all_data)
    
    # Save results
    print("\nSaving results...")
    extractor.save_results(all_data, categorized)
    
    print("\nExtraction complete!")


if __name__ == "__main__":
    main()