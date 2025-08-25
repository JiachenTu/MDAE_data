#!/usr/bin/env python3
"""
Extract and Merge Multi-Modality Data from Both Project Variants
================================================================
This script extracts data from both 'july_stratified_...' and 'stratified_...' 
projects and merges them for each benchmark.

Specifically handles:
- BraTS18: july_stratified_brats18_... and stratified_brats18_...
- UCSF: july_stratified_ucsf_... and stratified_ucsf_...
"""

import os
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import wandb
from tqdm import tqdm

class MergedBenchmarkExtractor:
    """Extract and merge data from both project variants for each benchmark."""
    
    # Define benchmark configurations with both project variants
    BENCHMARK_CONFIGS = {
        'brats18_lgg_vs_hgg': {
            'projects': [
                'july_stratified_brats18_lgg_vs_hgg_multi_flair_t1_t1ce_t2',
                'stratified_brats18_lgg_vs_hgg_multi_flair_t1_t1ce_t2'
            ],
            'modalities': ['flair', 't1', 't1ce', 't2'],
            'benchmark_name': 'BraTS18 LGG vs HGG (Multi-Modality Merged)'
        },
        'ucsf_pdgm_idh': {
            'projects': [
                'july_stratified_ucsf_pdgm_idh_classification_multi_t1_t1c_t2_flair_swi_asl',
                'stratified_ucsf_pdgm_idh_classification_multi_t1_t1c_t2_flair_adc_dwi'  # Different modalities!
            ],
            'modalities': ['t1', 't1c', 't2', 'flair', 'swi', 'asl', 'adc', 'dwi'],  # Combined modalities
            'benchmark_name': 'UCSF-PDGM IDH Classification (Multi-Modality Merged)'
        }
    }
    
    def __init__(self, entity: str = "t-jiachentu", output_date: str = None):
        """
        Initialize the merged extractor.
        
        Args:
            entity: WandB entity name
            output_date: Date folder for output (default: today's date)
        """
        self.entity = entity
        self.api = wandb.Api()
        self.base_path = Path(__file__).parent.parent
        self.output_date = output_date or datetime.now().strftime("%Y%m%d")
        
    def extract_runs(self, project: str, filters: Optional[Dict] = None) -> List[Dict]:
        """
        Extract runs from a WandB project.
        
        Args:
            project: WandB project name
            filters: Optional filters for runs
        
        Returns:
            List of run data dictionaries
        """
        runs_data = []
        project_path = f"{self.entity}/{project}"
        
        try:
            print(f"    Extracting from {project}...")
            runs = self.api.runs(project_path, filters=filters)
            
            for run in tqdm(runs, desc=f"      Processing runs", leave=False):
                run_data = {
                    'id': run.id,
                    'name': run.name,
                    'state': run.state,
                    'created_at': run.created_at,
                    'config': dict(run.config),
                    'summary': dict(run.summary),
                    'tags': run.tags,
                    'url': run.url,
                    'project': project,
                    'project_variant': 'july' if project.startswith('july_') else 'standard',
                    'notes': run.notes,
                }
                
                # Extract key metrics
                metrics = self._extract_metrics(run)
                run_data['metrics'] = metrics
                
                runs_data.append(run_data)
            
            print(f"      Found {len(runs_data)} runs")
                
        except Exception as e:
            print(f"      Error extracting runs from {project_path}: {e}")
            
        return runs_data
    
    def _extract_metrics(self, run) -> Dict:
        """
        Extract relevant metrics from a run.
        
        Args:
            run: WandB run object
        
        Returns:
            Dictionary of extracted metrics
        """
        metrics = {}
        
        # Common metric names to extract
        metric_keys = [
            'test/accuracy', 'test/auc', 'test/f1_score',
            'test/precision', 'test/recall', 'test/sensitivity',
            'test/specificity', 'val/best_accuracy', 'val/best_auc',
            'epoch', 'best_epoch',
            'Test_AUROC', 'Test_AP', 'Test_F1', 'Test_Balanced_Accuracy',
            'Test/AUROC', 'Test/AP', 'Test/F1', 'Test/Balanced_Accuracy',
            'Val_AUROC', 'Val_AP', 'Val_F1', 'Val_Balanced_Accuracy',
            'Val/AUROC', 'Val/AP', 'Val/F1', 'Val/Balanced_Accuracy'
        ]
        
        for key in metric_keys:
            if key in run.summary:
                metrics[key] = run.summary[key]
        
        # Also check for alternative naming conventions
        for key in run.summary.keys():
            if any(metric in key.lower() for metric in ['acc', 'auc', 'auroc', 'f1', 'precision', 'recall', 'ap', 'balanced']):
                if key not in metrics:
                    metrics[key] = run.summary[key]
        
        return metrics
    
    def extract_and_merge_benchmark(self, benchmark_key: str, benchmark_config: Dict) -> Dict:
        """
        Extract and merge data from all project variants for a benchmark.
        
        Args:
            benchmark_key: Benchmark identifier
            benchmark_config: Benchmark configuration with project list
        
        Returns:
            Dictionary containing merged data
        """
        print(f"\n{'='*60}")
        print(f"Processing: {benchmark_key}")
        print(f"{'='*60}")
        
        all_runs = []
        project_stats = {}
        
        # Extract from each project variant
        for project in benchmark_config['projects']:
            print(f"\n  Project: {project}")
            runs = self.extract_runs(project)
            
            if runs:
                all_runs.extend(runs)
                project_stats[project] = {
                    'run_count': len(runs),
                    'variant': 'july' if project.startswith('july_') else 'standard'
                }
        
        # Create merged dataset
        merged_data = {
            'benchmark': benchmark_key,
            'benchmark_name': benchmark_config['benchmark_name'],
            'projects': benchmark_config['projects'],
            'modalities': benchmark_config['modalities'],
            'extraction_date': datetime.now().isoformat(),
            'project_stats': project_stats,
            'total_runs': len(all_runs),
            'runs': all_runs
        }
        
        print(f"\n  Summary:")
        print(f"    Total runs extracted: {len(all_runs)}")
        for project, stats in project_stats.items():
            variant_label = "(july variant)" if stats['variant'] == 'july' else "(standard variant)"
            print(f"    - {project} {variant_label}: {stats['run_count']} runs")
        
        return merged_data
    
    def save_merged_data(self, data: Dict, benchmark_key: str):
        """
        Save merged data to files.
        
        Args:
            data: Merged data dictionary
            benchmark_key: Benchmark identifier
        """
        # Create output directory
        output_dir = self.base_path / "raw_data" / f"{self.output_date}_merged" / benchmark_key
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save full JSON data
        with open(output_dir / "merged_data.json", 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        # Convert to DataFrame and save as CSV
        if data['runs']:
            dfs = []
            for run in data['runs']:
                flat_run = self._flatten_run_data(run)
                flat_run['benchmark'] = benchmark_key
                flat_run['modalities'] = '_'.join(data['modalities'])
                dfs.append(flat_run)
            
            df = pd.DataFrame(dfs)
            
            # Save merged CSV
            df.to_csv(output_dir / "merged_runs.csv", index=False)
            
            # Also save separate CSVs for each project variant
            july_df = df[df['project_variant'] == 'july']
            standard_df = df[df['project_variant'] == 'standard']
            
            if not july_df.empty:
                july_df.to_csv(output_dir / "july_variant_runs.csv", index=False)
            if not standard_df.empty:
                standard_df.to_csv(output_dir / "standard_variant_runs.csv", index=False)
            
            print(f"\n  ✓ Data saved to {output_dir}")
            
            # Print statistics
            print(f"\n  Run Statistics by Method:")
            method_stats = self._analyze_methods(df)
            for method, stats in method_stats.items():
                print(f"    {method}: {stats['total']} runs (july: {stats['july']}, standard: {stats['standard']})")
        else:
            print(f"  ⚠ No runs found for {benchmark_key}")
        
        return output_dir
    
    def _flatten_run_data(self, run: Dict) -> Dict:
        """
        Flatten nested run data for DataFrame conversion.
        
        Args:
            run: Run data dictionary
        
        Returns:
            Flattened dictionary
        """
        flat = {
            'run_id': run['id'],
            'run_name': run['name'],
            'state': run['state'],
            'created_at': run['created_at'],
            'url': run['url'],
            'project': run['project'],
            'project_variant': run['project_variant'],
        }
        
        # Add metrics with standardized names
        metrics = run.get('metrics', {})
        
        # Standardize metric names
        metric_mapping = {
            'Test/AUROC': 'metric_Test_AUROC',
            'Test_AUROC': 'metric_Test_AUROC',
            'Test/AP': 'metric_Test_AP',
            'Test_AP': 'metric_Test_AP',
            'Test/F1': 'metric_Test_F1',
            'Test_F1': 'metric_Test_F1',
            'Test/Balanced_Accuracy': 'metric_Test_Balanced_Accuracy',
            'Test_Balanced_Accuracy': 'metric_Test_Balanced_Accuracy',
        }
        
        for original_key, standard_key in metric_mapping.items():
            if original_key in metrics and standard_key not in flat:
                flat[standard_key] = metrics[original_key]
        
        # Add any other metrics
        for key, value in metrics.items():
            metric_key = f'metric_{key.replace("/", "_")}'
            if metric_key not in flat:
                flat[metric_key] = value
        
        # Add selected config values
        config = run.get('config', {})
        config_keys = ['model', 'learning_rate', 'batch_size', 'epochs', 'method', 'trainer']
        for key in config_keys:
            if key in config:
                flat[f'config_{key}'] = config[key]
        
        # Extract method from run name
        run_name = run['name']
        if run_name.startswith('resenc_'):
            flat['method_extracted'] = run_name.split('_multi_')[0] if '_multi_' in run_name else run_name
        else:
            flat['method_extracted'] = run_name.split('_multi_')[0] if '_multi_' in run_name else run_name.split('_')[0]
        
        return flat
    
    def _analyze_methods(self, df: pd.DataFrame) -> Dict:
        """
        Analyze method distribution across project variants.
        
        Args:
            df: DataFrame with run data
        
        Returns:
            Dictionary with method statistics
        """
        method_stats = {}
        
        for method in df['method_extracted'].unique():
            method_df = df[df['method_extracted'] == method]
            july_count = len(method_df[method_df['project_variant'] == 'july'])
            standard_count = len(method_df[method_df['project_variant'] == 'standard'])
            
            method_stats[method] = {
                'total': len(method_df),
                'july': july_count,
                'standard': standard_count
            }
        
        return dict(sorted(method_stats.items(), key=lambda x: x[1]['total'], reverse=True))
    
    def extract_all_benchmarks(self):
        """Extract and merge data for all configured benchmarks."""
        print(f"\n{'='*60}")
        print(f"Multi-Modality Merged Data Extraction")
        print(f"Output folder: raw_data/{self.output_date}_merged/")
        print(f"Benchmarks to process: {len(self.BENCHMARK_CONFIGS)}")
        print(f"{'='*60}")
        
        results = []
        
        for benchmark_key, benchmark_config in self.BENCHMARK_CONFIGS.items():
            try:
                data = self.extract_and_merge_benchmark(benchmark_key, benchmark_config)
                output_dir = self.save_merged_data(data, benchmark_key)
                results.append((benchmark_key, True, f"Success - {data['total_runs']} total runs"))
            except Exception as e:
                print(f"\n  ✗ Error processing {benchmark_key}: {e}")
                results.append((benchmark_key, False, str(e)))
        
        # Print summary
        self._print_summary(results)
    
    def _print_summary(self, results: List[tuple]):
        """Print extraction summary."""
        print(f"\n{'='*60}")
        print("Extraction Summary:")
        print(f"{'='*60}")
        
        successful = [r for r in results if r[1]]
        failed = [r for r in results if not r[1]]
        
        print(f"✓ Successfully extracted: {len(successful)} benchmarks")
        print(f"⚠ Failed: {len(failed)} benchmarks")
        
        if successful:
            print("\nSuccessful extractions:")
            for benchmark, _, message in successful:
                print(f"  - {benchmark}: {message}")
        
        if failed:
            print("\nFailed extractions:")
            for benchmark, _, message in failed:
                print(f"  - {benchmark}: {message}")
        
        print(f"\nAll merged data saved to: raw_data/{self.output_date}_merged/")
        print(f"{'='*60}\n")


def main():
    """Main function to run the merged extraction."""
    parser = argparse.ArgumentParser(
        description="Extract and merge WandB data from both project variants"
    )
    parser.add_argument('--entity', type=str, default='t-jiachentu', help='WandB entity')
    parser.add_argument('--output-date', type=str, default=None,
                       help='Output date folder (default: today)')
    
    args = parser.parse_args()
    
    # Check for WandB authentication
    try:
        wandb.login()
    except Exception as e:
        print(f"Warning: Could not verify WandB authentication: {e}")
        print("Attempting to continue - authentication may be handled by .netrc file")
    
    # Initialize extractor
    extractor = MergedBenchmarkExtractor(entity=args.entity, output_date=args.output_date)
    
    # Run extraction
    extractor.extract_all_benchmarks()


if __name__ == "__main__":
    main()