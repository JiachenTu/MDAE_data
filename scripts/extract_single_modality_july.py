#!/usr/bin/env python3
"""
Extract Single-Modality Data from july_stratified Projects
===========================================================
This script extracts single-modality data from july_stratified projects
for all benchmarks to update the key benchmarks analysis.
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
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Thread-safe print lock
print_lock = threading.Lock()

def safe_print(message):
    """Thread-safe print function"""
    with print_lock:
        print(message)

# Define all benchmark-modality combinations
BENCHMARK_MODALITIES = {
    # T1/T2 benchmarks (in-distribution)
    'brats18_lgg_vs_hgg': ['t1', 't2', 't1ce', 'flair'],
    'brats23_gli_vs_men': ['t1n', 't1c', 't2w', 't2f'],
    'brats23_gli_vs_met': ['t1n', 't1c', 't2w', 't2f'],
    'brats23_men_vs_met': ['t1n', 't1c', 't2w', 't2f'],
    'rsna_miccai_mgmt_methylation': ['t1w', 't1wce', 't2w', 'flair'],
    'upenn_gbm_survival_18month': ['t1', 't1gd', 't2', 'flair'],
    'upenn_gbm_survival_1year': ['t1', 't1gd', 't2', 'flair'],
    'upenn_gbm_survival_2year': ['t1', 't1gd', 't2', 'flair'],
    'upenn_gbm_idh1_status': ['t1', 't1gd', 't2', 'flair'],
    
    # Generalization benchmarks (out-of-distribution)
    'ucsf_pdgm_idh_classification': ['t1', 't1c', 't2', 'flair', 'asl', 'swi', 'adc', 'dwi'],
    'upenn_gbm_age_group': ['t1', 't1gd', 't2', 'flair'],
    'upenn_gbm_gender': ['t1', 't1gd', 't2', 'flair'],
    'upenn_gbm_gtr_status': ['t1', 't1gd', 't2', 'flair'],
    'tcga_gbm_dss_1year': ['t1', 't1gd', 't2', 'flair'],  # mixed_contrasts handled separately
    'tcga_gbm_pfi_1year': ['t1', 't1gd', 't2', 'flair'],  # mixed_contrasts handled separately
}

class SingleModalityExtractor:
    """Extract single-modality data from july_stratified WandB projects."""
    
    def __init__(self, entity: str = "t-jiachentu", output_date: str = None):
        """
        Initialize the extractor.
        
        Args:
            entity: WandB entity name
            output_date: Date folder for output (default: today)
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
            runs = self.api.runs(project_path, filters=filters)
            
            for run in runs:
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
                    'notes': run.notes,
                }
                
                # Extract key metrics
                metrics = self._extract_metrics(run)
                run_data['metrics'] = metrics
                
                runs_data.append(run_data)
                
        except Exception as e:
            safe_print(f"  Warning: Could not extract from {project}: {e}")
            
        return runs_data
    
    def _extract_metrics(self, run) -> Dict:
        """Extract relevant metrics from a run."""
        metrics = {}
        
        # Common metric names to extract
        metric_keys = [
            'test/accuracy', 'test/auc', 'test/f1_score',
            'test/precision', 'test/recall', 'test/sensitivity',
            'test/specificity', 'val/best_accuracy', 'val/best_auc',
            'epoch', 'best_epoch',
            'Test_AUROC', 'Test_AP', 'Test_F1', 'Test_Balanced_Accuracy',
            'Test/AUROC', 'Test/AP', 'Test/F1', 'Test/Balanced_Accuracy',
            'Val_AUROC', 'Val_AP', 'Val_F1', 'Val_Balanced_Accuracy'
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
    
    def extract_benchmark_modality(self, benchmark: str, modality: str) -> Dict:
        """
        Extract data for a specific benchmark-modality combination.
        
        Args:
            benchmark: Benchmark name
            modality: Modality name
        
        Returns:
            Dictionary containing extracted data
        """
        # Construct project name
        project_name = f"july_stratified_{benchmark}_single_{modality}"
        
        safe_print(f"  Extracting {project_name}...")
        
        runs_data = self.extract_runs(project_name)
        
        return {
            'benchmark': benchmark,
            'modality': modality,
            'project': project_name,
            'extraction_date': datetime.now().isoformat(),
            'runs': runs_data,
            'run_count': len(runs_data)
        }
    
    def save_benchmark_data(self, data: Dict, benchmark: str, modality: str):
        """
        Save extracted data to files.
        
        Args:
            data: Extracted data
            benchmark: Benchmark name
            modality: Modality name
        """
        # Create output directory structure
        output_dir = self.base_path / "raw_data" / self.output_date / benchmark / modality
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save full JSON data
        with open(output_dir / "full_data.json", 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        # Convert runs to DataFrame and save as CSV
        if data['runs']:
            runs_df = pd.DataFrame(data['runs'])
            
            # Flatten metrics into columns
            if 'metrics' in runs_df.columns:
                metrics_df = pd.json_normalize(runs_df['metrics'])
                runs_df = pd.concat([runs_df.drop('metrics', axis=1), metrics_df], axis=1)
            
            runs_df.to_csv(output_dir / "runs_summary.csv", index=False)
            
            safe_print(f"    ✓ Saved {len(data['runs'])} runs to {output_dir}")
        else:
            safe_print(f"    ⚠ No runs found for {benchmark}/{modality}")
    
    def process_benchmark(self, benchmark: str, modalities: List[str]) -> List[tuple]:
        """
        Process all modalities for a benchmark.
        
        Args:
            benchmark: Benchmark name
            modalities: List of modalities
        
        Returns:
            List of (modality, success, message) tuples
        """
        results = []
        
        for modality in modalities:
            try:
                data = self.extract_benchmark_modality(benchmark, modality)
                if data['runs']:
                    self.save_benchmark_data(data, benchmark, modality)
                    results.append((modality, True, f"{data['run_count']} runs"))
                else:
                    results.append((modality, False, "No runs found"))
            except Exception as e:
                results.append((modality, False, str(e)))
        
        return results
    
    def extract_all_benchmarks(self, max_workers: int = 3):
        """
        Extract data for all benchmarks and modalities.
        
        Args:
            max_workers: Maximum number of parallel workers
        """
        safe_print(f"\n{'='*60}")
        safe_print(f"Single-Modality Data Extraction")
        safe_print(f"Output folder: raw_data/{self.output_date}/")
        safe_print(f"Benchmarks to process: {len(BENCHMARK_MODALITIES)}")
        safe_print(f"{'='*60}\n")
        
        # Process benchmarks
        overall_results = {}
        
        for benchmark, modalities in BENCHMARK_MODALITIES.items():
            safe_print(f"\nProcessing {benchmark}...")
            results = self.process_benchmark(benchmark, modalities)
            overall_results[benchmark] = results
        
        # Print summary
        self._print_summary(overall_results)
    
    def _print_summary(self, results: Dict[str, List[tuple]]):
        """Print extraction summary."""
        safe_print(f"\n{'='*60}")
        safe_print("Extraction Summary:")
        safe_print(f"{'='*60}")
        
        total_success = 0
        total_failed = 0
        
        for benchmark, modality_results in results.items():
            successful = sum(1 for _, success, _ in modality_results if success)
            failed = sum(1 for _, success, _ in modality_results if not success)
            
            total_success += successful
            total_failed += failed
            
            safe_print(f"\n{benchmark}:")
            safe_print(f"  ✓ Successful: {successful} modalities")
            if failed > 0:
                safe_print(f"  ⚠ Failed: {failed} modalities")
            
            for modality, success, message in modality_results:
                status = "✓" if success else "✗"
                safe_print(f"    {status} {modality}: {message}")
        
        safe_print(f"\n{'='*60}")
        safe_print(f"Total: {total_success} successful, {total_failed} failed")
        safe_print(f"Data saved to: raw_data/{self.output_date}/")
        safe_print(f"{'='*60}\n")


def main():
    """Main function to run the extraction."""
    parser = argparse.ArgumentParser(
        description="Extract single-modality data from july_stratified WandB projects"
    )
    parser.add_argument('--entity', type=str, default='t-jiachentu', 
                       help='WandB entity')
    parser.add_argument('--output-date', type=str, default=None,
                       help='Output date folder (default: today)')
    parser.add_argument('--max-workers', type=int, default=3,
                       help='Maximum parallel workers')
    
    args = parser.parse_args()
    
    # Check for WandB authentication
    try:
        wandb.login()
    except Exception as e:
        print(f"Warning: Could not verify WandB authentication: {e}")
        print("Attempting to continue - authentication may be handled by .netrc file")
    
    # Initialize extractor
    extractor = SingleModalityExtractor(entity=args.entity, output_date=args.output_date)
    
    # Run extraction
    extractor.extract_all_benchmarks(max_workers=args.max_workers)


if __name__ == "__main__":
    main()