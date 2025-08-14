#!/usr/bin/env python3
"""
Multi-Modality WandB Data Extraction Script
Extracts data from all july_*_multi_* projects for comprehensive multi-modality analysis
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


class MultiModalityDataExtractor:
    """Extract data from multi-modality WandB projects."""
    
    # Discovered multi-modality projects
    MULTI_MODALITY_PROJECTS = {
        'brats18_lgg_vs_hgg': {
            'project': 'july_stratified_brats18_lgg_vs_hgg_multi_flair_t1_t1ce_t2',
            'modalities': ['flair', 't1', 't1ce', 't2'],
            'benchmark_name': 'BraTS18 LGG vs HGG (Multi-Modality)'
        },
        'brats23_gli_vs_men': {
            'project': 'july_stratified_brats23_gli_vs_men_multi_t1n_t1c_t2w_t2f',
            'modalities': ['t1n', 't1c', 't2w', 't2f'],
            'benchmark_name': 'BraTS23 Glioma vs Meningioma (Multi-Modality)'
        },
        'brats23_gli_vs_met': {
            'project': 'july_stratified_brats23_gli_vs_met_multi_t1n_t1c_t2w_t2f',
            'modalities': ['t1n', 't1c', 't2w', 't2f'],
            'benchmark_name': 'BraTS23 Glioma vs Metastasis (Multi-Modality)'
        },
        'brats23_men_vs_met': {
            'project': 'july_stratified_brats23_men_vs_met_multi_t1n_t1c_t2w_t2f',
            'modalities': ['t1n', 't1c', 't2w', 't2f'],
            'benchmark_name': 'BraTS23 Meningioma vs Metastasis (Multi-Modality)'
        },
        'rsna_miccai_mgmt': {
            'project': 'july_stratified_rsna_miccai_brain_tumor_mgmt_methylation_multi_t1w_t1wce_t2w_flair',
            'modalities': ['t1w', 't1wce', 't2w', 'flair'],
            'benchmark_name': 'RSNA-MICCAI MGMT Methylation (Multi-Modality)'
        },
        'ucsf_pdgm_idh': {
            'project': 'july_stratified_ucsf_pdgm_idh_classification_multi_t1_t1c_t2_flair_swi_asl',
            'modalities': ['t1', 't1c', 't2', 'flair', 'swi', 'asl'],
            'benchmark_name': 'UCSF-PDGM IDH Classification (Multi-Modality)'
        },
        'tcga_gbm_dss_1year': {
            'project': 'july_stratified_tcga_gbm_dss_1year_multi_t1_t1gd_t2_flair',
            'modalities': ['t1', 't1gd', 't2', 'flair'],
            'benchmark_name': 'TCGA-GBM DSS 1-Year (Multi-Modality)'
        },
        'tcga_gbm_pfi_1year': {
            'project': 'july_stratified_tcga_gbm_pfi_1year_multi_t1_t1gd_t2_flair',
            'modalities': ['t1', 't1gd', 't2', 'flair'],
            'benchmark_name': 'TCGA-GBM PFI 1-Year (Multi-Modality)'
        },
        'upenn_gbm_age_group': {
            'project': 'july_stratified_upenn_gbm_age_group_multi_t1_t1gd_t2_flair',
            'modalities': ['t1', 't1gd', 't2', 'flair'],
            'benchmark_name': 'UPenn-GBM Age Group (Multi-Modality)'
        },
        'upenn_gbm_gender': {
            'project': 'july_stratified_upenn_gbm_gender_multi_t1_t1gd_t2_flair',
            'modalities': ['t1', 't1gd', 't2', 'flair'],
            'benchmark_name': 'UPenn-GBM Gender (Multi-Modality)'
        },
        'upenn_gbm_gtr_status': {
            'project': 'july_stratified_upenn_gbm_gtr_status_multi_t1_t1gd_t2_flair',
            'modalities': ['t1', 't1gd', 't2', 'flair'],
            'benchmark_name': 'UPenn-GBM GTR Status (Multi-Modality)'
        },
        'upenn_gbm_idh1_status': {
            'project': 'july_stratified_upenn_gbm_idh1_status_multi_t1_t1gd_t2_flair',
            'modalities': ['t1', 't1gd', 't2', 'flair'],
            'benchmark_name': 'UPenn-GBM IDH1 Status (Multi-Modality)'
        },
        'upenn_gbm_survival_18month': {
            'project': 'july_stratified_upenn_gbm_survival_18month_multi_t1_t1gd_t2_flair',
            'modalities': ['t1', 't1gd', 't2', 'flair'],
            'benchmark_name': 'UPenn-GBM 18-Month Survival (Multi-Modality)'
        },
        'upenn_gbm_survival_1year': {
            'project': 'july_stratified_upenn_gbm_survival_1year_multi_t1_t1gd_t2_flair',
            'modalities': ['t1', 't1gd', 't2', 'flair'],
            'benchmark_name': 'UPenn-GBM 1-Year Survival (Multi-Modality)'
        },
        'upenn_gbm_survival_2year': {
            'project': 'july_stratified_upenn_gbm_survival_2year_multi_t1_t1gd_t2_flair',
            'modalities': ['t1', 't1gd', 't2', 'flair'],
            'benchmark_name': 'UPenn-GBM 2-Year Survival (Multi-Modality)'
        }
    }
    
    def __init__(self, entity: str = "t-jiachentu", output_date: str = None):
        """
        Initialize the multi-modality data extractor.
        
        Args:
            entity: WandB entity name
            output_date: Date folder for output (default: today's date)
        """
        self.entity = entity
        self.api = wandb.Api()
        self.base_path = Path(__file__).parent
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
            
            for run in tqdm(runs, desc=f"Extracting {project}", position=None, leave=False):
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
            safe_print(f"Error extracting runs from {project_path}: {e}")
            
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
    
    def extract_benchmark_data(self, benchmark_key: str, benchmark_info: Dict,
                              start_date: Optional[str] = None,
                              end_date: Optional[str] = None) -> Dict:
        """
        Extract all data for a specific multi-modality benchmark.
        
        Args:
            benchmark_key: Benchmark identifier
            benchmark_info: Benchmark configuration
            start_date: Optional start date filter (YYYY-MM-DD)
            end_date: Optional end date filter (YYYY-MM-DD)
        
        Returns:
            Dictionary containing all extracted data
        """
        # Prepare filters
        filters = {}
        if start_date:
            filters['created_at'] = {'$gte': start_date}
        if end_date:
            if 'created_at' in filters:
                filters['created_at']['$lte'] = end_date
            else:
                filters['created_at'] = {'$lte': end_date}
        
        # Extract data
        all_data = {
            'benchmark': benchmark_key,
            'benchmark_name': benchmark_info['benchmark_name'],
            'project': benchmark_info['project'],
            'modalities': benchmark_info['modalities'],
            'extraction_date': datetime.now().isoformat(),
            'runs': []
        }
        
        safe_print(f"  Extracting from {benchmark_info['project']}")
        runs_data = self.extract_runs(benchmark_info['project'], filters)
        all_data['runs'] = runs_data
        
        return all_data
    
    def save_extracted_data(self, data: Dict, benchmark_key: str):
        """
        Save extracted data to files.
        
        Args:
            data: Extracted data dictionary
            benchmark_key: Benchmark identifier
        """
        # Use date folder for multi-modality
        raw_dir = self.base_path / "raw_data" / f"{self.output_date}_multi" / benchmark_key
        raw_dir.mkdir(parents=True, exist_ok=True)
        
        # Save full JSON data
        with open(raw_dir / "full_data.json", 'w') as f:
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
            df.to_csv(raw_dir / "runs_summary.csv", index=False)
            safe_print(f"  ✓ Data saved to {raw_dir}")
        else:
            safe_print(f"  ⚠ No runs found for {benchmark_key}")
        
        return raw_dir
    
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
        }
        
        # Add metrics
        for key, value in run.get('metrics', {}).items():
            flat[f'metric_{key}'] = value
        
        # Add selected config values
        config_keys = ['model', 'learning_rate', 'batch_size', 'epochs', 'method', 'trainer']
        for key in config_keys:
            if key in run.get('config', {}):
                flat[f'config_{key}'] = run['config'][key]
        
        return flat
    
    def process_benchmark(self, benchmark_key: str, benchmark_info: Dict,
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None) -> tuple:
        """
        Process a single benchmark.
        
        Args:
            benchmark_key: Benchmark identifier
            benchmark_info: Benchmark configuration
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            Tuple of (benchmark_key, success_status, message)
        """
        try:
            data = self.extract_benchmark_data(benchmark_key, benchmark_info, start_date, end_date)
            
            if data['runs']:
                self.save_extracted_data(data, benchmark_key)
                return (benchmark_key, True, f"Success - {len(data['runs'])} runs")
            else:
                return (benchmark_key, False, "No runs found")
                
        except Exception as e:
            return (benchmark_key, False, f"Error: {e}")
    
    def extract_all_benchmarks_parallel(self, start_date: Optional[str] = None,
                                       end_date: Optional[str] = None,
                                       max_workers: int = 3):
        """
        Extract data for all multi-modality benchmarks in parallel.
        
        Args:
            start_date: Optional start date filter
            end_date: Optional end date filter
            max_workers: Maximum number of parallel workers
        """
        safe_print(f"\n{'='*60}")
        safe_print(f"Multi-Modality Data Extraction")
        safe_print(f"Output folder: raw_data/{self.output_date}_multi/")
        safe_print(f"Benchmarks to process: {len(self.MULTI_MODALITY_PROJECTS)}")
        safe_print(f"Max parallel workers: {max_workers}")
        safe_print(f"{'='*60}\n")
        
        # Track results
        results = []
        
        # Process benchmarks in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_benchmark = {
                executor.submit(self.process_benchmark, key, info, start_date, end_date): key
                for key, info in self.MULTI_MODALITY_PROJECTS.items()
            }
            
            # Process completed tasks
            with tqdm(total=len(self.MULTI_MODALITY_PROJECTS), desc="Overall Progress") as pbar:
                for future in as_completed(future_to_benchmark):
                    benchmark_key = future_to_benchmark[future]
                    
                    try:
                        result = future.result()
                        results.append(result)
                        
                        if result[1]:  # Success
                            safe_print(f"✓ Completed: {result[0]} - {result[2]}")
                        else:
                            safe_print(f"⚠ Failed: {result[0]} - {result[2]}")
                            
                    except Exception as e:
                        safe_print(f"✗ Error: {benchmark_key} - {e}")
                        results.append((benchmark_key, False, str(e)))
                    
                    pbar.update(1)
        
        # Print summary
        self._print_summary(results)
    
    def _print_summary(self, results: List[tuple]):
        """Print extraction summary."""
        safe_print(f"\n{'='*60}")
        safe_print("Extraction Summary:")
        safe_print(f"{'='*60}")
        
        successful = [r for r in results if r[1]]
        failed = [r for r in results if not r[1]]
        
        safe_print(f"✓ Successfully extracted: {len(successful)} benchmarks")
        safe_print(f"⚠ Failed: {len(failed)} benchmarks")
        
        if successful:
            safe_print("\nSuccessful extractions:")
            for benchmark, _, message in successful:
                safe_print(f"  - {benchmark}: {message}")
        
        if failed:
            safe_print("\nFailed extractions:")
            for benchmark, _, message in failed:
                safe_print(f"  - {benchmark}: {message}")
        
        safe_print(f"\nAll data saved to: raw_data/{self.output_date}_multi/")
        safe_print(f"{'='*60}\n")


def main():
    """Main function to run the multi-modality extraction."""
    parser = argparse.ArgumentParser(
        description="Extract WandB data for multi-modality projects"
    )
    parser.add_argument('--start-date', type=str, help='Start date filter (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date filter (YYYY-MM-DD)')
    parser.add_argument('--entity', type=str, default='t-jiachentu', help='WandB entity')
    parser.add_argument('--output-date', type=str, default=None,
                       help='Output date folder (default: today)')
    parser.add_argument('--max-workers', type=int, default=3,
                       help='Maximum number of parallel workers (default: 3)')
    
    args = parser.parse_args()
    
    # Check for WandB authentication
    try:
        wandb.login()
    except Exception as e:
        print(f"Warning: Could not verify WandB authentication: {e}")
        print("Attempting to continue - authentication may be handled by .netrc file")
    
    # Initialize extractor
    extractor = MultiModalityDataExtractor(entity=args.entity, output_date=args.output_date)
    
    # Run parallel extraction
    extractor.extract_all_benchmarks_parallel(
        args.start_date, 
        args.end_date,
        args.max_workers
    )


if __name__ == "__main__":
    main()