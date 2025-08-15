#!/usr/bin/env python3
"""
Parallel WandB Data Extraction Script for july_stratified Projects
Extracts all july_stratified project data in parallel and saves to 20250813 folder
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


class ParallelWandBExtractor:
    """Extract data from WandB runs in parallel for july_stratified projects."""
    
    def __init__(self, entity: str = "t-jiachentu", output_date: str = "20250813"):
        """
        Initialize the parallel WandB data extractor.
        
        Args:
            entity: WandB entity name
            output_date: Date folder for output (default: 20250813)
        """
        self.entity = entity
        self.api = wandb.Api()
        self.base_path = Path(__file__).parent
        self.output_date = output_date
        
    def load_benchmark_config(self, benchmark_path: str) -> Dict:
        """
        Load benchmark configuration from JSON file.
        
        Args:
            benchmark_path: Path to benchmark JSON file (e.g., 'brats18/lgg_vs_hgg')
        
        Returns:
            Dictionary containing benchmark configuration
        """
        config_path = self.base_path / "benchmarks" / f"{benchmark_path}.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Benchmark config not found: {config_path}")
        
        with open(config_path, 'r') as f:
            return json.load(f)
    
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
            
            # Create a progress bar for this specific project
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
            'Test_AUROC', 'Test_AP', 'Test_F1', 'Test_Balanced_Accuracy'
        ]
        
        for key in metric_keys:
            if key in run.summary:
                metrics[key] = run.summary[key]
        
        # Also check for alternative naming conventions
        for key in run.summary.keys():
            if any(metric in key.lower() for metric in ['acc', 'auc', 'auroc', 'f1', 'precision', 'recall', 'ap']):
                if key not in metrics:
                    metrics[key] = run.summary[key]
        
        return metrics
    
    def extract_benchmark_data(self, benchmark_path: str, 
                              start_date: Optional[str] = None,
                              end_date: Optional[str] = None) -> Dict:
        """
        Extract all data for a specific benchmark (only july_stratified projects).
        
        Args:
            benchmark_path: Path to benchmark (e.g., 'brats18/lgg_vs_hgg')
            start_date: Optional start date filter (YYYY-MM-DD)
            end_date: Optional end date filter (YYYY-MM-DD)
        
        Returns:
            Dictionary containing all extracted data
        """
        # Load benchmark configuration
        config = self.load_benchmark_config(benchmark_path)
        
        # Prepare filters
        filters = {}
        if start_date:
            filters['created_at'] = {'$gte': start_date}
        if end_date:
            if 'created_at' in filters:
                filters['created_at']['$lte'] = end_date
            else:
                filters['created_at'] = {'$lte': end_date}
        
        # Extract data for each modality (only july_stratified projects)
        all_data = {
            'benchmark': config,
            'extraction_date': datetime.now().isoformat(),
            'modalities': {}
        }
        
        for modality, mod_config in config.get('modalities', {}).items():
            project = mod_config.get('wandb_project')
            # Only process july_stratified projects
            if project and project.startswith('july_stratified'):
                safe_print(f"  Extracting {modality} from {project}")
                runs_data = self.extract_runs(project, filters)
                all_data['modalities'][modality] = {
                    'config': mod_config,
                    'runs': runs_data
                }
        
        return all_data
    
    def save_extracted_data(self, data: Dict, benchmark_path: str):
        """
        Save extracted data to files in the 20250813 folder.
        
        Args:
            data: Extracted data dictionary
            benchmark_path: Benchmark identifier
        """
        # Use fixed date folder (20250813)
        raw_dir = self.base_path / "raw_data" / self.output_date / benchmark_path.replace('/', '_')
        raw_dir.mkdir(parents=True, exist_ok=True)
        
        with open(raw_dir / "full_data.json", 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        # Convert to DataFrame and save as CSV
        dfs = []
        for modality, mod_data in data['modalities'].items():
            for run in mod_data['runs']:
                flat_run = self._flatten_run_data(run)
                flat_run['modality'] = modality
                dfs.append(flat_run)
        
        if dfs:
            df = pd.DataFrame(dfs)
            df.to_csv(raw_dir / "runs_summary.csv", index=False)
            safe_print(f"  ✓ Data saved to {raw_dir}")
        else:
            safe_print(f"  ⚠ No runs found for {benchmark_path}")
        
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
    
    def process_benchmark(self, benchmark_file: Path, start_date: Optional[str] = None,
                         end_date: Optional[str] = None) -> tuple:
        """
        Process a single benchmark file.
        
        Args:
            benchmark_file: Path to benchmark JSON file
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            Tuple of (benchmark_path, success_status, message)
        """
        benchmark_path = str(benchmark_file.relative_to(self.base_path / "benchmarks").with_suffix(''))
        
        try:
            data = self.extract_benchmark_data(benchmark_path, start_date, end_date)
            
            # Only save if we have modality data
            if data['modalities']:
                self.save_extracted_data(data, benchmark_path)
                return (benchmark_path, True, "Success")
            else:
                return (benchmark_path, False, "No july_stratified projects found")
                
        except Exception as e:
            return (benchmark_path, False, f"Error: {e}")
    
    def extract_all_benchmarks_parallel(self, start_date: Optional[str] = None,
                                       end_date: Optional[str] = None,
                                       max_workers: int = 5):
        """
        Extract data for all benchmarks in parallel.
        
        Args:
            start_date: Optional start date filter
            end_date: Optional end date filter
            max_workers: Maximum number of parallel workers
        """
        benchmarks_dir = self.base_path / "benchmarks"
        benchmark_files = list(benchmarks_dir.rglob("*.json"))
        
        safe_print(f"\n{'='*60}")
        safe_print(f"Starting parallel extraction for {len(benchmark_files)} benchmarks")
        safe_print(f"Output folder: raw_data/{self.output_date}/")
        safe_print(f"Max parallel workers: {max_workers}")
        safe_print(f"{'='*60}\n")
        
        # Track results
        results = []
        
        # Process benchmarks in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_benchmark = {
                executor.submit(self.process_benchmark, bf, start_date, end_date): bf
                for bf in benchmark_files
            }
            
            # Process completed tasks
            with tqdm(total=len(benchmark_files), desc="Overall Progress") as pbar:
                for future in as_completed(future_to_benchmark):
                    benchmark_file = future_to_benchmark[future]
                    benchmark_name = benchmark_file.stem
                    
                    try:
                        result = future.result()
                        results.append(result)
                        
                        if result[1]:  # Success
                            safe_print(f"✓ Completed: {result[0]}")
                        else:
                            safe_print(f"⚠ Skipped: {result[0]} - {result[2]}")
                            
                    except Exception as e:
                        safe_print(f"✗ Failed: {benchmark_name} - {e}")
                        results.append((benchmark_name, False, str(e)))
                    
                    pbar.update(1)
        
        # Print summary
        safe_print(f"\n{'='*60}")
        safe_print("Extraction Summary:")
        safe_print(f"{'='*60}")
        
        successful = [r for r in results if r[1]]
        failed = [r for r in results if not r[1]]
        
        safe_print(f"✓ Successfully extracted: {len(successful)} benchmarks")
        safe_print(f"⚠ Skipped/Failed: {len(failed)} benchmarks")
        
        if failed:
            safe_print("\nSkipped/Failed benchmarks:")
            for benchmark, _, message in failed:
                safe_print(f"  - {benchmark}: {message}")
        
        safe_print(f"\nAll data saved to: raw_data/{self.output_date}/")
        safe_print(f"{'='*60}\n")


def main():
    """Main function to run the parallel extraction script."""
    parser = argparse.ArgumentParser(
        description="Extract WandB data for july_stratified projects in parallel"
    )
    parser.add_argument('--start-date', type=str, help='Start date filter (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date filter (YYYY-MM-DD)')
    parser.add_argument('--entity', type=str, default='t-jiachentu', help='WandB entity')
    parser.add_argument('--output-date', type=str, default='20250813', 
                       help='Output date folder (default: 20250813)')
    parser.add_argument('--max-workers', type=int, default=5,
                       help='Maximum number of parallel workers (default: 5)')
    
    args = parser.parse_args()
    
    # Check for WandB authentication
    try:
        # This will use .netrc file or WANDB_API_KEY if available
        wandb.login()
    except Exception as e:
        print(f"Warning: Could not verify WandB authentication: {e}")
        print("Attempting to continue - authentication may be handled by .netrc file")
    
    # Initialize extractor
    extractor = ParallelWandBExtractor(entity=args.entity, output_date=args.output_date)
    
    # Run parallel extraction
    extractor.extract_all_benchmarks_parallel(
        args.start_date, 
        args.end_date,
        args.max_workers
    )


if __name__ == "__main__":
    main()