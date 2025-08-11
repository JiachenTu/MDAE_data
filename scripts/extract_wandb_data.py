#!/usr/bin/env python3
"""
WandB Data Extraction Script for MDAE Benchmarking

This script extracts experiment metrics from WandB runs for analysis.
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


class WandBDataExtractor:
    """Extract and process data from WandB runs."""
    
    def __init__(self, entity: str = "t-jiachentu"):
        """
        Initialize the WandB data extractor.
        
        Args:
            entity: WandB entity name
        """
        self.entity = entity
        self.api = wandb.Api()
        self.base_path = Path(__file__).parent.parent
        
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
            
            for run in tqdm(runs, desc=f"Extracting {project}"):
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
                }
                
                # Extract key metrics
                metrics = self._extract_metrics(run)
                run_data['metrics'] = metrics
                
                runs_data.append(run_data)
                
        except Exception as e:
            print(f"Error extracting runs from {project_path}: {e}")
            
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
            'epoch', 'best_epoch'
        ]
        
        for key in metric_keys:
            if key in run.summary:
                metrics[key] = run.summary[key]
        
        # Also check for alternative naming conventions
        for key in run.summary.keys():
            if any(metric in key.lower() for metric in ['acc', 'auc', 'f1', 'precision', 'recall']):
                if key not in metrics:
                    metrics[key] = run.summary[key]
        
        return metrics
    
    def extract_benchmark_data(self, benchmark_path: str, 
                              start_date: Optional[str] = None,
                              end_date: Optional[str] = None) -> Dict:
        """
        Extract all data for a specific benchmark.
        
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
        
        # Extract data for each modality
        all_data = {
            'benchmark': config,
            'extraction_date': datetime.now().isoformat(),
            'modalities': {}
        }
        
        for modality, mod_config in config.get('modalities', {}).items():
            project = mod_config.get('wandb_project')
            if project:
                print(f"\nExtracting data for {modality} from {project}")
                runs_data = self.extract_runs(project, filters)
                all_data['modalities'][modality] = {
                    'config': mod_config,
                    'runs': runs_data
                }
        
        return all_data
    
    def save_extracted_data(self, data: Dict, benchmark_path: str):
        """
        Save extracted data to files.
        
        Args:
            data: Extracted data dictionary
            benchmark_path: Benchmark identifier
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save raw JSON data
        raw_dir = self.base_path / "raw_data" / timestamp / benchmark_path.replace('/', '_')
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
            print(f"\nData saved to {raw_dir}")
        
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
        config_keys = ['model', 'learning_rate', 'batch_size', 'epochs']
        for key in config_keys:
            if key in run.get('config', {}):
                flat[f'config_{key}'] = run['config'][key]
        
        return flat
    
    def extract_all_benchmarks(self, start_date: Optional[str] = None,
                              end_date: Optional[str] = None):
        """
        Extract data for all benchmarks.
        
        Args:
            start_date: Optional start date filter
            end_date: Optional end date filter
        """
        benchmarks_dir = self.base_path / "benchmarks"
        
        for benchmark_file in benchmarks_dir.rglob("*.json"):
            benchmark_path = str(benchmark_file.relative_to(benchmarks_dir).with_suffix(''))
            print(f"\n{'='*60}")
            print(f"Processing benchmark: {benchmark_path}")
            print(f"{'='*60}")
            
            try:
                data = self.extract_benchmark_data(benchmark_path, start_date, end_date)
                self.save_extracted_data(data, benchmark_path)
            except Exception as e:
                print(f"Error processing {benchmark_path}: {e}")


def main():
    """Main function to run the extraction script."""
    parser = argparse.ArgumentParser(description="Extract WandB data for MDAE benchmarking")
    parser.add_argument('--benchmark', type=str, help='Benchmark path (e.g., brats18/lgg_vs_hgg)')
    parser.add_argument('--all', action='store_true', help='Extract all benchmarks')
    parser.add_argument('--start-date', type=str, help='Start date filter (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date filter (YYYY-MM-DD)')
    parser.add_argument('--entity', type=str, default='t-jiachentu', help='WandB entity')
    
    args = parser.parse_args()
    
    # Check for WandB API key
    if not os.environ.get('WANDB_API_KEY'):
        print("Error: WANDB_API_KEY environment variable not set")
        print("Please set it with: export WANDB_API_KEY=your_api_key")
        return
    
    # Initialize extractor
    extractor = WandBDataExtractor(entity=args.entity)
    
    # Run extraction
    if args.all:
        extractor.extract_all_benchmarks(args.start_date, args.end_date)
    elif args.benchmark:
        data = extractor.extract_benchmark_data(args.benchmark, args.start_date, args.end_date)
        extractor.save_extracted_data(data, args.benchmark)
    else:
        print("Please specify --benchmark or --all")
        parser.print_help()


if __name__ == "__main__":
    main()