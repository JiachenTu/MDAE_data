#!/usr/bin/env python3
"""
Discover Multi-Modality WandB Projects
This script searches for all WandB projects matching the pattern july_*_multi_*
"""

import wandb
import re
from pathlib import Path
from typing import List, Dict
import json
from datetime import datetime

class MultiModalityProjectDiscovery:
    """Discover multi-modality projects in WandB."""
    
    def __init__(self, entity: str = "t-jiachentu"):
        """Initialize the discovery tool."""
        self.entity = entity
        self.api = wandb.Api()
        self.base_path = Path(__file__).parent
        
    def discover_projects(self) -> List[Dict]:
        """
        Discover all projects matching july_*_multi_* pattern.
        
        Returns:
            List of project information dictionaries
        """
        print(f"Discovering multi-modality projects for entity: {self.entity}")
        print("=" * 60)
        
        multi_modality_projects = []
        
        try:
            # Get all projects for the entity
            projects = self.api.projects(self.entity)
            
            for project in projects:
                # Check if project name matches multi-modality pattern
                if project.name.startswith('july_') and '_multi_' in project.name:
                    project_info = {
                        'name': project.name,
                        'url': f"https://wandb.ai/{self.entity}/{project.name}",
                        'run_count': project.run_count if hasattr(project, 'run_count') else 'N/A',
                        'created_at': project.created_at if hasattr(project, 'created_at') else 'N/A',
                        'description': project.description if hasattr(project, 'description') else '',
                    }
                    
                    # Extract benchmark and modality info from project name
                    # Pattern: july_[stratified_]benchmark_multi_modalities
                    parts = project.name.split('_')
                    
                    # Try to identify the benchmark name
                    if 'stratified' in parts:
                        stratified_idx = parts.index('stratified')
                        multi_idx = parts.index('multi')
                        benchmark_parts = parts[stratified_idx+1:multi_idx]
                    else:
                        multi_idx = parts.index('multi')
                        benchmark_parts = parts[1:multi_idx]
                    
                    # Identify modality combination
                    modality_parts = parts[multi_idx+1:]
                    
                    project_info['benchmark'] = '_'.join(benchmark_parts)
                    project_info['modalities'] = '_'.join(modality_parts)
                    
                    multi_modality_projects.append(project_info)
                    print(f"Found: {project.name}")
                    
        except Exception as e:
            print(f"Error discovering projects: {e}")
            
        return multi_modality_projects
    
    def save_discovery_results(self, projects: List[Dict]):
        """
        Save discovered projects to a JSON file.
        
        Args:
            projects: List of project information
        """
        output_dir = self.base_path / "multi_modality_discovery"
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"discovered_projects_{timestamp}.json"
        
        discovery_data = {
            'discovery_date': datetime.now().isoformat(),
            'entity': self.entity,
            'total_projects': len(projects),
            'projects': projects
        }
        
        with open(output_file, 'w') as f:
            json.dump(discovery_data, f, indent=2, default=str)
            
        print(f"\nDiscovery results saved to: {output_file}")
        
        # Also create a summary CSV
        if projects:
            import pandas as pd
            df = pd.DataFrame(projects)
            csv_file = output_dir / f"discovered_projects_{timestamp}.csv"
            df.to_csv(csv_file, index=False)
            print(f"CSV summary saved to: {csv_file}")
        
        return output_file
    
    def analyze_discoveries(self, projects: List[Dict]):
        """
        Analyze and summarize discovered projects.
        
        Args:
            projects: List of project information
        """
        print("\n" + "=" * 60)
        print("DISCOVERY SUMMARY")
        print("=" * 60)
        
        print(f"Total multi-modality projects found: {len(projects)}")
        
        if projects:
            # Group by benchmark
            benchmarks = {}
            for project in projects:
                benchmark = project.get('benchmark', 'unknown')
                if benchmark not in benchmarks:
                    benchmarks[benchmark] = []
                benchmarks[benchmark].append(project)
            
            print(f"\nBenchmarks with multi-modality experiments:")
            for benchmark, benchmark_projects in benchmarks.items():
                print(f"\n  {benchmark}:")
                for proj in benchmark_projects:
                    modalities = proj.get('modalities', 'unknown')
                    run_count = proj.get('run_count', 'N/A')
                    print(f"    - Modalities: {modalities} (Runs: {run_count})")
            
            # List unique modality combinations
            modality_combinations = set()
            for project in projects:
                modalities = project.get('modalities', '')
                if modalities:
                    modality_combinations.add(modalities)
            
            print(f"\nUnique modality combinations found:")
            for combo in sorted(modality_combinations):
                print(f"  - {combo}")


def main():
    """Main function to run the discovery."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Discover multi-modality WandB projects"
    )
    parser.add_argument('--entity', type=str, default='t-jiachentu',
                       help='WandB entity name')
    
    args = parser.parse_args()
    
    # Check for WandB authentication
    try:
        wandb.login()
    except Exception as e:
        print(f"Warning: Could not verify WandB authentication: {e}")
        print("Attempting to continue - authentication may be handled by .netrc file")
    
    # Initialize discovery tool
    discovery = MultiModalityProjectDiscovery(entity=args.entity)
    
    # Discover projects
    projects = discovery.discover_projects()
    
    if projects:
        # Save results
        discovery.save_discovery_results(projects)
        
        # Analyze findings
        discovery.analyze_discoveries(projects)
    else:
        print("\nNo multi-modality projects found matching pattern 'july_*_multi_*'")
    
    print("\n" + "=" * 60)
    print("Discovery complete!")


if __name__ == "__main__":
    main()