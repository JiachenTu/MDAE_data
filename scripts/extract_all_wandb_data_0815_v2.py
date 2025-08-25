#!/usr/bin/env python3
"""
Comprehensive WandB Data Extraction Script - Version 0815_v2

This script extracts ALL data from ALL projects in a WandB account.
No filters, no priorities - complete data extraction.
Modified to output to 0815_v2 folder for the second extraction version.
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
import time
import traceback


class ComprehensiveWandBExtractor:
    """Extract all data from all WandB projects."""
    
    def __init__(self, entity: str = "t-jiachentu", output_base: str = None):
        """
        Initialize the comprehensive WandB data extractor.
        
        Args:
            entity: WandB entity name
            output_base: Base path for output (default: /home/jtu9/Documents/MDAE/MDAE_data/raw_data_full)
        """
        self.entity = entity
        self.api = wandb.Api()
        
        # Set output base path
        if output_base:
            self.output_base = Path(output_base)
        else:
            self.output_base = Path("/home/jtu9/Documents/MDAE/MDAE_data/raw_data_full")
        
        # Use fixed folder name "0815_v2" for version 2
        self.date_str = "0815_v2"
        self.output_dir = self.output_base / self.date_str
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Track extraction statistics
        self.stats = {
            'total_projects': 0,
            'successful_projects': 0,
            'failed_projects': 0,
            'total_runs': 0,
            'extraction_start': datetime.now().isoformat(),
            'extraction_version': '0815_v2',
            'projects': {}
        }
    
    def get_all_projects(self) -> List[str]:
        """
        Get all projects from the WandB account.
        
        Returns:
            List of project names
        """
        print(f"Fetching all projects for entity: {self.entity}")
        projects = []
        
        try:
            # Get all projects for the entity
            for project in self.api.projects(entity=self.entity):
                projects.append(project.name)
                print(f"  Found project: {project.name}")
        except Exception as e:
            print(f"Error fetching projects: {e}")
            
        return projects
    
    def extract_project_data(self, project_name: str) -> Dict:
        """
        Extract all data from a specific project.
        
        Args:
            project_name: Name of the WandB project
        
        Returns:
            Dictionary containing all project data
        """
        project_path = f"{self.entity}/{project_name}"
        print(f"\nExtracting data from project: {project_path}")
        
        project_data = {
            'project_name': project_name,
            'entity': self.entity,
            'extraction_timestamp': datetime.now().isoformat(),
            'extraction_version': '0815_v2',
            'runs': [],
            'project_metadata': {},
            'statistics': {
                'total_runs': 0,
                'completed_runs': 0,
                'failed_runs': 0,
                'running_runs': 0
            }
        }
        
        try:
            # Get project metadata
            try:
                project = self.api.project(name=project_name, entity=self.entity)
                project_data['project_metadata'] = {
                    'created_at': str(project.created_at) if hasattr(project, 'created_at') else None,
                    'updated_at': str(project.updated_at) if hasattr(project, 'updated_at') else None,
                    'description': project.description if hasattr(project, 'description') else None,
                }
            except Exception as e:
                print(f"  Warning: Could not fetch project metadata: {e}")
            
            # Get all runs from the project
            runs = self.api.runs(project_path)
            
            # Process each run
            for run in tqdm(runs, desc=f"Processing {project_name}"):
                run_data = self._extract_run_data(run)
                project_data['runs'].append(run_data)
                
                # Update statistics
                project_data['statistics']['total_runs'] += 1
                if run.state == 'finished':
                    project_data['statistics']['completed_runs'] += 1
                elif run.state == 'failed':
                    project_data['statistics']['failed_runs'] += 1
                elif run.state == 'running':
                    project_data['statistics']['running_runs'] += 1
            
            print(f"  Successfully extracted {project_data['statistics']['total_runs']} runs")
            
        except Exception as e:
            print(f"  Error extracting project data: {e}")
            traceback.print_exc()
            
        return project_data
    
    def _extract_run_data(self, run) -> Dict:
        """
        Extract comprehensive data from a single run.
        
        Args:
            run: WandB run object
        
        Returns:
            Dictionary containing all run data
        """
        run_data = {
            # Basic information
            'id': run.id,
            'name': run.name,
            'state': run.state,
            'created_at': str(run.created_at) if run.created_at else None,
            'updated_at': str(run.updated_at) if hasattr(run, 'updated_at') else None,
            'url': run.url,
            
            # Configuration and parameters
            'config': dict(run.config) if run.config else {},
            
            # All summary metrics (no filtering)
            'summary': dict(run.summary) if run.summary else {},
            
            # Metadata
            'tags': list(run.tags) if run.tags else [],
            'notes': run.notes if run.notes else "",
            'group': run.group if hasattr(run, 'group') else None,
            'job_type': run.job_type if hasattr(run, 'job_type') else None,
            
            # System metrics
            'system_metrics': {},
            
            # User and system info
            'user': run.user.username if hasattr(run, 'user') and run.user else None,
            'host': run.host if hasattr(run, 'host') else None,
        }
        
        # Try to get system metrics
        try:
            if hasattr(run.summary, '_runtime'):
                run_data['system_metrics']['runtime'] = run.summary._runtime
            if hasattr(run.summary, '_step'):
                run_data['system_metrics']['total_steps'] = run.summary._step
        except:
            pass
        
        return run_data
    
    def save_project_data(self, project_name: str, data: Dict):
        """
        Save project data to files.
        
        Args:
            project_name: Name of the project
            data: Project data dictionary
        """
        # Create project directory
        project_dir = self.output_dir / project_name
        project_dir.mkdir(parents=True, exist_ok=True)
        
        # Save full data as JSON
        json_path = project_dir / "full_data.json"
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        print(f"  Saved full data to: {json_path}")
        
        # Save project info separately
        project_info_path = project_dir / "project_info.json"
        project_info = {
            'project_name': data['project_name'],
            'entity': data['entity'],
            'extraction_timestamp': data['extraction_timestamp'],
            'extraction_version': data.get('extraction_version', '0815_v2'),
            'project_metadata': data['project_metadata'],
            'statistics': data['statistics']
        }
        with open(project_info_path, 'w') as f:
            json.dump(project_info, f, indent=2, default=str)
        
        # Create runs summary CSV
        if data['runs']:
            runs_df = self._create_runs_dataframe(data['runs'])
            csv_path = project_dir / "runs_summary.csv"
            runs_df.to_csv(csv_path, index=False)
            print(f"  Saved runs summary to: {csv_path}")
    
    def _create_runs_dataframe(self, runs: List[Dict]) -> pd.DataFrame:
        """
        Create a summary DataFrame from runs data.
        
        Args:
            runs: List of run data dictionaries
        
        Returns:
            DataFrame with flattened run information
        """
        rows = []
        for run in runs:
            row = {
                'run_id': run['id'],
                'run_name': run['name'],
                'state': run['state'],
                'created_at': run['created_at'],
                'url': run['url'],
                'tags': ', '.join(run['tags']) if run['tags'] else '',
                'group': run['group'],
                'user': run['user'],
            }
            
            # Add key config values (flatten nested config)
            for key, value in run.get('config', {}).items():
                if isinstance(value, (str, int, float, bool)):
                    row[f'config_{key}'] = value
                elif isinstance(value, dict) and 'value' in value:
                    row[f'config_{key}'] = value['value']
            
            # Add all summary metrics (flatten)
            for key, value in run.get('summary', {}).items():
                if isinstance(value, (str, int, float, bool)):
                    row[f'metric_{key}'] = value
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def extract_all_projects(self):
        """
        Extract data from all projects in the account.
        """
        print("\n" + "="*60)
        print("Starting comprehensive WandB data extraction - Version 0815_v2")
        print(f"Entity: {self.entity}")
        print(f"Output directory: {self.output_dir}")
        print(f"Extraction started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
        # Get all projects
        projects = self.get_all_projects()
        self.stats['total_projects'] = len(projects)
        
        if not projects:
            print("No projects found!")
            return
        
        print(f"\nFound {len(projects)} projects to extract")
        
        # Extract data from each project
        for i, project_name in enumerate(projects, 1):
            print(f"\n{'='*40}")
            print(f"Project {i}/{len(projects)}: {project_name}")
            print(f"{'='*40}")
            
            try:
                # Extract project data
                project_data = self.extract_project_data(project_name)
                
                # Save project data
                self.save_project_data(project_name, project_data)
                
                # Update statistics
                self.stats['successful_projects'] += 1
                self.stats['total_runs'] += project_data['statistics']['total_runs']
                self.stats['projects'][project_name] = {
                    'status': 'success',
                    'runs': project_data['statistics']['total_runs'],
                    'completed_runs': project_data['statistics']['completed_runs'],
                    'failed_runs': project_data['statistics']['failed_runs']
                }
                
            except Exception as e:
                print(f"  ERROR: Failed to extract project {project_name}: {e}")
                traceback.print_exc()
                self.stats['failed_projects'] += 1
                self.stats['projects'][project_name] = {
                    'status': 'failed',
                    'error': str(e)
                }
            
            # Small delay to avoid API rate limits
            time.sleep(1)
        
        # Save extraction summary
        self.save_extraction_summary()
        
        # Print final summary
        self.print_summary()
    
    def save_extraction_summary(self):
        """Save overall extraction summary."""
        self.stats['extraction_end'] = datetime.now().isoformat()
        
        summary_path = self.output_dir / "extraction_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(self.stats, f, indent=2, default=str)
        
        print(f"\nSaved extraction summary to: {summary_path}")
    
    def print_summary(self):
        """Print extraction summary."""
        print("\n" + "="*60)
        print("EXTRACTION COMPLETE - Version 0815_v2")
        print("="*60)
        print(f"Total projects processed: {self.stats['total_projects']}")
        print(f"Successful: {self.stats['successful_projects']}")
        print(f"Failed: {self.stats['failed_projects']}")
        print(f"Total runs extracted: {self.stats['total_runs']}")
        print(f"Output directory: {self.output_dir}")
        print(f"Extraction ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)


def main():
    """Main function to run the comprehensive extraction."""
    parser = argparse.ArgumentParser(
        description="Extract ALL data from ALL WandB projects - Version 0815_v2"
    )
    parser.add_argument(
        '--entity', 
        type=str, 
        default='t-jiachentu',
        help='WandB entity name'
    )
    parser.add_argument(
        '--output-base',
        type=str,
        default='/home/jtu9/Documents/MDAE/MDAE_data/raw_data_full',
        help='Base directory for output'
    )
    parser.add_argument(
        '--project',
        type=str,
        help='Extract only a specific project (for testing)'
    )
    
    args = parser.parse_args()
    
    # Check for WandB API key or authentication
    try:
        wandb.login()
    except Exception as e:
        print(f"Warning: Could not verify WandB authentication: {e}")
        print("Attempting to continue - authentication may be handled by .netrc file")
    
    # Initialize extractor
    extractor = ComprehensiveWandBExtractor(
        entity=args.entity,
        output_base=args.output_base
    )
    
    # Run extraction
    if args.project:
        # Extract single project (for testing)
        print(f"Extracting single project: {args.project}")
        data = extractor.extract_project_data(args.project)
        extractor.save_project_data(args.project, data)
    else:
        # Extract all projects
        extractor.extract_all_projects()


if __name__ == "__main__":
    main()