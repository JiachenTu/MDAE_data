#!/usr/bin/env python3
"""
Regenerate CSV files with enhanced format for MDAE data analysis.

This script improves the CSV format by:
- Removing unnecessary columns (url, created_at, state, unwanted test metrics)
- Adding key test metrics (Test_Balanced_Accuracy, Test_F1, Test_AUROC, Test_AP)
- Adding wandb_notes column extracted from actual WandB notes
- Better column organization for analysis
"""

import json
import pandas as pd
import os
from pathlib import Path
from typing import Dict, List



def process_benchmark_csv(benchmark_dir: Path) -> bool:
    """
    Process and enhance CSV file for a benchmark.
    
    Args:
        benchmark_dir: Path to benchmark directory
        
    Returns:
        True if successful, False otherwise
    """
    csv_path = benchmark_dir / "runs_summary.csv"
    json_path = benchmark_dir / "full_data.json"
    
    if not csv_path.exists() or not json_path.exists():
        print(f"Missing files in {benchmark_dir}")
        return False
    
    try:
        # Load existing CSV
        df = pd.read_csv(csv_path)
        
        # Load JSON for additional metrics
        with open(json_path, 'r') as f:
            full_data = json.load(f)
        
        # Create enhanced dataframe
        enhanced_rows = []
        
        for _, row in df.iterrows():
            # Find corresponding run in JSON data
            run_id = row['run_id']
            run_data = None
            
            for modality_data in full_data.get('modalities', {}).values():
                for run in modality_data.get('runs', []):
                    if run['id'] == run_id:
                        run_data = run
                        break
                if run_data:
                    break
            
            if not run_data:
                continue
                
            # Extract all available test metrics
            # Test metrics are stored in the summary section, not metrics section
            summary = run_data.get('summary', {})
            metrics = run_data.get('metrics', {})  # Keep for backward compatibility
            config = run_data.get('config', {})
            
            # Build enhanced row
            enhanced_row = {
                'run_id': row['run_id'],
                'run_name': row['run_name'],
                'project': row['project'],
                'modality': row.get('modality', 'unknown'),
                'model': config.get('model', row.get('config_model', 'unknown')),
                'learning_rate': config.get('learning_rate', row.get('config_learning_rate', None)),
                'batch_size': config.get('batch_size', row.get('config_batch_size', None)),
                'epochs': config.get('epochs', row.get('config_epochs', None)),
                'epoch': summary.get('epoch', metrics.get('epoch', row.get('metric_epoch', None))),
            }
            
            # Add test metrics - updated to match WandB format exactly (excluding unwanted columns)
            test_metrics = [
                ('Test_Balanced_Accuracy', ['Test/Balanced_Accuracy']),
                ('Test_F1', ['Test/F1']),
                ('Test_AUROC', ['Test/AUROC']),
                ('Test_AP', ['Test/AP']),
            ]
            
            for metric_name, possible_keys in test_metrics:
                value = None
                for key in possible_keys:
                    # First check summary section (where WandB stores final metrics)
                    if key in summary:
                        value = summary[key]
                        break
                    # Then check metrics section for backward compatibility
                    if key in metrics:
                        value = metrics[key]
                        break
                    # Also check with metric_ prefix from original CSV
                    metric_key = f'metric_{key}'
                    if metric_key in row:
                        value = row[metric_key]
                        break
                enhanced_row[metric_name] = value
            
            # Add validation metrics - updated to match WandB format
            val_metrics = [
                ('Val_Balanced_Accuracy', ['Val/Balanced_Accuracy']),
                ('Val_F1', ['Val/F1']),
                ('Val_AUROC', ['Val/AUROC']),
                ('Val_AP', ['Val/AP']),
            ]
            
            for metric_name, possible_keys in val_metrics:
                value = None
                for key in possible_keys:
                    # First check summary section (where WandB stores final metrics)
                    if key in summary:
                        value = summary[key]
                        break
                    # Then check metrics section for backward compatibility
                    if key in metrics:
                        value = metrics[key]
                        break
                    # Also check with metric_ prefix from original CSV
                    metric_key = f'metric_{key}'
                    if metric_key in row:
                        value = row[metric_key]
                        break
                enhanced_row[metric_name] = value
            
            # Extract actual WandB notes from run data
            enhanced_row['wandb_notes'] = run_data.get('notes', '')
            
            enhanced_rows.append(enhanced_row)
        
        # Create new DataFrame
        enhanced_df = pd.DataFrame(enhanced_rows)
        
        # Reorder columns for better readability - updated metric names (removed unwanted columns)
        column_order = [
            'run_id', 'run_name', 'project', 'modality', 'model', 
            'learning_rate', 'batch_size', 'epochs', 'epoch',
            'Test_Balanced_Accuracy', 'Test_F1', 'Test_AUROC', 'Test_AP',
            'Val_Balanced_Accuracy', 'Val_F1', 'Val_AUROC', 'Val_AP',
            'wandb_notes'
        ]
        
        # Only include columns that exist
        available_columns = [col for col in column_order if col in enhanced_df.columns]
        enhanced_df = enhanced_df[available_columns]
        
        # Save enhanced CSV
        enhanced_df.to_csv(csv_path, index=False)
        
        # Print summary of extracted metrics
        test_metrics_found = sum(1 for col in ['Test_Balanced_Accuracy', 'Test_F1', 'Test_AUROC', 'Test_AP'] 
                               if col in enhanced_df.columns and enhanced_df[col].notna().sum() > 0)
        print(f"Enhanced CSV saved for {benchmark_dir.name}: {len(enhanced_df)} rows, {test_metrics_found}/4 test metrics populated")
        
        return True
        
    except Exception as e:
        print(f"Error processing {benchmark_dir}: {e}")
        return False


def main():
    """Main function to regenerate all CSV files."""
    base_path = Path(__file__).parent.parent
    data_dir = base_path / "raw_data" / "20250811"
    
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return
    
    print("Regenerating CSV files with enhanced format...")
    print("=" * 60)
    
    successful = 0
    total = 0
    
    for benchmark_dir in data_dir.iterdir():
        if benchmark_dir.is_dir():
            total += 1
            print(f"\nProcessing: {benchmark_dir.name}")
            if process_benchmark_csv(benchmark_dir):
                successful += 1
            else:
                print(f"Failed to process: {benchmark_dir.name}")
    
    print("\n" + "=" * 60)
    print(f"Regeneration complete: {successful}/{total} benchmarks processed successfully")


if __name__ == "__main__":
    main()