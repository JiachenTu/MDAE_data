#!/usr/bin/env python3
"""
Consolidate Per-Modality Data for Comprehensive Analysis
=========================================================
This script consolidates the per-modality extracted data into the format
expected by the comprehensive analysis script.
"""

import pandas as pd
from pathlib import Path
import json
from typing import Dict, List
import argparse
from datetime import datetime

def consolidate_benchmark(benchmark_dir: Path) -> pd.DataFrame:
    """
    Consolidate all modality data for a benchmark into a single DataFrame.
    
    Args:
        benchmark_dir: Path to benchmark directory containing modality subdirs
    
    Returns:
        Consolidated DataFrame with all modalities
    """
    all_dfs = []
    
    # Iterate through modality subdirectories
    for modality_dir in benchmark_dir.iterdir():
        if not modality_dir.is_dir():
            continue
            
        modality_name = modality_dir.name
        csv_path = modality_dir / 'runs_summary.csv'
        
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            # Add modality column if not present
            if 'modality' not in df.columns:
                df['modality'] = modality_name
            # Rename 'name' to 'run_name' for compatibility
            if 'name' in df.columns and 'run_name' not in df.columns:
                df['run_name'] = df['name']
            all_dfs.append(df)
            print(f"    Added {len(df)} runs from {modality_name}")
    
    if all_dfs:
        consolidated_df = pd.concat(all_dfs, ignore_index=True)
        return consolidated_df
    else:
        return pd.DataFrame()

def main():
    """Main consolidation function."""
    parser = argparse.ArgumentParser(description='Consolidate per-modality data')
    parser.add_argument('--input-dir', type=Path, default=Path('raw_data/20250815'),
                       help='Input directory with per-modality data')
    parser.add_argument('--output-dir', type=Path, default=Path('raw_data/20250815_consolidated'),
                       help='Output directory for consolidated data')
    
    args = parser.parse_args()
    
    print(f"{'='*60}")
    print(f"CONSOLIDATING MODALITY DATA")
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"{'='*60}\n")
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each benchmark
    benchmark_count = 0
    for benchmark_dir in sorted(args.input_dir.iterdir()):
        if not benchmark_dir.is_dir():
            continue
            
        benchmark_name = benchmark_dir.name
        print(f"Processing {benchmark_name}...")
        
        # Consolidate modality data
        consolidated_df = consolidate_benchmark(benchmark_dir)
        
        if not consolidated_df.empty:
            # Create output directory for benchmark
            output_benchmark_dir = args.output_dir / benchmark_name
            output_benchmark_dir.mkdir(parents=True, exist_ok=True)
            
            # Save consolidated data
            consolidated_df.to_csv(output_benchmark_dir / 'runs_summary.csv', index=False)
            
            # Also save a full_data.json for compatibility
            full_data = {
                'benchmark': benchmark_name,
                'extraction_date': datetime.now().isoformat(),
                'modalities': {},
                'consolidated': True
            }
            
            # Group by modality for the JSON structure
            for modality in consolidated_df['modality'].unique():
                modality_df = consolidated_df[consolidated_df['modality'] == modality]
                full_data['modalities'][modality] = {
                    'runs': modality_df.to_dict('records')
                }
            
            with open(output_benchmark_dir / 'full_data.json', 'w') as f:
                json.dump(full_data, f, indent=2, default=str)
            
            print(f"  ✓ Saved {len(consolidated_df)} total runs")
            benchmark_count += 1
        else:
            print(f"  ⚠ No data found")
    
    print(f"\n{'='*60}")
    print(f"CONSOLIDATION COMPLETE")
    print(f"Processed {benchmark_count} benchmarks")
    print(f"Output saved to: {args.output_dir}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()