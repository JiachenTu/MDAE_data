#!/usr/bin/env python3
"""
Check all MDAE-related patterns in the raw data to fix missing results.
"""

import pandas as pd
from pathlib import Path
import re

def find_all_mdae_patterns():
    """Find all unique MDAE-related patterns in the data."""
    
    data_dir = Path('/home/t-jiachentu/repos/benchmarking/misc/data/raw_data/20250811')
    
    all_patterns = set()
    mdae_runs = []
    
    for benchmark_dir in data_dir.iterdir():
        if benchmark_dir.is_dir():
            csv_path = benchmark_dir / 'runs_summary.csv'
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                
                # Find all MDAE-related runs
                for _, row in df.iterrows():
                    run_name = row['run_name']
                    
                    # Check for MDAE patterns
                    if 'MDAE' in run_name or 'mdae' in run_name.lower():
                        # Extract pattern (remove timestamp)
                        pattern = re.sub(r'_\d{4}-\d{2}-\d{2}_.*$', '', run_name)
                        all_patterns.add(pattern)
                        
                        mdae_runs.append({
                            'benchmark': benchmark_dir.name,
                            'run_name': run_name,
                            'pattern': pattern,
                            'auroc': row.get('Test_AUROC', None),
                            'modality': row.get('modality', None)
                        })
                    
                    # Also check for time-conditioned variants
                    if 'time_conditioned' in run_name or 'multimodal_mm_mdae' in run_name:
                        pattern = re.sub(r'_\d{4}-\d{2}-\d{2}_.*$', '', run_name)
                        all_patterns.add(pattern)
    
    print("=" * 60)
    print("ALL UNIQUE MDAE-RELATED PATTERNS FOUND:")
    print("=" * 60)
    
    for pattern in sorted(all_patterns):
        if 'MDAE' in pattern or 'mdae' in pattern.lower() or 'time_conditioned' in pattern:
            print(f"  - {pattern}")
    
    print("\n" + "=" * 60)
    print("MDAE RUNS WITH HIGH AUROC (>0.8) THAT MIGHT BE MISSED:")
    print("=" * 60)
    
    # Find high-performing MDAE runs
    mdae_df = pd.DataFrame(mdae_runs)
    if not mdae_df.empty:
        high_perf = mdae_df[mdae_df['auroc'] > 0.8].sort_values('auroc', ascending=False)
        
        for _, row in high_perf.head(20).iterrows():
            print(f"  {row['benchmark']:30s} | {row['modality']:10s} | AUROC={row['auroc']:.3f} | {row['pattern']}")
    
    print("\n" + "=" * 60)
    print("RECOMMENDED PATTERN UPDATES:")
    print("=" * 60)
    
    # Analyze patterns that need to be added
    current_patterns = {
        'MDAE': r'^resenc_MDAETrainer_RandomMask_Flow_BS48_2000ep_pretrained',
        'MDAE (TC)': r'^(resenc_time_conditioned|resenc_multimodal_mm_mdae)',
    }
    
    missing_patterns = []
    
    for pattern in all_patterns:
        if 'MDAE' in pattern:
            # Check if this pattern would be matched by current regex
            matched = False
            for method, regex in current_patterns.items():
                if re.match(regex, pattern):
                    matched = True
                    break
            
            if not matched:
                missing_patterns.append(pattern)
    
    if missing_patterns:
        print("The following MDAE patterns are NOT currently matched:")
        for pattern in sorted(missing_patterns):
            print(f"  - {pattern}")
        
        print("\nSuggested pattern update:")
        print("  'MDAE': r'^(resenc_MDAETrainer_RandomMask_Flow_BS48_2000ep_pretrained|resenc_MDAE_pretrained)'")
    else:
        print("All MDAE patterns are properly matched.")
    
    return all_patterns, mdae_df


if __name__ == "__main__":
    patterns, mdae_df = find_all_mdae_patterns()
    
    # Save detailed analysis
    output_path = Path('/home/t-jiachentu/repos/benchmarking/misc/data/processed_data/mdae_pattern_analysis.csv')
    if not mdae_df.empty:
        mdae_df.to_csv(output_path, index=False)
        print(f"\nDetailed analysis saved to: {output_path}")