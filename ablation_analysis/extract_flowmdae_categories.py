#!/usr/bin/env python3
"""
Extract FlowMDAE Categories into Separate Files
Separates Fixed Masking and Random Sampling FlowMDAE runs
"""

import pandas as pd
import re
from pathlib import Path

def extract_flowmdae_params(notes):
    """Extract FlowMDAE parameters from Notes field"""
    match = re.search(r'RESENC_FLOWMDAE_M(\d+)_N(\d+)', notes)
    if match:
        return {
            'masking_ratio': int(match.group(1)),
            'noise_level': int(match.group(2)),
            'param_combo': f'M{match.group(1)}_N{match.group(2)}'
        }
    return {'masking_ratio': None, 'noise_level': None, 'param_combo': None}

def main():
    """Extract and save FlowMDAE categories"""
    base_path = Path(__file__).parent
    raw_data_path = base_path / 'raw_data_extracted'
    
    print("Extracting FlowMDAE Categories")
    print("="*60)
    
    # 1. Load Random Sampling from flowmdae_raw.csv
    print("\n1. Loading Random Sampling FlowMDAE...")
    random_df = pd.read_csv(raw_data_path / 'flowmdae_raw.csv')
    random_df['category'] = 'Random Sampling'
    random_df['masking_behavior'] = 'Random [1%, M%]'
    print(f"   Found {len(random_df)} random sampling runs")
    print(f"   Benchmarks: {random_df['benchmark'].unique()}")
    
    # 2. Extract Fixed Masking from other_ablations_raw.csv
    print("\n2. Extracting Fixed Masking FlowMDAE...")
    other_df = pd.read_csv(raw_data_path / 'other_ablations_raw.csv')
    
    # Filter for FIXED masking FlowMDAE
    fixed_mask = other_df['notes'].str.contains('FIXED.*Masking.*FLOWMDAE', case=False, na=False)
    fixed_df = other_df[fixed_mask].copy()
    
    # Extract parameters for fixed masking
    params_data = fixed_df['notes'].apply(lambda x: pd.Series(extract_flowmdae_params(x)))
    fixed_df['masking_ratio'] = params_data['masking_ratio']
    fixed_df['noise_level'] = params_data['noise_level']
    fixed_df['param_combo'] = params_data['param_combo']
    fixed_df['category'] = 'Fixed Masking'
    fixed_df['masking_behavior'] = 'Fixed at M%'
    
    # Rename columns to match random_df structure if needed
    if 'noise_level' not in random_df.columns:
        random_df['noise_level'] = random_df.apply(
            lambda row: int(row['param_combo'].split('_N')[1]) if pd.notna(row['param_combo']) else None, 
            axis=1
        )
    
    print(f"   Found {len(fixed_df)} fixed masking runs")
    print(f"   Benchmarks: {fixed_df['benchmark'].nunique()} unique")
    
    # 3. Save separate category files
    print("\n3. Saving extracted data...")
    
    # Save Fixed Masking
    fixed_output_path = raw_data_path / 'flowmdae_fixed_masking.csv'
    fixed_df.to_csv(fixed_output_path, index=False)
    print(f"   Saved fixed masking to: {fixed_output_path}")
    
    # Save Random Sampling (with additional metadata)
    random_output_path = raw_data_path / 'flowmdae_random_sampling.csv'
    random_df.to_csv(random_output_path, index=False)
    print(f"   Saved random sampling to: {random_output_path}")
    
    # 4. Extract BraTS18-specific comparison
    print("\n4. Creating BraTS18-specific comparison...")
    
    # Filter for BraTS18
    brats18_random = random_df[random_df['benchmark'] == 'brats18_lgg_vs_hgg'].copy()
    brats18_fixed = fixed_df[fixed_df['benchmark'] == 'brats18_lgg_vs_hgg'].copy()
    
    print(f"   BraTS18 Random: {len(brats18_random)} runs")
    print(f"   BraTS18 Fixed: {len(brats18_fixed)} runs")
    
    # Combine for comparison
    brats18_combined = pd.concat([brats18_random, brats18_fixed], ignore_index=True)
    
    # Save BraTS18 comparison
    brats18_output_path = raw_data_path / 'flowmdae_brats18_comparison.csv'
    brats18_combined.to_csv(brats18_output_path, index=False)
    print(f"   Saved BraTS18 comparison to: {brats18_output_path}")
    
    # 5. Print summary statistics
    print("\n" + "="*60)
    print("EXTRACTION SUMMARY")
    print("="*60)
    
    # Overall statistics
    print("\nOverall Statistics:")
    print(f"  Total Fixed Masking: {len(fixed_df)}")
    print(f"  Total Random Sampling: {len(random_df)}")
    print(f"  Total Combined: {len(fixed_df) + len(random_df)}")
    
    # Parameter coverage
    print("\nParameter Coverage:")
    print(f"  Fixed - Unique combinations: {fixed_df['param_combo'].nunique()}")
    print(f"  Random - Unique combinations: {random_df['param_combo'].nunique()}")
    
    # BraTS18 specific
    print("\nBraTS18 Dataset:")
    print(f"  Fixed runs: {len(brats18_fixed)}")
    print(f"  Random runs: {len(brats18_random)}")
    
    if len(brats18_fixed) > 0:
        print(f"  Fixed modalities: {sorted(brats18_fixed['modality'].unique())}")
    if len(brats18_random) > 0:
        print(f"  Random modalities: {sorted(brats18_random['modality'].unique())}")
    
    # Performance preview
    print("\nPerformance Preview (Mean AUROC):")
    if len(fixed_df) > 0:
        print(f"  Fixed Masking: {fixed_df['test_auroc'].mean():.4f} ± {fixed_df['test_auroc'].std():.4f}")
    if len(random_df) > 0:
        print(f"  Random Sampling: {random_df['test_auroc'].mean():.4f} ± {random_df['test_auroc'].std():.4f}")
    
    if len(brats18_fixed) > 0 and len(brats18_random) > 0:
        print(f"\nBraTS18 Performance:")
        print(f"  Fixed: {brats18_fixed['test_auroc'].mean():.4f} ± {brats18_fixed['test_auroc'].std():.4f}")
        print(f"  Random: {brats18_random['test_auroc'].mean():.4f} ± {brats18_random['test_auroc'].std():.4f}")
    
    print("\nExtraction complete!")
    
    return fixed_df, random_df, brats18_combined

if __name__ == "__main__":
    main()