#!/usr/bin/env python3
"""
Extract FlowMDAE ablation metrics for contour analysis.
Processes masking ratio (M) and noise corruption level (N) experiments.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import re

def parse_config_from_notes(notes):
    """Extract M and N values from notes string."""
    pattern = r'RESENC_FLOWMDAE_M(\d+)_N(\d+)'
    match = re.search(pattern, notes)
    if match:
        m_val = int(match.group(1))
        n_val = int(match.group(2))
        return m_val, n_val
    return None, None

def extract_modality_from_notes(notes):
    """Extract modality from notes string."""
    if 'with t1ce' in notes:
        return 'T1CE'
    elif 'with t1' in notes:
        return 'T1'
    elif 'with t2' in notes:
        return 'T2'
    elif 'with flair' in notes:
        return 'FLAIR'
    return None

def main():
    # Load data
    data_path = Path('/home/t-jiachentu/repos/benchmarking/misc/data/raw_data/20250813/brats18_lgg_vs_hgg/full_data.json')
    output_dir = Path('/home/t-jiachentu/repos/benchmarking/misc/data/ablation/contour_analysis/data')
    
    print("Loading data...")
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # Initialize storage for metrics
    # Structure: {modality: {(M, N): {metrics}}}
    ablation_data = defaultdict(lambda: defaultdict(dict))
    
    # Process each modality
    for modality_key, modality_data in data['modalities'].items():
        for run in modality_data['runs']:
            # Check if this is an ablation run
            notes = run.get('notes', '')
            if 'FlowMDAE FIXED Masking Ablation' not in notes:
                continue
            
            # Parse configuration
            m_val, n_val = parse_config_from_notes(notes)
            modality = extract_modality_from_notes(notes)
            
            if m_val is None or n_val is None or modality is None:
                continue
            
            # Extract metrics
            metrics = run.get('metrics', {})
            summary = run.get('summary', {})
            
            # Combine metrics and summary for comprehensive data
            all_metrics = {**metrics, **summary}
            
            # Extract relevant metrics
            extracted = {
                'run_id': run['id'],
                'run_name': run['name'],
                'state': run['state'],
                # Test metrics
                'Test_AUROC': all_metrics.get('Test_AUROC') or all_metrics.get('Test/AUROC'),
                'Test_AP': all_metrics.get('Test_AP') or all_metrics.get('Test/AP'),
                'Test_F1': all_metrics.get('Test_F1') or all_metrics.get('Test/F1'),
                'Test_Balanced_Accuracy': all_metrics.get('Test_Balanced_Accuracy') or all_metrics.get('Test/Balanced_Accuracy'),
                # Validation metrics
                'Val_AUROC': all_metrics.get('Val_AUROC') or all_metrics.get('Val/AUROC'),
                'Val_AP': all_metrics.get('Val_AP') or all_metrics.get('Val/AP'),
                'Val_F1': all_metrics.get('Val_F1') or all_metrics.get('Val/F1'),
                'Val_Balanced_Accuracy': all_metrics.get('Val_Balanced_Accuracy') or all_metrics.get('Val/Balanced_Accuracy'),
            }
            
            # Store or update if better (based on Test_AUROC)
            config_key = (m_val, n_val)
            if config_key not in ablation_data[modality]:
                ablation_data[modality][config_key] = extracted
            else:
                # Keep the run with higher Test_AUROC
                current_auroc = ablation_data[modality][config_key].get('Test_AUROC', 0) or 0
                new_auroc = extracted.get('Test_AUROC', 0) or 0
                if new_auroc > current_auroc:
                    ablation_data[modality][config_key] = extracted
    
    # Convert to grid format for plotting
    masking_ratios = [25, 50, 75, 95]
    noise_levels = [25, 50, 75, 100]
    modalities = ['T1', 'T1CE', 'T2', 'FLAIR']
    
    # Create grids for each metric and modality
    grids = {}
    for modality in modalities:
        grids[modality] = {}
        for metric in ['Test_AUROC', 'Test_AP', 'Test_F1', 'Test_Balanced_Accuracy',
                      'Val_AUROC', 'Val_AP', 'Val_F1', 'Val_Balanced_Accuracy']:
            grid = np.zeros((len(noise_levels), len(masking_ratios)))
            grid[:] = np.nan  # Initialize with NaN for missing data
            
            for i, n_val in enumerate(noise_levels):
                for j, m_val in enumerate(masking_ratios):
                    config_key = (m_val, n_val)
                    if config_key in ablation_data[modality]:
                        value = ablation_data[modality][config_key].get(metric)
                        if value is not None:
                            grid[i, j] = value
            
            grids[modality][metric] = grid
    
    # Save extracted data
    print("\nSaving extracted metrics...")
    
    # Save raw extracted data as JSON
    json_output = {}
    for modality, configs in ablation_data.items():
        json_output[modality] = {}
        for (m_val, n_val), metrics in configs.items():
            key = f"M{m_val}_N{n_val}"
            json_output[modality][key] = metrics
    
    with open(output_dir / 'extracted_metrics.json', 'w') as f:
        json.dump(json_output, f, indent=2, default=str)
    
    # Save grids as NumPy arrays
    np.savez(output_dir / 'processed_grids.npz', 
             grids=grids,
             masking_ratios=masking_ratios,
             noise_levels=noise_levels,
             modalities=modalities)
    
    # Print summary
    print("\n" + "="*60)
    print("EXTRACTION SUMMARY")
    print("="*60)
    
    for modality in modalities:
        print(f"\n{modality}:")
        configs_found = len(ablation_data[modality])
        print(f"  Configurations found: {configs_found}/16")
        
        # Check Test_AUROC grid
        test_grid = grids[modality]['Test_AUROC']
        valid_values = np.sum(~np.isnan(test_grid))
        print(f"  Valid Test_AUROC values: {valid_values}/16")
        
        if valid_values > 0:
            print(f"  Test_AUROC range: {np.nanmin(test_grid):.3f} - {np.nanmax(test_grid):.3f}")
            print(f"  Test_AUROC mean: {np.nanmean(test_grid):.3f}")
            
            # Find best configuration
            if not np.all(np.isnan(test_grid)):
                best_idx = np.nanargmax(test_grid)
                best_i, best_j = np.unravel_index(best_idx, test_grid.shape)
                best_value = test_grid[best_i, best_j]
                print(f"  Best config: M{masking_ratios[best_j]}_N{noise_levels[best_i]} = {best_value:.3f}")
    
    print(f"\nData saved to: {output_dir}")
    print("  - extracted_metrics.json: Raw metrics for each configuration")
    print("  - processed_grids.npz: NumPy grids ready for plotting")

if __name__ == "__main__":
    main()