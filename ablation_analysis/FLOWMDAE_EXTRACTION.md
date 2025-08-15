# FlowMDAE Grid Search Extraction Guide

## Overview
FlowMDAE is a variant of MDAE that systematically explores the parameter space of:
- **M** (Masking Ratio): Percentage of input masked during training
- **N** (Max Corruption Noise Level): Maximum noise level for corruption during denoising

## Notes Field Pattern Recognition

### Pattern Structure
The FlowMDAE runs are identified through specific patterns in the WandB Notes field:

```
FlowMDAE Ablation RESENC_FLOWMDAE_M{masking}_N{noise} pretrained on {benchmark} {task} with {modality}
```

### Example Notes Patterns
```
1. "FlowMDAE Ablation RESENC_FLOWMDAE_M25_N50 pretrained on brats18 lgg_vs_hgg with flair"
2. "FlowMDAE Ablation RESENC_FLOWMDAE_M75_N100 pretrained on upenn_gbm survival_2year with t2"
3. "FlowMDAE FIXED Masking Ablation RESENC_FLOWMDAE_M95_N75 pretrained on ucsf_pdgm idh_classification with t1"
```

## Extraction Process

### Step 1: Pattern Matching
```python
import re

def extract_flowmdae_params(notes):
    """
    Extract FlowMDAE parameters from Notes field
    
    Args:
        notes: WandB run Notes field string
        
    Returns:
        dict with masking_ratio, noise_level, and other metadata
    """
    
    # Primary pattern for FlowMDAE identification
    flowmdae_pattern = r'RESENC_FLOWMDAE_M(\d+)_N(\d+)'
    
    match = re.search(flowmdae_pattern, notes)
    if match:
        return {
            'is_flowmdae': True,
            'masking_ratio': int(match.group(1)),
            'noise_level': int(match.group(2)),
            'param_combo': f'M{match.group(1)}_N{match.group(2)}'
        }
    
    return {'is_flowmdae': False}
```

### Step 2: Grid Search Parameters

The FlowMDAE grid search explores a 4×4 parameter grid:

| Parameter | Values | Description |
|-----------|--------|-------------|
| **M (Masking)** | 25, 50, 75, 95 | Masking percentage (fixed or max) |
| **N (Noise Level)** | 25, 50, 75, 100 | Max corruption noise level |

**Total Combinations**: 16 (4 × 4) per category

### Step 3: Filtering Logic
```python
def filter_flowmdae_runs(all_runs):
    """
    Filter and categorize FlowMDAE runs from all ablation runs
    """
    flowmdae_runs = []
    
    for run in all_runs:
        notes = run.get('Notes', '')
        
        # Check for FlowMDAE pattern
        if 'FLOWMDAE' in notes and 'RESENC_FLOWMDAE' in notes:
            # Extract parameters
            params = extract_flowmdae_params(notes)
            
            if params['is_flowmdae']:
                # Add extracted parameters to run data
                run['masking_ratio'] = params['masking_ratio']
                run['noise_level'] = params['noise_level']
                run['param_combo'] = params['param_combo']
                
                # Categorize by variation type
                if 'FIXED' in notes:
                    run['variation'] = 'fixed_masking'
                else:
                    run['variation'] = 'standard'
                
                flowmdae_runs.append(run)
    
    return flowmdae_runs
```

## Two Distinct FlowMDAE Categories

### Category 1: Fixed Masking FlowMDAE
- **Pattern**: `FlowMDAE FIXED Masking Ablation RESENC_FLOWMDAE_*`
- **Behavior**: Masking ratio is **fixed at exactly M%** for all samples
- **Example**: "FlowMDAE FIXED Masking Ablation RESENC_FLOWMDAE_M75_N50"
- **Interpretation**: With M=75, exactly 75% of each input is masked

### Category 2: Random Sampling FlowMDAE  
- **Pattern**: `FlowMDAE Ablation RESENC_FLOWMDAE_*` (without "FIXED")
- **Behavior**: Masking ratio is **randomly sampled from [1%, M%]** for each sample
- **Example**: "FlowMDAE Ablation RESENC_FLOWMDAE_M50_N75"
- **Interpretation**: With M=50, masking ratio varies randomly between 1% and 50%

### Key Difference
- **Fixed**: Consistent masking ratio across all samples (deterministic)
- **Random**: Variable masking ratio per sample (stochastic)

## Data Processing Pipeline

### Complete Extraction Code
```python
import pandas as pd
import re
from typing import List, Dict

class FlowMDAEExtractor:
    """Extract and process FlowMDAE grid search runs"""
    
    def __init__(self):
        self.grid_m = [25, 50, 75, 95]
        self.grid_n = [25, 50, 75, 100]
        self.expected_combos = {
            f'M{m}_N{n}' for m in self.grid_m for n in self.grid_n
        }
    
    def extract_from_notes(self, notes: str) -> Dict:
        """Extract FlowMDAE parameters from Notes field"""
        
        # Check if this is a FlowMDAE run
        if 'RESENC_FLOWMDAE' not in notes:
            return None
        
        # Extract M and N parameters
        pattern = r'RESENC_FLOWMDAE_M(\d+)_N(\d+)'
        match = re.search(pattern, notes)
        
        if not match:
            return None
        
        masking = int(match.group(1))
        noise = int(match.group(2))
        
        # Extract benchmark and modality
        benchmark_pattern = r'pretrained on (\S+) (\S+)'
        modality_pattern = r'with (\S+)'
        
        benchmark_match = re.search(benchmark_pattern, notes)
        modality_match = re.search(modality_pattern, notes)
        
        return {
            'masking_ratio': masking,
            'noise_level': noise,
            'param_combo': f'M{masking}_N{noise}',
            'benchmark': benchmark_match.group(1) if benchmark_match else None,
            'task': benchmark_match.group(2) if benchmark_match else None,
            'modality': modality_match.group(1) if modality_match else None,
            'is_fixed': 'FIXED' in notes,
            'notes': notes
        }
    
    def process_runs(self, runs: List[Dict]) -> pd.DataFrame:
        """Process all runs and extract FlowMDAE experiments"""
        
        flowmdae_data = []
        
        for run in runs:
            notes = run.get('Notes', '')
            params = self.extract_from_notes(notes)
            
            if params:
                # Combine run data with extracted parameters
                run_data = {
                    'run_id': run['id'],
                    'run_name': run['name'],
                    'created_at': run['created_at'],
                    'state': run['state'],
                    **params,  # Add extracted parameters
                    # Add metrics
                    'test_auroc': run.get('metrics', {}).get('Test_AUROC'),
                    'test_ap': run.get('metrics', {}).get('Test_AP'),
                    'test_f1': run.get('metrics', {}).get('Test_F1'),
                    'val_auroc': run.get('metrics', {}).get('Val_AUROC'),
                }
                flowmdae_data.append(run_data)
        
        df = pd.DataFrame(flowmdae_data)
        
        # Validate grid coverage
        self.validate_grid_coverage(df)
        
        return df
    
    def validate_grid_coverage(self, df: pd.DataFrame):
        """Check which parameter combinations are covered"""
        
        found_combos = set(df['param_combo'].unique())
        missing_combos = self.expected_combos - found_combos
        
        print(f"Grid Coverage Report:")
        print(f"  Expected combinations: {len(self.expected_combos)}")
        print(f"  Found combinations: {len(found_combos)}")
        print(f"  Coverage: {len(found_combos)/len(self.expected_combos)*100:.1f}%")
        
        if missing_combos:
            print(f"\nMissing combinations:")
            for combo in sorted(missing_combos):
                print(f"    - {combo}")
```

## Usage Example

```python
# Load raw data
import json
with open('ablation_analysis/raw_data/all_ablations_combined.json', 'r') as f:
    data = json.load(f)

# Initialize extractor
extractor = FlowMDAEExtractor()

# Extract all runs
all_runs = []
for project in data['projects']:
    all_runs.extend(project.get('runs', []))

# Process FlowMDAE runs
flowmdae_df = extractor.process_runs(all_runs)

# Save processed data
flowmdae_df.to_csv('ablation_analysis/raw_data_extracted/flowmdae_raw.csv', index=False)

# Analyze results
print(f"\nFlowMDAE Analysis Summary:")
print(f"Total runs: {len(flowmdae_df)}")
print(f"Benchmarks covered: {flowmdae_df['benchmark'].nunique()}")
print(f"Parameter combinations: {flowmdae_df['param_combo'].nunique()}")

# Find optimal configuration
best_config = flowmdae_df.groupby('param_combo')['test_auroc'].mean().idxmax()
best_score = flowmdae_df.groupby('param_combo')['test_auroc'].mean().max()
print(f"\nBest configuration: {best_config} (AUROC: {best_score:.4f})")
```

## Comparative Analysis Results

### Coverage Analysis
- **Fixed Masking**: 217 runs across 4 benchmarks
- **Random Sampling**: 64 runs across 1 benchmark (BraTS18)
- **Grid coverage**: Both achieve 16/16 parameter combinations

### Performance Comparison
| Category | Mean AUROC | Std Dev | Best Config | Best AUROC |
|----------|------------|---------|-------------|------------|
| **Fixed Masking** | 0.6259 | ±0.1344 | M75_N100 | 0.6584 |
| **Random Sampling** | 0.5893 | ±0.1218 | M75_N75 | 0.6575 |
| **Difference** | +0.0366 | - | - | - |

### Head-to-Head Results
- **Fixed Masking wins**: 13/16 parameter combinations (81.25%)
- **Random Sampling wins**: 3/16 parameter combinations (18.75%)
- **Statistical significance**: p=0.0516 (marginally significant)

### Key Findings
1. **Fixed masking generally outperforms random sampling** by 3.66% AUROC on average
2. **Optimal M value**: 75% for both strategies
3. **Optimal N value**: 100 for fixed, 75 for random
4. **Random sampling benefits**: Better exploration during training, but lower average performance
5. **Fixed masking benefits**: Consistent training signal, more stable convergence

## Notes Field Validation

### Valid FlowMDAE Notes Format
✅ **Correct**:
```
"FlowMDAE Ablation RESENC_FLOWMDAE_M50_N75 pretrained on brats18 lgg_vs_hgg with t2"
```

❌ **Invalid** (would be filtered out):
```
"RESENC_FLOWMDAE pretrained on brats18"  # Missing M and N parameters
"FlowMDAE M50_N75 experiment"             # Missing RESENC_FLOWMDAE pattern
"RESENC_MDAE_FLOWMDAE_M50"               # Different pattern structure
```

## Troubleshooting

### Common Issues

1. **Missing FlowMDAE runs**
   - Check Notes field is populated in WandB
   - Verify pattern matching regex
   - Ensure case sensitivity in pattern matching

2. **Incomplete grid coverage**
   - Some parameter combinations may have failed runs
   - Check `state == 'finished'` filter
   - Review WandB for crashed/cancelled runs

3. **Parameter extraction errors**
   - Validate regex patterns against actual Notes
   - Handle variations in Notes formatting
   - Check for typos in WandB Notes field

---

*This document explains the complete FlowMDAE extraction process from Notes field identification to final data processing.*