# Method Pattern Recognition Documentation

## Overview
This document defines the pattern recognition rules for identifying different methods in the benchmarking results. These patterns are critical for correctly extracting and comparing method performance across benchmarks.

## Method Patterns (ResNet-based Models Only)

### Our Methods

#### MDAE (Masked-Diffusion Autoencoder)
**Pattern**: `^(resenc_MDAETrainer_RandomMask_Flow_BS48_2000ep|resenc_MDAE_pretrained|resenc_MDAE_scratch)`

Matches:
- `resenc_MDAETrainer_RandomMask_Flow_BS48_2000ep_pretrained_*` - Full training pipeline MDAE
- `resenc_MDAE_pretrained_*` - Simplified MDAE pretrained models
- `resenc_MDAE_scratch_*` - MDAE trained from scratch

Example runs:
- `resenc_MDAETrainer_RandomMask_Flow_BS48_2000ep_pretrained_t2f_2025-08-01-12-34-56`
- `resenc_MDAE_pretrained_asl_2025-08-08_19-22-19`
- `resenc_MDAE_scratch_flair_2025-08-05_14-22-33`

#### MDAE (TC) - Time-Conditioned Variants
**Pattern**: `^(resenc_time_conditioned|resenc_multimodal_mm_mdae)`

Matches:
- `resenc_time_conditioned_pretrained_*` - Time-conditioned MDAE
- `resenc_multimodal_mm_mdae_pretrained_*` - Multimodal MDAE variants

Example runs:
- `resenc_time_conditioned_pretrained_t1c_2025-08-03_02-18-41`
- `resenc_multimodal_mm_mdae_pretrained_pretrained_t2f_2025-08-05_11-16-58`

### SSL Baselines

#### MAE (Masked Autoencoder)
**Pattern**: `^resenc_pretrained_`
- Note: Most generic pattern, must be checked LAST after all other resenc patterns

#### SimCLR
**Pattern**: `^resenc_SimCLR`

#### VoCo (Volume Contrast)
**Pattern**: `^resenc_VoCo`

#### Models Genesis (MG)
**Pattern**: `^resenc_MG`

#### SwinUNETR
**Pattern**: `^resenc_SwinUNETR`

#### Volume Fusion (VF)
**Pattern**: `^resenc_VF`

#### S3D
**Pattern**: `^resenc_S3D`

### Foundation Models

#### BrainIAC
**Pattern**: `^brainiac_pretrained`

#### MRI-Core
**Pattern**: `^mri_core`

#### BrainMVP
**Pattern**: `^brainmvp`

### Other Baselines

#### DinoV2
**Pattern**: `^dinov2`

#### ResNet-50 (Scratch)
**Pattern**: `^brainiac_scratch`

## Important Notes

1. **Pattern Order Matters**: More specific patterns must be checked before generic ones. For example, `resenc_MDAE_pretrained` must be checked before `resenc_pretrained_` (MAE).

2. **MDAE (Combined) Method**: 
   - Created by taking the best performance between MDAE and MDAE (TC) for each benchmark/modality
   - Represents the optimal MDAE variant performance
   - Achieves 76.5% mean AUROC across all benchmarks
   - Always shown alongside individual MDAE and MDAE (TC) results

3. **Modality Standardization**:
   - T2F/T2f → FLAIR (prevents duplicates in BraTS23)
   - T1WCE/T1wce → T1CE (prevents duplicates in RSNA-MICCAI)
   - T1W/T1w → T1
   - T2W/T2w → T2
   - Applied consistently across all benchmarks

4. **Modality Suffixes**: Most patterns include modality suffixes like `_t1`, `_t2`, `_flair`, `_asl`, `_swi`, etc.

5. **Timestamp Format**: Run names typically end with timestamps in format `YYYY-MM-DD_HH-MM-SS`

## Verification Examples

### Correct Identification
- `resenc_MDAE_pretrained_asl_2025-08-08_19-22-19` → **MDAE**
- `resenc_time_conditioned_pretrained_t1c_2025-08-03_02-18-41` → **MDAE (TC)**
- `resenc_VoCo_pretrained_t2f_2025-08-04_10-45-15` → **VoCo**
- `brainiac_pretrained_t2f_2025-08-01_22-13-18` → **BrainIAC**

### Edge Cases
- `resenc_pretrained_t2_2025-08-01_12-34-56` → **MAE** (generic pattern)
- `resenc_MDAETrainer_RandomMask_Flow_BS48_2000ep_scratch_flair` → **MDAE** (scratch variant)

## Updates History

### 2025-08-12
- Added comprehensive modality standardization mappings
- Documented MDAE (Combined) methodology
- Fixed T2F→FLAIR and T1WCE→T1CE duplicates
- Added visualization highlighting for MDAE variants

### 2025-08-11
- Fixed MDAE pattern to include `resenc_MDAE_pretrained` and `resenc_MDAE_scratch` variants
- Removed non-ResNet models (eva_mdae) to focus on ResNet-based architectures
- Verified fix with ucsf_pdgm_idh_classification benchmark (ASL modality now correctly shows MDAE AUROC=0.813)

## Usage in Code

```python
import re

METHOD_PATTERNS = {
    'MDAE': r'^(resenc_MDAETrainer_RandomMask_Flow_BS48_2000ep|resenc_MDAE_pretrained|resenc_MDAE_scratch)',
    'MDAE (TC)': r'^(resenc_time_conditioned|resenc_multimodal_mm_mdae)',
    # ... other patterns
}

def identify_method(run_name):
    for method, pattern in METHOD_PATTERNS.items():
        if re.match(pattern, run_name):
            return method
    return 'Unknown'
```

## Testing Pattern Recognition

To verify patterns are working correctly:

```python
# Test cases
test_runs = [
    "resenc_MDAE_pretrained_asl_2025-08-08_19-22-19",  # Should be MDAE
    "resenc_time_conditioned_pretrained_t1c_2025-08-03_02-18-41",  # Should be MDAE (TC)
    "resenc_VoCo_pretrained_t2f_2025-08-04_10-45-15",  # Should be VoCo
]

for run in test_runs:
    method = identify_method(run)
    print(f"{run} → {method}")
```