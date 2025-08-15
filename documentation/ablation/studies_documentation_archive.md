# Ablation Studies Documentation

## Date: August 14, 2025

## Overview
This document captures our findings about ablation studies in the MDAE benchmarking experiments. These ablations are stored in the WandB run Notes field and were not fully extracted in previous data pulls.

## Ablation Study Types

### 1. Noise Corruption Type Ablation
Tests different noise corruption methods used in the diffusion process.

**Types Identified:**
- **VE** (Variance Exploding)
- **VP** (Variance Preserving)  
- **Flow** (Flow-based)

**Identification Pattern in Notes:**
- Pattern: `"MDAE ablation Noise Corruption [TYPE] M[RATIO]"`
- Example: `"MDAE ablation Noise Corruption Flow M90 on brats18 lgg_vs_hgg with flair"`
- Fixed masking ratio: M90 (90% masking)

**Coverage in BraTS18:**
- Flow_M90: 9 runs across 4 modalities
- VE_M90: 8 runs across 4 modalities  
- VP_M90: 11 runs across 4 modalities
- **Total**: 28 noise corruption ablation runs

### 2. Masking Type Ablation
Tests different spatial masking strategies.

**Types Identified:**
- **Random patch-based masking** - Random patches across the volume
- **Block masking** - Contiguous block regions
- **Tube masking** - Tubular or cylindrical masks

**Identification Pattern in Notes:**
- Pattern: `"Masking Type Ablation: [TYPE]"`
- Examples:
  - `"Masking Type Ablation: Random patch-based masking - brats18 lgg_vs_hgg flair"`
  - `"Masking Type Ablation: Standard blocky masking with 90% ratio - brats18 lgg_vs_hgg flair"`
- Fixed masking ratio: 90%

**Coverage in BraTS18:**
- Random_patch: 4 runs across 4 modalities
- Block: 4 runs across 4 modalities
- Other/Tube: 4 runs across 4 modalities
- **Total**: 12 masking type ablation runs

### 3. Masking Ratio Ablation
Tests different percentages of input masking.

**Ratios Identified:**
- 75%, 80%, 85%, 90%, 95%

**Identification Pattern in Notes:**
- Pattern: `"Masking Ratio Ablation: [NOISE_TYPE] [RATIO]%"`
- Example: `"Masking Ratio Ablation: Flow 80% - brats18 lgg_vs_hgg flair"`
- Fixed noise type: Flow

**Coverage in BraTS18:**
- 75%_Flow: 4 runs across 4 modalities
- 80%_Flow: 4 runs across 4 modalities
- 85%_Flow: 4 runs across 4 modalities
- 90%_Flow: 4 runs across 4 modalities
- 95%_Flow: 4 runs across 4 modalities
- **Total**: 20 masking ratio ablation runs

## Additional Ablation Studies (FlowMDAE)

We also found FlowMDAE ablation studies with different parameter combinations:

**Pattern in Notes:**
- `"FlowMDAE Ablation RESENC_FLOWMDAE_M[MASK]_N[NOISE]"`
- Example: `"FlowMDAE Ablation RESENC_FLOWMDAE_M25_N75 pretrained on brats18 lgg_vs_hgg"`

**Parameter Grid Found:**
- Masking ratios (M): 25%, 50%, 75%, 95%
- Noise levels (N): 25%, 50%, 75%, 100%
- **Total**: ~40 additional FlowMDAE ablation runs in BraTS18

## Benchmark Coverage

### Confirmed Benchmarks with Ablation Studies
All 15 single-modality benchmarks have ablation studies:

1. **BraTS18**: brats18_lgg_vs_hgg (4 modalities)
2. **BraTS23**: 
   - brats23_gli_vs_men (4 modalities)
   - brats23_gli_vs_met (4 modalities)
   - brats23_men_vs_met (4 modalities)
3. **RSNA-MICCAI**: rsna_miccai_brain_tumor_mgmt_methylation (4 modalities)
4. **TCGA-GBM**:
   - tcga_gbm_dss_1year (5 modalities)
   - tcga_gbm_pfi_1year (2 modalities)
5. **UCSF-PDGM**: ucsf_pdgm_idh_classification (6 modalities)
6. **UPenn-GBM**:
   - upenn_gbm_age_group (4 modalities)
   - upenn_gbm_gender (4 modalities)
   - upenn_gbm_gtr_status (4 modalities)
   - upenn_gbm_idh1_status (4 modalities)
   - upenn_gbm_survival_18month (4 modalities)
   - upenn_gbm_survival_1year (4 modalities)
   - upenn_gbm_survival_2year (4 modalities)

**Total Projects**: 61 single-modality projects with prefix `july_stratified_*_single_*`

## Data Extraction Requirements

### Critical Fields to Extract
1. **Notes field** - Contains ablation type and parameters (MOST IMPORTANT)
2. **Run name** - Identifies the model and timestamp
3. **Metrics**:
   - Test_AUROC
   - Test_AP
   - Test_F1
   - Test_Balanced_Accuracy
   - Val_AUROC
   - Val_AP
4. **Config parameters** - Model configuration used
5. **Timestamps** - When the run was created
6. **State** - Whether run completed successfully

### Parsing Logic for Notes Field

```python
# Noise Corruption Type
if 'Noise Corruption' in notes:
    noise_type = re.search(r'(VE|VP|Flow)', notes)
    mask_ratio = re.search(r'M(\d+)', notes)

# Masking Type
if 'Masking Type Ablation' in notes:
    if 'Random patch' in notes: type = 'Random'
    elif 'Block' in notes or 'blocky' in notes: type = 'Block'
    elif 'Tube' in notes: type = 'Tube'

# Masking Ratio
if 'Masking Ratio Ablation' in notes:
    ratio = re.search(r'(\d+)%', notes)
    noise = re.search(r'(Flow|VE|VP)', notes)

# FlowMDAE
if 'FlowMDAE Ablation' in notes:
    mask = re.search(r'M(\d+)', notes)
    noise = re.search(r'N(\d+)', notes)
```

## Key Insights

1. **Systematic Coverage**: Ablations are systematically tested across all 4 modalities for BraTS18
2. **Controlled Variables**: Each ablation type fixes other parameters (e.g., masking type ablation fixes ratio at 90%)
3. **Complete Grid**: Masking ratio ablation covers 75-95% range in 5% increments
4. **Noise Types**: Three distinct noise corruption methods tested (VE, VP, Flow)
5. **Extensive Testing**: Over 100 ablation runs just for BraTS18, likely 500+ across all benchmarks

## Next Steps

1. **Extract Complete Data**: Re-extract all 61 single-modality projects with full Notes field
2. **Parse Ablation Parameters**: Use regex patterns to categorize runs
3. **Structure Analysis**: Create matrices for each ablation type
4. **Statistical Analysis**: Compare performance across parameters
5. **Optimal Parameters**: Identify best configurations per benchmark/modality

## Important Notes

- Previous extractions may have incomplete or missing Notes field
- The Notes field is critical for identifying ablation studies
- Some runs may have variations in note formatting that need handling
- FlowMDAE ablations use different parameter naming (M/N instead of explicit percentages)