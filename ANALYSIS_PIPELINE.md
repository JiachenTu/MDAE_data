# MDAE Analysis Pipeline Documentation

## Overview
This document provides a complete guide for generating analysis results from raw WandB data for the MDAE paper.

## Pipeline Steps

### Step 1: Raw Data Structure
The raw data is located in `raw_data/20250811/` with the following structure:
```
raw_data/20250811/
├── brats18_lgg_vs_hgg/
├── brats23_gli_vs_men/
├── brats23_gli_vs_met/
├── brats23_men_vs_met/
├── rsna_miccai_mgmt_methylation/
├── tcga_gbm_dss_1year/
├── tcga_gbm_pfi_1year/
├── ucsf_pdgm_idh_classification/
├── upenn_gbm_age_group/
├── upenn_gbm_gender/
├── upenn_gbm_gtr_status/
├── upenn_gbm_idh1_status/
├── upenn_gbm_survival_18month/
├── upenn_gbm_survival_1year/
└── upenn_gbm_survival_2year/
```

Each benchmark directory contains:
- `full_data.json`: Complete WandB experiment data
- `runs_summary.csv`: Flattened metrics for each run

### Step 2: Running the Analysis

#### Option A: Simplified Combined Mode (Default - Paper Ready)
```bash
# Generates results with single best MDAE method
python run_comprehensive_analysis.py --verbose

# Output: processed_data_combined/
# - Shows MDAE as single method (76.5% mean AUROC)
# - Best for paper figures and tables
```

#### Option B: Full Variant Analysis
```bash
# Shows MDAE, MDAE (TC), and MDAE (Combined) separately
python run_comprehensive_analysis.py --no-combine-mdae --verbose

# Output: processed_data/comprehensive_analysis/
# - Shows all three MDAE variants
# - Best for detailed analysis
```

### Step 3: Key Processing Features

#### Modality Standardization
The pipeline automatically standardizes modality names to prevent duplicates:
- T2F → FLAIR (BraTS23 benchmarks)
- T1WCE → T1CE (RSNA-MICCAI benchmark)
- T1W → T1
- T2W → T2

#### Method Recognition
Methods are identified using regex patterns (see `METHOD_PATTERNS` in script):
- **MDAE**: `resenc_MDAETrainer_*`, `resenc_MDAE_pretrained_*`, `resenc_MDAE_scratch_*`
- **MDAE (TC)**: `resenc_time_conditioned_*`, `resenc_multimodal_mm_mdae_*`
- **SSL Methods**: SimCLR, VoCo, MG, SwinUNETR, VF, S3D, MAE
- **Foundation Models**: BrainIAC, MRI-Core, BrainMVP
- **Others**: DinoV2, ResNet-50

#### MDAE Combination Logic
When `--combine-mdae` is used (default):
1. For each benchmark/modality, takes max(MDAE, MDAE_TC) AUROC
2. Labels the result as just "MDAE"
3. Removes individual MDAE and MDAE (TC) entries

### Step 4: Output Structure

```
processed_data_combined/
├── benchmarks/
│   ├── [Benchmark_Name]/
│   │   ├── [modality]/
│   │   │   ├── metrics_table.csv         # All metrics for this modality
│   │   │   ├── threshold_independent_metrics.png  # AUROC & AP plots
│   │   │   └── all_metrics.png           # All 4 metrics visualization
│   │   ├── cross_modality_comparison.png # Heatmap across modalities
│   │   └── cross_modality_table.csv      # Data for heatmap
├── comprehensive_metrics_table.csv       # All methods × all benchmarks
├── overall_summary_statistics.csv        # Mean±std across benchmarks
└── overall_performance.png               # Overall ranking visualization
```

### Step 5: Key Metrics

#### Primary (Threshold-Independent)
- **Test_AUROC**: Area Under ROC Curve
- **Test_AP**: Average Precision

#### Secondary (Threshold-Dependent)
- **Test_F1**: F1 Score
- **Test_Balanced_Accuracy**: Balanced Accuracy

### Step 6: Visualization Features
- **Color Coding**: MDAE variants highlighted in blue (#45B7D1)
- **Bold Labels**: MDAE methods shown with bold text
- **Black Borders**: MDAE bars have 2px black borders
- **Sorted Display**: Methods sorted by AUROC (descending)

## Reproducibility Checklist

1. ✅ Raw data in `raw_data/20250811/`
2. ✅ Run `python run_comprehensive_analysis.py`
3. ✅ Results in `processed_data_combined/`
4. ✅ Overall statistics in `overall_summary_statistics.csv`
5. ✅ MDAE achieves 76.5% mean AUROC (rank #1)

## Paper-Specific Outputs

### Main Results Table
```python
# Load comprehensive results
import pandas as pd
df = pd.read_csv('processed_data_combined/overall_summary_statistics.csv')
print(df[['Method', 'Mean_AUROC', 'Std_AUROC']].head(10))
```

### Benchmark-Specific Results
```python
# Load specific benchmark
bench_df = pd.read_csv('processed_data_combined/comprehensive_metrics_table.csv')
# Rows: Methods, Columns: Benchmarks
```

### Statistical Significance
The pipeline currently provides mean±std. Additional statistical tests (e.g., paired t-tests, Wilcoxon signed-rank) can be added for paper requirements.

## Troubleshooting

### Issue: Method not recognized
- Check `METHOD_PATTERNS` dictionary in `run_comprehensive_analysis.py`
- Verify run name matches expected pattern

### Issue: Missing modality
- Check raw data CSV for available modalities
- Verify modality standardization mapping

### Issue: Duplicate results
- Pipeline automatically keeps best AUROC for duplicates
- Check `detect_and_merge_duplicates()` function

## Contact
For questions about the pipeline, refer to the main README.md or the code comments in `run_comprehensive_analysis.py`.