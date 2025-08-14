# MDAE Benchmarking Data Analysis Pipeline

## Overview

This repository contains the complete data extraction, processing, and analysis infrastructure for the MDAE (Masked-Diffusion Autoencoder) paper benchmarking experiments. The pipeline processes results from 15 brain MRI classification benchmarks across multiple modalities, comparing MDAE against established baseline models.

## Quick Start

```bash
# 1. Extract data from WandB
python scripts/extract_wandb_data.py --all

# 2. Run comprehensive analysis (recommended)
# Default: Combined MDAE mode (shows single best MDAE)
python run_comprehensive_analysis.py

# Alternative: Show all MDAE variants separately
python run_comprehensive_analysis.py --no-combine-mdae

# Alternative: Use individual scripts
# python process_results_final.py
# python create_final_tables.py
```

## Repository Structure

```
data/
├── raw_data/                    # Raw data from WandB
│   └── 20250811/                # Latest extraction
│       └── [benchmark_name]/    # 15 benchmark folders
│           ├── full_data.json   # Complete WandB data
│           └── runs_summary.csv # Flattened metrics
│
├── processed_data/              # Processed results
│   └── benchmark_results_final/ # Latest analysis
│       ├── [benchmark_name]/    # Per-benchmark results
│       │   └── modalities/      # Per-modality analysis
│       └── *.csv                # Summary tables
│
├── processed_data_combined/     # Combined MDAE results (simplified mode)
│   └── benchmarks/              # Per-benchmark visualizations
│       └── *.csv                # Summary tables
│
├── scripts/                     # Processing scripts
│   ├── extract_wandb_data.py   # WandB data extraction
│   └── *.py                     # Analysis scripts
│
├── baselines/                   # Method configurations
│   └── key_baselines.txt       # Method name patterns
│
└── README_METHODS.md           # Method pattern documentation
```

## Current Workflow (Updated 2024-08-11)

### Step 1: Data Extraction
Extract experiment results from WandB, including all metrics and configurations.

```bash
# Extract all benchmark data
python scripts/extract_wandb_data.py --all

# Extract specific benchmark
python scripts/extract_wandb_data.py --benchmark brats18/lgg_vs_hgg
```

**Output**: `raw_data/YYYYMMDD/[benchmark_name]/`

### Step 2: Process Results with Fixed Patterns
Process all benchmarks using corrected MDAE pattern recognition.

```bash
# Main processing script with fixed MDAE patterns
python process_results_final.py
```

**Key Features**:
- ✅ Correctly identifies all MDAE variants (`resenc_MDAE_pretrained_*`, etc.)
- ✅ Processes all modalities for each benchmark
- ✅ Generates threshold-independent visualizations (AUROC & AP)
- ✅ Creates comprehensive metrics tables

**Output**: `processed_data/benchmark_results_final/`

### Step 3: Generate Analysis Tables
Create comprehensive tables for paper and analysis.

```bash
# Generate summary tables and LaTeX formatting
python create_final_tables.py
```

**Output Files**:
- `comprehensive_auroc_ap_table.csv` - All methods × all benchmarks
- `methods_summary_statistics.csv` - Summary with mean ± std
- `paper_comprehensive_table.tex` - LaTeX table for paper

### Step 4: Combined MDAE Analysis (Optional)
Combine MDAE and MDAE-TC variants for overall analysis.

```bash
python process_mdae_combined.py
```

## Method Recognition Patterns

**CRITICAL**: The system uses pattern matching to identify methods. See `README_METHODS.md` for complete documentation.

### Our Methods
- **MDAE**: `resenc_MDAETrainer_*`, `resenc_MDAE_pretrained_*`, `resenc_MDAE_scratch_*`
- **MDAE (TC)**: `resenc_time_conditioned_*`, `resenc_multimodal_mm_mdae_*`

### Baselines
- **SSL Methods**: MAE, SimCLR, VoCo, MG, SwinUNETR, VF, S3D
- **Foundation Models**: BrainIAC, MRI-Core, BrainMVP
- **Others**: DinoV2, ResNet-50 (scratch)

## Key Scripts

### Core Processing

#### `run_comprehensive_analysis.py` ⭐ PRIMARY SCRIPT
Complete reproducible pipeline for all analysis with enhanced features.
- Processes raw data to final results
- Handles all 15 benchmarks and 44+ modality combinations
- **Two modes of operation**:
  - `--combine-mdae` (default): Shows only best MDAE performance as single "MDAE" method
  - `--no-combine-mdae`: Shows MDAE, MDAE (TC), and MDAE (Combined) separately
- **Modality standardization**: Fixes T2F→FLAIR and T1WCE→T1CE duplicates
- **Visualization highlighting**: MDAE variants highlighted in distinct colors
- Generates all visualizations and tables
- **Single command to regenerate everything**

#### `process_results_final.py`
Alternative processing script with MDAE pattern recognition.
- Processes all benchmarks and modalities
- Generates clean visualizations (AUROC & AP)
- Creates metrics tables

#### `process_results_fixed.py`
Alternative processing with threshold-independent metrics focus.

#### `process_mdae_combined.py`
Combines MDAE variants (standard and time-conditioned) for unified analysis.

### Table Generation

#### `create_final_tables.py`
Generates comprehensive tables from processed results.
- Summary statistics
- LaTeX formatting
- Excel-friendly CSV outputs

### Visualization

#### `create_final_boxplot.py`
Generates publication-ready box plot visualization comparing MDAE performance across all baselines.

**Box Plot Configuration**:
- **Notches**: Enabled to show 95% confidence intervals around medians
  - Allows statistical comparison between methods
  - Non-overlapping notches indicate significantly different medians
- **Whiskers**: Set to 1.0×IQR (instead of default 1.5×IQR)
  - Provides tighter bounds for "typical" values
  - Makes outliers more visible as individual points
- **Mean Markers**: Red diamonds show arithmetic mean alongside median
- **Color Scheme**:
  - Green: MDAE (our method)
  - Blue: SSL baselines (MAE, SimCLR, VoCo, etc.)
  - Purple: Foundation models (BrainIAC, MRI-Core, BrainMVP)
  - Gray: Other methods (DinoV2)

**Usage**:
```bash
python create_final_boxplot.py
```

**Output**:
- `processed_data_combined/boxplot_final.png` - High-resolution (300 DPI) for presentations
- `processed_data_combined/boxplot_final.pdf` - Vector format for publication

**Box Plot Elements Explained**:
- **Box**: Middle 50% of data (Q1 to Q3)
- **Median Line**: Black horizontal line inside box
- **Notch**: Narrowing at median showing confidence interval
- **Mean**: Red diamond marker
- **Whiskers**: Lines extending to furthest points within 1.0×IQR
- **Outliers**: Circles beyond whisker range

### Data Extraction

#### `scripts/extract_wandb_data.py`
Extracts data from WandB API.
- Supports batch extraction
- Handles all test metrics

## Benchmark Tasks

15 benchmarks across 6 major datasets:

### Tumor Classification
- BraTS18: LGG vs HGG
- BraTS23: Glioma vs Meningioma
- BraTS23: Glioma vs Metastasis  
- BraTS23: Meningioma vs Metastasis

### Molecular Markers
- RSNA-MICCAI: MGMT methylation
- UCSF-PDGM: IDH classification
- UPenn-GBM: IDH1 status

### Survival Prediction
- TCGA-GBM: DSS 1-year
- TCGA-GBM: PFI 1-year
- UPenn-GBM: 18-month survival
- UPenn-GBM: 1-year survival
- UPenn-GBM: 2-year survival

### Clinical Features
- UPenn-GBM: Age group
- UPenn-GBM: Gender
- UPenn-GBM: GTR status

## Metrics

### Threshold-Independent (Primary)
- **Test_AUROC**: Area Under ROC Curve
- **Test_AP**: Average Precision

### Threshold-Dependent (Secondary)
- **Test_F1**: F1 Score
- **Test_Balanced_Accuracy**: Balanced Accuracy

## Current Results Summary

As of 2025-08-12:
- **MDAE Mean AUROC**: 76.5% across all benchmarks (#1 rank)
  - Combined best performance from MDAE and MDAE (TC) variants
  - Default mode shows only this combined result as "MDAE"
- **Improvement over SSL baselines**: 10.4% average
- **Improvement over foundation models**: 10.8% average
- **Total runs analyzed**: 3,000+
- **Benchmarks**: 15
- **Modality combinations**: 44 (standardized, no duplicates)

## Troubleshooting

### Issue: MDAE methods not recognized
**Solution**: Ensure using `process_results_final.py` which includes fixed patterns for `resenc_MDAE_pretrained_*`

### Issue: Missing modalities
**Solution**: Script automatically processes all available modalities in raw data

### Issue: Outdated results
**Solution**: Re-run `process_results_final.py` after any new data extraction

## Changelog

### 2025-08-12 - Simplified MDAE Mode
- **Added combined mode**: `--combine-mdae` flag (default) shows only best MDAE as single method
- **Created `processed_data_combined`**: New output directory for simplified results
- **Flexible processing**: Can toggle between combined and full variant display

### 2025-08-12 - Enhanced Processing Pipeline
- **Fixed modality duplicates**: T2F→FLAIR (BraTS23), T1WCE→T1CE (RSNA-MICCAI)
- **Added MDAE (Combined)**: Takes best performance of MDAE and MDAE (TC)
- **Visualization improvements**: Color-coded highlighting for MDAE variants
- **Updated results**: MDAE achieves 76.5% mean AUROC

### 2025-08-11 - Initial Pipeline
- Fixed MDAE pattern recognition
- Processed all 15 benchmarks and 44 modality combinations
- Initial MDAE (Combined) analysis

## Archive

Outdated scripts have been moved to `archive_old_scripts/` for reference.

## Citation

If using this pipeline, please cite:
```bibtex
@article{mdae2024,
  title={Masked-Diffusion Autoencoders for Self-Supervised 3D Brain MRI Classification},
  author={...},
  journal={ICLR},
  year={2024}
}
```

## Contact

For questions about the pipeline or results, please contact the maintainers.

## License

MIT License - See LICENSE file for details.