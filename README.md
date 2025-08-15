# MDAE Benchmarking Data Analysis Repository

## Overview

This repository contains comprehensive data analysis for MDAE (Masked Denoising Autoencoder) benchmarking on medical imaging tasks. The analysis includes performance comparisons across multiple self-supervised learning methods, ablation studies, and multi-modality evaluations across 15 brain MRI classification benchmarks.

## Directory Structure

```
.
├── ablation_analysis/          # Ablation study results
│   ├── raw_data/              # Original JSON extractions
│   ├── raw_data_extracted/    # Processed CSV files
│   ├── visualizations/        # Generated figures (PNG)
│   └── tables/                # LaTeX/Markdown/CSV tables
│
├── documentation/              # All documentation
│   ├── ablation/              # Ablation study docs
│   │   └── README.md          # Main ablation documentation
│   ├── multi_modality/        # Multi-modality analysis docs
│   │   └── README.md          # Multi-modality findings
│   └── methods/               # Method descriptions
│
├── scripts/                   # Python scripts
│   ├── ablation/              # Ablation analysis scripts
│   │   ├── extract_raw_data.py
│   │   ├── generate_visualizations.py
│   │   └── generate_tables.py
│   ├── multi_modality/        # Multi-modality scripts
│   │   └── extract_data.py
│   └── legacy/                # Archived/old scripts
│
├── raw_data/                  # Original extracted data
│   ├── 20250811/             # Single-modality extractions
│   └── 20250814_multi/       # Multi-modality extractions
│
└── processed_results/         # Processed analysis results
```

## Quick Start

### Running Ablation Analysis

```bash
# 1. Extract raw ablation data
python scripts/ablation/extract_ablation_raw_data.py

# 2. Generate visualizations
python scripts/ablation/generate_ablation_visualizations.py

# 3. Create publication tables
python scripts/ablation/generate_ablation_tables.py
```

### Running Multi-Modality Analysis

```bash
# Extract and analyze multi-modality data
python scripts/multi_modality/extract_multi_modality_data.py
python scripts/multi_modality/analyze_multi_modality_results.py
```

## Key Results

### Ablation Studies
- **425 ablation runs** analyzed across 37 projects
- **Optimal Configuration**: VP noise, 75% masking ratio, block masking
- **Best FlowMDAE**: M75_N75 (75% masking, 75% noise)
- Full documentation: [documentation/ablation/README.md](documentation/ablation/README.md)

### Multi-Modality Analysis
- **680 multi-modality runs** from 15 benchmarks
- **MDAE achieves 81.1%** mean AUROC (best among all methods)
- **33% of benchmarks** benefit from multi-modality
- Full documentation: [documentation/multi_modality/README.md](documentation/multi_modality/README.md)

## Benchmarks Analyzed

### 15 Brain MRI Classification Tasks

1. **BraTS18**: LGG vs HGG tumor classification
2. **BraTS23**: 3 tasks (GLI vs MEN, GLI vs MET, MEN vs MET)
3. **RSNA-MICCAI**: MGMT methylation status
4. **TCGA-GBM**: DSS and PFI survival prediction
5. **UCSF-PDGM**: IDH mutation classification
6. **UPenn-GBM**: 7 tasks (age, gender, GTR status, IDH1, survival predictions)

## Methods Compared

| Method | Type | Mean AUROC | Description |
|--------|------|------------|-------------|
| **MDAE** | SSL | 0.811 | Masked Denoising Autoencoder (Our method) |
| **Supervised** | - | 0.793 | Standard supervised baseline |
| **MAE** | SSL | 0.739 | Masked Autoencoder |
| **SimCLR** | SSL | 0.691 | Contrastive Learning |
| **SwAV** | SSL | 0.689 | Swapping Assignments |
| **DINO** | SSL | 0.641 | Self-Distillation |

## Visualizations

All generated visualizations are stored in `ablation_analysis/visualizations/`:
- **Noise Corruption**: UPenn-focused performance comparison
- **Masking Ratio**: Trend analysis with 95% confidence intervals
- **Masking Type**: Block vs Random comparison with significance
- **FlowMDAE Grid**: MDAE (Flow SDE) parameter optimization heatmap

## Tables

Publication-ready tables available in three formats:
- **LaTeX**: `ablation_analysis/tables/latex/`
- **Markdown**: `ablation_analysis/tables/markdown/`
- **CSV**: `ablation_analysis/tables/csv/`

## Data Sources

- **WandB Entity**: `t-jiachentu`
- **Project Patterns**: 
  - Single-modality: `july_stratified_*`
  - Multi-modality: `july_*_multi_*`
- **Extraction Date**: August 14, 2025

## Requirements

```bash
pip install wandb pandas numpy matplotlib seaborn scipy tabulate
```

## Performance Metrics

All evaluations use threshold-independent metrics:
- **AUROC**: Area Under ROC Curve (primary metric)
- **AP**: Average Precision
- **F1 Score**: Harmonic mean of precision and recall
- **Balanced Accuracy**: Average of sensitivity and specificity

## Citation

This analysis is part of the MDAE benchmarking project for medical imaging, submitted to ICLR 2025.

## Repository Maintenance

- **Active Scripts**: Located in `scripts/ablation/` and `scripts/multi_modality/`
- **Legacy Scripts**: Archived in `scripts/legacy/`
- **Documentation**: Consolidated in `documentation/`
- **Latest Results**: August 14, 2025 extraction and analysis

## Contact

For questions about the analysis or data, please refer to the documentation in the respective folders.

---

*Last Updated: August 14, 2025*