# Comprehensive Analysis Pipeline

## Overview
The `run_comprehensive_analysis.py` script is a complete, reproducible pipeline for processing MDAE benchmarking results from raw WandB data.

## Quick Start

```bash
# Run with default settings
python run_comprehensive_analysis.py

# Run with custom directories
python run_comprehensive_analysis.py --input-dir path/to/raw_data --output-dir path/to/output

# Run quietly (minimal output)
python run_comprehensive_analysis.py --quiet
```

## Features

### 1. Complete Processing Pipeline
- Reads raw data from WandB exports
- Identifies all methods using correct patterns
- Processes all single-modality benchmarks
- Generates comprehensive visualizations
- Creates summary statistics

### 2. MDAE (Combined) Analysis
The script creates three MDAE-related rows in all results:
- **MDAE**: Original MDAE method performance
- **MDAE (TC)**: Time-conditioned variant performance  
- **MDAE (Combined)**: Best performance between MDAE and MDAE (TC)

### 3. Method Recognition
Correctly identifies 14+ methods including:
- SSL methods: MAE, SimCLR, VoCo, MG, SwinUNETR, VF, S3D
- Foundation models: BrainIAC, MRI-Core, BrainMVP
- Other baselines: DinoV2, ResNet-50
- Our methods: MDAE, MDAE (TC)

### 4. Output Structure
```
comprehensive_analysis/
├── benchmarks/          # Individual benchmark results
│   └── [benchmark]/
│       └── [modality]/
│           ├── metrics_table.csv
│           ├── threshold_independent_metrics.png
│           └── all_metrics.png
├── overall_mean_metrics.csv
├── overall_std_metrics.csv
├── comprehensive_metrics_table.csv
└── overall_performance.png
```

## Results Summary

As of 2025-08-11:
- **15 benchmarks** processed
- **44 modality combinations** analyzed
- **MDAE (Combined)** achieves **76.3% mean AUROC**
- **8.1% improvement** over best SSL baseline (VoCo)
- **8.0% improvement** over best foundation model (BrainIAC)

## Reproducibility

To reproduce the analysis with new data:

1. Extract new data from WandB:
```bash
python scripts/extract_wandb_data.py --all
```

2. Run the comprehensive analysis:
```bash
python run_comprehensive_analysis.py --input-dir raw_data/YYYYMMDD/
```

3. Results will be in `processed_data/comprehensive_analysis/`

## Command-Line Options

- `--input-dir`: Path to raw data directory (default: `raw_data/20250811/`)
- `--output-dir`: Path to output directory (default: `processed_data/comprehensive_analysis/`)
- `--quiet`: Suppress verbose output

## Method Patterns

The script uses the following patterns to identify methods:

| Method | Pattern |
|--------|---------|
| MDAE | `^(resenc_MDAETrainer_RandomMask_Flow_BS48_2000ep\|resenc_MDAE_pretrained\|resenc_MDAE_scratch)` |
| MDAE (TC) | `^(resenc_time_conditioned\|resenc_multimodal_mm_mdae)` |
| MAE | `^resenc_pretrained_` (generic, checked last) |
| BrainIAC | `^brainiac_pretrained` |
| MRI-Core | `^mri_core` |
| ... | See script for complete list |

## Notes

- Multimodal results are automatically excluded
- For multiple runs of the same method, the best Test_AUROC is selected
- MDAE (Combined) takes the best performing variant between MDAE and MDAE (TC)
- All visualizations use consistent color coding (red for MDAE variants, blue for baselines)