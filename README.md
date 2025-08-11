# MDAE Data Repository

This repository contains the data extraction, processing, and analysis infrastructure for the MDAE (Masked Denoising Autoencoder) paper benchmarking experiments.

## Overview

This repository organizes and tracks experimental results from various brain MRI classification benchmarks, comparing MDAE against established baseline models including BrainMVP, BrainIAC, MRI-CORE, and OpenMind-based methods.

## Repository Structure

```
MDAE_data/
├── benchmarks/          # Benchmark task definitions and metadata
├── baselines/           # Baseline model configurations
├── scripts/             # Data extraction and analysis scripts
├── raw_data/            # Raw extracted data from WandB
├── processed_data/      # Cleaned and aggregated results
├── outputs/             # Final tables, figures, and reports
└── docs/                # Documentation
```

## Setup

### Prerequisites

- Python 3.8+
- WandB account with access to experiment runs
- Environment variable `WANDB_API_KEY` set

### Installation

```bash
# Clone the repository
git clone /home/t-jiachentu/repos/benchmarking/misc/data MDAE_data
cd MDAE_data

# Install dependencies
pip install -r requirements.txt

# Set up WandB API key (if not already set)
export WANDB_API_KEY=your_api_key_here
```

## Usage

### 1. Extract Data from WandB

```bash
# Extract data for a specific benchmark
python scripts/extract_wandb_data.py --benchmark brats18/lgg_vs_hgg

# Extract all benchmark data
python scripts/extract_wandb_data.py --all

# Extract with specific date range
python scripts/extract_wandb_data.py --benchmark brats18/lgg_vs_hgg --start-date 2024-07-01
```

### 2. Aggregate Results

```bash
# Aggregate results across all baselines for a benchmark
python scripts/aggregate_results.py --benchmark brats18/lgg_vs_hgg

# Generate comparison tables
python scripts/generate_tables.py --format latex --output outputs/tables/
```

### 3. Generate Reports

```bash
# Generate comprehensive report
python scripts/generate_report.py --output outputs/reports/mdae_benchmark_report.pdf
```

## Benchmark Tasks

The repository tracks 15 benchmark tasks across multiple datasets:

1. **BRATS18**: LGG vs HGG classification
2. **BRATS23**: Multiple tumor type classifications
3. **RSNA-MICCAI**: MGMT methylation prediction
4. **TCGA-GBM**: Survival predictions
5. **UCSF-PDGM**: IDH classification
6. **UPenn-GBM**: Various clinical outcome predictions

See `docs/baselines_and_benchmarks.md` for complete details.

## Baseline Models

### Foundation Models
- **BrainMVP**: Brain Masked Vision Pretraining
- **BrainIAC**: Foundation model for generalized brain MRI analysis
- **MRI-CORE**: Core MRI analysis model

### OpenMind-based Methods
- VF (Vision Foundation)
- VoCo (Volume Contrastive)
- MG (Multi-Grid)
- SimCLR
- SwinUNETR
- MAE (Masked Autoencoder)

## Data Organization

### Benchmark Metadata
Each benchmark is defined with:
- Task description
- Modalities tested
- WandB project URLs
- Evaluation metrics
- Priority levels (indicated by ★ ratings)

### Extracted Data Format
- **Raw data**: Complete WandB run information in JSON
- **Processed data**: Standardized CSV/Parquet files
- **Aggregated results**: Summary statistics and comparisons

## Key Scripts

### `extract_wandb_data.py`
Connects to WandB API and extracts experiment metrics.

### `aggregate_results.py`
Combines results across modalities and baselines.

### `generate_tables.py`
Creates publication-ready tables in LaTeX/Markdown format.

## Contributing

When adding new benchmarks or baselines:

1. Create appropriate JSON configuration in `benchmarks/` or `baselines/`
2. Update `docs/baselines_and_benchmarks.md`
3. Run extraction scripts to populate data
4. Commit changes with descriptive message

## WandB Projects

All experiments are tracked under the Microsoft Research WandB organization:
- Entity: `t-jiachentu`
- Projects: `july_stratified_*`

## Citation

If you use this data or infrastructure, please cite:
```bibtex
@article{mdae2024,
  title={Masked Denoising Autoencoder for Brain MRI Analysis},
  author={...},
  year={2024}
}
```

## Contact

For questions or issues, please contact the repository maintainer.

## License

This project is licensed under the MIT License - see the LICENSE file for details.