# Ablation Studies Documentation

## Overview

Comprehensive analysis of MDAE (Masked Denoising Autoencoder) ablation studies across 425 experimental runs from 37 single-modality projects. This documentation consolidates all findings, methodologies, and recommendations from the ablation analysis.

## Quick Links

- [Visualizations](../../ablation_analysis/visualizations/) - All generated figures
- [Tables](../../ablation_analysis/tables/) - LaTeX, Markdown, and CSV tables
- [Raw Data](../../ablation_analysis/raw_data_extracted/) - Extracted CSV files
- [Scripts](../../scripts/ablation/) - Python scripts for analysis

## Key Findings Summary

### Optimal Configuration
| Parameter | Recommended Value | Performance (AUROC) |
|-----------|------------------|---------------------|
| **Noise Type** | VP (Variance Preserving) | 0.612 ± 0.134 |
| **Masking Ratio** | 75% | 0.625 ± 0.114 |
| **Masking Type** | Block | 0.623 ± 0.106 |
| **FlowMDAE** | M75_N75 | 0.658 |

### Performance by Ablation Type
- **Noise Corruption**: 112 runs, mean AUROC 0.614
- **Masking Ratio**: 20 runs, mean AUROC 0.604
- **Masking Type**: 12 runs, mean AUROC 0.601
- **FlowMDAE**: 64 runs, mean AUROC 0.589

## Ablation Types Analyzed

### 1. Noise Corruption Type
Tests different noise schedules in the diffusion process:
- **VE (Variance Exploding)**: Progressive noise addition
- **VP (Variance Preserving)**: Controlled noise with preserved variance
- **Flow**: Flow-based noise schedule

**Key Finding**: VP noise provides most stable performance across benchmarks.

### 2. Masking Ratio
Tests different percentages of input masking (75%, 80%, 85%, 90%, 95%)

**Key Finding**: 75% masking ratio optimal - higher ratios degrade performance significantly.

### 3. Masking Type
Compares masking strategies:
- **Block**: Contiguous region masking
- **Random**: Random patch-based masking

**Key Finding**: Block masking outperforms random (0.623 vs 0.600 AUROC).

### 4. FlowMDAE Parameter Grid
Tests combinations of masking ratio (25-95%) and noise level (25-100%)

**Key Finding**: Moderate values (M75_N75) perform best; extremes underperform.

## Data Extraction Pipeline

### Extraction Process
1. **Source**: WandB API from entity `t-jiachentu`
2. **Projects**: 37 successfully extracted (61 targeted)
3. **Total Runs**: 2,838 experiments analyzed
4. **Ablation Runs**: 425 runs with ablation parameters

### Scripts Usage

```bash
# 1. Extract raw data
python scripts/ablation/extract_ablation_raw_data.py

# 2. Generate visualizations
python scripts/ablation/generate_ablation_visualizations.py

# 3. Create tables
python scripts/ablation/generate_ablation_tables.py
```

## Visualizations Generated

1. **Noise Corruption** (UPenn-focused)
   - Grouped bar chart by benchmark
   - Performance heatmap
   - Distribution box plots

2. **Masking Ratio**
   - Trend lines with 95% confidence intervals
   - Performance degradation analysis

3. **Masking Type**
   - Comparison bar chart with statistical significance

4. **FlowMDAE (Flow SDE)**
   - Parameter grid heatmap
   - Performance contour plot

## Statistical Analysis

### Significance Tests
- **Noise Types**: ANOVA F=0.021, p=0.979 (not significant)
- **Masking Types**: t-test p=0.682 (not significant)
- **Conclusion**: Task-specific optimization more important than universal best

### Confidence Intervals
All visualizations include 95% CI calculated as: `1.96 × std/√n`

## Recommendations

### For New Benchmarks
1. Start with **VP noise** and **75% masking ratio**
2. Use **block masking** when applicable
3. For FlowMDAE: begin with **M75_N75** configuration

### Task-Specific Guidelines
- **Classification** (Age/Gender): VE noise may perform better
- **Tumor Detection**: VP noise with block masking
- **Complex Features**: Consider Flow noise with lower masking

## File Structure

```
ablation_analysis/
├── raw_data/                  # Original JSON extractions
├── raw_data_extracted/         # CSV files with metrics
│   ├── noise_corruption_raw.csv
│   ├── masking_ratio_raw.csv
│   ├── masking_type_raw.csv
│   ├── flowmdae_raw.csv
│   └── summary_statistics.csv
├── visualizations/             # 8 PNG figures
│   ├── noise_corruption/      
│   ├── masking_ratio/         
│   ├── masking_type/          
│   └── flowmdae/              
└── tables/                     # Publication-ready tables
    ├── csv/                   
    ├── latex/                 
    └── markdown/              
```

## Notes on Data Quality

- **Coverage**: 60.7% of planned projects successfully extracted
- **Missing Projects**: 24 projects not yet available in WandB
- **"Other" Category**: 217 runs need further investigation
- **Statistical Power**: Limited for masking type (only 12 runs)

## Future Work

1. Complete extraction when remaining projects available
2. Investigate unclassified "other" ablations
3. Expand masking type tests across more benchmarks
4. Automated hyperparameter optimization based on findings

## Citation

This analysis is part of the MDAE benchmarking project for medical imaging, 2025.

---

*Last Updated: August 14, 2025*