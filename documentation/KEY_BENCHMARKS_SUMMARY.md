# Key Benchmarks Summary for ICLR Paper

## Overview
This document summarizes the key benchmark results for the MDAE ICLR paper submission, focusing on 16 representative benchmark/modality combinations.

## Results Summary

### Category 1: T1/T2 Modalities (In-distribution)
**10 combinations** from benchmarks where T1 and T2 are the primary modalities used in pretraining.

| Benchmark | Modalities | MDAE Performance |
|-----------|------------|------------------|
| BraTS18: LGG_vs_HGG | T1, T2 | ✓ |
| BraTS23: Glio_vs_Met | T1, T2 | ✓ |
| RSNA-MICCAI: MGMT | T1, T2 | ✓ |
| UPenn-GBM: Surv_18M | T1, T2 | ✓ |
| UPenn-GBM: IDH1 | T1, T2 | ✓ |

**Result**: MDAE achieves **72.8% ± 14.5%** mean AUROC

### Category 2: Generalization Test
**6 combinations** testing generalization to less common modalities and mixed contrasts.

| Benchmark | Modalities | MDAE Performance |
|-----------|------------|------------------|
| UCSF-PDGM: IDH | ASL | ✓ |
| UCSF-PDGM: IDH | SWI | ✓ |
| UPenn-GBM: Age | FLAIR | ✓ |
| UPenn-GBM: GTR | T1GD | ✓ |
| TCGA-GBM: DSS_1Y | Mixed | ✓ |
| TCGA-GBM: PFI_1Y | Mixed | ✓ |

**Result**: MDAE achieves **76.9% ± 10.3%** mean AUROC

### Overall Key Benchmarks
**All 16 combinations** combined.

**Result**: MDAE achieves **74.8% ± 12.0%** mean AUROC

## Command Usage

```bash
# Generate results for T1/T2 category (Table 1)
python run_comprehensive_analysis.py --key-benchmarks --key-benchmarks-category t1t2

# Generate results for generalization category (Table 2)
python run_comprehensive_analysis.py --key-benchmarks --key-benchmarks-category generalization

# Generate all key benchmarks
python run_comprehensive_analysis.py --key-benchmarks
```

## Output Location
Results are saved to: `processed_data_key_benchmarks/[category]/`

## Key Fixes Applied
- **T1n Mapping**: Fixed T1n (native T1) to map to standardized 't1' modality
- **Modality Standardization**: Ensures consistent naming across all benchmarks
- **BraTS23 Fix**: Now correctly includes both T1 and T2 for BraTS23_Glio_vs_Met

## Performance Comparison

| Method | T1/T2 (In-dist) | Generalization | Overall |
|--------|-----------------|----------------|---------|
| MDAE | **72.8%** | **76.9%** | **74.8%** |
| VoCo | 63.8% | 69.2% | 66.5% |
| BrainIAC | 66.0% | 66.6% | 66.2% |
| DinoV2 | 61.7% | 71.1% | 66.4% |

MDAE consistently outperforms all baselines across both categories, demonstrating strong performance on in-distribution T1/T2 modalities and excellent generalization to other imaging sequences.