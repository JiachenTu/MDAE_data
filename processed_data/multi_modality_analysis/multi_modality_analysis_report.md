# Multi-Modality Analysis Report

Generated: 2025-08-14 18:54:49

## Executive Summary

- **Average Improvement**: -0.35%
- **Best Improvement**: tcga_gbm_pfi_1year (34.34%)
- **Worst Performance**: upenn_gbm_gtr_status (-16.18%)
- **Positive Improvements**: 5/15 benchmarks

## Detailed Results

| Benchmark                  |   Single_MDAE_AUROC |   Multi_MDAE_AUROC |   Improvement |   Improvement_% |   Multi_MAE_AUROC |   Multi_SimCLR_AUROC |   Multi_VoCo_AUROC |   Multi_BrainMVP_AUROC |
|:---------------------------|--------------------:|-------------------:|--------------:|----------------:|------------------:|---------------------:|-------------------:|-----------------------:|
| tcga_gbm_pfi_1year         |            0.744361 |           1        |    0.255639   |       34.3434   |          1        |             1        |           1        |               0.7      |
| ucsf_pdgm_idh              |            0.813187 |           0.929792 |    0.116606   |       14.3393   |          0.849206 |             0.802198 |           0.915751 |               0.802198 |
| brats18_lgg_vs_hgg         |            0.84127  |           0.92381  |    0.0825397  |        9.81132  |          0.900794 |             0.879365 |           0.847619 |               0.852381 |
| upenn_gbm_survival_2year   |            0.649823 |           0.675532 |    0.0257092  |        3.95633  |          0.573582 |             0.581782 |           0.498892 |               0.566489 |
| brats23_men_vs_met         |            0.964865 |           0.969595 |    0.00472975 |        0.490198 |          0.961487 |             0.758784 |           0.790878 |               0.531757 |
| brats23_gli_vs_men         |            0.978844 |           0.976971 |   -0.00187266 |       -0.191313 |          0.961939 |             0.96938  |           0.998583 |               0.896852 |
| tcga_gbm_dss_1year         |            0.75974  |           0.75     |   -0.00974023 |       -1.28205  |          1        |             1        |           0.625    |               0        |
| brats23_gli_vs_met         |            0.972566 |           0.944757 |   -0.027809   |       -2.85934  |          0.695974 |             0.517463 |           0.883006 |               0.645646 |
| upenn_gbm_gender           |            0.870213 |           0.843769 |   -0.0264438  |       -3.03878  |          0.839514 |             0.864438 |           0.863526 |               0.560942 |
| upenn_gbm_age_group        |            0.917824 |           0.884259 |   -0.0335649  |       -3.65701  |          0.874421 |             0.860243 |           0.85489  |               0.785012 |
| upenn_gbm_idh1_status      |            0.695876 |           0.641875 |   -0.054001   |       -7.76014  |          0.730486 |             0.71674  |           0.487973 |               0.631075 |
| upenn_gbm_survival_1year   |            0.646521 |           0.590065 |   -0.0564563  |       -8.73232  |          0.587559 |             0.57046  |           0.595224 |               0.647258 |
| upenn_gbm_survival_18month |            0.69458  |           0.630315 |   -0.0642649  |       -9.25234  |          0.586498 |             0.555501 |           0.610841 |               0.438494 |
| rsna_miccai_mgmt           |            0.643499 |           0.545455 |   -0.0980443  |      -15.2361   |          0.542283 |             0.539641 |           0.572939 |               0.553118 |
| upenn_gbm_gtr_status       |            0.608929 |           0.510417 |   -0.0985119  |      -16.1779   |          0.60878  |             0.533036 |           0.563095 |               0.534524 |

## Per-Benchmark Analysis

### Brats18 Lgg Vs Hgg

**Multi-Modality Top 5 Methods:**
- MDAE: 0.9238
- MDAE (TC): 0.9206
- MAE: 0.9008
- S3D: 0.8937
- SimCLR: 0.8794

**MDAE Improvement:** 0.0825 (9.81%)

### Brats23 Gli Vs Men

**Multi-Modality Top 5 Methods:**
- VoCo: 0.9986
- VF: 0.9957
- SwinUNETR: 0.9942
- MDAE (TC): 0.9880
- MDAE: 0.9770

**MDAE Improvement:** -0.0019 (-0.19%)

### Brats23 Gli Vs Met

**Multi-Modality Top 5 Methods:**
- MDAE (TC): 0.9669
- MDAE: 0.9448
- VoCo: 0.8830
- VF: 0.7798
- ResNet-50 (Scratch): 0.7327

**MDAE Improvement:** -0.0278 (-2.86%)

### Brats23 Men Vs Met

**Multi-Modality Top 5 Methods:**
- MDAE (TC): 0.9723
- MDAE: 0.9696
- ResNet-50 (Scratch): 0.9625
- MAE: 0.9615
- BrainIAC: 0.9176

**MDAE Improvement:** 0.0047 (0.49%)

### Rsna Miccai Mgmt

**Multi-Modality Top 5 Methods:**
- MG: 0.5832
- VoCo: 0.5729
- BrainMVP: 0.5531
- MDAE: 0.5455
- MAE: 0.5423

**MDAE Improvement:** -0.0980 (-15.24%)

### Tcga Gbm Dss 1Year

**Multi-Modality Top 5 Methods:**
- MAE: 1.0000
- MDAE (TC): 1.0000
- SimCLR: 1.0000
- SwinUNETR: 1.0000
- Unknown: 0.8750

**MDAE Improvement:** -0.0097 (-1.28%)

### Tcga Gbm Pfi 1Year

**Multi-Modality Top 5 Methods:**
- MAE: 1.0000
- MDAE: 1.0000
- VoCo: 1.0000
- SimCLR: 1.0000
- SwinUNETR: 0.8000

**MDAE Improvement:** 0.2556 (34.34%)

### Ucsf Pdgm Idh

**Multi-Modality Top 5 Methods:**
- MDAE: 0.9298
- MG: 0.9231
- VoCo: 0.9158
- VF: 0.9017
- SwinUNETR: 0.8651

**MDAE Improvement:** 0.1166 (14.34%)

### Upenn Gbm Age Group

**Multi-Modality Top 5 Methods:**
- SwinUNETR: 0.8883
- MDAE: 0.8843
- MAE: 0.8744
- MDAE (TC): 0.8672
- SimCLR: 0.8602

**MDAE Improvement:** -0.0336 (-3.66%)

### Upenn Gbm Gender

**Multi-Modality Top 5 Methods:**
- SwinUNETR: 0.8875
- SimCLR: 0.8644
- VoCo: 0.8635
- MDAE (TC): 0.8599
- MDAE: 0.8438

**MDAE Improvement:** -0.0264 (-3.04%)

### Upenn Gbm Gtr Status

**Multi-Modality Top 5 Methods:**
- MAE: 0.6088
- VF: 0.5790
- SwinUNETR: 0.5658
- VoCo: 0.5631
- MDAE (TC): 0.5618

**MDAE Improvement:** -0.0985 (-16.18%)

### Upenn Gbm Idh1 Status

**Multi-Modality Top 5 Methods:**
- MAE: 0.7305
- SimCLR: 0.7167
- SwinUNETR: 0.6672
- MDAE: 0.6419
- BrainMVP: 0.6311

**MDAE Improvement:** -0.0540 (-7.76%)

### Upenn Gbm Survival 18Month

**Multi-Modality Top 5 Methods:**
- SwinUNETR: 0.6921
- VF: 0.6352
- MDAE (TC): 0.6323
- MDAE: 0.6303
- VoCo: 0.6108

**MDAE Improvement:** -0.0643 (-9.25%)

### Upenn Gbm Survival 1Year

**Multi-Modality Top 5 Methods:**
- BrainMVP: 0.6473
- VF: 0.6427
- SwinUNETR: 0.6232
- VoCo: 0.5952
- MDAE: 0.5901

**MDAE Improvement:** -0.0565 (-8.73%)

### Upenn Gbm Survival 2Year

**Multi-Modality Top 5 Methods:**
- SwinUNETR: 0.7092
- MDAE: 0.6755
- MDAE (TC): 0.6164
- SimCLR: 0.5818
- VF: 0.5769

**MDAE Improvement:** 0.0257 (3.96%)