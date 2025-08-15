# Ablation Study - All Tables

## Table of Contents

- [Noise Corruption](#noise-corruption)
- [Masking Ratio](#masking-ratio)
- [Flowmdae Top](#flowmdae-top)
- [Summary Comparison](#summary-comparison)
- [Best Configs](#best-configs)

---

## Noise Corruption

| benchmark                                | Flow              | VE                | VP                 | Best   |
|:-----------------------------------------|:------------------|:------------------|:-------------------|:-------|
| brats18_lgg_vs_hgg                       | 0.578 ± 0.146 (9) | 0.544 ± 0.129 (8) | 0.594 ± 0.102 (11) | VP     |
| rsna_miccai_brain_tumor_mgmt_methylation | 0.478 ± nan (1)   | 0.469 ± nan (1)   | 0.566 ± nan (1)    | VP     |
| ucsf_pdgm_idh_classification             | 0.725 ± 0.074 (6) | 0.749 ± 0.048 (6) | 0.722 ± 0.088 (6)  | VE     |
| upenn_gbm_age_group                      | 0.801 ± 0.020 (3) | 0.865 ± 0.064 (3) | 0.825 ± 0.051 (3)  | VE     |
| upenn_gbm_gender                         | 0.824 ± 0.023 (3) | 0.811 ± 0.011 (3) | 0.838 ± 0.001 (2)  | VP     |
| upenn_gbm_gtr_status                     | 0.477 ± 0.069 (3) | 0.465 ± 0.059 (3) | 0.488 ± 0.007 (3)  | VP     |
| upenn_gbm_idh1_status                    | 0.526 ± 0.071 (3) | 0.477 ± 0.061 (3) | 0.495 ± 0.065 (3)  | Flow   |
| upenn_gbm_survival_18month               | 0.540 ± 0.097 (3) | 0.531 ± 0.067 (3) | 0.547 ± 0.047 (3)  | VP     |
| upenn_gbm_survival_1year                 | 0.545 ± 0.076 (3) | 0.543 ± 0.066 (3) | 0.563 ± 0.076 (3)  | VP     |
| upenn_gbm_survival_2year                 | 0.512 ± 0.041 (3) | 0.580 ± 0.028 (3) | 0.477 ± 0.074 (3)  | VE     |

---

## Masking Ratio

|    | 75%               | 80%               | 85%               | 90%               | 95%               |   Optimal |
|---:|:------------------|:------------------|:------------------|:------------------|:------------------|----------:|
|  0 | 0.625 ± 0.105 (4) | 0.610 ± 0.126 (4) | 0.623 ± 0.126 (4) | 0.595 ± 0.163 (4) | 0.568 ± 0.097 (4) |       nan |

---

## Flowmdae Top

|    | Config   | Benchmark          | Modality   |   AUROC |     AP |     F1 |
|---:|:---------|:-------------------|:-----------|--------:|-------:|-------:|
| 18 | M25_N25  | brats18_lgg_vs_hgg | t1ce       |  0.8032 | 0.8013 | 0.6842 |
| 22 | M50_N25  | brats18_lgg_vs_hgg | t1ce       |  0.8032 | 0.7678 | 0.7331 |
| 23 | M50_N75  | brats18_lgg_vs_hgg | t1ce       |  0.7873 | 0.7233 | 0.7159 |
| 17 | M50_N50  | brats18_lgg_vs_hgg | t1ce       |  0.7619 | 0.7135 | 0.646  |
| 28 | M75_N100 | brats18_lgg_vs_hgg | t1ce       |  0.7397 | 0.689  | 0.6299 |
| 20 | M75_N50  | brats18_lgg_vs_hgg | t1ce       |  0.7349 | 0.6882 | 0.6752 |
| 24 | M95_N75  | brats18_lgg_vs_hgg | t1ce       |  0.7254 | 0.6712 | 0.6495 |
| 16 | M25_N75  | brats18_lgg_vs_hgg | t1ce       |  0.7175 | 0.6589 | 0.5884 |
| 31 | M95_N50  | brats18_lgg_vs_hgg | t1ce       |  0.7095 | 0.673  | 0.4815 |
| 32 | M25_N75  | brats18_lgg_vs_hgg | t1         |  0.7016 | 0.679  | 0.4242 |
| 61 | M75_N75  | brats18_lgg_vs_hgg | t2         |  0.6952 | 0.6387 | 0.609  |
| 25 | M50_N100 | brats18_lgg_vs_hgg | t1ce       |  0.6873 | 0.6471 | 0.667  |
| 26 | M75_N25  | brats18_lgg_vs_hgg | t1ce       |  0.681  | 0.6425 | 0.5875 |
| 63 | M95_N50  | brats18_lgg_vs_hgg | t2         |  0.6746 | 0.6219 | 0.6381 |
| 54 | M50_N50  | brats18_lgg_vs_hgg | t2         |  0.6706 | 0.6429 | 0.4722 |

---

## Summary Comparison

|    | Ablation Type     |   N |   Mean AUROC |   Std Dev |    Min |    Max |   Median |
|---:|:------------------|----:|-------------:|----------:|-------:|-------:|---------:|
|  0 | Noise: Flow       |  37 |       0.6138 |    0.1429 | 0.3746 | 0.8516 |   0.5587 |
|  1 | Noise: VE         |  36 |       0.6148 |    0.152  | 0.354  | 0.9178 |   0.5904 |
|  2 | Noise: VP         |  38 |       0.613  |    0.1337 | 0.3927 | 0.8695 |   0.5856 |
|  3 | Mask Ratio: 75%   |   4 |       0.6246 |    0.1052 | 0.5413 | 0.7746 |   0.5913 |
|  4 | Mask Ratio: 80%   |   4 |       0.6101 |    0.1261 | 0.4381 | 0.7397 |   0.6313 |
|  5 | Mask Ratio: 85%   |   4 |       0.623  |    0.1262 | 0.5016 | 0.7873 |   0.6016 |
|  6 | Mask Ratio: 90%   |   4 |       0.5948 |    0.163  | 0.4984 | 0.8381 |   0.5214 |
|  7 | Mask Ratio: 95%   |   4 |       0.5683 |    0.0969 | 0.4286 | 0.6524 |   0.596  |
|  8 | Mask Type: Block  |   4 |       0.6226 |    0.1709 | 0.4254 | 0.8413 |   0.6119 |
|  9 | Mask Type: Random |   4 |       0.6002 |    0.0977 | 0.4563 | 0.6651 |   0.6397 |

---

## Best Configs

|    | Benchmark                                | Noise        | Ratio       | Type          | FlowMDAE        |
|---:|:-----------------------------------------|:-------------|:------------|:--------------|:----------------|
|  0 | upenn_gbm_survival_1year                 | VP (0.563)   | -           | -             | -               |
|  1 | upenn_gbm_survival_2year                 | VE (0.580)   | -           | -             | -               |
|  2 | upenn_gbm_gtr_status                     | VP (0.488)   | -           | -             | -               |
|  3 | upenn_gbm_gender                         | VP (0.838)   | -           | -             | -               |
|  4 | ucsf_pdgm_idh_classification             | VE (0.749)   | -           | -             | -               |
|  5 | upenn_gbm_idh1_status                    | Flow (0.526) | -           | -             | -               |
|  6 | upenn_gbm_age_group                      | VE (0.865)   | -           | -             | -               |
|  7 | upenn_gbm_survival_18month               | VP (0.547)   | -           | -             | -               |
|  8 | rsna_miccai_brain_tumor_mgmt_methylation | VP (0.566)   | -           | -             | -               |
|  9 | brats18_lgg_vs_hgg                       | VP (0.594)   | 75% (0.625) | Block (0.623) | M25_N25 (0.803) |

---

