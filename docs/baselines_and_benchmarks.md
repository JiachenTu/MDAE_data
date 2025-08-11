# MDAE Baselines and Benchmarks

## Pretraining Dataset
- **OpenMind Dataset** - Reference: `/home/t-jiachentu/repos/benchmarking/misc/MDAE_Paper/related_articles/OpenMind_3D_SSL.pdf`

## Baseline Models

### Foundation Models
1. **BrainMVP** - Reference: `/home/t-jiachentu/repos/benchmarking/misc/MDAE_Paper/related_articles/BrainMVP.pdf`
2. **BrainIAC** - Reference: `/home/t-jiachentu/repos/benchmarking/misc/MDAE_Paper/related_articles/BrainIAC_A foundation model for generalized brain MRI analysis.pdf`
3. **MRI-CORE** - Reference: `/home/t-jiachentu/repos/benchmarking/misc/MDAE_Paper/related_articles/MRI_Core.pdf`

### OpenMind-based Methods
Trained under the same settings as described in the OpenMind paper:
- VF (Vision Foundation)
- VoCo (Volume Contrastive)
- MG (Multi-Grid)
- SimCLR
- SwinUNETR
- MAE (Masked Autoencoder)

## Benchmark Tasks

*Note: Star ratings (★) indicate relative importance/priority for the paper*

### 1. BRATS18: LGG vs HGG Classification
**Focus Modalities:** T1, T2

| Modality | WandB Project | Priority |
|----------|---------------|----------|
| T1 | [july_stratified_brats18_lgg_vs_hgg_single_t1](https://microsoft-research.wandb.io/t-jiachentu/july_stratified_brats18_lgg_vs_hgg_single_t1) | ★★ |
| T1ce | [july_stratified_brats18_lgg_vs_hgg_single_t1ce](https://microsoft-research.wandb.io/t-jiachentu/july_stratified_brats18_lgg_vs_hgg_single_t1ce) | ★★★★★ |
| T2 | [july_stratified_brats18_lgg_vs_hgg_single_t2](https://microsoft-research.wandb.io/t-jiachentu/july_stratified_brats18_lgg_vs_hgg_single_t2) | ★★ |
| FLAIR | [july_stratified_brats18_lgg_vs_hgg_single_flair](https://microsoft-research.wandb.io/t-jiachentu/july_stratified_brats18_lgg_vs_hgg_single_flair) | - |

### 2. BRATS23: Glioma vs Meningioma
**Focus:** Top 2 performing modalities

| Modality | WandB Project | Priority |
|----------|---------------|----------|
| T1c | [july_stratified_brats23_gli_vs_men_single_t1c](https://microsoft-research.wandb.io/t-jiachentu/july_stratified_brats23_gli_vs_men_single_t1c) | ★★★ |
| T2w | [july_stratified_brats23_gli_vs_men_single_t2w](https://microsoft-research.wandb.io/t-jiachentu/july_stratified_brats23_gli_vs_men_single_t2w) | ★★★ |

### 3. BRATS23: Meningioma vs Metastasis
**Note:** Potential generalization test

| Modality | WandB Project | Priority |
|----------|---------------|----------|
| T1n | [july_stratified_brats23_men_vs_met_single_t1n](https://microsoft-research.wandb.io/t-jiachentu/july_stratified_brats23_men_vs_met_single_t1n) | ★★ |
| T1c | [july_stratified_brats23_men_vs_met_single_t1c](https://microsoft-research.wandb.io/t-jiachentu/july_stratified_brats23_men_vs_met_single_t1c) | ★★ |
| T2w | [july_stratified_brats23_men_vs_met_single_t2w](https://microsoft-research.wandb.io/t-jiachentu/july_stratified_brats23_men_vs_met_single_t2w) | ★ |
| T2f | [july_stratified_brats23_men_vs_met_single_t2f](https://microsoft-research.wandb.io/t-jiachentu/july_stratified_brats23_men_vs_met_single_t2f) | ★ |

### 4. BRATS23: Glioma vs Metastasis

| Modality | WandB Project | Priority |
|----------|---------------|----------|
| T1n | [july_stratified_brats23_gli_vs_met_single_t1n](https://microsoft-research.wandb.io/t-jiachentu/july_stratified_brats23_gli_vs_met_single_t1n) | ★ |
| T1c | [july_stratified_brats23_gli_vs_met_single_t1c](https://microsoft-research.wandb.io/t-jiachentu/july_stratified_brats23_gli_vs_met_single_t1c) | ★ |
| T2f | [july_stratified_brats23_gli_vs_met_single_t2f](https://microsoft-research.wandb.io/t-jiachentu/july_stratified_brats23_gli_vs_met_single_t2f) | ★★ |
| T2w | [july_stratified_brats23_gli_vs_met_single_t2w](https://microsoft-research.wandb.io/t-jiachentu/july_stratified_brats23_gli_vs_met_single_t2w) | ★ |

### 5. RSNA-MICCAI Brain Tumor: MGMT Methylation
**Focus:** All modalities

| Modality | WandB Project | Priority |
|----------|---------------|----------|
| T1w | [july_stratified_rsna_miccai_brain_tumor_mgmt_methylation_single_t1w](https://microsoft-research.wandb.io/t-jiachentu/july_stratified_rsna_miccai_brain_tumor_mgmt_methylation_single_t1w) | ★ |
| T1wce | [july_stratified_rsna_miccai_brain_tumor_mgmt_methylation_single_t1wce](https://microsoft-research.wandb.io/t-jiachentu/july_stratified_rsna_miccai_brain_tumor_mgmt_methylation_single_t1wce) | ★★ |
| T2w | [july_stratified_rsna_miccai_brain_tumor_mgmt_methylation_single_t2w](https://microsoft-research.wandb.io/t-jiachentu/july_stratified_rsna_miccai_brain_tumor_mgmt_methylation_single_t2w) | ★ |
| FLAIR | [july_stratified_rsna_miccai_brain_tumor_mgmt_methylation_single_flair](https://microsoft-research.wandb.io/t-jiachentu/july_stratified_rsna_miccai_brain_tumor_mgmt_methylation_single_flair) | ★★ |

### 6. TCGA-GBM: DSS 1-Year Survival
**Note:** Mixed contrasts

| Modality | WandB Project |
|----------|---------------|
| Mixed | [july_stratified_tcga_gbm_dss_1year_single_mixed_contrasts](https://microsoft-research.wandb.io/t-jiachentu/july_stratified_tcga_gbm_dss_1year_single_mixed_contrasts) |

### 7. TCGA-GBM: PFI 1-Year Survival
**Note:** Mixed contrasts

| Modality | WandB Project |
|----------|---------------|
| Mixed | [july_stratified_tcga_gbm_pfi_1year_single_mixed_contrasts](https://microsoft-research.wandb.io/t-jiachentu/july_stratified_tcga_gbm_pfi_1year_single_mixed_contrasts) |

### 8. UCSF-PDGM: IDH Classification
**Note:** Generalization test

| Modality | WandB Project |
|----------|---------------|
| SWI | [july_stratified_ucsf_pdgm_idh_classification_single_swi](https://microsoft-research.wandb.io/t-jiachentu/july_stratified_ucsf_pdgm_idh_classification_single_swi) |
| ASL | [july_stratified_ucsf_pdgm_idh_classification_single_asl](https://microsoft-research.wandb.io/t-jiachentu/july_stratified_ucsf_pdgm_idh_classification_single_asl) |

### 9. UPenn-GBM: 1-Year Survival
**Focus Modalities:** T1, T2

| Modality | WandB Project | Priority |
|----------|---------------|----------|
| T1 | [july_stratified_upenn_gbm_survival_1year_single_t1](https://microsoft-research.wandb.io/t-jiachentu/july_stratified_upenn_gbm_survival_1year_single_t1) | ★★★ |
| T1gd | [july_stratified_upenn_gbm_survival_1year_single_t1gd](https://microsoft-research.wandb.io/t-jiachentu/july_stratified_upenn_gbm_survival_1year_single_t1gd) | ★ |
| T2 | [july_stratified_upenn_gbm_survival_1year_single_t2](https://microsoft-research.wandb.io/t-jiachentu/july_stratified_upenn_gbm_survival_1year_single_t2) | ★ |
| FLAIR | [july_stratified_upenn_gbm_survival_1year_single_flair](https://microsoft-research.wandb.io/t-jiachentu/july_stratified_upenn_gbm_survival_1year_single_flair) | ★★ |

### 10. UPenn-GBM: 18-Month Survival
**Focus Modalities:** T1, T2

| Modality | WandB Project | Priority |
|----------|---------------|----------|
| T1 | [july_stratified_upenn_gbm_survival_18month_single_t1](https://microsoft-research.wandb.io/t-jiachentu/july_stratified_upenn_gbm_survival_18month_single_t1) | ★ |
| T1gd | [july_stratified_upenn_gbm_survival_18month_single_t1gd](https://microsoft-research.wandb.io/t-jiachentu/july_stratified_upenn_gbm_survival_18month_single_t1gd) | ★★★★★★ |
| T2 | [july_stratified_upenn_gbm_survival_18month_single_t2](https://microsoft-research.wandb.io/t-jiachentu/july_stratified_upenn_gbm_survival_18month_single_t2) | ★★ |
| FLAIR | [july_stratified_upenn_gbm_survival_18month_single_flair](https://microsoft-research.wandb.io/t-jiachentu/july_stratified_upenn_gbm_survival_18month_single_flair) | ★ |

### 11. UPenn-GBM: 2-Year Survival
**Focus Modalities:** T1, T2

| Modality | WandB Project | Priority |
|----------|---------------|----------|
| T1 | [july_stratified_upenn_gbm_survival_2year_single_t1](https://microsoft-research.wandb.io/t-jiachentu/july_stratified_upenn_gbm_survival_2year_single_t1) | ★★★ |
| T1gd | [july_stratified_upenn_gbm_survival_2year_single_t1gd](https://microsoft-research.wandb.io/t-jiachentu/july_stratified_upenn_gbm_survival_2year_single_t1gd) | ★ |
| T2 | [july_stratified_upenn_gbm_survival_2year_single_t2](https://microsoft-research.wandb.io/t-jiachentu/july_stratified_upenn_gbm_survival_2year_single_t2) | ★ |
| FLAIR | [july_stratified_upenn_gbm_survival_2year_single_flair](https://microsoft-research.wandb.io/t-jiachentu/july_stratified_upenn_gbm_survival_2year_single_flair) | - |

### 12. UPenn-GBM: Age Group Classification

| Modality | WandB Project | Priority |
|----------|---------------|----------|
| T1 | [july_stratified_upenn_gbm_age_group_single_t1](https://microsoft-research.wandb.io/t-jiachentu/july_stratified_upenn_gbm_age_group_single_t1) | ★★ |
| T1gd | [july_stratified_upenn_gbm_age_group_single_t1gd](https://microsoft-research.wandb.io/t-jiachentu/july_stratified_upenn_gbm_age_group_single_t1gd) | ★★★ |
| T2 | [july_stratified_upenn_gbm_age_group_single_t2](https://microsoft-research.wandb.io/t-jiachentu/july_stratified_upenn_gbm_age_group_single_t2) | ★★★ |
| FLAIR | [july_stratified_upenn_gbm_age_group_single_flair](https://microsoft-research.wandb.io/t-jiachentu/july_stratified_upenn_gbm_age_group_single_flair) | ★ |

### 13. UPenn-GBM: Gender Classification
**Focus:** T1, T2 - all metrics

| Modality | WandB Project | Priority |
|----------|---------------|----------|
| T1 | [july_stratified_upenn_gbm_gender_single_t1](https://microsoft-research.wandb.io/t-jiachentu/july_stratified_upenn_gbm_gender_single_t1) | ★★ |
| T2 | [july_stratified_upenn_gbm_gender_single_t2](https://microsoft-research.wandb.io/t-jiachentu/july_stratified_upenn_gbm_gender_single_t2) | ★★★ |

### 14. UPenn-GBM: IDH1 Status
**Focus Modalities:** T1, T2

| Modality | WandB Project | Priority |
|----------|---------------|----------|
| T1 | [july_stratified_upenn_gbm_idh1_status_single_t1](https://microsoft-research.wandb.io/t-jiachentu/july_stratified_upenn_gbm_idh1_status_single_t1) | ★ |
| T2 | [july_stratified_upenn_gbm_idh1_status_single_t2](https://microsoft-research.wandb.io/t-jiachentu/july_stratified_upenn_gbm_idh1_status_single_t2) | ★ |

### 15. UPenn-GBM: GTR Status
**Note:** Generalization test

| Modality | WandB Project | Priority |
|----------|---------------|----------|
| T1gd | [july_stratified_upenn_gbm_gtr_status_single_t1gd](https://microsoft-research.wandb.io/t-jiachentu/july_stratified_upenn_gbm_gtr_status_single_t1gd) | ★ |
| FLAIR | [july_stratified_upenn_gbm_gtr_status_single_flair](https://microsoft-research.wandb.io/t-jiachentu/july_stratified_upenn_gbm_gtr_status_single_flair) | ★ |

---

## Dataset Details
Details for all benchmarks can be found at: `/home/t-jiachentu/repos/benchmarking/SSL3D_classification/datasets`