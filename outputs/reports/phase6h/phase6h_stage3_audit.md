# Phase 6H Stage 3: Spatial Feature Pre-Benchmark Audit Report

**Timestamp:** 2026-10-20T07:07:15.488751  
**Audit Status:** **PASS**  
**Critical Failures:** 0  

## 1. Executive Summary

This audit report validates the complete dataset and feature infrastructure prior to executing the Phase 6H S/T/P feature ablation benchmark across the full 7,843 eligible farm cohort.

| Gate / Verification Item | Target Standard | Observed Status | Verdict |
| :--- | :--- | :--- | :---: |
| **1. Protected Artifact Hashes** | 34 historical files bit-for-bit intact | 34 verified | PASS |
| **2. Cohort Exact Alignment** | 7,843 eligible farms (0 missing, 0 extra) | 7843 farms, 0 duplicates | PASS |
| **3. GLCM 20-Feature Schema** | 5 Haralick x 4 phenological windows | 20 features confirmed | PASS |
| **4. Feature Numerical Sanity** | Zero infs, non-constant, NaN preserved | 0 infs, 0 constant, valid bounds | PASS |
| **5. Dataset Merge Preservation** | Unaltered historical features & labels | 100% exact DataFrame equality | PASS |
| **6. Grouped 5-Fold Separation** | Zero Mandal train-val overlap | 0 Mandal overlap across all 5 folds | PASS |
| **7. Phase 6F Baseline Reproduction** | Exp C Macro-F1 0.7038, Acc 0.7877 | Macro-F1 0.7038, Acc 0.7877 | PASS |

## 2. GLCM Feature Distributions & Missingness

| Feature | Finite Rate (%) | NaN Count | Mean | Std | Min | Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `glcm_contrast_early` | 93.94% | 475 | 0.6485 | 0.5276 | 0.0000 | 8.3839 |
| `glcm_contrast_mid` | 98.69% | 103 | 0.6763 | 0.5395 | 0.0022 | 8.2094 |
| `glcm_contrast_late` | 97.86% | 168 | 0.6976 | 0.5720 | 0.0000 | 8.2202 |
| `glcm_contrast_full` | 100.0% | 0 | 0.5195 | 0.3606 | 0.0000 | 4.7112 |
| `glcm_dissimilarity_early` | 93.94% | 475 | 0.4615 | 0.2214 | 0.0000 | 2.2477 |
| `glcm_dissimilarity_mid` | 98.69% | 103 | 0.4766 | 0.2261 | 0.0022 | 2.1687 |
| `glcm_dissimilarity_late` | 97.86% | 168 | 0.4846 | 0.2285 | 0.0000 | 2.0808 |
| `glcm_dissimilarity_full` | 100.0% | 0 | 0.4088 | 0.1801 | 0.0000 | 1.5897 |
| `glcm_homogeneity_early` | 93.94% | 475 | 0.7872 | 0.0868 | 0.3532 | 1.0000 |
| `glcm_homogeneity_mid` | 98.69% | 103 | 0.7810 | 0.0875 | 0.3957 | 0.9989 |
| `glcm_homogeneity_late` | 97.86% | 168 | 0.7781 | 0.0873 | 0.3821 | 1.0000 |
| `glcm_homogeneity_full` | 100.0% | 0 | 0.8065 | 0.0751 | 0.4681 | 1.0000 |
| `glcm_energy_early` | 93.94% | 475 | 0.5113 | 0.1451 | 0.1624 | 1.0000 |
| `glcm_energy_mid` | 98.69% | 103 | 0.5011 | 0.1414 | 0.1768 | 0.9975 |
| `glcm_energy_late` | 97.86% | 168 | 0.4972 | 0.1405 | 0.1649 | 1.0000 |
| `glcm_energy_full` | 100.0% | 0 | 0.5377 | 0.1362 | 0.2060 | 1.0000 |
| `glcm_correlation_early` | 93.94% | 475 | 0.4870 | 0.1272 | -0.0151 | 1.0000 |
| `glcm_correlation_mid` | 98.69% | 103 | 0.4835 | 0.1286 | -0.1614 | 0.9485 |
| `glcm_correlation_late` | 97.86% | 168 | 0.4885 | 0.1255 | -0.2585 | 1.0000 |
| `glcm_correlation_full` | 100.0% | 0 | 0.4525 | 0.1288 | -0.1614 | 1.0000 |

## 3. Missingness by Crop Cohort

| Crop | Total Farms | Early NaN | Mid NaN | Late NaN | Full-Season NaN |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Bengal Gram | 938 | 4 (0.4%) | 0 (0.0%) | 4 (0.4%) | 0 (0.0%) |
| Chillies | 601 | 34 (5.7%) | 17 (2.8%) | 41 (6.8%) | 0 (0.0%) |
| Cotton | 1679 | 93 (5.5%) | 34 (2.0%) | 55 (3.3%) | 0 (0.0%) |
| Groundnut | 768 | 22 (2.9%) | 0 (0.0%) | 2 (0.3%) | 0 (0.0%) |
| Maize | 1376 | 137 (10.0%) | 31 (2.3%) | 29 (2.1%) | 0 (0.0%) |
| Paddy | 2260 | 137 (6.1%) | 19 (0.8%) | 36 (1.6%) | 0 (0.0%) |
| Red Gram | 221 | 48 (21.7%) | 2 (0.9%) | 1 (0.5%) | 0 (0.0%) |

## 4. Methodological Implementation Conformance

- **Input Band & Resolution:** B8 (NIR Broad, 842nm, 10m native)
- **Quantization & Scaling:** 32 levels [0, 31] linear scaling of surface reflectance [0, 10000]
- **Kernel Size:** 3x3 window (30m x 30m)
- **Directional Averaging:** Isotropic omnidirectional average across 0, 45, 90, 135 deg
- **Texture Descriptors:** contrast, dissimilarity, homogeneity, energy (sqrt(ASM)), correlation
- **Aggregation & Missingness:** ee.Reducer.mean() over exact farm parcel geometry, cloudy windows strictly preserved as NaN (zero fabricated zeros)

## 5. Audit Decision & Next Steps

All integrity gates passed with zero critical failures. The spatial texture features, dataset merge, grouped cross-validation folds, and historical baseline reproduction are certified for the Stage 3 feature ablation benchmark.
