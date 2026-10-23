# Phase 8: Cloud Attrition, Selection Bias, and Scope Boundary Audit

**Project Title:** Multi-Temporal Sentinel-2 Spectral-Temporal Features for Seven-Class Crop Classification in Telangana: Spatial Cross-Validation and Independent Test Evaluation  
**Auditor Roles:** Senior Research Engineer, ML Experiment Auditor, Scientific Editor  
**Date:** 2026-10-23  
**Status:** COMPLETED & EMPIRICALLY RECONCILED  

---

## 1. Executive Summary

This audit establishes the explicit mathematical accounting of satellite feature extraction failures on `Test.csv`, analyzes crop-dependent selection bias, and formalizes strict operational scope boundaries to prevent scientific over-claiming.

**Core Findings:**
1. **Explicit Metric Denominators:**  
   The reported independent test metrics (Overall Accuracy: **81.55%**, Macro-F1: **0.7308**, Balanced Accuracy: **0.7326**, Weighted-F1: **0.8137**) apply strictly to the **$N = 2,265$ successfully extracted parcels (83.33%)**, not the full cohort of 2,718 surveyed parcels.
2. **Crop-Dependent Selection Bias:**  
   Cloud and cloud-shadow exclusion is strongly non-uniform. Monsoon flood-irrigated crops suffer severe extraction loss (Paddy: **25.91% exclusion**, 185 parcels lost), whereas dry-season winter crops experience high coverage (Bengal Gram: **4.17% exclusion**, 2 parcels lost). Consequently, the evaluated subset is not a uniform random sample of Telangana agriculture.
3. **Strict Scope Demarcation (Crop Classification vs Future Extensions):**  
   The verified empirical contribution of this study is **seven-class crop-type classification**. Secondary target modeling (`category` for disease/pest diagnosis), weather reanalysis integration (ERA5-Land), crop evapotranspiration ($ET_c$), and FAO-56 Penman-Monteith irrigation advisory algorithms have not been executed or evaluated in this codebase. They must be presented strictly as unvalidated conceptual future work.
4. **Temporal Generalization Boundary:**  
   All imagery and ground truth correspond to the **2023 cropping cycle**. Inter-annual stability across drought or extreme rainfall years has not been tested.

---

## 2. Authoritative Test Parcel Extraction Accounting

Source: `outputs/data/phase7/test_predictions.csv` ($N = 2,718$)

| Crop Class | Evaluated Parcels (`success`) | Cloud-Excluded Parcels (`failed_cloud_obscured`) | Total Surveyed Parcels | Extraction Coverage (%) | Exclusion Rate (%) | Primary Agronomic Context |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Bengal Gram** | 46 | 2 | 48 | **95.83%** | **4.17%** | Dry-season Rabi pulse; minimal cloud cover. |
| **Groundnut** | 241 | 25 | 266 | **90.60%** | **9.40%** | Transitional late Kharif / post-monsoon oilseed. |
| **Maize** | 631 | 68 | 699 | **90.27%** | **9.73%** | Versatile cereal with diverse sowing windows. |
| **Red Gram** | 35 | 6 | 41 | **85.37%** | **14.63%** | Long-duration Kharif pulse (150–180 days). |
| **Cotton** | 472 | 97 | 569 | **82.95%** | **17.05%** | Rainfed monsoon Kharif fiber crop. |
| **Chillies** | 311 | 70 | 381 | **81.63%** | **18.37%** | High-value irrigated spice crop. |
| **Paddy** | 529 | 185 | 714 | **74.09%** | **25.91%** | Heavy monsoon Kharif flood irrigation; persistent cloud/shadow. |
| **Total Test Cohort** | **2,265** | **453** | **2,718** | **83.33%** | **16.67%** | **Over 6× disparity between Bengal Gram and Paddy.** |

---

## 3. Methodological Disclosures Required for Manuscript

1. **Denominators in Abstract and Results:**  
   Every presentation of test performance must state $N = 2,265$ and explicitly disclose the 453 cloud-excluded parcels:  
   *"On the $N = 2,265$ farm parcels (83.33% of the surveyed cohort) with usable multi-temporal Sentinel-2 acquisitions..."*
2. **Denial-of-Service Risk in Optical Remote Sensing:**  
   The manuscript’s discussion must highlight that relying exclusively on optical surface reflectance creates a 16.67% operational data deficit during monsoon seasons, demonstrating why operational crop monitoring systems require all-weather Synthetic Aperture Radar (SAR) fusion (e.g., Sentinel-1 C-band backscatter).
3. **Disclosing Class Distribution Shift:**  
   The test cohort features higher Maize proportions (27.86% evaluated test vs 17.54% training) and lower Bengal Gram proportions (2.03% evaluated test vs 11.96% training). Combined with Paddy’s 25.91% cloud exclusion, this distribution shift inflates overall accuracy and weighted F1 on the test set.

---

## 4. Scope Governance Matrix

| Framework Module | Claimed in Initial Title / Vision | Status in Present Repository | Governance Action in Manuscript |
| :--- | :--- | :--- | :--- |
| **Crop-Type Classification** | Primary 7-class model | **Fully Implemented & Evaluated** (CV Macro-F1: 0.7226; Test Macro-F1: 0.7308) | Present as core validated empirical contribution. |
| **Crop Health Diagnosis** | Secondary target `category` | **Not Trained** (Survey labels preserved, but no ML model trained) | Describe strictly as planned future work; remove any implication of current operational diagnosis. |
| **Meteorological Integration** | ERA5 / CHIRPS ingestion | **Not Implemented** (No weather tables or reanalysis pipelines exist) | Move to future extensions section. |
| **Evapotranspiration ($ET_c$)** | FAO-56 Penman-Monteith | **Not Implemented** (No dual crop coefficient code executed) | Describe as prospective application for crop-type maps. |
| **Irrigation Advisory Engine** | Dynamic scheduling | **Not Implemented** (No advisory generation scripts exist) | Frame as conceptual system vision only. |
| **Multi-Year Validation** | Inter-annual robustness | **Not Evaluated** (Single 2023 cropping cycle) | Explicitly disclose as temporal limitation. |
