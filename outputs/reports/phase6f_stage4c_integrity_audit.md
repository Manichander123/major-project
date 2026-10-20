# Phase 6F Stage 4C: Scientific Integrity Audit & Verification Report

**Project:** AI-Based Remote Sensing Framework for Crop Monitoring and Irrigation Decision Support Using Multi-Temporal Sentinel-2 Satellite Imagery  
**Audit Stage:** Phase 6F — Stage 4C (Read-Only Scientific Benchmark Integrity Audit)  
**Execution Mode:** GSD + Ponytail Full (Read-Only)  
**Audit Status:** **PASS WITH VERIFIED SCIENTIFIC ADDENDA**  

---

## Executive Summary

A rigorous, independent, read-only audit of all Phase 6F Stage 4B benchmark outputs was conducted across all 15 model-run combinations, feature matrices, cross-validation folds, and stored out-of-fold predictions. All core computational results and data pipelines passed integrity checks, while four critical scientific nuances were identified, mathematically reconciled, and formally documented.

### Key Audit Findings:
1. **Metric Reproducibility (PASS):** All reported pooled macro-F1, accuracy, balanced accuracy, and fold-level metrics across all 15 models were reproduced to 4 decimal places directly from the 94,846 stored out-of-fold predictions.
2. **Complete-Trajectory Discrepancy Reconciled (PASS):** The difference between the 7,107 four-window image metadata count and the 7,099 complete-trajectory modeling cohort was traced to 8 specific farms with historical partial-band nulls in raw satellite extraction.
3. **Red Gram Variability Misattribution Corrected (DOCUMENTATION CORRECTION):** The primary benchmark report previously cited $\pm 0.0230$ and $\pm 0.0112$ as Red Gram variability; these were actually the overall fold macro-F1 standard deviations. The actual per-fold Red Gram F1 standard deviation was audited as **$\pm 0.1309$** in Exp C and **$\pm 0.1918$** in Exp D.
4. **Experiment B Model Rank Reversal Explained (DOCUMENTATION CORRECTION):** Random Forest outperformed XGBoost on the full cohort (0.5101 vs 0.5058), but XGBoost outperformed RF on the complete-case cohort (0.5286 vs 0.5058). This reversal is driven by a structural shift in class distribution—31.7% of Red Gram farms were dropped due to monsoon clouds—not purely by imputation behavior.
5. **Experiment D Provenance Clarified (OPERATIONAL BOUNDARY):** Calendar features (`sowing_month`, `sowing_doy`) originate from farmer survey records (`SDate`). Experiment D must be cited as a Multimodal benchmark (satellite + farmer registration), not an autonomous satellite-only monitoring pipeline.
6. **Protected Baselines (PASS):** All 24 baseline files remained 100% bit-for-bit identical to recorded sha256 hashes.

---

## 1. Population & Cohort Accounting Verification

| Metric | Expected | Actual | Audit Status |
| :--- | :---: | :---: | :---: |
| Harmonized Cohort Rows | 7,888 | 7888 | **PASS** |
| Unique Farm IDs | 7,888 | 7888 | **PASS** |
| Excluded Cloud Failures | 45 | 45 | **PASS** |
| Eligible Modeling Cohort | 7,843 | 7843 | **PASS** |
| Mandal Leakage Across Folds | 0 | 0 Mandals Overlap | **PASS** |
| 7 Classes Present in All Folds | Yes | Yes (All 5 Folds) | **PASS** |

---

## 2. Mathematical Reconciliation: Complete-Trajectory Counts (7,107 vs 7,099)

The Stage 4A preflight reported **7,107** farms with complete 4-window trajectories, while Experiment C complete-trajectory sensitivity evaluated **7,099** farms. The 8 missing farms were audited at the raw band level:

| FarmID | True Crop | District | Sub-District | Missing Feature Group | Root Cause |
| :---: | :---: | :---: | :---: | :--- | :--- |
| 1304367 | Maize | Mahabubabad | Bayyaram | `B5_mid_mean, B5_mid_std, B6_mid_mean...` | Partial band nulls in raw Phase 6D/6E bands |
| 1311790 | Maize | Mahabubabad | Gangaram | `B2_early_mean, B2_early_std, B3_early_mean...` | Partial band nulls in raw Phase 6D/6E bands |
| 1317244 | Paddy | Mahabubabad | Thorrur | `B5_early_mean, B5_early_std, B6_early_mean...` | Partial band nulls in raw Phase 6D/6E bands |
| 1317978 | Cotton | Adilabad | Inderavelly | `B2_late_mean, B2_late_std, B3_late_mean...` | Partial band nulls in raw Phase 6D/6E bands |
| 1320213 | Maize | Mahabubabad | Thorrur | `B2_late_mean, B2_late_std, B3_late_mean...` | Partial band nulls in raw Phase 6D/6E bands |
| 1325016 | Maize | Mahabubabad | Gangaram | `B5_early_mean, B5_early_std, B6_early_mean...` | Partial band nulls in raw Phase 6D/6E bands |
| 1331153 | Maize | Nirmal | Mudhole | `B5_late_mean, B5_late_std, B6_late_mean...` | Partial band nulls in raw Phase 6D/6E bands |
| 1331755 | Red Gram | Adilabad | Utnur | `B2_early_mean, B2_early_std, B3_early_mean...` | Partial band nulls in raw Phase 6D/6E bands |

> **Audit Conclusion:** 7,107 is the valid-image metadata count (`n_images_valid > 0` in all 4 windows). However, 8 farms suffered from partial band nulls during historical satellite extraction (e.g. 10m bands present, but 20m red-edge/SWIR bands null). Therefore, **7,099** is the strictly complete feature matrix cohort with 0 NaNs across all 140 numerical features.

---

## 3. Rare-Class (Red Gram) Precision, Recall & True Variability Audit

The primary benchmark report inadvertently cited the overall macro-F1 standard deviation ($\pm 0.0230$) in discussion of Red Gram. The audited, true per-fold Red Gram statistics are:

### Experiment C (Multi-Temporal Satellite XGBoost)
Pooled Red Gram: Precision = **0.6329**, Recall = **0.2262**, **F1 = 0.3333** (Support = 221)

| Fold | Validation Support | Precision | Recall | Red Gram F1 | Overall Fold Macro-F1 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Fold 0 | 56 | 0.7000 | 0.2500 | **0.3684** | *Ref overall macro* |
| Fold 1 | 26 | 0.2500 | 0.0769 | **0.1176** | *Ref overall macro* |
| Fold 2 | 21 | 0.2727 | 0.1429 | **0.1875** | *Ref overall macro* |
| Fold 3 | 59 | 0.6818 | 0.2542 | **0.3704** | *Ref overall macro* |
| Fold 4 | 59 | 0.8889 | 0.2712 | **0.4156** | *Ref overall macro* |

- **Fold Red Gram F1 Mean:** 0.2919
- **Actual Red Gram Across-Fold Std Dev (ddof=0):** **± 0.1171** (Sample ddof=1: **± 0.1310**)
- **Overall Fold Macro-F1 Std Dev:** ± 0.0230

### Experiment D (Multimodal XGBoost)
Pooled Red Gram: Precision = **0.7733**, Recall = **0.5249**, **F1 = 0.6253** (Support = 221)

| Fold | Validation Support | Precision | Recall | Red Gram F1 |
| :---: | :---: | :---: | :---: | :---: |
| Fold 0 | 56 | 0.7447 | 0.6250 | **0.6796** |
| Fold 1 | 26 | 0.5714 | 0.1538 | **0.2424** |
| Fold 2 | 21 | 0.6429 | 0.4286 | **0.5143** |
| Fold 3 | 59 | 0.8205 | 0.5424 | **0.6531** |
| Fold 4 | 59 | 0.8372 | 0.6102 | **0.7059** |

- **Fold Red Gram F1 Mean:** 0.5591
- **Actual Red Gram Across-Fold Std Dev (ddof=0):** **± 0.1716** (Sample ddof=1: **± 0.1918**)

> **Scientific Note:** Red Gram exhibits wide across-fold F1 variance ($\sigma = 0.1309$ in satellite-only and $\sigma = 0.1918$ in multimodal) because validation support varies heavily across mandals (from 21 in Fold 2 to 59 in Fold 3/4). This confirms that Red Gram uncertainty is substantially higher than the overall model macro-F1 variance.

---

## 4. Experiment B Sensitivity Audit & Model Rank Reversal

- **Primary Cohort ($N=7,843$):** RF Macro-F1 = **0.5101** | XGB Macro-F1 = **0.5058** (Ranking: **RF > XGB**)
- **Complete-Case Cohort ($N=7,298$):** RF Macro-F1 = **0.5058** | XGB Macro-F1 = **0.5286** (Ranking: **XGB > RF**)

### Population Discrepancy Analysis:
- 545 farms were excluded due to early-season cloud cover.
- **70 of the 221 total Red Gram farms (31.67%) were excluded** from the complete-case cohort, compared to only 9 of 938 Bengal Gram farms (0.96%).
- Because the hardest, rarest class was disproportionately removed, the complete-case cohort was structurally easier for gradient boosting.
- **Audit Conclusion:** The rank reversal cannot be cited as proof of imputation superiority. The two runs evaluated fundamentally different populations.

---

## 5. Experiment D Feature Provenance & Operational Availability

| Feature | Source | Operational Availability | Deployment Boundary |
| :--- | :--- | :--- | :--- |
| `Season` | Survey record (`Train.csv`) | Pre-sowing calendar | Valid pre-sowing |
| `sowing_month` | Derived from `SDate` | Available only if farmer declares sowing date | Requires farmer interaction; unavailable in autonomous statewide imagery mapping |
| `sowing_doy` | Derived from `SDate` | Available only if farmer declares sowing date | Same as above |
| `area_ha` | Planar polygon geometry | Farm registration boundary | Fully operational pre-sowing |
| `perimeter_m` | Planar polygon geometry | Farm registration boundary | Fully operational pre-sowing |
| `compactness` | Polsby-Popper quotient | Farm registration boundary | Fully operational pre-sowing |

> **Operational Recommendation:** Experiment D should be formally presented as a **Multimodal Farm-Assisted Benchmark** combining remote sensing with cadastral boundaries and farmer-declared sowing timing, distinct from autonomous remote sensing.

---

## 6. Scientific Tone & Claim Corrections

The following language guidelines are established for paper write-up and downstream documentation:
1. **Retrospective vs Early-Season:** Full-season composite classification (Exp A, macro-F1 0.6174) is strictly retrospective and must not be described as an in-season operational tool.
2. **Irrigation Separation:** High crop classification accuracy proves crop type separability, but does not prove optimal irrigation scheduling; irrigation recommendation requires separate soil moisture and hydrological modeling.
3. **Spatial Extrapolation:** Grouped spatial validation prevents local Mandal leakage, but does not guarantee generalization to agro-climatic zones outside Telangana.
4. **Degrees of Freedom Transparency:** When citing fold-level standard deviations, specify whether population degrees of freedom (`ddof=0`, e.g. $\pm 0.0230$) or sample degrees of freedom (`ddof=1`, e.g. $\pm 0.0257$) are used.

---

## 7. Artifacts Generated

- Audit JSON Report: [`outputs/reports/phase6f_stage4c_integrity_audit.json`](file:///c:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/reports/phase6f_stage4c_integrity_audit.json)
- Audit Markdown Report: [`outputs/reports/phase6f_stage4c_integrity_audit.md`](file:///c:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/reports/phase6f_stage4c_integrity_audit.md)
- Audit Verification Script: [`scripts/audit_phase6f_stage4_benchmark.py`](file:///c:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/scripts/audit_phase6f_stage4_benchmark.py)
- Audit Test Suite: [`tests/test_phase6f_stage4c_audit.py`](file:///c:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/tests/test_phase6f_stage4c_audit.py)
