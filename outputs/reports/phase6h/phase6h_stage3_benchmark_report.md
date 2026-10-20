# Phase 6H Stage 3: Spatial Feature Audit and S/T/P Ablation Benchmark Report

**Timestamp:** 2026-10-20T07:10:16.543557  
**Cohort:** N = 7843 eligible farms across 7 crop classes  
**Cross-Validation:** 5-Fold StratifiedGroupKFold grouped strictly by Sub-District (Mandal)  
**Common Model:** Regularized XGBoost (Trial 4: `max_depth=7`, `colsample=0.75`, `subsample=0.80`, `reg_lambda=5.0`, `min_child_weight=3`, `gamma=0.2`, `reg_alpha=0.2`, `learning_rate=0.05`, `n_estimators=300`)  
**Class Weighting:** Disabled (Unweighted) to evaluate pure feature contributions  

---

## 1. Executive Summary & Core Research Findings

### Primary Research Question:
> *Does adding GLCM spatial texture to multi-temporal Sentinel-2 features improve five-fold spatially grouped crop-classification performance over the existing S+T model, and for which crop classes is the improvement most evident?*

### Key Findings:
1. **Full Feature Fusion (S+T+P vs S+T):**
   - **S+T (140 features):** Macro-F1 = **0.7069**, Accuracy = **0.7873**
   - **S+T+P (160 features):** Macro-F1 = **0.7038**, Accuracy = **0.79**
   - **Net Spatial Contribution:** Macro-F1 delta = **-0.0031**, Accuracy delta = **+0.0027**.

2. **Minority-Class Recognition Impact:**
   - **Red Gram (N=221):** S+T F1 = 0.3683 -> S+T+P F1 = 0.3333 (Delta: **-0.0350**)
   - **Chillies (N=601):** S+T F1 = 0.5062 -> S+T+P F1 = 0.493 (Delta: **-0.0132**)

3. **Historical Reproduction Check:**
   - Historical Phase 6F Experiment C (untuned XGBoost): Macro-F1 = 0.7038, Accuracy = 0.7877.
   - S+T with Trial 4 regularized parameters achieves Macro-F1 = 0.7069 and Accuracy = 0.7873.

---

## 2. Seven-Experiment Controlled Benchmark Table

| Experiment ID | Feature Description | Feature Count | Pooled Macro-F1 | Pooled Accuracy | Balanced Accuracy | Weighted-F1 | Fold Macro-F1 (Mean ± Std) | Runtime |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **S** | Spectral-only full season (34 features) | 34 | **0.6227** | 0.7365 | 0.6219 | 0.7246 | 0.6212 ± 0.0322 | 27.58s |
| **T** | Temporal-only additional features (106 features) | 106 | **0.6948** | 0.7815 | 0.6807 | 0.7733 | 0.6851 ± 0.0194 | 111.29s |
| **P** | Spatial texture only (20 features) | 20 | **0.2945** | 0.3899 | 0.2956 | 0.366 | 0.2905 ± 0.0204 | 43.95s |
| **S+T** | Spectral + Temporal (140 features) | 140 | **0.7069** | 0.7873 | 0.6914 | 0.7796 | 0.6997 ± 0.0228 | 155.02s |
| **S+P** | Spectral + Spatial texture (54 features) | 54 | **0.6282** | 0.7379 | 0.6238 | 0.7263 | 0.625 ± 0.0334 | 57.76s |
| **T+P** | Temporal + Spatial texture (126 features) | 126 | **0.6935** | 0.782 | 0.6782 | 0.773 | 0.6853 ± 0.0143 | 125.61s |
| **S+T+P** | Full feature fusion: Spectral + Temporal + Spatial (160 features) | 160 | **0.7038** | 0.79 | 0.6895 | 0.7818 | 0.6965 ± 0.0173 | 164.82s |

---

## 3. Per-Class F1-Score Breakdown across Feature Groups

| Crop Class | Support | S (34) | T (106) | P (20) | S+T (140) | S+P (54) | T+P (126) | S+T+P (160) | Delta (S+T+P vs S+T) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bengal Gram** | 938 | 0.7520 | 0.8230 | 0.3113 | 0.8278 | 0.7599 | 0.8361 | **0.8385** | **+0.0107** |
| **Chillies** | 601 | 0.4119 | 0.4841 | 0.0731 | 0.5062 | 0.3926 | 0.4553 | **0.4930** | **-0.0132** |
| **Cotton** | 1679 | 0.7455 | 0.7980 | 0.3857 | 0.8016 | 0.7427 | 0.7962 | **0.8060** | **+0.0044** |
| **Groundnut** | 768 | 0.8188 | 0.8643 | 0.4244 | 0.8675 | 0.8200 | 0.8550 | **0.8712** | **+0.0037** |
| **Maize** | 1376 | 0.6573 | 0.7029 | 0.2780 | 0.7049 | 0.6535 | 0.7107 | **0.7105** | **+0.0056** |
| **Paddy** | 2260 | 0.8483 | 0.8672 | 0.5141 | 0.8717 | 0.8556 | 0.8670 | **0.8740** | **+0.0023** |
| **Red Gram** | 221 | 0.1250 | 0.3238 | 0.0749 | 0.3683 | 0.1727 | 0.3344 | **0.3333** | **-0.0350** |

---

## 4. Fold-Level Metrics & Spatial Variability

### S+T Model (140 features):
| Fold | Validation Mandals | N Val | Accuracy | Macro-F1 | Balanced Accuracy | Weighted-F1 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Fold 0 | Sub-District group | 1508 | 0.7798 | 0.7091 | 0.696 | 0.7702 |
| Fold 1 | Sub-District group | 1600 | 0.855 | 0.7369 | 0.7263 | 0.8463 |
| Fold 2 | Sub-District group | 1595 | 0.8013 | 0.6959 | 0.6857 | 0.7938 |
| Fold 3 | Sub-District group | 1530 | 0.7556 | 0.6683 | 0.6548 | 0.7423 |
| Fold 4 | Sub-District group | 1610 | 0.7435 | 0.6883 | 0.6764 | 0.7399 |

### S+T+P Model (160 features):
| Fold | Validation Mandals | N Val | Accuracy | Macro-F1 | Balanced Accuracy | Weighted-F1 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Fold 0 | Sub-District group | 1508 | 0.7832 | 0.7048 | 0.6928 | 0.7719 |
| Fold 1 | Sub-District group | 1600 | 0.8475 | 0.7258 | 0.7158 | 0.8377 |
| Fold 2 | Sub-District group | 1595 | 0.81 | 0.6879 | 0.6784 | 0.8016 |
| Fold 3 | Sub-District group | 1530 | 0.7627 | 0.6759 | 0.6655 | 0.7506 |
| Fold 4 | Sub-District group | 1610 | 0.7453 | 0.6879 | 0.6764 | 0.7417 |

---

## 5. Confusion Matrices (OOF Pooled N=7843)

### S+T Model (140 Features):
```
Predicted ->    Bengal  Chilli  Cotton  Ground   Maize   Paddy  Red Gr
Bengal Gram  [    798       7      30       2      88       5       8 ]
Chillies     [      1     265     113      23      79     116       4 ]
Cotton       [     37      50    1428      31      52      67      14 ]
Groundnut    [      0      18      54     658      12      26       0 ]
Maize        [    118      54      92      20     933     151       8 ]
Paddy        [      6      51      72      12      82    2035       2 ]
Red Gram     [     30       1      95       3      25       9      58 ]
```

### S+T+P Model (160 Features):
```
Predicted ->    Bengal  Chilli  Cotton  Ground   Maize   Paddy  Red Gr
Bengal Gram  [    815       7      25       2      76       5       8 ]
Chillies     [      2     263     118      20      74     120       4 ]
Cotton       [     34      58    1440      26      52      59      10 ]
Groundnut    [      0      19      57     653      12      26       1 ]
Maize        [    119      59      85      19     934     150      10 ]
Paddy        [      7      56      69       9      78    2040       1 ]
Red Gram     [     29       4     100       2      27       8      51 ]
```

---

## 6. Top Discriminative Features in S+T+P

| Rank | Feature Name | Group | Feature Importance |
| :---: | :--- | :---: | :---: |
| 1 | `cig_early` | T (Temporal) | 0.037369 |
| 2 | `B6_full_mean` | S (Spectral) | 0.030661 |
| 3 | `msi_full` | S (Spectral) | 0.025987 |
| 4 | `ndre1_full` | S (Spectral) | 0.025564 |
| 5 | `B11_full_mean` | S (Spectral) | 0.024594 |
| 6 | `B6_early_mean` | T (Temporal) | 0.022341 |
| 7 | `B12_full_mean` | S (Spectral) | 0.019921 |
| 8 | `ndmi_full` | S (Spectral) | 0.018978 |
| 9 | `B11_full_std` | S (Spectral) | 0.018886 |
| 10 | `B11_mid_mean` | T (Temporal) | 0.017106 |
| 11 | `cire_full` | S (Spectral) | 0.016182 |
| 12 | `B12_full_std` | S (Spectral) | 0.015588 |
| 13 | `bsi_full` | S (Spectral) | 0.015240 |
| 14 | `cig_full` | S (Spectral) | 0.014387 |
| 15 | `B6_mid_mean` | T (Temporal) | 0.014031 |

---

## 7. Methodological Discussion & Scientific Nuance

- **Spatial Texture Standalone Performance:** The P-only model (20 GLCM features) achieves a standalone Macro-F1 of 0.2945 (Accuracy: 0.3899), demonstrating that canopy texture contains non-trivial crop discriminatory signal even without spectral indices or temporal curves.
- **Complementarity with Temporal Curves:** When combined with temporal trajectory features (T+P vs T), Macro-F1 moves from 0.6948 to 0.6935.
- **Full Satellite Fusion (S+T+P):** Moving from S+T to S+T+P changes Macro-F1 by -0.0031 and accuracy by +0.0027.
- **Boundary & Parcel Contamination Effects:** For small parcels (<1,000 m²), a 3x3 moving window (30m x 30m) incorporates edge pixels and field boundaries, which can introduce mixed canopy-soil noise. For larger consolidated fields, canopy homogeneity and correlation provide clean architectural separation.
