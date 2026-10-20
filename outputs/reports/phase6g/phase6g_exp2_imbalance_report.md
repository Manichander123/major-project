# Phase 6G Experiment 2: Imbalance-Aware Pilot Report (Outer Fold 0)

**Project:** AI-Based Remote Sensing Framework for Crop Monitoring and Irrigation Decision Support Using Multi-Temporal Sentinel-2 Satellite Imagery  
**Phase:** Phase 6G — Scientific Model Improvement  
**Experiment:** Experiment 2 (Class-Imbalance Sample Weighting Pilot on Outer Fold 0)  
**Fixed Baseline:** XGBoost Trial 4 Hyperparameters (`depth=7, lr=0.05, colsample=0.75, subsample=0.8, reg_lambda=5.0`)  
**Validation Design:** 3-Fold Inner `StratifiedGroupKFold` on 93 Mandals ($N=6,335$) $\rightarrow$ Final Refit $\rightarrow$ Untouched Outer Fold 0 ($N=1,508$)  
**Evaluation Scope:** **EXPLORATORY PILOT RESULT ONLY. NOT A FINAL 5-FOLD GENERALIZATION ESTIMATE.**  

---

## 1. Executive Summary & Outer Fold 0 Comparison

Seven class-weighting strategies were evaluated within Outer Fold 0's training partition to test whether training-partition-only sample weighting can mitigate Red Gram recall deficits without causing false-positive cascade in majority classes.

| Metric | Phase 6F Untuned Baseline | Phase 6G Unweighted (Trial 4) | Phase 6G Imbalance-Weighted (Selected) | Delta vs Unweighted |
| :--- | :---: | :---: | :---: | :---: |
| **Outer Validation Macro-F1** | 0.7095 | 0.7024 | **0.7339** | **+0.0315** |
| **Outer Validation Accuracy** | 0.7818 | 0.7798 | 0.7845 | +0.0047 |
| **Balanced Accuracy** | 0.6973 | 0.6950 | **0.7336** | **+0.0386** |
| **Red Gram F1 Score** | 0.3684 | 0.3636 | **0.5263** | **+0.1627** |
| **Red Gram Recall** | 0.2500 (14/56) | 0.2500 (14/56) | **0.4464 (25/56)** | **+0.1964** |
| **Red Gram Precision** | 0.7000 | 0.6667 | 0.6410 | -0.0257 |
| **Red Gram $\rightarrow$ Cotton Confusion** | 35 / 56 | 36 / 56 | **12 / 56** | — |

---

## 2. Inner Grouped Optimization Ledger (7 Weighting Strategies)

All strategies evaluated strictly via 3-Fold `StratifiedGroupKFold` on the 93 training Mandals ($N=6,335$):

| Strategy ID | Name | Weight Formula | Inner Macro-F1 | Inner Acc | Inner Red Gram F1 | Inner Red Gram Recall | RG $\rightarrow$ Cotton Confusion |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `unweighted_ref` | Trial 4 Unweighted (Reference) | `w_c = 1.0` | **0.6797** | 0.7699 | 0.3022 | 0.2061 | 70 / 165 |
| `balanced_inv_freq` | Inverse-Frequency Balanced **(SELECTED)** | `w_c = N / (K * N_c)` | **0.7020** | 0.7725 | 0.4211 | 0.3394 | 56 / 165 |
| `sqrt_balanced` | Smoothed Square-Root Inverse-Frequency | `w_c = sqrt(N_max / N_c)` | **0.6858** | 0.7706 | 0.3291 | 0.2364 | 64 / 165 |
| `red_gram_1.5x` | Targeted Red Gram Upweight (1.5x) | `w_RedGram = 1.5, others = 1.0` | **0.6758** | 0.7687 | 0.2931 | 0.2061 | 69 / 165 |
| `red_gram_2.0x` | Targeted Red Gram Upweight (2.0x) | `w_RedGram = 2.0, others = 1.0` | **0.6861** | 0.7705 | 0.3605 | 0.2545 | 65 / 165 |
| `targeted_1.5x_1.3x` | Targeted Red Gram (1.5x) + Chillies (1.3x) | `w_RedGram = 1.5, w_Chillies = 1.3, others = 1.0` | **0.6799** | 0.7699 | 0.2996 | 0.2061 | 67 / 165 |
| `targeted_2.0x_1.3x` | Targeted Red Gram (2.0x) + Chillies (1.3x) | `w_RedGram = 2.0, w_Chillies = 1.3, others = 1.0` | **0.6779** | 0.7686 | 0.2882 | 0.2000 | 71 / 165 |

> **Selection Rationale:** Strategy `balanced_inv_freq` (`Inverse-Frequency Balanced`) achieved the highest inner validation Macro-F1 (0.7020) and was selected for outer refit. The full outer training partition sample weights were:

```json
{
  "Bengal Gram": 1.2019,
  "Chillies": 1.832,
  "Cotton": 0.6744,
  "Groundnut": 1.3568,
  "Maize": 0.8168,
  "Paddy": 0.5011,
  "Red Gram": 5.4848
}
```

---

## 3. Outer Fold 0 Per-Class Breakdown

| Crop Class | Support | Precision | Recall | F1 Score | Unweighted Pilot F1 | Delta |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bengal Gram** | 185 | 0.8061 | 0.8541 | **0.8294** | *Ref fold 0* | — |
| **Chillies** | 107 | 0.5566 | 0.5514 | **0.5540** | *Ref fold 0* | — |
| **Cotton** | 337 | 0.8040 | 0.8279 | **0.8158** | *Ref fold 0* | — |
| **Groundnut** | 101 | 0.7966 | 0.9307 | **0.8584** | *Ref fold 0* | — |
| **Maize** | 268 | 0.7397 | 0.6679 | **0.7020** | *Ref fold 0* | — |
| **Paddy** | 454 | 0.8457 | 0.8568 | **0.8512** | *Ref fold 0* | — |
| **Red Gram** | 56 | 0.6410 | 0.4464 | **0.5263** | *Ref fold 0* | — |

---

## 4. Scientific Findings & Trade-Off Analysis

1. **Inverse-Frequency Balanced Strategy Wins Decisively:** Contrary to the concern that full inverse-frequency weights ($w_{\text{RedGram}} \approx 5.5\times$ to $6.3\times$) would trigger catastrophic false-positive precision collapse, the regularized XGBoost architecture (`reg_lambda=5.0, min_child_weight=3`) effectively controlled spurious boundary shifts. Red Gram inner precision remained stable at 55.45% (vs 56.67% unweighted) while inner recall surged from 20.61% to 33.94%.
2. **Substantial Outer Fold 0 Gains:** On the unseen Outer Fold 0 validation partition ($N=1,508$), `balanced_inv_freq` elevated Macro-F1 to **0.7339** (+0.0315 over unweighted Trial 4, and +0.0244 over the Phase 6F baseline). Balanced accuracy rose to **0.7336** (+0.0386).
3. **Major Reduction in Cotton Confusion:** On Outer Fold 0, Red Gram true positives nearly doubled from 14/56 to 25/56 (Recall: 44.64%, F1: 0.5263), while misclassifications into Cotton were cut by nearly two-thirds from 36/56 down to 12/56. Chillies recall also improved from 42.51% to 50.81% in inner folds.
4. **Generalization Scope Boundary:** These findings are strictly exploratory pilot results on Outer Fold 0. While highly promising, a full 5-fold nested grouped cross-validation is required before declaring a final, definitive generalization claim across all regions.

---

## 5. Artifact Registry

- Strategy Ledger: [`outputs/reports/phase6g/exp2_imbalance_pilot_ledger.csv`](file:///c:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/reports/phase6g/exp2_imbalance_pilot_ledger.csv)
- Experiment 2 JSON Report: [`outputs/reports/phase6g/phase6g_exp2_imbalance_report.json`](file:///c:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/reports/phase6g/phase6g_exp2_imbalance_report.json)
- Experiment 2 Markdown Report: [`outputs/reports/phase6g/phase6g_exp2_imbalance_report.md`](file:///c:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/reports/phase6g/phase6g_exp2_imbalance_report.md)
- Outer Fold 0 Predictions: [`outputs/data/phase6g/exp2_imbalance_pilot_oof_predictions.csv`](file:///c:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/data/phase6g/exp2_imbalance_pilot_oof_predictions.csv)
- Selected Model Checkpoint: [`outputs/models/phase6g/xgb_exp2_selected.joblib`](file:///c:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/models/phase6g/xgb_exp2_selected.joblib)
