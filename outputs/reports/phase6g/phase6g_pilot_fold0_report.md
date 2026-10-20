# Phase 6G Pilot Report: Bounded Nested Grouped Cross-Validation (Outer Fold 0)

**Project:** AI-Based Remote Sensing Framework for Crop Monitoring and Irrigation Decision Support Using Multi-Temporal Sentinel-2 Satellite Imagery  
**Phase:** Phase 6G — Scientific Model Improvement  
**Stage:** Pilot Stage (Outer Fold 0 Bounded Hyperparameter Optimization)  
**Design:** 3-Fold Inner `StratifiedGroupKFold` on 93 Mandals ($N=6,335$) $\rightarrow$ Final Refit $\rightarrow$ Untouched Outer Fold 0 Validation ($N=1,508$, 23 Mandals)  
**Evaluation Scope:** **EXPLORATORY PILOT RESULT ONLY. NOT A FINAL 5-FOLD GENERALIZATION ESTIMATE.**  

---

## 1. Executive Summary & Outer Fold 0 Comparison

The nested grouped cross-validation pilot was executed successfully with 10 bounded XGBoost candidate configurations on Outer Fold 0 without spatial data leakage. Zero Sub-District groups crossed training/validation boundaries in any inner or outer split.

| Metric | Phase 6F Untuned Baseline (Fold 0) | Phase 6G Tuned Pilot (Fold 0) | Delta |
| :--- | :---: | :---: | :---: |
| **Outer Validation Macro-F1** | **0.7095** | **0.7024** | **-0.0071** |
| **Outer Validation Accuracy** | 0.7818 | 0.7798 | -0.0020 |
| **Balanced Accuracy** | 0.6973 | 0.6917 | -0.0056 |
| **Weighted-F1** | 0.7723 | 0.7695 | -0.0028 |
| **Red Gram F1 Score** | 0.3684 | 0.3636 | -0.0048 |
| **Red Gram Recall** | 0.2500 | 0.2500 | +0.0000 |

---

## 2. Inner Grouped Search: 10-Trial Optimization Ledger

Selected strictly by **Mean Inner Validation Macro-F1** across 3 inner grouped folds on the 93 outer training Mandals ($N=6,335$):

| Trial ID | Configuration Name | Depth | LR | Colsample | Subsample | L2 Reg | Mean Trees | Inner Macro-F1 | Inner Acc | Red Gram F1 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | Phase 6F Baseline (Reference) | 7 | 0.05 | 1.0 | 0.8 | 1.0 | 159 | **0.6649** | 0.7631 | 0.2559 |
| 1 | Depth 6 | Feature Subsample 0.8 | L2 Regularized | 6 | 0.05 | 0.8 | 0.8 | 3.0 | 247 | **0.6738** | 0.7686 | 0.2740 |
| 2 | Depth 5 | Feature Subsample 0.75 | Conservative Trees | 5 | 0.05 | 0.75 | 0.85 | 3.0 | 277 | **0.6742** | 0.7694 | 0.2752 |
| 3 | Depth 6 | Feature Subsample 0.70 | Slower Learning Rate (0.03) | 6 | 0.03 | 0.7 | 0.8 | 5.0 | 398 | **0.6719** | 0.7684 | 0.2727 |
| 4 | Depth 7 | Feature Subsample 0.75 | L2 Regularized (5.0) **(SELECTED)** | 7 | 0.05 | 0.75 | 0.8 | 5.0 | 270 | **0.6797** | 0.7699 | 0.3022 |
| 5 | Depth 8 | Feature Subsample 0.70 | High Leaf Weight (5) | 8 | 0.04 | 0.7 | 0.8 | 5.0 | 354 | **0.6736** | 0.7694 | 0.2743 |
| 6 | Depth 6 | Strong Decorrelation (colsample 0.65) | Faster LR (0.06) | 6 | 0.06 | 0.65 | 0.75 | 2.0 | 169 | **0.6670** | 0.7667 | 0.2407 |
| 7 | Depth 5 | colsample 0.80 | Fast Rate (0.08) | 5 | 0.08 | 0.8 | 0.85 | 1.0 | 150 | **0.6706** | 0.7675 | 0.2559 |
| 8 | Depth 7 | colsample 0.85 | Heavy Regularization (L1=1, L2=10) | 7 | 0.05 | 0.85 | 0.8 | 10.0 | 350 | **0.6751** | 0.7702 | 0.2805 |
| 9 | Depth 6 | colsample 0.85 | Balanced Mild Regularization | 6 | 0.05 | 0.85 | 0.8 | 1.0 | 175 | **0.6649** | 0.7667 | 0.2254 |

> **Selection Rationale:** Trial 4 (`Depth 7 | Feature Subsample 0.75 | L2 Regularized (5.0)`) achieved the highest inner validation macro-F1 (0.6797) across all 3 inner grouped splits. Refit was performed with 270 trees on all 6,335 training farms.

---

## 3. Outer Fold 0 Per-Class Performance Breakdown

| Crop Class | Support | Precision | Recall | F1 Score | Phase 6F Baseline F1 | F1 Delta |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bengal Gram** | 185 | 0.8201 | 0.8378 | **0.8289** | *Ref fold 0* | — |
| **Chillies** | 107 | 0.6081 | 0.4206 | **0.4972** | *Ref fold 0* | — |
| **Cotton** | 337 | 0.7880 | 0.8605 | **0.8227** | *Ref fold 0* | — |
| **Groundnut** | 101 | 0.8246 | 0.9307 | **0.8744** | *Ref fold 0* | — |
| **Maize** | 268 | 0.7068 | 0.6567 | **0.6809** | *Ref fold 0* | — |
| **Paddy** | 454 | 0.8154 | 0.8855 | **0.8490** | *Ref fold 0* | — |
| **Red Gram** | 56 | 0.6667 | 0.2500 | **0.3636** | *Ref fold 0* | — |

---

## 4. Integrity, Leakage & Safety Audit

1. **Outer Validation Isolation:** Outer Fold 0 validation farms ($N=1,508$) were never evaluated during inner hyperparameter search or early stopping. Predictions were generated exactly once after final refit.
2. **Zero Spatial Group Leakage:** 0 Mandals overlap between inner training and inner validation across all 3 splits; 0 Mandals overlap between outer training (93 Mandals) and outer validation (23 Mandals).
3. **Protected Baseline Immutability:** All 24 historical baseline files verified 100% bit-for-bit identical to recorded sha256 hashes.
4. **Runtime & Computational Budget:** Total pilot runtime was 604.98 seconds (~10.1 minutes) for 30 inner fits + 1 outer refit.

---

## 5. Artifact Registry

- Pilot Trial Ledger: [`outputs/reports/phase6g/pilot_fold0_trial_ledger.csv`](file:///c:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/reports/phase6g/pilot_fold0_trial_ledger.csv)
- Pilot JSON Report: [`outputs/reports/phase6g/phase6g_pilot_fold0_report.json`](file:///c:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/reports/phase6g/phase6g_pilot_fold0_report.json)
- Pilot Markdown Report: [`outputs/reports/phase6g/phase6g_pilot_fold0_report.md`](file:///c:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/reports/phase6g/phase6g_pilot_fold0_report.md)
- Outer Fold 0 Predictions: [`outputs/data/phase6g/pilot_fold0_oof_predictions.csv`](file:///c:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/data/phase6g/pilot_fold0_oof_predictions.csv)
- Trained Refit Checkpoint: [`outputs/models/phase6g/xgb_pilot_fold0_selected.joblib`](file:///c:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/models/phase6g/xgb_pilot_fold0_selected.joblib)
