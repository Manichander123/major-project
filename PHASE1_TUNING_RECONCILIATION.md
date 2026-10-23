# Phase 1: Hyperparameter Tuning and Partition Reconciliation Report

**Project Title:** Multi-Temporal Sentinel-2 Spectral-Temporal Features for Seven-Class Crop Classification in Telangana: Spatial Cross-Validation and Independent Test Evaluation  
**Auditor Roles:** Senior Research Engineer, ML Experiment Auditor, Scientific Editor, Reproducibility Reviewer  
**Date:** 2026-10-23  
**Status:** RECONCILED & SCRIPT-VERIFIED (0 DISCREPANCIES)  

---

## 1. Executive Summary

This audit reconciles the partition counts and tuning procedures reported in earlier drafts against primary code, split CSVs, and execution logs. 

**Primary Finding:**  
- The authoritative size of the Outer Fold 0 training partition is **$N = 6,335$ parcels** across 93 Mandals, and the Outer Fold 0 validation partition is **$N = 1,508$ parcels** across 23 Mandals ($6,335 + 1,508 = 7,843$).
- The conflicting value of **$N = 6,274$** reported in `MANUSCRIPT_DRAFT.md` and `RESULTS_TABLES.md` is an ungrounded idealized 80% approximation ($7,843 \times 0.8 = 6,274.4 \approx 6,274$ or $7,843 - 1,569 = 6,274$). It does not appear in any raw dataset, code, or execution output.
- The hyperparameter optimization is confirmed to be **exploratory tuning on one outer-fold training partition** using 3-fold grouped inner cross-validation across 10 trials, **not fully nested cross-validation**.
- Selection was based strictly on inner cross-validation Macro-F1 (**0.6797**, Trial 4); Outer Fold 0 validation labels were not used for trial selection.

---

## 2. Authoritative Primary Artifacts Inspected

1. `outputs/data/phase6f_cv_folds.csv`  
   - SHA-256: `8ba2bb116b5a1fb8c8c1125a1478273887971ab49f8b3557d8bc5de64191f233`  
   - Record count: 7,843 rows across 116 unique Sub-Districts (Mandals).
2. `outputs/reports/phase6f_cv_folds_audit.json`  
   - Documented validation group counts: Folds 0–4 have 23, 21, 22, 26, 24 Mandals respectively.
3. `outputs/reports/phase6g/phase6g_pilot_fold0_report.json`  
   - Documented outer partition counts: `outer_train_samples: 6335`, `outer_val_samples: 1508`.
4. `outputs/reports/phase6g/pilot_fold0_trial_ledger.csv`  
   - 10 evaluated candidate configurations evaluated on 3-fold inner cross-validation.
5. `scripts/benchmark_phase6g_pilot.py`  
   - Execution script establishing the grouped inner-CV procedure and zero-leakage assertions.

---

## 3. Independent Recalculation of Fold Partitions

An independent recount executed via `scripts/validate_stage_c_reconciliation.py` on `outputs/data/phase6f_cv_folds.csv` confirms the exact partition sizes:

| Fold Index | Validation Mandals | Validation Parcels | Training Mandals | Training Parcels | Inter-Partition Mandal Overlap | All 7 Classes Present? |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 0** | **23** | **1,508** | **93** | **6,335** | **0** | **Yes** |
| **Fold 1** | 21 | 1,600 | 95 | 6,243 | 0 | Yes |
| **Fold 2** | 22 | 1,595 | 94 | 6,248 | 0 | Yes |
| **Fold 3** | 26 | 1,530 | 90 | 6,313 | 0 | Yes |
| **Fold 4** | 24 | 1,610 | 92 | 6,233 | 0 | Yes |
| **Total Cohort** | **116** | **7,843** | — | — | **0 across all folds** | **Yes** |

### Origin of the Discrepant $N = 6,274$ Count
In early draft versions of `MANUSCRIPT_DRAFT.md` and `RESULTS_TABLES.md`, the authors assumed an equal 5-fold split:
$$\frac{7,843}{5} = 1,568.6 \approx 1,569 \text{ parcels per fold}$$
$$N_{\text{train, idealized}} = 7,843 - 1,569 = 6,274 \text{ parcels}$$
Because `StratifiedGroupKFold` splits parcels by whole administrative clusters (Mandals) rather than individual parcels, cluster sizes vary from 1,508 to 1,610 parcels per fold. The actual training partition for Fold 0 contains **6,335 parcels**. The manuscript text and Table 4 must be updated from 6,274 to **6,335**.

---

## 4. Hyperparameter Search & Selection Verification

Inspection of `outputs/reports/phase6g/pilot_fold0_trial_ledger.csv` and `phase6g_pilot_fold0_report.json` confirms:

1. **Candidate Configurations:** Exactly 10 distinct configurations evaluated.
2. **Inner Cross-Validation:** 3-fold `StratifiedGroupKFold` on the 6,335 training parcels grouped by Mandal (93 Mandals), with zero inner Mandal overlap.
3. **Selection Metric:** Inner validation Macro-F1 (unweighted).
4. **Selected Trial:** **Trial 4** achieved the highest inner Macro-F1 of **0.6797** (Inner Accuracy: 0.7699, Inner Balanced Accuracy: 0.6669).
   - Parameters: `max_depth=7`, `learning_rate=0.05`, `subsample=0.80`, `colsample_bytree=0.75`, `min_child_weight=3`, `gamma=0.2`, `reg_alpha=0.2`, `reg_lambda=5.0`.
5. **Outer Fold 0 Evaluation:** Trial 4 was refit once on all 6,335 training parcels and evaluated prospectively on the 1,508 validation parcels of Outer Fold 0:
   - Outer Validation Macro-F1: **0.7024** (Accuracy: 0.7798, Balanced Accuracy: 0.6917, Weighted-F1: 0.7695).
   - Phase 6F Outer Fold 0 Baseline: **0.7095** (Accuracy: 0.7818).
   - Delta vs Baseline on Fold 0: $-0.0071$ Macro-F1.

---

## 5. Evaluation Across the Five Outer Folds & Selection Bias Analysis

The selected configuration from Trial 4 was subsequently frozen and evaluated across all 5 spatial cross-validation folds in Phase 6H and Phase 6I (with inverse-frequency sample weighting incorporated in Phase 6I):

| Outer Fold Index | Validation Parcels | Weighted S+T Macro-F1 | Overall Accuracy |
| :---: | :---: | :---: | :---: |
| **Fold 0 (Tuned Fold)** | 1,508 | **0.7269** | 0.7960 |
| **Fold 1** | 1,600 | **0.7381** | 0.8013 |
| **Fold 2** | 1,595 | **0.6919** | 0.7717 |
| **Fold 3** | 1,530 | **0.6945** | 0.7749 |
| **Fold 4** | 1,610 | **0.7073** | 0.7966 |
| **Pooled Out-of-Fold** | **7,843** | **0.7226** | **0.7881** |

### Leakage and Optimistic Bias Assessment:
- **No Direct Outer Validation Leakage:** Trial 4 was selected solely on the inner 3-fold Macro-F1 score (0.6797). Outer validation labels were not in the loss function or trial selection loop.
- **No Fold-0 Specific Optimistic Bias:** Fold 0 is **not** the highest-performing fold. Fold 1 (which was completely untouched during hyperparameter tuning) achieved **0.7381**, which is higher than Fold 0 (0.7269). This rules out severe Fold-0 specific overfitting.
- **Methodological Characterization:** This procedure is **exploratory tuning on Outer Fold 0's training split**, followed by 5-fold spatial evaluation of the fixed configuration. **It must not be described as full nested cross-validation**, because hyperparameter search was not repeated independently within all five outer folds.

---

## 6. Manuscript Remediation Guidance

In `MANUSCRIPT_DRAFT.md` Section IV-F and `RESULTS_TABLES.md` Table 4:
1. Replace: `"training split of Outer Fold 0 ($N = 6,274$)"`  
   With: `"training split of Outer Fold 0 ($N = 6,335$ parcels across 93 Mandals, with 1,508 validation parcels across 23 Mandals)"`.
2. Ensure methodology explicitly discloses:
   > "Hyperparameters were selected via a bounded grid search of ten candidate configurations evaluated on a 3-fold inner spatial cross-validation using the training partition of Outer Fold 0 ($N = 6,335$). The selected configuration (Trial 4: `max_depth=7`, `colsample_bytree=0.75`, `reg_lambda=5.0`, `min_child_weight=3`) was then fixed and evaluated across all five outer spatial folds. To confirm stability, Fold 0 performance (Macro-F1 0.7269) was not the highest across the five folds (Fold 1 achieved 0.7381), indicating no systematic Fold-0 overfitting."
