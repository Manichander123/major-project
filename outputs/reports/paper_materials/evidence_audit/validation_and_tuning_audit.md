# Validation and Tuning Audit

**Project:** AI-Based Remote Sensing — Seven-Class Crop Classification, Telangana  
**Task C:** Audit hyperparameter tuning and validation leakage  
**Audit date:** 2026-10-23 (read-only)

---

## 1. Summary Classification

**Classification: Category 2 — Exploratory tuning that used an outer validation fold.**  
The methodology is described as "nested CV on Fold 0" but the implementation is a bounded manual grid search on Outer Fold 0 of the outer CV, which means the outer validation labels from Fold 0 were accessible to the researcher during trial selection. This is NOT a full nested cross-validation. The impact is quantified below.

---

## 2. What the Phase 6G Script Actually Did

Source: `outputs/reports/phase6g/phase6g_pilot_fold0_report.json`

| Step | Evidence |
|---|---|
| Outer fold used | Fold 0 only (`outer_fold: 0`) |
| Outer training samples | 6,335 |
| Outer validation samples | 1,508 |
| Inner CV folds | 3-fold grouped CV on the 6,335 training samples |
| Trials evaluated | 10 distinct hyperparameter configurations |
| Trial selection criterion | Best **inner** 3-fold macro-F1 (not outer val macro-F1) |
| Selected trial | Trial 4 (inner macro-F1 = 0.6797) |
| Outer val macro-F1 of Trial 4 | 0.7024 |
| Phase 6F baseline outer val macro-F1 | 0.7095 |

The inner CV used only the 6,335 training samples of Fold 0. The selection criterion was the inner macro-F1, which did not access Fold 0 validation labels directly.

However: the researcher **did see** the outer validation metric (0.7024) for Trial 4 when it was reported, and the experimental design was not modified after seeing it. Whether any trial was eliminated or selected based on outer val performance requires script inspection.

### Inspection of trial ledger

From `pilot_fold0_trial_ledger.csv` and the report JSON:

- All 10 trials were evaluated on the inner CV only.
- Trial 4 was selected because it had the highest **inner** macro-F1 (0.6797).
- The outer validation metric was computed *after* selection, as a prospective evaluation.
- There is no evidence that any trial was retroactively added or removed based on outer val results.

**Conclusion:** The selection used the inner CV metric, not the outer val metric. This is exploratory tuning (one outer fold, not a full nested grid), but the selection criterion itself was properly isolated from outer val labels.

---

## 3. Leakage Risk Assessment

| Leakage vector | Status |
|---|---|
| Outer val labels used to select Trial 4 | **No evidence** — selection was by inner macro-F1 |
| Selected hyperparameters re-evaluated across all 5 folds | **Yes** — in Phase 6H/6I |
| Feature selection used outer val results | Not applicable — no feature selection was performed |
| Class weighting computed inside fold | **Verified** — `project_evidence_and_discrepancies.md` §4 |
| Preprocessing leakage | **None** — XGBoost uses native sparsity; no imputer fit on val set |

---

## 4. Impact on Reported Score

Phase 6G Fold 0 metric (0.7339, from `phase6g_exp2_imbalance_report`) is explicitly labelled a pilot result and is NOT reported as the generalization estimate. The 5-fold scores from Phase 6H/6I are independent and not touched by the Fold 0 tuning process.

The concern is mild optimistic bias: the hyperparameters were tuned on Fold 0, so Fold 0 may be slightly optimistic compared with the other four folds.

Inspection of fold-level macro-F1 for weighted S+T:

| Fold | Macro-F1 |
|---|---|
| Fold 0 | 0.7269 |
| Fold 1 | 0.7381 |
| Fold 2 | 0.6919 |
| Fold 3 | 0.6945 |
| Fold 4 | 0.7073 |

Fold 0 (the tuned fold) is NOT the highest-performing fold. Fold 1 (which was never used for tuning) is highest at 0.7381. This is evidence against systematic optimistic bias in Fold 0.

---

## 5. Accurate Methodology Wording for the Paper

**Do not write:** "We performed nested cross-validation to tune hyperparameters."

**Write instead:**
> "Hyperparameters were selected via a bounded grid search of ten configurations evaluated on a 3-fold inner spatial CV using the training partition of Outer Fold 0 (6,335 farms). The selected configuration (Trial 4: max_depth=7, colsample_bytree=0.75, reg_lambda=5.0) was then fixed and evaluated independently across all five outer spatial folds in Phases 6H–6I, yielding the reported pooled macro-F1 of 0.7226. To confirm stability, Fold 0 performance (macro-F1 0.7269) was not the highest across the five folds, providing no indication of Fold-0-specific over-fitting."

---

## 6. Zero-Leakage Verification

Source: `project_evidence_and_discrepancies.md` §4

1. Target and identity leakage: 0 administrative, coordinate, or target columns in any feature set.
2. Spatial autocorrelation: 0 Mandal overlaps between training and validation across all 5 folds (verified by `phase6i_robustness_audit.json`, `zero_mandal_overlap_confirmed: true`).
3. Class weighting: weights computed strictly inside fold-internal training partitions.
4. Preprocessing: XGBoost native sparsity routing — no imputer step that could leak val statistics.
