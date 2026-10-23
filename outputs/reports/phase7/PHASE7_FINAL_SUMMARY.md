# Phase 7 — Final Summary: Independent Test Set Evaluation

**Phase:** 7 — Test.csv Satellite Feature Extraction, Model Inference & Independent Evaluation  
**Date Completed:** 2026-10-23  
**Status:** COMPLETE (all plans verified)

---

## 1. Objective

Evaluate the frozen Phase 6I champion crop-classification model on an independent held-out test set (`datasss/Test.csv`) using satellite-derived features extracted from Sentinel-2 imagery. This phase tests the model's ability to classify crop types across geographically unseen administrative units (Mandals) in Telangana, India, with zero retraining, hyperparameter tuning, threshold recalibration, or test-feature selection.

**Scope clarification:** Phase 7 evaluates **crop-type classification** (7 classes: Bengal Gram, Chillies, Cotton, Groundnut, Maize, Paddy, Red Gram). It does not evaluate crop-health prediction, moisture-stress detection, or irrigation decision support, which are described in the broader project framework but are not implemented or validated by this phase.

---

## 2. Evaluation Protocol

- **Model:** Frozen `Weighted S+T` XGBoost (`outputs/models/phase6h/xgb_weighted_s+t.joblib`)
  - 140 Spectral-Temporal features (exact schema in `phase6h_feature_schema_manifest.json`)
  - Inverse-frequency class weighting with L2 regularization
  - Trained on `Train.csv` (7,843 farms, 116 Mandals) via 5-fold spatial cross-validation
- **Inference:** `model.predict()` and `model.predict_proba()` applied to the evaluable test subset
- **Leakage safeguards:** Forbidden columns (`Crop`, `category`, `FarmID`, `District`, `Sub-District`, `geometry`, `area_ha`, `SDate`, `HDate`, `CropCoveredArea`) strictly excluded from the predictor matrix
- **NaN handling:** Isolated NaN cells in otherwise-successful extractions imputed with per-column median of the evaluable set; fully cloud-obscured parcels excluded entirely

---

## 3. Test Cohort

| Property | Value |
|:---|:---|
| Total Test.csv farms | 2,718 |
| Test Mandals (Sub-Districts) | 35 |
| Training Mandals | 116 |
| Reported Mandal overlap | 0 |
| Successfully extracted farms | 2,265 (83.33%) |
| Cloud-excluded farms | 453 (16.67%) |

### Cloud Exclusion Details

453 farms had persistent cloud or shadow obscuration across all four temporal windows, rendering satellite feature extraction infeasible. These farms received no predicted labels and are excluded from all supervised evaluation metrics. The cloud exclusion rate varies by crop class — Paddy is disproportionately affected (185/714 = 25.9% excluded) compared to Bengal Gram (2/48 = 4.2%). This means the evaluated subset is not a uniform random sample of Test.csv.

---

## 4. Verified Headline Metrics (N = 2,265)

All metrics below were independently recomputed from `outputs/data/phase7/test_predictions.csv` using `sklearn.metrics` and verified against the saved JSON by `scripts/audit_phase7_metrics.py`. Zero discrepancies were found.

| Metric | Independent Test ($N=2,265$) | 5-Fold Spatial CV ($N=7,843$) | Delta |
|:---|:---:|:---:|:---:|
| **Macro-F1** | **0.7308** | 0.7226 | +0.0082 |
| **Overall Accuracy** | **0.8155 (81.55%)** | 0.7881 (78.81%) | +0.0274 |
| **Balanced Accuracy** | **0.7326** | 0.7180 | +0.0146 |
| **Weighted-F1** | **0.8137** | 0.7857 | +0.0280 |
| **Correct / Incorrect** | 1,847 / 418 | — | — |

---

## 5. Verified Per-Class Performance

| Crop | Precision | Recall | F1-Score | Evaluated Support | Cloud Excluded | Total Test Support |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Bengal Gram | 0.6792 | 0.7826 | 0.7273 | 46 | 2 | 48 |
| Chillies | 0.7409 | 0.7814 | 0.7606 | 311 | 70 | 381 |
| Cotton | 0.7505 | 0.8157 | 0.7817 | 472 | 97 | 569 |
| Groundnut | 0.8340 | 0.8755 | 0.8543 | 241 | 25 | 266 |
| Maize | 0.9016 | 0.7845 | 0.8390 | 631 | 68 | 699 |
| Paddy | 0.8468 | 0.8885 | 0.8672 | 529 | 185 | 714 |
| Red Gram | 0.5000 | 0.2000 | 0.2857 | 35 | 6 | 41 |

---

## 6. Confusion Matrix Analysis

```
Predicted ->
True Crop        BG    CH   COT    GN    MZ   PAD    RG | Support
------------------------------------------------------------------
Bengal Gram      36     1     4     0     2     1     2 |      46
Chillies          0   243    40     5    12    11     0 |     311
Cotton            7    14   385    20    15    28     3 |     472
Groundnut         0     9    11   211     6     4     0 |     241
Maize             6    41    36    12   495    40     1 |     631
Paddy             1    18    16     5    18   470     1 |     529
Red Gram          3     2    21     0     1     1     7 |      35
------------------------------------------------------------------
Total Predicted  53   328   513   253   549   555    14 |    2265
```

**Key observations:**

1. **Red Gram** has the lowest recall (7/35 = 20.0%, F1 = 0.2857). Of 35 evaluated Red Gram parcels, 21 are misclassified as Cotton and 3 as Bengal Gram. This is consistent with spectral similarity between Red Gram and Cotton canopies in the Sentinel-2 bands, compounded by very low support (35 farms).

2. **Maize–Chillies confusion:** 41 Maize farms are misclassified as Chillies and 40 as Paddy, likely reflecting overlapping growing seasons and spectral signatures during the vegetative phase.

3. **Cotton** is the most frequent false-positive target (513 total predictions vs 472 true support), absorbing misclassifications from Red Gram (21), Chillies (40), and Maize (36).

4. **Paddy** and **Groundnut** show the strongest per-class performance (F1 = 0.8672 and 0.8543 respectively), benefiting from distinctive temporal moisture and greenness signatures.

---

## 7. Scientific Interpretation

### 7.1 Comparison with Spatial Cross-Validation

The independent test metrics (Macro-F1 = 0.7308, accuracy = 81.55%) are numerically comparable to the 5-fold spatial cross-validation baseline (Macro-F1 = 0.7226, accuracy = 78.81%). However, **the higher test scores should not be interpreted as evidence of superior geographic generalization.** Several factors complicate direct comparison:

1. **Class-distribution shift:** The test set has a substantially different crop-class distribution than the training set. Maize represents 27.9% of evaluated test farms vs 17.6% of training farms. Bengal Gram drops from 12.0% to 2.0%. These shifts affect weighted and macro-averaged metrics.

2. **Cloud-related selection bias:** 16.67% of Test.csv farms were excluded due to persistent cloud obscuration. Paddy is disproportionately excluded (25.9% cloud loss). The evaluated 2,265 farms are not a uniform random sample of the full 2,718-farm cohort.

3. **Cohort size difference:** The spatial-CV baseline evaluates 7,843 farms across 116 Mandals; the test evaluation covers 2,265 farms across 35 Mandals, potentially yielding higher variance in the estimate.

4. **Different evaluation designs:** Spatial CV uses Mandal-based fold isolation within the training Mandals; the test set uses geographically distinct Mandals absent from training.

### 7.2 What Can Be Concluded

- The model does not exhibit catastrophic performance collapse on geographically unseen Mandals, which supports (but does not prove) spatial robustness of the 140-feature Spectral-Temporal representation.
- Per-class performance follows a pattern consistent with spatial CV: majority crops (Paddy, Groundnut, Maize) are classified reliably, while minority crops with spectral ambiguity (Red Gram) remain challenging.
- The observed metric parity between test and CV baselines is consistent with, but not conclusive evidence of, robust geographic generalization.

---

## 8. Metric Audit Results

| Check | Outcome |
|:---|:---:|
| Confusion matrix recomputation from raw predictions | ✅ Identical |
| Per-class precision/recall/F1 from `sklearn.metrics.classification_report` | ✅ All 7 classes match JSON |
| CM-derived precision (TP/col_sum) matches JSON precision | ✅ All match |
| Headline metrics recomputation | ✅ All within rounding tolerance |
| correct + incorrect = 2,265 | ✅ |
| Cloud-excluded records preserved as NaN | ✅ 453 rows with no predictions |
| Frozen model integrity (classes=[0..6], n_features=140) | ✅ |
| Zero leakage in predictor matrix | ✅ |
| 140-feature schema matches Phase 6H manifest | ✅ |

Audit script: `scripts/audit_phase7_metrics.py`

---

## 9. Test Results

| Suite | Collected | Passed | Skipped | Failed |
|:---|:---:|:---:|:---:|:---:|
| Phase 7 inference tests (`tests/test_phase7_inference.py`) | 9 | 9 | 0 | 0 |
| Full regression suite (`tests/`) | 121 | 120 | 1 | 0 |
| **Total** | **121** | **120** | **1** | **0** |

The single skipped test (`test_phase6h_stage2_extraction.py::test_05_pilot_nan_preserved_for_failed`) is an existing skip unrelated to Phase 7.

---

## 10. Reproducibility Instructions

```bash
# 1. Run Sentinel-2 feature extraction for Test.csv (requires GEE authentication)
python scripts/extract_s2_test_phase7.py

# 2. Run frozen model inference and evaluation
python scripts/run_phase7_test_inference.py

# 3. Run metric consistency audit
python -X utf8 scripts/audit_phase7_metrics.py

# 4. Run Phase 7 tests
python -m pytest tests/test_phase7_inference.py -v

# 5. Run full regression suite
python -m pytest tests/ -v
```

**Environment requirements:** Python 3.14, scikit-learn, xgboost, pandas, numpy, matplotlib, joblib, earthengine-api, shapely.

---

## 11. Artifact Inventory

### Data Files
| File | Description | Rows |
|:---|:---|:---:|
| `outputs/data/phase7/s2_test_raw_bands.csv` | Raw Sentinel-2 band means per temporal window | 2,718 |
| `outputs/data/phase7/s2_test_140_features.csv` | 140 S+T feature predictor matrix | 2,718 |
| `outputs/data/phase7/s2_test_full_features_with_metadata.csv` | Full features with metadata | 2,718 |
| `outputs/data/phase7/test_predictions.csv` | Predictions, probabilities, and ground truth | 2,718 |

### Reports
| File | Description |
|:---|:---|
| `outputs/reports/phase7/phase7_test_preflight_audit.md` | Plan 1: Read-only preflight audit |
| `outputs/reports/phase7/phase7_evaluation_protocol.md` | Plan 1: Leakage-safe evaluation protocol |
| `outputs/reports/phase7/phase7_test_extraction_report.md` | Plan 2: GEE extraction report |
| `outputs/reports/phase7/phase7_test_metrics.json` | Plan 3: Verified metrics JSON |
| `outputs/reports/phase7/phase7_confusion_matrix.png` | Plan 3: 7×7 confusion matrix figure |
| `outputs/reports/phase7/phase7_test_evaluation_report.md` | Plan 3: Detailed evaluation report |
| `outputs/reports/phase7/PHASE7_PLAN3_SUMMARY.md` | Plan 3: Execution summary |
| `outputs/reports/phase7/PHASE7_FINAL_SUMMARY.md` | Plan 4: This document |

### Scripts & Tests
| File | Description |
|:---|:---|
| `scripts/extract_s2_test_phase7.py` | GEE extraction pipeline for Test.csv |
| `scripts/run_phase7_test_inference.py` | Frozen model inference & evaluation |
| `scripts/audit_phase7_metrics.py` | Post-hoc metric consistency audit |
| `tests/test_phase7_inference.py` | 9 quality-control tests |

---

## 12. Limitations

1. **Cloud-dependent evaluation subset:** Only 2,265 of 2,718 farms (83.33%) could be evaluated. The cloud-exclusion rate is crop-dependent (Paddy: 25.9%, Bengal Gram: 4.2%), meaning the evaluated subset does not represent the full Test.csv population uniformly.

2. **Minority-class instability:** Red Gram has only 35 evaluated farms (1.5% of the cohort) and an F1 of 0.2857. This score is statistically unreliable due to small support and should be interpreted with appropriate uncertainty.

3. **Class-distribution shift:** The test and training sets have different crop-class proportions. Metrics sensitive to class balance (macro-F1, balanced accuracy) may not be directly comparable between the two evaluation designs.

4. **Single model, single test set:** The evaluation uses a single frozen model on a single held-out test set. No confidence intervals, bootstrap estimates, or multiple-model comparisons are reported for the test-set metrics.

5. **Geographic coverage:** The test set covers 35 Mandals, all from Telangana. Performance on other Indian states or agro-climatic zones is not evaluated.

6. **Temporal generalization:** Both training and test data are from the same Rabi/Kharif seasons. The model's performance on different years or seasons is unknown.

7. **Not an end-to-end irrigation system evaluation:** This phase evaluates crop-type classification only. The broader project's irrigation decision support, moisture-stress detection, and crop-health monitoring objectives are not validated by Phase 7 results.

---

## 13. Git Status

```
Working tree: Modified (.planning/ROADMAP.md, .planning/STATE.md)
              Staged (paper_materials/* from earlier phases)
              Untracked (outputs/reports/phase7/*, scripts/*_phase7*.py, tests/test_phase7_inference.py)
Commits:      NONE made during Phase 7 (per user directive)
Pushes:       NONE
```

No files have been committed or pushed. All Phase 7 artifacts exist only in the working tree.
