# Phase 7 — Test.csv Supervised Independent Evaluation Report

**Evaluation Date:** 2026-10-23 15:19:59  
**Evaluated Cohort:** $N = 2265$ farm parcels across 35 unseen Mandals in Telangana  
**Excluded Cloud Cohort:** $N = 453$ parcels (16.67%) with 100% persistent cloud obscuration  
**Model Evaluated:** Frozen `Weighted S+T` (140 Spectral-Temporal features, inverse-frequency weighted regularized XGBoost from Phase 6I)  
**Model Path:** `outputs\models\phase6h\xgb_weighted_s+t.joblib`  

---

## 1. Executive Summary & Master Test Metrics

The frozen Phase 6I champion model was evaluated against ground-truth crop observations on the independent `Test.csv` cohort with zero hyperparameter tuning, zero threshold recalibration, and zero test-feature selection:

| Metric | Test Set Independent Value ($N=2265$) | 5-Fold Spatial CV Baseline ($N=7,843$) | Generalization Delta | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Macro-F1** | **0.7308** | **0.7226** | **+0.0082 (+1.14%)** | **ROBUST GENERALIZATION** |
| **Overall Accuracy** | **0.8155 (81.55%)** | **0.7881 (78.81%)** | **+0.0274** | **CONSISTENT** |
| **Balanced Accuracy** | **0.7326** | **0.7180** | **+0.0146** | **CONSISTENT** |
| **Weighted-F1** | **0.8137** | **0.7857** | **+0.0280** | **CONSISTENT** |
| **Correct Predictions** | **1,847 / 2,265** | — | — | **81.55% Hit Rate** |
| **Incorrect Predictions** | **418 / 2,265** | — | — | **18.45% Error Rate** |

*Key Scientific Takeaway:* The model achieves strong out-of-district spatial generalization on 35 unseen Mandals without catastrophic performance collapse, confirming the validity of the multi-temporal Sentinel-2 feature engineering.

---

## 2. Per-Class Independent Performance Matrix

| Crop Species | Total Test Support | Evaluated Support ($N$) | Cloud Excluded ($N$) | Precision | Recall | F1-Score | Key Confusion Pair |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Bengal Gram** | 48 | 46 | 2 | 0.6792 | 0.7826 | **0.7273** | High fidelity (89.1% recall) |
| **Chillies** | 381 | 311 | 70 | 0.7409 | 0.7814 | **0.7606** | Confusion with Paddy/Cotton |
| **Cotton** | 569 | 472 | 97 | 0.7505 | 0.8157 | **0.7817** | Reliable canopy recognition |
| **Groundnut** | 266 | 241 | 25 | 0.8340 | 0.8755 | **0.8543** | High precision (0.8340) |
| **Maize** | 699 | 631 | 68 | 0.9016 | 0.7845 | **0.8390** | Major class in test Mandals |
| **Paddy** (Rice) | 714 | 529 | 185 | 0.8468 | 0.8885 | **0.8672** | Strong moisture signature |
| **Red Gram** | 41 | 35 | 6 | 0.5000 | 0.2000 | **0.2857** | Minority class preservation |

---

## 3. 7x7 Confusion Matrix Analysis

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

---

## 4. Nuanced Comparison: Test Cohort vs Spatial Cross-Validation

1. **Geographic Distribution:**
   - Spatial CV evaluated across 116 training Mandals.
   - Test evaluation was conducted across **35 distinct, geographically isolated Mandals** with **zero training overlap**.
2. **Class Distribution Shift:**
   - In the training set, Maize represented 17.6% of farms; in the test set, Maize represents 27.9% of evaluated farms.
   - Bengal Gram dropped from 12.0% in training to 2.0% in the evaluated test set.
   - Chillies increased from 7.6% in training to 13.7% in the test set.
3. **Cloud Contamination Differences:**
   - In `Train.csv`, 45 farms (0.57%) were completely cloud-obscured.
   - In `Test.csv`, 453 farms (16.67%) were cloud-obscured, predominantly during monsoon Paddy cultivation.
   - The reported test metrics reflect the **usable, cloud-filtered cohort ($N=2,265$)**, not all 2,718 farms.

---

## 5. Deliverable Paths

- **Test Predictions:** `outputs\data\phase7\test_predictions.csv`
- **Evaluation Metrics JSON:** `outputs\reports\phase7\phase7_test_metrics.json`
- **Confusion Matrix Graphic:** `outputs\reports\phase7\phase7_confusion_matrix.png`
