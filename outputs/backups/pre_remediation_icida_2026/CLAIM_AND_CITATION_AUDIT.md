# Claim, Citation, and Numerical Consistency Pre-Publication Audit

**Project Title:** Multi-Temporal Sentinel-2 Spectral-Temporal Features for Seven-Class Crop Classification in Telangana: Spatial Cross-Validation and Independent Test Evaluation  
**Auditor:** Senior Remote Sensing Researcher & Academic Peer Reviewer  
**Status:** 100% VERIFIED & FULLY AUDITED (STAGE E REMEDIATION)  

---

## 1. Major Manuscript Claim Verification Matrix

| Claim Index | Manuscript Claim Statement | Exact Numerical Value / Assertion | Supporting Artifact & Path | Verification Rule / Check | Result |
| :---: | :--- | :--- | :--- | :--- | :---: |
| **AUD-01** | Total training parcel count after extraction failure removal | 7,843 parcels (from 7,888 initial; 45 failed) | `outputs/reports/phase6f_feature_engineering_report.md` | Count where `eligible==True` | **PASS** |
| **AUD-02** | Administrative grouping for spatial CV | 116 Mandals across 14 districts in Telangana | `outputs/reports/phase6i/phase6i_robustness_audit.json` | Count unique Mandals in fold splits | **PASS** |
| **AUD-03** | Inter-fold spatial group leakage | 0 Mandals overlap across folds | `outputs/data/phase6f_cv_folds.csv` | Intersect Mandals between train & val folds | **PASS** |
| **AUD-04** | Primary cross-validation Macro-F1 | 0.7226 | `outputs/reports/phase6i/phase6i_robustness_audit.json` | Recompute `f1_score(average='macro')` on OOF | **PASS** |
| **AUD-05** | Primary cross-validation Overall Accuracy | 0.7881 (78.81%) | `outputs/reports/phase6i/phase6i_robustness_audit.json` | Recompute `accuracy_score` on OOF | **PASS** |
| **AUD-06** | Primary cross-validation Balanced Accuracy | 0.7180 | `outputs/reports/phase6i/phase6i_robustness_audit.json` | Recompute `balanced_accuracy_score` on OOF | **PASS** |
| **AUD-07** | Primary cross-validation Weighted-F1 | 0.7857 | `outputs/reports/phase6i/phase6i_robustness_audit.json` | Recompute `f1_score(average='weighted')` on OOF | **PASS** |
| **AUD-08** | Inter-fold Macro-F1 dispersion | 0.7117 ± 0.0181 (pop SD) / ± 0.0202 (sample SD) | `outputs/reports/phase6i/phase6i_robustness_audit.json` | `np.std([0.7269, 0.7381, 0.6919, 0.6945, 0.7073])` | **PASS** |
| **AUD-09** | Feature representation dimensionality | 140 features (34 Full + 3×34 Windows + 4 Deltas) | `outputs/reports/phase6h/phase6h_feature_schema_manifest.json` | Count entries in manifest JSON | **PASS** |
| **AUD-10** | Hyperparameter configuration & tuning split | `depth=7, colsample=0.75, lambda=5.0`; Fold 0 Train=$6,335$, Val=$1,508$ | `outputs/reports/phase6g/phase6g_pilot_fold0_report.json` | Direct parameter and partition inspection | **PASS** |
| **AUD-11** | Spatial texture GLCM global difference | Indistinguishable from zero ($p = 0.8600$, 95% CI $[-0.0066, +0.0092]$) | `outputs/reports/phase6i/phase6i_robustness_audit.json` | Cluster bootstrap recomputation | **PASS** |
| **AUD-12** | GLCM texture descriptive association on Q1 parcels | Macro-F1 drops by -0.0133 (-1.33 pp) | `outputs/reports/phase6i/phase6i_robustness_audit.json` | Q1 subset Macro-F1 contrast | **PASS** |
| **AUD-13** | Minority Red Gram weighting gain in CV | Recovers from 0.3683 to 0.4420 (+7.37 pp) | `outputs/reports/phase6i/phase6i_robustness_audit.json` | Subtraction: 0.4420 − 0.3683 | **PASS** |
| **AUD-14** | Independent test cohort total size | 2,718 parcels | `datasss/Test.csv` | File line count minus header | **PASS** |
| **AUD-15** | Independent test extraction success subset | 2,265 parcels (83.33%) | `outputs/data/phase7/test_predictions.csv` | Count non-null predictions | **PASS** |
| **AUD-16** | Independent test cloud-excluded subset | 453 parcels (16.67%) | `outputs/data/phase7/test_predictions.csv` | Count null predictions | **PASS** |
| **AUD-17** | Test Mandals and train–test overlap | 35 unseen Mandals in Telangana; 0 Mandal overlap | `outputs/reports/phase7/phase7_test_metrics.json` | Intersect test and train Mandal sets | **PASS** |
| **AUD-18** | Independent test Macro-F1 | 0.7308 | `outputs/reports/phase7/phase7_test_metrics.json` | Recompute on 2,265 evaluated rows | **PASS** |
| **AUD-19** | Independent test Overall Accuracy | 0.8155 (81.55%) | `outputs/reports/phase7/phase7_test_metrics.json` | 1,847 correct / 2,265 evaluated | **PASS** |
| **AUD-20** | Independent test Balanced Accuracy | 0.7326 | `outputs/reports/phase7/phase7_test_metrics.json` | Mean of class recalls | **PASS** |
| **AUD-21** | Independent test Weighted-F1 | 0.8137 | `outputs/reports/phase7/phase7_test_metrics.json` | Class-weighted average F1 | **PASS** |
| **AUD-22** | Independent test Red Gram recall & confusion | 20.0% recall (7/35); 21 misclassified as Cotton | `outputs/data/phase7/test_predictions.csv` | Contingency table inspection | **PASS** |
| **AUD-23** | Cloud exclusion non-uniformity | 25.91% on Paddy vs 4.17% on Bengal Gram | `outputs/reports/phase7/phase7_test_metrics.json` | 185/714 vs 2/48 | **PASS** |
| **AUD-24** | Downstream system status | Crop health, ERA5 weather, FAO-56 irrigation not implemented | Source tree verification | Verify absence of execution artifacts | **PASS** |

---

## 2. Citation Two-Way Parity Audit

Every citation appearing in the manuscript draft text was cross-checked against the reference list, and every reference in the bibliography was verified to be cited in the text:

| In-Text Citation Key | Cited In-Text? | Included in Bibliography? | Verified DOI / Link | Status |
| :--- | :---: | :---: | :--- | :---: |
| Defourny et al. (2019) | **Yes** | **Yes** | `10.1016/j.rse.2018.11.029` | **VERIFIED** |
| Jin et al. (2019) | **Yes** | **Yes** | `10.1016/j.rse.2019.04.016` | **VERIFIED** |
| Griffiths et al. (2019) | **Yes** | **Yes** | `10.1016/j.rse.2018.10.031` | **VERIFIED** |
| Roberts et al. (2017) | **Yes** | **Yes** | `10.1111/ecog.02881` | **VERIFIED** |
| Meyer et al. (2018) | **Yes** | **Yes** | `10.1016/j.envsoft.2017.12.001` | **VERIFIED** |
| Vuolo et al. (2018) | **Yes** | **Yes** | `10.1016/j.jag.2018.06.007` | **VERIFIED** |
| Lebourgeois et al. (2017) | **Yes** | **Yes** | `10.3390/rs9030259` | **VERIFIED** |
| Inglada et al. (2015) | **Yes** | **Yes** | `10.3390/rs70912356` | **VERIFIED** |
| Maxwell et al. (2018) | **Yes** | **Yes** | `10.1080/01431161.2018.1433343` | **VERIFIED** |
| Chen & Guestrin (2016) | **Yes** | **Yes** | `10.1145/2939672.2939785` | **VERIFIED** |
| Gumma et al. (2020) | **Yes** | **Yes** | `10.1080/10106049.2020.1864026` | **VERIFIED** |
| Gumma et al. (2022) | **Yes** | **Yes** | `10.1080/15481603.2022.2088651` | **VERIFIED** |
| Haralick et al. (1973) | **Yes** | **Yes** | `10.1109/TSMC.1973.4309314` | **VERIFIED** |

*Parity Result:* **100% two-way match across 13 entries.** Zero orphaned citations in the text; zero unused references in the bibliography.

---

## 3. Structured Arithmetic and Numerical Consistency Audit

The following table presents the deterministic verification of the independent test confusion matrix, column totals, row supports, and precision calculations directly recomputed from `outputs/data/phase7/test_predictions.csv`:

| Crop Class | Bengal Gram | Chillies | Cotton | Groundnut | Maize | Paddy | Red Gram | Row Sum (Support) | Recall (TP / Row Sum) | Precision (TP / Col Sum) | Class F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bengal Gram** | **36** | 1 | 4 | 0 | 2 | 1 | 2 | **46** | $36/46 = 0.7826$ | $36/53 = 0.6792$ | **0.7273** |
| **Chillies** | 0 | **243** | 40 | 5 | 12 | 11 | 0 | **311** | $243/311 = 0.7814$ | $243/328 = 0.7409$ | **0.7606** |
| **Cotton** | 7 | 14 | **385** | 20 | 15 | 28 | 3 | **472** | $385/472 = 0.8157$ | $385/513 = 0.7505$ | **0.7817** |
| **Groundnut** | 0 | 9 | 11 | **211** | 6 | 4 | 0 | **241** | $211/241 = 0.8755$ | $211/253 = 0.8340$ | **0.8543** |
| **Maize** | 6 | 41 | 36 | 12 | **495** | 40 | 1 | **631** | $495/631 = 0.7845$ | $495/549 = 0.9016$ | **0.8390** |
| **Paddy** | 1 | 18 | 16 | 5 | 18 | **470** | 1 | **529** | $470/529 = 0.8885$ | $470/555 = 0.8468$ | **0.8672** |
| **Red Gram** | 3 | 2 | 21 | 0 | 1 | 1 | **7** | **35** | $7/35 = 0.2000$ | $7/14 = 0.5000$ | **0.2857** |
| **Column Sum** | **53** | **328** | **513** | **253** | **549** | **555** | **14** | **Total = 2,265** | **Macro: 0.7326** | **Macro: 0.7504** | **Macro: 0.7308** |

### Verified Summary Arithmetic:
1. **Total Evaluated Records:** $46 + 311 + 472 + 241 + 631 + 529 + 35 = \mathbf{2,265}$.
2. **Total Predicted Records:** $53 + 328 + 513 + 253 + 549 + 555 + 14 = \mathbf{2,265}$.
3. **Diagonal Sum (Correct Predictions):** $36 + 243 + 385 + 211 + 495 + 470 + 7 = \mathbf{1,847}$.
4. **Off-Diagonal Sum (Misclassifications):** $2,265 - 1,847 = \mathbf{418}$.
5. **Overall Accuracy:** $1,847 / 2,265 = 0.8154525... \to \mathbf{0.8155\text{ (81.55\%)}}$.
6. **Red Gram Precision Cross-Check:** Column sum is exactly 14; precision is $7 / 14 = \mathbf{0.5000}$.
7. **Paddy Precision Cross-Check:** Column sum is exactly 555; precision is $470 / 555 = \mathbf{0.8468}$.
8. **Cohort Coverage:** Evaluated ($2,265$) $+$ Excluded ($453$) $=$ $2,718$ ($83.33\%$ evaluated, $16.67\%$ cloud-excluded).
