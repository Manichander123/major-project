# Research Evidence Audit & Source-of-Truth Ledger

**Project Title:** Multi-Temporal Sentinel-2 Spectral-Temporal Features for Seven-Class Crop Classification in Telangana: Spatial Cross-Validation and Independent Test Evaluation  
**Auditor Roles:** Senior Remote Sensing Researcher, Machine Learning Experimental Auditor, Scientific Peer Reviewer  
**Date of Audit:** October 2026  
**Status:** COMPLETE — ALL CLAIMS CATEGORIZED AND RECONCILED  

---

## 1. Executive Audit Summary

This document establishes the authoritative source-of-truth evidence ledger for all empirical statements, dataset metrics, model configurations, spatial validation outcomes, ablation results, and independent held-out test evaluations across the Telangana crop-classification study.

Every claim is categorized into one of five rigorous verification states:
1. **[Directly Verified]**: Confirmed by primary numerical artifacts, raw predictions, or deterministic script execution.
2. **[Supported by Secondary Report]**: Documented in official phase summary reports with matching provenance.
3. **[Inferred from Verified Evidence]**: Derived arithmetically from verified raw data (e.g., delta calculations, percentages).
4. **[Not Verified]**: Unsubstantiated or unsupported by repository evidence (excluded from paper claims).
5. **[Contradicted by Evidence]**: Refuted by artifact analysis and corrected in the paper drafting ledger.

---

## 2. Source-of-Truth Evidence Ledger

| Claim ID | Exact Claim / Property | Numerical Value | Source Artifact Path | Exact Section / Field | Verification State | Caveats / Discrepancy Resolution | Proposed Section |
| :--- | :--- | :--- | :--- | :--- | :---: | :--- | :--- |
| **CLM-01** | Raw Training Parcel Count | 7,888 | `datasss/Train.csv` | File row count | [Directly Verified] | Pre-extraction raw surveyed cohort | Sec. III (Dataset) |
| **CLM-02** | Training Extraction Failures | 45 | `outputs/reports/phase6f_feature_engineering_report.md` | Feature extraction audit | [Directly Verified] | Failed due to persistent cloud/shadow in all windows | Sec. III (Dataset) |
| **CLM-03** | Eligible Training Cohort ($N_{train}$) | 7,843 | `outputs/data/s2_phase6f_features.csv` | Row count where `eligible=True` | [Directly Verified] | Evaluated in 5-fold spatial cross-validation | Sec. III, IV, V |
| **CLM-04** | Training Administrative Units | 116 Mandals | `outputs/reports/phase6i/phase6i_robustness_audit.json` | `reconciled_models.weighted_S+T.mandals_count` | [Directly Verified] | Spanned across 14 Telangana districts | Sec. III (Study Area) |
| **CLM-05** | Training Spatial Overlap | 0 Mandals | `outputs/reports/phase6i/phase6i_robustness_audit.json` | `zero_mandal_overlap_confirmed: true` | [Directly Verified] | Zero spatial group leakage across 5 folds | Sec. III, IV |
| **CLM-06** | Total Test Parcels ($N_{test,total}$) | 2,718 | `datasss/Test.csv`, `outputs/data/phase7/test_predictions.csv` | Total row count | [Directly Verified] | Held-out evaluation cohort | Sec. III, V |
| **CLM-07** | Test Mandals | 35 Mandals | `outputs/reports/phase7/phase7_test_metrics.json` | `cohort.test_mandals_count` | [Directly Verified] | Completely distinct from 116 training Mandals | Sec. III, IV |
| **CLM-08** | Train–Test Mandal Overlap | 0 Mandals | `outputs/reports/phase7/phase7_test_metrics.json` | `cohort.mandal_overlap_with_train` | [Directly Verified] | Independent intra-state administrative holdout across unseen Mandals within Telangana; 10 overlapping districts | Sec. IV, V |
| **CLM-09** | Evaluated Test Subset ($N_{test,eval}$) | 2,265 | `outputs/data/phase7/test_predictions.csv` | Non-null `predicted_crop` rows | [Directly Verified] | 83.33% extraction coverage | Sec. III, V |
| **CLM-10** | Cloud-Excluded Test Parcels | 453 | `outputs/data/phase7/test_predictions.csv` | Null `predicted_crop` rows | [Directly Verified] | 16.67% persistent cloud/shadow failure | Sec. III, V, Limitations |
| **CLM-11** | Primary Target Variable | `Crop` (7 classes) | `datasss/Train.csv`, `datasss/Test.csv` | Column header `Crop` | [Directly Verified] | Verified ground-truth crop identity | Sec. I, III |
| **CLM-12** | Secondary Target (`category`) | Health diagnosis | `datasss/Train.csv` | Column header `category` | [Directly Verified] | **NOT TRAINED**; designated for future work | Sec. I, Limitations |
| **CLM-13** | Feature Representation Dimensionality | 140 features | `phase6h_feature_schema_manifest.json` | Total feature count | [Directly Verified] | 34 full + 102 window (3×34) + 4 transition | Sec. III (Methodology) |
| **CLM-14** | Transition Delta Features | 4 features | `phase6h_feature_schema_manifest.json` | `groups.T` | [Directly Verified] | `ndvi_growth_delta`, `peak_to_early_ratio`, `ndvi_senescence_delta`, `moisture_loss_delta` | Sec. III (Methodology) |
| **CLM-15** | Primary Classifier Family | XGBoost | `outputs/models/phase6h/xgb_weighted_s+t.joblib` | Scikit-learn estimator inspection | [Directly Verified] | `XGBClassifier` with histogram tree method | Sec. III (Methodology) |
| **CLM-16** | Model Hyperparameters | Depth=7, Colsample=0.75, $\gamma$=0.20, $\alpha$=0.20, $\lambda$=5.0 | `outputs/reports/phase6g_pilot_fold0_report.json` | Best params from Trial 4 | [Directly Verified] | Evaluated via 3-fold inner CV on Outer Fold 0 training partition only ($N=6,335$) | Sec. III, IV |
| **CLM-17** | Class Weighting Strategy | Inverse-Frequency | `scripts/run_phase7_test_inference.py`, Phase 6G | Sample weighting formulation | [Directly Verified] | Computed fold-internally during training | Sec. III (Methodology) |
| **CLM-18** | Spatial CV Pooled Macro-F1 | 0.7226 | `outputs/reports/phase6i/phase6i_robustness_audit.json` | `reconciled_models.weighted_S+T.pooled_metrics.macro_f1` | [Directly Verified] | 5-Fold StratifiedGroupKFold on 7,843 farms | Sec. V (Results) |
| **CLM-19** | Spatial CV Overall Accuracy | 0.7881 (78.81%) | `outputs/reports/phase6i/phase6i_robustness_audit.json` | `reconciled_models.weighted_S+T.pooled_metrics.accuracy` | [Directly Verified] | Pooled out-of-fold accuracy | Sec. V (Results) |
| **CLM-20** | Spatial CV Balanced Accuracy | 0.7180 | `outputs/reports/phase6i/phase6i_robustness_audit.json` | `reconciled_models.weighted_S+T.pooled_metrics.balanced_accuracy` | [Directly Verified] | Unweighted mean of recall across 7 classes | Sec. V (Results) |
| **CLM-21** | Spatial CV Weighted-F1 | 0.7857 | `outputs/reports/phase6i/phase6i_robustness_audit.json` | `reconciled_models.weighted_S+T.pooled_metrics.weighted_f1` | [Directly Verified] | Corrected from early typo "0.7847" | Sec. V (Results) |
| **CLM-22** | Spatial CV Fold Mean Macro-F1 | 0.7117 ± 0.0181 | `outputs/reports/phase6i/phase6i_robustness_audit.json` | `fold_summary.macro_f1_mean` / `std` | [Directly Verified] | Population SD (ddof=0) is 0.0181; sample SD (ddof=1) is 0.0202 | Sec. V (Results) |
| **CLM-23** | Unweighted S+T Macro-F1 | 0.7069 | `outputs/reports/phase6i/phase6i_robustness_audit.json` | `reconciled_models.unweighted_S+T.pooled_metrics.macro_f1` | [Directly Verified] | Same 140 features, uniform sample weights | Sec. V (Results) |
| **CLM-24** | Weighting Macro-F1 Gain | +0.0157 (+1.57 pp) | Derived arithmetically: 0.7226 − 0.7069 | Audit ledger | [Inferred from Verified Evidence] | Statistically verified across folds | Sec. V (Results) |
| **CLM-25** | Spatial Texture Feature Count | 20 GLCM rasters | `phase6h_feature_schema_manifest.json` | `groups.P` | [Directly Verified] | 5 metrics × 4 windows × 1 kernel ($3\times3$) | Sec. III, V |
| **CLM-26** | Weighted S+T+P Macro-F1 | 0.7228 | `outputs/reports/phase6i/phase6i_robustness_audit.json` | `reconciled_models.weighted_S+T+P.pooled_metrics.macro_f1` | [Directly Verified] | 160 features (140 S+T + 20 GLCM) | Sec. V (Results) |
| **CLM-27** | S+T vs S+T+P Statistical Parity | $p = 0.8600$, CI $[-0.0066, +0.0092]$ | `outputs/reports/phase6i/phase6i_robustness_audit.json` | `paired_significance_audit.mandal_cluster_bootstrap` | [Directly Verified] | 1,000 cluster bootstrap iterations; paired t-test $p=0.7765$ | Sec. V (Results) |
| **CLM-28** | Texture Boundary Degradation (Q1) | S+T: 0.6598 vs S+T+P: 0.6465 ($\Delta = -0.0133$) | `outputs/reports/phase6i/phase6i_robustness_audit.json` | `area_quartile_metrics.Q1_small` | [Directly Verified] | Parcels $<1,257\text{ m}^2$ suffer boundary contamination | Sec. V (Discussion) |
| **CLM-29** | Red Gram F1 (Unweighted S+T) | 0.3683 | `outputs/reports/phase6i/phase6i_robustness_audit.json` | `unweighted_S+T.per_class_metrics.Red Gram.f1_score` | [Directly Verified] | Reconciled: corrected from orphan "0.3478" | Sec. V (Results) |
| **CLM-30** | Red Gram F1 (Weighted S+T) | 0.4420 | `outputs/reports/phase6i/phase6i_robustness_audit.json` | `weighted_S+T.per_class_metrics.Red Gram.f1_score` | [Directly Verified] | Rescue of extreme minority pulse crop | Sec. V (Results) |
| **CLM-31** | Red Gram Weighting Gain | +0.0737 (+7.37 pp) | Derived arithmetically: 0.4420 − 0.3683 | Audit ledger | [Inferred from Verified Evidence] | Corrected from early unverified "+9.42 pp" | Sec. V (Results) |
| **CLM-32** | Independent Test Macro-F1 | 0.7308 | `outputs/reports/phase7/phase7_test_metrics.json` | `metrics.macro_f1` | [Directly Verified] | Evaluated on $N=2,265$ across 35 unseen Mandals | Sec. V (Results) |
| **CLM-33** | Independent Test Overall Accuracy | 0.8155 (81.55%) | `outputs/reports/phase7/phase7_test_metrics.json` | `metrics.accuracy` | [Directly Verified] | 1,847 correct / 418 incorrect | Sec. V (Results) |
| **CLM-34** | Independent Test Balanced Accuracy | 0.7326 | `outputs/reports/phase7/phase7_test_metrics.json` | `metrics.balanced_accuracy` | [Directly Verified] | Unweighted class recall mean | Sec. V (Results) |
| **CLM-35** | Independent Test Weighted-F1 | 0.8137 | `outputs/reports/phase7/phase7_test_metrics.json` | `metrics.weighted_f1` | [Directly Verified] | Class-support weighted F1 | Sec. V (Results) |
| **CLM-36** | Test Red Gram Recall | 0.2000 (7/35) | `outputs/reports/phase7/phase7_test_metrics.json` | `per_class.Red Gram.recall` | [Directly Verified] | 21/35 misclassified as Cotton; 3 as Bengal Gram | Sec. V, VI |
| **CLM-37** | Test Cloud Loss on Paddy | 25.91% (185/714) | `outputs/reports/phase7/phase7_test_metrics.json` | `per_class.Paddy.cloud_excluded` | [Directly Verified] | Monsoon flood/cloud obscuration | Sec. V, Limitations |
| **CLM-38** | Test Cloud Loss on Bengal Gram | 4.17% (2/48) | `outputs/reports/phase7/phase7_test_metrics.json` | `per_class.Bengal Gram.cloud_excluded` | [Directly Verified] | Rabi dry-season pulse crop | Sec. V, Limitations |
| **CLM-39** | Full-Season Single-Window Baseline | Macro-F1 = 0.6174 | `outputs/reports/phase6f_stage4_benchmark_report.json` | Line 424 | [Directly Verified] | $N=7,843$, 34 full-season features | Sec. V (Ablations) |
| **CLM-40** | Early-Season Single-Window Baseline | Macro-F1 = 0.5058 | `outputs/reports/phase6f_stage4_benchmark_report.json` | Line 1028 | [Directly Verified] | $N=7,843$, 34 early-season features | Sec. V (Ablations) |
| **CLM-41** | Pilot Satellite-Only Baseline | Macro-F1 = 0.5511 | `outputs/reports/phase6d_model_benchmark.json` | `XGB_satellite_only` | [Directly Verified] | Exploratory pilot cohort $N=714$ | Sec. V (Progression) |
| **CLM-42** | Downstream Irrigation Engine | Not Implemented | Repository source tree | Code search for FAO-56 / ETc algorithms | [Contradicted by Evidence] | **EXCLUDED**; reserved as future work | Sec. I, VI |
| **CLM-43** | Multi-Year Cross-Season Generalization | Not Tested | Dataset manifest | Single agricultural cycle (2023) | [Contradicted by Evidence] | Explicitly disclosed as temporal limitation | Sec. VI (Limitations) |

---

## 3. Methodological Caveats & Boundary Conditions

1. **Hyperparameter Tuning Transparency:** Tuning was conducted exclusively using a 10-trial parameter grid evaluated via 3-fold grouped inner cross-validation on the training split of Outer Fold 0. The selected parameters (`max_depth=7`, `colsample_bytree=0.75`, `reg_lambda=5.0`) were subsequently fixed and deployed across all 5 spatial cross-validation folds and the final frozen test inference model. This protocol must be explicitly documented as *exploratory tuning on a single outer fold partition* rather than nested cross-validation.
2. **Generalization Claim Governance:** The numeric increase from 5-fold spatial CV (Macro-F1 0.7226, Accuracy 78.81%) to independent test evaluation (Macro-F1 0.7308, Accuracy 81.55%) **must not be claimed as proof of superior generalization**. It reflects an altered class distribution (Maize represents 27.86% of evaluated test farms vs 17.54% of training; Bengal Gram represents 2.03% vs 11.96%) and selective attrition from cloud cover (453 excluded parcels).
3. **Cohort Size and Geographic Coverage:** Training comprises 7,843 parcels across 116 Mandals; independent test comprises 2,265 evaluated parcels across 35 Mandals. Both cohorts are situated within Telangana, India. Geographic transfer beyond Telangana's semi-arid agro-ecological zone has not been experimentally tested.
