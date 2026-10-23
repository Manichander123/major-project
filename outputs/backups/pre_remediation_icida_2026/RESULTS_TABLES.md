# Publication-Ready Results Tables and Benchmark Matrices

**Project Title:** Multi-Temporal Sentinel-2 Spectral-Temporal Features for Seven-Class Crop Classification in Telangana: Spatial Cross-Validation and Independent Test Evaluation  
**Auditor:** Senior Remote Sensing Researcher & Experimental Auditor  
**Status:** FULLY RECONCILED AGAINST PRIMARY REPOSITORY ARTIFACTS (STAGE E REMEDIATION)  

---

## Table 1: Cohort Composition and Spatial Partitioning

| Cohort Property | Training Cohort (`Train.csv`) | Independent Held-Out Test Cohort (`Test.csv`) |
| :--- | :---: | :---: |
| **Initial Surveyed Parcels** | 7,888 | 2,718 |
| **Persistent Cloud / Shadow Failures** | 45 (0.57%) | 453 (16.67%) |
| **Eligible Evaluated Parcels** | **7,843 (99.43%)** | **2,265 (83.33%)** |
| **Administrative Units (Sub-Districts / Mandals)** | 116 | 35 |
| **Districts Covered** | 14 | Distinct non-overlapping sub-regions |
| **Inter-Cohort Mandal Overlap** | — | **0 Mandals (Disjoint Mandal Holdout within Telangana)** |
| **Validation Design** | 5-Fold `StratifiedGroupKFold` | Frozen Model Out-of-Mandal Inference |
| **Target Variable** | `Crop` (7 Ground-Truth Classes) | `Crop` (7 Ground-Truth Classes) |
| **Secondary Target (`category`)** | Preserved (Health status; not trained) | Preserved (Health status; not evaluated) |

*Note on Geographic Domain:* Both the 116 training Mandals and the 35 test Mandals are situated within the state of Telangana, India. While there is zero Mandal group overlap between cohorts, an administrative holdout does not guarantee zero spatial autocorrelation; nearby Mandals may remain geographically close or share agro-climatic characteristics. This evaluation assesses prediction across unseen administrative units within Telangana, not a cross-state, contiguous spatial-block, or multi-year transfer test.

---

## Table 2: Parcel-Level Feature Extraction Coverage & Cloud Exclusion Accounting

| Crop Class | Total Test Parcels | Evaluated Parcels | Cloud-Excluded Parcels | Extraction Coverage (%) | Exclusion Rate (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Bengal Gram** | 48 | 46 | 2 | 95.83% | 4.17% |
| **Groundnut** | 266 | 241 | 25 | 90.60% | 9.40% |
| **Maize** | 699 | 631 | 68 | 90.27% | 9.73% |
| **Red Gram** | 41 | 35 | 6 | 85.37% | 14.63% |
| **Cotton** | 569 | 472 | 97 | 82.95% | 17.05% |
| **Chillies** | 381 | 311 | 70 | 81.63% | 18.37% |
| **Paddy** | 714 | 529 | 185 | 74.09% | **25.91%** |
| **Total Cohort** | **2,718** | **2,265** | **453** | **83.33%** | **16.67%** |

*Note on Selection Bias:* Cloud exclusion is strongly crop-dependent. Monsoon Kharif crops (especially flood-irrigated Paddy) experience an exclusion rate over six times higher than dry-season Rabi pulses (Bengal Gram). Excluded parcels receive null predictions and are excluded from supervised evaluation to prevent synthetic imputation bias; consequently, the evaluated subset is not a uniform random sample of all surveyed parcels.

---

## Table 3: 140-Dimensional Spectral-Temporal Feature Representation

| Feature Group | Temporal Scope / Window | Feature Description | Count |
| :--- | :--- | :--- | :---: |
| **Spectral & Biophysical Indices (S)** | Full Season (Jun 01 – Dec 31) | 10 Sentinel-2 L2A surface reflectance bands (B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12) + 14 spectral indices (NDVI, EVI, NDMI, NDWI, GNDVI, NDRE, MSAVI, CI Green, CI Red Edge, MSI, NDBI, SAVI, SIPI, NMDI). Extracted via median aggregation. | 34 |
| **Windowed Phenology (W1: Early)** | Early Season (Jun 01 – Jul 31) | Identical 34 spectral bands and vegetation/water indices extracted during crop establishment / vegetative emergence. | 34 |
| **Windowed Phenology (W2: Mid)** | Mid Season (Aug 01 – Sep 30) | Identical 34 spectral bands and vegetation/water indices extracted during peak vegetative vigor / flowering. | 34 |
| **Windowed Phenology (W3: Late)** | Late Season (Oct 01 – Nov 30) | Identical 34 spectral bands and vegetation/water indices extracted during maturity / senescence. | 34 |
| **Dynamic Transition Deltas (T)** | Multi-Window Trajectories | 1. `ndvi_growth_delta`: $\text{NDVI}_{\text{Mid}} - \text{NDVI}_{\text{Early}}$<br>2. `peak_to_early_ratio`: $\text{NDVI}_{\text{Mid}} / (\text{NDVI}_{\text{Early}} + \epsilon)$<br>3. `ndvi_senescence_delta`: $\text{NDVI}_{\text{Late}} - \text{NDVI}_{\text{Mid}}$<br>4. `moisture_loss_delta`: $\text{NDMI}_{\text{Late}} - \text{NDMI}_{\text{Mid}}$ | 4 |
| **Total Dimensionality** | — | **Full Spectral-Temporal Vector Space ($S + T$)** | **140** |

---

## Table 4: XGBoost Hyperparameter Specification and Tuning Protocol

| Hyperparameter | Selected Value | Search Space Evaluated (10 Trials) | Tuning Protocol & Context |
| :--- | :---: | :---: | :--- |
| **`n_estimators`** | 300 | [100, 200, 300, 500] | Fixed during exploration |
| **`max_depth`** | 7 | [5, 6, 7, 8] | Selected for non-linear canopy interactions |
| **`learning_rate` ($\eta$)** | 0.05 | [0.03, 0.05, 0.10] | Conservative shrinkage for generalization |
| **`subsample`** | 0.80 | [0.70, 0.80, 0.90] | Row sub-sampling per boosting iteration |
| **`colsample_bytree`** | 0.75 | [0.60, 0.75, 0.80, 1.00] | Feature sub-sampling per tree split |
| **`min_child_weight`** | 3 | [1, 3, 5] | Prevents fitting spurious smallholder clusters |
| **`reg_lambda` ($\text{L}_2$)** | 5.0 | [1.0, 3.0, 5.0, 10.0] | Heavy L2 regularization against spatial noise |
| **`reg_alpha` ($\text{L}_1$)** | 0.1 | [0.0, 0.1, 0.5] | Mild sparsity enforcement |
| **`tree_method`** | `hist` | [`hist`] | Fast exact histogram binning |
| **Class Weighting** | Inverse-Frequency | [None, Balanced, Inverse-Freq] | $w_c = N / (C \cdot N_c)$; computed fold-internally |

*Methodological Note:* Hyperparameters were tuned via an exploratory bounded search of ten configurations evaluated via 3-fold grouped inner cross-validation strictly on the training partition of Outer Fold 0 ($N = 6,335$ parcels across 93 Mandals; Outer Fold 0 validation partition has $N = 1,508$ parcels across 23 Mandals). Selection was based on inner Macro-F1 (0.6797). The chosen configuration was frozen and evaluated across all 5 spatial outer folds and the independent test set.

---

## Table 5: 5-Fold Spatial Cross-Validation Performance (`Weighted S+T`, $N=7,843$)

| Spatial Fold ID | Validation Mandals | Validation Parcels | Macro-F1 | Overall Accuracy | Balanced Accuracy | Weighted-F1 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 0** | 23 | 1,508 | 0.7269 | 0.7960 | 0.7203 | 0.7932 |
| **Fold 1** | 21 | 1,600 | 0.7381 | 0.8013 | 0.7329 | 0.7997 |
| **Fold 2** | 22 | 1,595 | 0.6919 | 0.7717 | 0.6865 | 0.7681 |
| **Fold 3** | 26 | 1,530 | 0.6945 | 0.7749 | 0.6911 | 0.7728 |
| **Fold 4** | 24 | 1,610 | 0.7073 | 0.7966 | 0.7102 | 0.7946 |
| **Fold Mean ± Pop. SD** | 23.2 | 1,568.6 | **0.7117 ± 0.0181** | **0.7881 ± 0.0123** | **0.7082 ± 0.0177** | **0.7857 ± 0.0128** |
| **Fold Mean ± Sample SD**| 23.2 | 1,568.6 | **0.7117 ± 0.0202** | **0.7881 ± 0.0137** | **0.7082 ± 0.0198** | **0.7857 ± 0.0143** |
| **Pooled Out-of-Fold** | **116** | **7,843** | **0.7226** | **0.7881 (78.81%)** | **0.7180** | **0.7857** |

*Note on Fold Counts:* Partition counts reflect actual administrative cluster sizes in `phase6f_cv_folds.csv`. Standard deviation is disclosed under both population ($\text{ddof}=0$) and sample ($\text{ddof}=1$) conventions.

---

## Table 6: Model Evolution, Feature Ablations, and Significance Testing

| Model / Pipeline Stage | Features | Cohort ($N$) | Weighting Strategy | Spatial Macro-F1 | Overall Accuracy | Balanced Accuracy | Statistical Contrast vs `Weighted S+T` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Phase 6D Satellite Pilot** | 34 | 714 (Pilot) | Unweighted | 0.5511 | 0.6653 | 0.5312 | Exploratory cohort; baseline |
| **Early-Season Single Window** | 34 | 7,843 | Unweighted | 0.5058 | 0.6358 | 0.4902 | $\Delta = -0.2168$ (-21.68 pp) |
| **Full-Season Single Window** | 34 | 7,843 | Unweighted | 0.6174 | 0.7208 | 0.6015 | $\Delta = -0.1052$ (-10.52 pp) |
| **Unweighted Multi-Temporal S+T** | 140 | 7,843 | Unweighted | 0.7069 | 0.7924 | 0.6982 | $\Delta = -0.0157$ (-1.57 pp) |
| **Unweighted S+T+P (+20 GLCM)** | 160 | 7,843 | Unweighted | 0.7038 | 0.7900 | 0.6895 | $\Delta = -0.0188$ (-1.88 pp) |
| **Final Weighted S+T (Winner)** | **140** | **7,843** | **Inverse-Freq** | **0.7226** | **0.7881** | **0.7180** | **Reference Champion Pipeline** |
| **Weighted S+T+P (+20 GLCM)** | 160 | 7,843 | Inverse-Freq | 0.7228 | 0.7905 | 0.7188 | $\Delta = +0.0002$ ($p=0.8600$, 95% CI $[-0.0066, +0.0092]$) |

*Key Takeaway:* Adding 20 GLCM spatial texture rasters yields no statistically supported performance gain over pure Spectral-Temporal features ($p = 0.8600$ via cluster bootstrap; paired $t$-test $p = 0.7765$). S+T was selected via parsimony (Occam's razor).

---

## Table 7: Independent Test Performance vs Spatial Cross-Validation Parity

| Metric | 5-Fold Spatial CV (Pooled $N=7,843$) | Independent Test Set (Evaluated $N=2,265$) | Observed Delta | Interpretation / Context |
| :--- | :---: | :---: | :---: | :--- |
| **Macro-F1** | 0.7226 | **0.7308** | +0.0082 | Confirms consistent performance across 35 unseen Mandals |
| **Overall Accuracy** | 0.7881 (78.81%) | **0.8155 (81.55%)** | +0.0274 | Influenced by higher proportion of Maize in test set |
| **Balanced Accuracy** | 0.7180 | **0.7326** | +0.0146 | Steady unweighted per-class discrimination |
| **Weighted-F1** | 0.7857 | **0.8137** | +0.0280 | Reflects high accuracy on dominant test classes |
| **Evaluated Sample Size** | 7,843 | 2,265 | — | 116 training Mandals vs 35 unseen test Mandals |
| **Excluded Sample Size** | 45 (0.57%) | 453 (16.67%) | — | Higher monsoon cloud impact in test cohort |

---

## Table 8: Verified Class-Wise Performance Metrics (Spatial CV vs Independent Test)

| Crop Class | Spatial CV Precision | Spatial CV Recall | Spatial CV F1-Score | Test Precision | Test Recall | Test F1-Score | Test Eval Support | Test Total Support | Test Cloud Loss (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bengal Gram** | 0.7916 | 0.8667 | 0.8275 | 0.6792 | 0.7826 | 0.7273 | 46 | 48 | 4.17% |
| **Chillies** | 0.5249 | 0.5441 | 0.5343 | 0.7409 | 0.7814 | 0.7606 | 311 | 381 | 18.37% |
| **Cotton** | 0.7939 | 0.8261 | 0.8097 | 0.7505 | 0.8157 | 0.7817 | 472 | 569 | 17.05% |
| **Groundnut** | 0.8597 | 0.8698 | 0.8647 | 0.8340 | 0.8755 | 0.8543 | 241 | 266 | 9.40% |
| **Maize** | 0.7492 | 0.6730 | 0.7090 | 0.9016 | 0.7845 | 0.8390 | 631 | 699 | 9.73% |
| **Paddy** | 0.8664 | 0.8752 | 0.8708 | 0.8468 | 0.8885 | 0.8672 | 529 | 714 | 25.91% |
| **Red Gram** | 0.5467 | 0.3710 | **0.4420** | 0.5000 | 0.2000 | **0.2857** | 35 | 41 | 14.63% |
| **Macro Average** | **0.7332** | **0.7180** | **0.7226** | **0.7504** | **0.7326** | **0.7308** | **2,265** | **2,718** | **16.67%** |

---

## Table 9: Independent Test Set Confusion Matrix ($N=2,265$)

| True Crop $\downarrow$ \ Predicted $\rightarrow$ | Bengal Gram | Chillies | Cotton | Groundnut | Maize | Paddy | Red Gram | Total True Support |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bengal Gram** | **36** | 1 | 4 | 0 | 2 | 1 | 2 | 46 |
| **Chillies** | 0 | **243** | 40 | 5 | 12 | 11 | 0 | 311 |
| **Cotton** | 7 | 14 | **385** | 20 | 15 | 28 | 3 | 472 |
| **Groundnut** | 0 | 9 | 11 | **211** | 6 | 4 | 0 | 241 |
| **Maize** | 6 | 41 | 36 | 12 | **495** | 40 | 1 | 631 |
| **Paddy** | 1 | 18 | 16 | 5 | 18 | **470** | 1 | 529 |
| **Red Gram** | 3 | 2 | 21 | 0 | 1 | 1 | **7** | 35 |
| **Total Predicted** | **53** | **328** | **513** | **253** | **549** | **555** | **14** | **2,265** |

*Verified Column Sums and Precision Cross-Check:*
- Red Gram Precision = 7 / 14 = 0.5000 (50.0%)
- Paddy Precision = 470 / 555 = 0.8468 (84.68%)
- Maize Precision = 495 / 549 = 0.9016 (90.16%)
- Groundnut Precision = 211 / 253 = 0.8340 (83.40%)
- Cotton Precision = 385 / 513 = 0.7505 (75.05%)
- Chillies Precision = 243 / 328 = 0.7409 (74.09%)
- Bengal Gram Precision = 36 / 53 = 0.6792 (67.92%)

---

## Table 10: Parcel Area Quartile Stratification: GLCM Spatial Texture vs Spectral-Temporal

| Quartile | Parcel Area Range ($\text{m}^2$) | Area Description | Eligible Parcels ($N$) | `Weighted S+T` Macro-F1 | `Weighted S+T+P` Macro-F1 | Delta ($\Delta_{\text{P - ST}}$) | Physical Interpretation / Hypothesis |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Q1** | 54 – 1,257 | Smallest Parcels | 1,961 | **0.6598** | 0.6465 | **-0.0133 (-1.33 pp)** | **Consistent with Boundary Contamination:** A $3\times3$ GLCM kernel ($900\text{ m}^2$) constitutes a large fraction of small fields, potentially sampling uncropped borders. |
| **Q2** | 1,257 – 2,783 | Medium-Low Area | 1,961 | **0.7332** | 0.7184 | **-0.0148 (-1.48 pp)** | Intermediate parcel scale; texture feature inclusion associated with slight decrease. |
| **Q3** | 2,784 – 5,517 | Medium-High Area | 1,960 | 0.7400 | **0.7530** | **+0.0130 (+1.30 pp)** | Interior parcel pixels exceed boundary footprint; texture captures canopy pattern. |
| **Q4** | 5,528 – 69,491 | Largest Holdings | 1,961 | 0.7253 | 0.7219 | **-0.0034 (-0.34 pp)** | Homogeneous interior canopy; negligible difference. |
| **Full** | **54 – 69,491** | **Entire Cohort** | **7,843** | **0.7226** | **0.7228** | **+0.0002 (+0.02 pp)** | Global difference is statistically indistinguishable ($p = 0.8600$). |

*Methodological Note:* The quartile analysis is an exploratory descriptive breakdown without separate subgroup hypothesis testing. The minor numerical gap between Q3 upper bound ($5,517\text{ m}^2$) and Q4 lower bound ($5,528\text{ m}^2$) is a documented discrete binning artifact of `pandas.qcut` on discrete parcel area measurements.
