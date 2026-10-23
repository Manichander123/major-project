# Research Integrity and Citations Audit

**Task I & J:** Verify completed vs planned work; bibliography integrity  
**Sources:** All Phase 6 report files, `project_evidence_and_discrepancies.md`, `related_work_matrix.md`  
**Audit date:** 2026-10-23 (read-only)

---

## 1. Implemented and Evaluated Work (Safe to Claim)

| Claim | Evidence | Status |
|---|---|---|
| Sentinel-2 satellite features extracted for 7,843 farms | `project_evidence_and_discrepancies.md` §3.1, SHA-256 hashes verified | **VERIFIED** |
| 5-Fold StratifiedGroupKFold spatial CV grouped by Mandal | `phase6i_robustness_audit.json`, `zero_mandal_overlap_confirmed: true` | **VERIFIED** |
| Pooled macro-F1 = 0.7226, balanced acc = 0.7180, OA = 0.7881, weighted-F1 = 0.7857 | `phase6i_robustness_audit.json` `reconciled_models.weighted_S+T.pooled_metrics` | **VERIFIED** |
| Fold-mean macro-F1 = 0.7117 ± 0.0181 | `phase6i_robustness_audit.json` `fold_summary` (population SD, ddof=0) | **VERIFIED** |
| 20 GLCM texture features evaluated (S+T+P model, 160 features) | `phase6h_feature_schema_manifest.json` | **VERIFIED** |
| Paired tests: t-test p=0.7765, bootstrap p=0.8600, CI [−0.0066, +0.0092] | `phase6i_robustness_audit.json` `paired_significance_audit` | **VERIFIED** |
| Parcel quartile analysis (Q1–Q4) | `phase6i_robustness_audit.json` `area_quartile_metrics` | **VERIFIED (descriptive, no CIs)** |
| 45 cloud-obscured farms excluded | `project_evidence_and_discrepancies.md` DISC-06 | **VERIFIED** |
| Zero coordinate/target feature leakage | `phase6i_robustness_audit.json` `leakage_check_passed: true` | **VERIFIED** |

---

## 2. Planned / Not-Yet-Implemented Work (Must NOT Be Claimed as Results)

| Item | Status | Wording constraint |
|---|---|---|
| Satellite feature extraction for Test.csv (2,718 farms) | **NOT done** | Describe as "out-of-scope in this study" or "future work" |
| Blind test submission / test set score | **NOT done** | Do not claim any test set macro-F1 |
| Crop health classification (`category` column) | **NOT done** | "Planned secondary phase" only |
| Soil moisture estimation / FAO-56 ETc calculation | **NOT done** | "Broader system vision" only |
| Irrigation recommendation engine | **NOT done** | "Future extension" only |
| Multi-year generalization (any year other than 2023–2024) | **NOT done** | "Single-season study" limitation |
| Full 5×5 nested CV grid search | **NOT done** | See `validation_and_tuning_audit.md` |

---

## 3. Bibliography Integrity Assessment

### Papers confirmed in published literature (publication details verifiable)

| # | Authors (Year) | Venue | Core claim | Status |
|---|---|---|---|---|
| 1 | Defourny et al. (2019) | *Remote Sensing of Environment*, Vol. 234 | Sentinel-2 national crop mapping, 80–92% OA | Verified journal/volume — plausible |
| 2 | Jin et al. (2019) | *Remote Sensing of Environment*, Vol. 234 | Smallholder maize mapping with S2+PlanetScope | Verified journal/volume — plausible |
| 3 | Griffiths et al. (2019) | *Remote Sensing of Environment*, Vol. 230 | Phenological metrics + RF, OA 88.5% | Verified journal/volume — plausible |
| 4 | Roberts et al. (2017) | *Ecography*, Vol. 40, Issue 8 | Spatial autocorrelation in geospatial CV | Verified journal/volume — plausible |
| 5 | Meyer et al. (2018) | *Ecological Informatics*, Vol. 45 | Spatial overfitting in env. remote sensing | Verified journal/volume — plausible |
| 6 | Vuolo et al. (2018) | *Int. J. Applied Earth Obs.*, Vol. 72 | GLCM texture marginal gain on large parcels | Verified journal/volume — plausible |
| 7 | Lebourgeois et al. (2017) | *Remote Sensing*, Vol. 9, Issue 12 | Smallholder mapping in Madagascar | Verified journal/volume — plausible |
| 8 | Inglada et al. (2015) | *Remote Sensing*, Vol. 7, Issue 9 | Sen2-Agri operational mapping, 75–85% OA | Verified journal/volume — plausible |
| 9 | Maxwell et al. (2018) | *Remote Sensing*, Vol. 10, Issue 8 | Class imbalance handling review | Verified journal/volume — plausible |
| 10 | Chen & Guestrin (2016) | *ACM SIGKDD* | XGBoost scalable tree boosting | Verified — seminal, widely cited |

**CAUTION:** "Plausible" means the journal name and volume number are internally consistent with the publication year and topic. This audit did NOT independently verify DOIs, page numbers, author lists, or actual metric values cited from each paper. Before submission, DOIs must be verified for papers 1–9 using a scholarly database (Google Scholar, Semantic Scholar, or the journal website).

### Claims attributed to cited papers — spot check

| Claim | Paper | Risk |
|---|---|---|
| "Random CV underestimates error by 20–60%" | Roberts et al. (2017) | Plausible but the exact percentage range needs verification |
| "Texture added +0.5% to +1.2% OA on large parcels" | Vuolo et al. (2018) | Plausible; verify against the actual paper |
| "30–40% boundary contamination on fields < 0.25 ha" | Lebourgeois et al. (2017) | Plausible; verify wording |

None of these appear to be fabricated values, but they should be verified by reading the actual papers before submission.

---

## 4. Established Methods Reused (Not Novel Contributions)

- StratifiedGroupKFold spatial CV — standard scikit-learn.
- XGBoost with L1/L2 regularization — Chen & Guestrin (2016).
- Inverse-frequency sample weighting — standard cost-sensitive learning.
- GLCM texture extraction — well-established image analysis technique.
- NDVI, EVI, NDWI, NDMI, BSI, etc. — standard remote sensing indices.
- Sentinel-2 SCL cloud masking — ESA standard.

---

## 5. Empirical Findings Supported by This Study's Experiments

- GLCM texture adds no significant macro-F1 gain under Mandal-grouped spatial CV (p=0.8600).
- Inverse-frequency weighting improves macro-F1 by +1.57 pp over unweighted training on 7,843-farm cohort.
- Parcel area correlates with GLCM effectiveness (size-stratified association, exploratory).
- 84.6% of Red Gram parcels are in Adilabad district — spatial confounding documented.
- Random CV produces artificially inflated macro-F1 due to spatial autocorrelation (Phase 6A discarded baseline: 0.9230).

---

## 6. Potentially Novel Findings Requiring Literature Verification

- Parcel-area quartile ablation of GLCM features under Mandal-grouped spatial CV.
- Quantification of temporal-window phenological deltas for Telangana seven-class classification.
- Empirical bootstrap of spatial cluster uncertainty on the S+T vs S+T+P difference.

These are novel in the Telangana smallholder context but may have been done elsewhere. A targeted literature search is recommended before claiming novelty.
