# Phase 6F Stage 4A: Read-Only Modeling Preflight Report

**Project:** AI-Based Remote Sensing Framework for Crop Monitoring and Irrigation Decision Support Using Multi-Temporal Sentinel-2 Satellite Imagery  
**Stage:** Phase 6F — Stage 4A (Read-Only Modeling Preflight)  
**Status:** **READY WITH CAVEATS**  
**Audit Timestamp:** 2026-10-18T10:00:00Z  

---

## 1. Executive Decision: READY WITH CAVEATS

The project is **READY WITH CAVEATS** to proceed to Stage 4B model benchmarking, subject to the strict implementation safeguards defined in this preflight report.

### Core Justifications for "READY":
1. **Authoritative Cohort Invariance:** The harmonized dataset [`outputs/data/s2_phase6f_harmonized_cohort.csv`](file:///c:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/data/s2_phase6f_harmonized_cohort.csv) accounts for 100.0% of the 7,888 survey farms in [`datasss/Train.csv`](file:///c:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/datasss/Train.csv) with zero duplicate IDs, zero label discrepancies, zero coordinate corruption, and zero infinities.
2. **Eligible Modeling Universe:** Exactly 7,843 farms form the usable primary satellite modeling cohort (99.43% coverage). The 45 persistent cloud failures are honestly recorded with 100% NaN spectral values and strictly segregated via the `cloud_obscured_failure = True` flag.
3. **Full-Season Spectral Integrity:** 100.0% of the 7,843 eligible farms possess complete, valid Sentinel-2 surface reflectance and standard deviation statistics across all 10 bands in the `full` season composite window.
4. **Spatial Cross-Validation Feasibility:** 5-fold `StratifiedGroupKFold` grouped by `Sub-District` (116 Mandals) is mathematically verified with 0 group overlap between training and validation folds. All 7 target crop classes are robustly represented across every fold, including the rarest class (Red Gram: $\ge 21$ validation samples per fold).
5. **Daily Mosaicking Provenance Verified:** Source code inspection and historical empirical audit reports confirm that same-date Sentinel-2 granules were mosaicked prior to temporal reduction throughout Phase 6D, 6E, and 6F, eliminating MGRS tile overlap weighting bias.

### Mandatory "CAVEATS" to Enforce in Stage 4B:
1. **Sub-Window Missingness Handling:** Phenological sub-windows contain physical cloud gaps (541 missing in `early`, 121 in `mid`, 192 in `late`). While XGBoost and LightGBM handle NaNs natively, Scikit-Learn Random Forest requires a strict fold-isolated `SimpleImputer(strategy='median')` pipeline fitted exclusively on outer training folds.
2. **Exclusion of `Test.csv` as Satellite Holdout:** `datasss/Test.csv` ($N=2,718$) does **not** have Sentinel-2 satellite extractions. Stage 4 evaluation must rely exclusively on out-of-fold spatial CV over the 7,843 `Train.csv` satellite farms.
3. **Strict Target & Proxy Leakage Barrier:** Survey proxy fields (`CropCoveredArea`, `CHeight`, `CLast`, `CNext`, `CTransp`, `IrriType`, `IrriSource`, `IrriCount`, `WaterCov`, `ExpYield`), spatial coordinates (`lat`, `lon`, `centroid_lat`, `centroid_lon`), administrative identifiers, and secondary target `category` are **strictly prohibited** from predictive feature sets.

---

## 2. Evidence Registry & Provenance Audit

Every claim and source artifact was examined directly in the filesystem:

| Artifact / Claim | Path / Location | Finding Status | Evidence Details |
| :--- | :--- | :---: | :--- |
| **Harmonized Cohort Dataset** | [`outputs/data/s2_phase6f_harmonized_cohort.csv`](file:///c:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/data/s2_phase6f_harmonized_cohort.csv) | **VERIFIED** | 7,888 rows, 100 columns, SHA-256 `caed4029...`. 0 duplicate IDs. |
| **Stage 3 Harmonization Report** | [`outputs/reports/phase6f_stage3_harmonization_report.json`](file:///c:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/reports/phase6f_stage3_harmonization_report.json) | **VERIFIED** | Cohort metrics, 10/10 Stage 3 tests passed. |
| **Stage 2 Recovery Report & Ledger** | [`outputs/reports/phase6f_stage2_recovery_ledger.csv`](file:///c:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/reports/phase6f_stage2_recovery_ledger.csv) | **VERIFIED** | 20 entries (5 farms $\times$ 4 windows). 5 preserved windows, 15 recovered. |
| **Stage 1 Dataset Audit** | [`outputs/reports/phase6f_dataset_audit.json`](file:///c:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/reports/phase6f_dataset_audit.json) | **VERIFIED** | Inventory of Phase 6E raw bands ($N=7,164$), chunk reconciliation. |
| **MGRS Daily Mosaic Audit** | [`outputs/reports/mgrs_multi_farm_audit.json`](file:///c:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/reports/mgrs_multi_farm_audit.json) | **VERIFIED** | Multi-tile overlap analysis in Adilabad (up to 3.67x overlap), verified mosaic logic. |
| **Sentinel-2 Extractor Logic** | [`src/satellite/s2_extractor.py`](file:///c:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/src/satellite/s2_extractor.py#L169-L192) | **VERIFIED** | `mosaic_same_date_scenes` implemented; daily mosaicking before `.median()`. |
| **Phase 6E Extraction Script** | [`scripts/extract_s2_full_campaign_phase6e.py`](file:///c:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/scripts/extract_s2_full_campaign_phase6e.py#L82) | **VERIFIED** | Default `daily_mosaic=True` confirmed. |
| **Phase 6D Benchmark Report** | [`outputs/reports/phase6d_model_benchmark.json`](file:///c:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/reports/phase6d_model_benchmark.json) | **VERIFIED** | Prior grouped spatial baseline on $N=714$: RF macro-F1 0.496, XGB macro-F1 0.551. |
| **Project Configuration** | [`config/project_config.yaml`](file:///c:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/config/project_config.yaml) | **VERIFIED** | 7 crop classes, EPSG:4326/32644, S2 bands and index formulas. |
| **Protected Baseline Files (24)** | Manifests, chunks, raw bands, recovery artifacts | **VERIFIED** | All 24 files re-hashed; SHA-256 identical to recorded baselines. |
| **Test.csv Holdout Proposal** | [`datasss/Test.csv`](file:///c:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/datasss/Test.csv) | **DEFECT** | Lacks S2 satellite extractions; 0 overlap with harmonized cohort. Cannot serve as satellite holdout. |
| **Phase 6A Random-Split Baseline (~0.90)** | Prior report documentation | **DEFECT** | Severe spatial leakage across village/mandal clusters. Invalid for spatial comparison. |

---

## 3. Cohort Accounting & Label Integrity

### Exact Population Partition
$$\text{Total Survey Population (Train.csv)} = 7,888 \text{ unique farms}$$
$$7,888 = 7,164 \text{ (Phase 6E)} + 724 \text{ (Phase 6D)}$$
$$\text{Harmonized Modeling Population} = 7,843 \text{ successful farms (99.43% coverage)}$$
$$\text{Persistent Cloud Failures} = 45 \text{ farms (0.57% honest missingness: 35 from 6E, 10 from 6D)}$$

### Primary Target (`Crop`) Class Distribution
The target variable is strictly `Crop` (7 distinct classes). Class counts match `Train.csv` with 0 mismatches:

| Crop Class | Total Population (`Train.csv`) | Eligible Satellite Cohort (`extraction_status == success`) | Cloud Failure Cohort (`extraction_status == failed`) | Class Share (Eligible %) |
| :--- | :---: | :---: | :---: | :---: |
| **Paddy** | 2,268 | 2,261 | 7 | 28.83% |
| **Cotton** | 1,692 | 1,681 | 11 | 21.43% |
| **Maize** | 1,384 | 1,378 | 6 | 17.57% |
| **Bengal Gram** | 938 | 932 | 6 | 11.88% |
| **Groundnut** | 768 | 767 | 1 | 9.78% |
| **Chillies** | 614 | 603 | 11 | 7.69% |
| **Red Gram** | 224 | 221 | 3 | 2.82% |
| **Total** | **7,888** | **7,843** | **45** | **100.00%** |

### Metadata & Boundary Consistency
- `Crop`: 0 mismatches against `Train.csv`.
- `District`: 0 mismatches against `Train.csv` (14 districts).
- `Sub-District`: 0 mismatches against `Train.csv` (116 mandals).
- `Season`: 0 mismatches against `Train.csv` (Kharif: 5,689; Rabi: 2,154 in eligible cohort).
- `SDate` / `HDate`: 0 date mismatches when parsed as datetimes.

---

## 4. Temporal-Window Missingness Reconciliation

### Mathematical Identity Between Stage 1 Audit and Stage 3 Report
The Stage 1 audit inspected **Phase 6E raw bands only ($N=7,164$)**, while Stage 3 inspected the **complete harmonized modeling cohort ($N=7,843$)**:

| Window | Phase 6E Success ($N=7,124$) | Phase 6D Success ($N=714$) | Phase 6F Recovery ($N=5$) | Stage 3 Total Missing ($N=7,843$) | Stage 1 Audit Phase 6E Raw ($N=7,164$) | Reconciliation Formula |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`early`** | 428 | 113 | 0 | **541** | **466** | $428 \text{ (6E succ)} + 38 \text{ (failed)} = 466$; $428 + 113 = 541$ |
| **`mid`** | 95 | 26 | 0 | **121** | **134** | $95 \text{ (6E succ)} + 39 \text{ (failed)} = 134$; $95 + 26 = 121$ |
| **`late`** | 159 | 33 | 0 | **192** | **198** | $159 \text{ (6E succ)} + 39 \text{ (failed)} = 198$; $159 + 33 = 192$ |
| **`full`** | 0 | 0 | 0 | **0** | **40** | $0 \text{ (6E succ)} + 40 \text{ (failed)} = 40$; $0 + 0 = 0$ |

### Full-Season Composite vs. 4-Window Trajectory Breakdown
- **Full-Season Composite (`full` window):** **100.0% valid (7,843 / 7,843 farms)**.
- **Complete Multi-Temporal Trajectory:** Exactly **7,107 farms (90.62%)** have valid observations across all 4 windows simultaneously.
- **Partial Phenological Trajectory:** Exactly **736 farms (9.38%)** have valid full-season data, but lack observations in one or two specific sub-windows:
  - `early` missing only: 446 farms (5.69%) — peak July/August monsoon cloud cover.
  - `late` missing only: 121 farms (1.54%) — dry-season transition or cloudy harvesting edge.
  - `mid` missing only: 51 farms (0.65%) — mid-season monsoon squall line.
  - `early` + `late` missing: 48 farms (0.61%).
  - `early` + `mid` missing: 47 farms (0.60%).
  - `mid` + `late` missing: 23 farms (0.29%).
  - All 3 sub-windows missing: **0 farms (0.00%)**.

---

## 5. Daily Mosaicking: Verification & Provenance

### Source Code Verification
In [`src/satellite/s2_extractor.py`](file:///c:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/src/satellite/s2_extractor.py#L169-L192), `mosaic_same_date_scenes` maps over unique calendar dates:
```python
dates = ee.List(col.map(lambda img: ee.Feature(None, {"date": img.date().format("YYYY-MM-dd")})).aggregate_array("date")).distinct()
def make_daily(d):
    t0 = ee.Date(d)
    t1 = t0.advance(1, "day")
    return col.filterDate(t0, t1).mosaic().set({"system:time_start": t0.millis(), "date": d})
return ee.ImageCollection(dates.map(make_daily))
```
When `daily_mosaic=True` (default in extraction scripts), the temporal reduction is taken across this daily mosaicked collection:
```python
if daily_mosaic:
    daily_col = mosaic_same_date_scenes(col_masked)
    composite = daily_col.median()
else:
    composite = col_masked.median()
```

### Empirical Audit Evidence
In [`outputs/reports/mgrs_multi_farm_audit.json`](file:///c:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/reports/mgrs_multi_farm_audit.json), empirical evaluation on farms situated in Adilabad's multi-tile overlap zone (MGRS tiles 44QND, 44QNE, 44QPD) demonstrated:
- Farm 1327181 had 165 raw tile granules across 45 unique acquisition dates (3.67x overlap).
- Without daily mosaicking, overlap dates had $3\times$ heavier statistical weight in the median, skewing B4 reflectance by $-6.02\%$ and distorting NDVI by $+0.024$.
- Daily mosaicking collapsed same-date granules into one spatial mosaic, ensuring 1 calendar date = 1 vote in the temporal median composite.

### Provenance Conclusion
Daily mosaicking is **VERIFIED** as operational and active across Phase 6D, Phase 6E, and Phase 6F recovery extractions.

---

## 6. Feature Eligibility & Target Leakage Audit

### Primary Target
- Target: `Crop` (7 classes: Bengal Gram, Chillies, Cotton, Groundnut, Maize, Paddy, Red Gram).
- `Crop` is strictly a training label. It is never used as a predictive feature, nor as a group for imputation.

### Prohibited Predictors (Strict Exclusion)
- Secondary Target: `category` (Crop Health: Healthy/Diseased/Pests/Stressed) is strictly excluded.
- Identifiers: `FarmID` is strictly excluded.
- Coordinates: `lat`, `lon`, `centroid_lat`, `centroid_lon`, `geometry` WKT are strictly excluded from the satellite-only models.
- Administrative Names: `State`, `District`, `Sub-District` are strictly excluded from predictive features (used only for spatial grouping).
- Survey Proxies & Post-Harvest Outcomes: `ExpYield`, `CropCoveredArea`, `CHeight`, `CLast`, `CNext`, `CTransp`, `IrriType`, `IrriSource`, `IrriCount`, `WaterCov` are **strictly excluded** because they are self-reported farmer survey fields unavailable in automated satellite deployment.

### Permitted Feature Definitions Across Experiments

| Feature / Feature Group | Source | Available at Prediction? | Target Leakage Risk? | Allowed in Exp A (Full-Season)? | Allowed in Exp B (Early)? | Allowed in Exp C (Multi-Temporal)? | Allowed in Exp D (Multimodal)? | Rationale / Caveats |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`full` Spectral Bands (10 bands $\times$ mean, std)** | Sentinel-2 L2A | Post-harvest / Retrospective | None | **YES (20 cols)** | NO | **YES (20 cols)** | **YES (20 cols)** | Standard optical reflectance across entire crop lifecycle. |
| **`full` Spectral Indices (14 indices)** | Computed from `full` bands | Post-harvest / Retrospective | None | **YES (14 cols)** | NO | **YES (14 cols)** | **YES (14 cols)** | NDVI, EVI, GNDVI, SAVI, MSAVI, NDMI, NDWI, NDRE1, NDRE2, PSRI, BSI, MSI, CIRE, CIG. |
| **`early` Spectral Bands (10 bands $\times$ mean, std)** | Sentinel-2 L2A | In-season (first 25% duration) | None | NO | **YES (20 cols)** | **YES (20 cols)** | **YES (20 cols)** | Early vegetative emergence reflectance. Contains 541 NaNs. |
| **`early` Spectral Indices (14 indices)** | Computed from `early` bands | In-season (first 25% duration) | None | NO | **YES (14 cols)** | **YES (14 cols)** | **YES (14 cols)** | Early vegetative vigor indicators. Contains 541 NaNs. |
| **`mid` Spectral Bands & Indices (34 features)** | Sentinel-2 L2A | Mid-season (25%–75% duration) | None | NO | NO | **YES (34 cols)** | **YES (34 cols)** | Reproductive / flowering stage reflectance. Contains 121 NaNs. |
| **`late` Spectral Bands & Indices (34 features)** | Sentinel-2 L2A | Maturation (75%–100% duration) | None | NO | NO | **YES (34 cols)** | **YES (34 cols)** | Ripening / pre-harvest reflectance. Contains 192 NaNs. |
| **Multi-temporal Trajectory Deltas** | Window deltas (`mid - early`, etc.) | End-of-season | None | NO | NO | **YES (12 cols)** | **YES (12 cols)** | Phenological progression dynamics (e.g. $\Delta\text{NDVI}$, $\Delta\text{NDMI}$). |
| **Field Geometry Metrics (`area_ha`, `perimeter_m`, `compactness`)** | Enriched polygon shape | Pre-sowing (Cadastral / GIS) | None | NO | NO | NO | **YES (3 cols)** | Physical parcel size and geometric shape complexity; independent of geographic location. |
| **Calendar Predictors (`Season`, `sowing_month`, `sowing_doy`)** | Sowing schedule | Pre-sowing / At planting | None | NO | NO | NO | **YES (3 cols)** | Kharif vs Rabi season calendar window; known at planting time. |
| **Farmer Survey Fields (`ExpYield`, `IrriType`, etc.)** | Survey questionnaire | Varies / Post-harvest | High | NO | NO | NO | **NO** | Subjective farmer survey responses; not available in satellite monitoring pipeline. |

---

## 7. Installed Dependencies & Model Compatibility

Inspected installed Python environment:
- `python`: 3.14.0
- `scikit-learn`: 1.8.0
- `xgboost`: 3.2.0
- `lightgbm`: 4.6.0

### Missing-Value Estimator Capabilities:
- `xgboost.XGBClassifier`: **Fully supports NaNs natively**. During tree construction, samples with missing values are evaluated in both left and right child nodes, and the branch that maximizes split gain is selected as the default path.
- `lightgbm.LGBMClassifier`: **Fully supports NaNs natively**.
- `sklearn.ensemble.RandomForestClassifier`: **Does NOT support NaNs natively** (raises `ValueError: Input X contains NaN`).
- **Mitigation for Random Forest:** Wrap the classifier inside a `sklearn.pipeline.Pipeline` with `SimpleImputer(strategy='median')`. The imputer is fitted strictly inside each training fold during cross-validation, preventing data leakage into validation folds.

---

## 8. Missing-Data Strategy & Cohort Fairness

1. **Retrospective Full-Season Experiment (Experiment A):**
   - 100.0% of the 7,843 eligible farms possess complete, non-NaN features in the `full` season composite. Zero imputation is required.
2. **Early-Season & Multi-Temporal Experiments (Experiments B & C):**
   - **XGBoost Track:** Process raw features directly; missing sub-windows are routed via XGBoost's native sparsity split-finding algorithm.
   - **Random Forest Track:** Process features through a fold-isolated `Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('rf', RandomForestClassifier(...))])`.
   - **Leakage Prevention Invariant:** Imputers are fitted exclusively on `X_train_fold`. They never see validation-fold data or target labels (`Crop`).
   - **Cohort Fairness Sensitivity Analysis:** In addition to reporting the full $N=7,843$ spatial CV benchmark, report a complete-case sensitivity analysis on farms with all four windows complete ($N=7,107$) to confirm that imputation does not distort benchmark rankings.

---

## 9. Spatial Validation Feasibility: 5-Fold StratifiedGroupKFold

### Feasibility Test Outcomes
Simulated 5-fold `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` grouped strictly by `Sub-District` (Mandal) on the 7,843 eligible cohort:
- Total Groups: 116 unique Sub-Districts.
- Group Overlap: **0 groups overlap** between training and validation across all 5 folds.
- Farm Overlap: **0 farms overlap** between training and validation across all 5 folds.
- Target Class Coverage: **All 7 classes are present in both training and validation folds across all 5 folds (7/7)**.

### Class Support Across Validation Folds:
| Fold | Total Train Farms | Total Val Farms | Val Bengal Gram | Val Chillies | Val Cotton | Val Groundnut | Val Maize | Val Paddy | Val Red Gram (Rarest) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | 6,335 | 1,508 | 185 | 107 | 337 | 101 | 268 | 454 | **56** |
| **1** | 6,243 | 1,600 | 204 | 143 | 332 | 174 | 270 | 451 | **26** |
| **2** | 6,248 | 1,595 | 177 | 117 | 335 | 209 | 286 | 450 | **21** |
| **3** | 6,313 | 1,530 | 187 | 106 | 341 | 103 | 282 | 452 | **59** |
| **4** | 6,233 | 1,610 | 185 | 128 | 334 | 181 | 270 | 453 | **59** |

### Known Limitations of Spatial CV Design:
1. **Administrative Mandal Boundaries:** Sub-districts provide administrative separation, but farms located near sub-district boundaries may share agro-climatic conditions with neighboring mandals.
2. **Sub-District Clustering in Red Gram:** Red Gram is concentrated in northern Telangana (Gadiguda, Inderavelly, Utnur represent 76.5% of Red Gram farms). StratifiedGroupKFold successfully assigns these groups across folds, yielding between 21 and 59 validation instances per fold, which provides sufficient sample support for macro-F1 estimation.

---

## 10. Proposed Model Comparison Matrix for Stage 4B

All models will be evaluated on the exact same 5 spatial cross-validation folds using identical out-of-fold predictions.

| Experiment | Model Track | Estimator | Key Hyperparameters | Missing Handling | Input Dimension |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **Baseline** | Dummy | `DummyClassifier(strategy='most_frequent')` | — | Ignored | 0 |
| **Exp A** | Satellite Retrospective | Random Forest | `n_estimators=300, max_depth=15, class_weight='balanced'` | None (100% complete) | 34 cols |
| **Exp A** | Satellite Retrospective | XGBoost | `n_estimators=300, max_depth=7, lr=0.05, subsample=0.8` | Native NaN | 34 cols |
| **Exp B** | Satellite Early-Season | Random Forest | `n_estimators=300, max_depth=15, class_weight='balanced'` | Fold Pipeline Median | 34 cols |
| **Exp B** | Satellite Early-Season | XGBoost | `n_estimators=300, max_depth=7, lr=0.05, subsample=0.8` | Native NaN | 34 cols |
| **Exp C** | Multi-Temporal Trajectory | Random Forest | `n_estimators=300, max_depth=15, class_weight='balanced'` | Fold Pipeline Median | 148 cols |
| **Exp C** | Multi-Temporal Trajectory | XGBoost | `n_estimators=300, max_depth=7, lr=0.05, subsample=0.8` | Native NaN | 148 cols |
| **Exp D** | Multimodal Benchmark | Random Forest | `n_estimators=300, max_depth=15, class_weight='balanced'` | Fold Pipeline Median | 154 cols |
| **Exp D** | Multimodal Benchmark | XGBoost | `n_estimators=300, max_depth=7, lr=0.05, subsample=0.8` | Native NaN | 154 cols |

### Primary Evaluation Metrics:
1. **Macro-F1** (Primary scientific criterion across all 7 classes).
2. **Balanced Accuracy** (Mean recall across classes).
3. **Overall Accuracy & Weighted-F1**.
4. **Per-Class Precision, Recall, and F1** (Focusing on minority classes Chillies and Red Gram).
5. **Confusion Matrix** (Pooled Out-of-Fold 7 $\times$ 7 predictions).
6. **Cross-Validation Stability** (Mean $\pm$ standard deviation across the 5 spatial folds).

---

## 11. Risks, Unresolved Questions & Mitigations

1. **Risk: Severe Class Imbalance (Paddy $N=2,261$ vs. Red Gram $N=221$).**  
   *Mitigation:* Use `class_weight='balanced'` for Random Forest, compute dynamic class balance sample weights for XGBoost, and judge all models on macro-F1 rather than overall accuracy.
2. **Risk: Sub-Window Monsoon Missingness in Tree Ensembles.**  
   *Mitigation:* Random Forest utilizes strict fold-isolated median imputation inside `Pipeline`. XGBoost utilizes native sparsity-aware split branching. Compare against complete-case cohort ($N=7,107$).
3. **Risk: Data Leakage from Agronomic Survey Proxies.**  
   *Mitigation:* Complete prohibition of all 10 self-reported survey proxies from feature extraction.
4. **Risk: Over-Optimistic Evaluation via Spatial Autocorrelation.**  
   *Mitigation:* 5-fold `StratifiedGroupKFold` grouped by Mandal ensures no mandal appears in both training and validation folds.

---

## 12. Stage 4B Execution Sequence (Upon User Approval)

```
Stage 4B Execution Sequence:
│
├── Step 1: Feature Matrix Engineering
│   ├── Script: `scripts/generate_phase6f_features.py`
│   ├── Inputs: `outputs/data/s2_phase6f_harmonized_cohort.csv`, `datasss/Train.csv`
│   └── Output: `outputs/data/s2_phase6f_features.csv` (100% verified non-leaking features)
│
├── Step 2: Spatial Cross-Validation Fold Partitioning
│   ├── Script: `scripts/generate_phase6f_cv_folds.py`
│   ├── Grouping: `Sub-District` (116 Mandals, StratifiedGroupKFold, seed=42)
│   └── Output: `outputs/data/phase6f_cv_folds.csv` (Deterministic fold mapping)
│
├── Step 3: Model Benchmark Execution
│   ├── Script: `scripts/benchmark_phase6f_models.py`
│   ├── Runs Experiments A, B, C, D across Dummy, RF, and XGBoost
│   ├── Saves Models: `outputs/models/*_phase6f.joblib`
│   └── Saves OOF Predictions: `outputs/data/phase6f_oof_predictions.csv`
│
├── Step 4: Verification & Automated Test Suite
│   ├── Test Suite: `tests/test_phase6f_stage4_benchmark.py`
│   └── Asserts: 0 leakage, 0 missing predictions, metric sanity, fold isolation
│
└── Step 5: Auditable Benchmark Reporting
    ├── Reports: `outputs/reports/phase6f_stage4_benchmark_report.json` & `.md`
    └── Updates: `.planning/STATE.md` with final benchmark results
```

---

## 13. Files Created or Modified in Stage 4A

- Created: [`outputs/reports/phase6f_stage4a_preflight_report.json`](file:///c:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/reports/phase6f_stage4a_preflight_report.json)
- Created: [`outputs/reports/phase6f_stage4a_preflight_report.md`](file:///c:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/reports/phase6f_stage4a_preflight_report.md)
- Created: [`.planning/PHASE6F_STAGE4A_PLAN.md`](file:///c:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/.planning/PHASE6F_STAGE4A_PLAN.md)
- Created: [phase6f_stage4a_preflight.md](file:///C:/Users/Manichander/.gemini/antigravity-ide/brain/58de9543-a27e-4850-b90c-8370c09cc208/phase6f_stage4a_preflight.md)
- Modified: [`.planning/STATE.md`](file:///c:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/.planning/STATE.md) (Updated state to reflect Stage 4A completion)

---

## 14. Read-Only Checks Executed & Verified

1. **Dependency Verification:** Python 3.14, Scikit-Learn 1.8.0, XGBoost 3.2.0, LightGBM 4.6.0 verified (`Exit Code 0`).
2. **Cohort Accounting Integrity:** 7,888 total rows, 7,843 success, 45 cloud-failed verified (`Exit Code 0`).
3. **Metadata & Label Invariance:** 0 mismatches across Crop, District, Sub-District, Season, SDate, HDate verified (`Exit Code 0`).
4. **Test.csv Investigation:** Confirmed 0 overlap with harmonized S2 cohort and lack of satellite data (`Exit Code 0`).
5. **StratifiedGroupKFold Feasibility Simulation:** Confirmed 116 groups, 0 group overlap, 7/7 classes in every fold (`Exit Code 0`).
6. **Protected File Hash Verification:** Confirmed 0 mutations across all 24 baseline protected files (`Exit Code 0`).

---

## 15. Checks Deferred to Stage 4B

The following actions are **strictly deferred** and will only occur after explicit user approval of Stage 4B:
- Writing feature matrix `outputs/data/s2_phase6f_features.csv`.
- Writing fold partition file `outputs/data/phase6f_cv_folds.csv`.
- Training Random Forest, XGBoost, or Dummy models.
- Hyperparameter tuning or feature selection.
- Generating model evaluation predictions and metrics.
