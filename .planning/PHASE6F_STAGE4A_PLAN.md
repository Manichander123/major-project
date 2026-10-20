# Phase 6F Stage 4A Plan & Stage 4B Execution Blueprint

## Stage 4A Outcome: READY WITH CAVEATS

The read-only modeling preflight for Phase 6F Stage 4 is complete. All prerequisites for model benchmarking on the 7,843 eligible satellite cohort have been verified against the physical filesystem and source code.

---

## 1. Verified Modeling Parameters
- **Authoritative Universe:** `datasss/Train.csv` (7,888 unique farms).
- **Primary Modeling Cohort:** 7,843 farms with valid Sentinel-2 extractions.
- **Excluded Extraction Failures:** 45 farms with persistent cloud obscuration (`cloud_obscured_failure = True`).
- **Primary Classification Target:** `Crop` (7 classes: Paddy, Cotton, Maize, Bengal Gram, Groundnut, Chillies, Red Gram).
- **Validation Design:** 5-fold `StratifiedGroupKFold` grouped strictly by `Sub-District` (116 Mandals). Zero group overlap.
- **Primary Scientific Metric:** `macro-F1` across all 7 classes.

---

## 2. Leakage Protection Protocol
1. **Target Leakage:** `Crop` is strictly a training label. `category` is completely excluded.
2. **Identification Leakage:** `FarmID` is strictly an index; excluded from features.
3. **Spatial Coordinates:** `lat`, `lon`, `centroid_lat`, `centroid_lon`, and WKT geometry strings are excluded from features.
4. **Survey Proxies:** All 10 farmer-reported survey fields (`ExpYield`, `CropCoveredArea`, `CHeight`, `CLast`, `CNext`, `CTransp`, `IrriType`, `IrriSource`, `IrriCount`, `WaterCov`) are strictly excluded.
5. **Preprocessing Isolation:** Imputers and scalers must be fitted exclusively inside each training fold via `sklearn.pipeline.Pipeline`.

---

## 3. Four Core Benchmark Experiments (Stage 4B)
- **Experiment A (Satellite-Only Retrospective Full-Season):**
  - Features: 10 bands (mean + std) + 14 vegetation/moisture indices from `full` season composite = 34 features.
  - Completeness: 100.0% (0 NaNs across all 7,843 farms).
- **Experiment B (Satellite-Only In-Season Early):**
  - Features: 10 bands + 14 indices from `early` phenological window = 34 features.
  - Completeness: 93.10% valid (541 farms missing early window due to monsoon clouds). Evaluated with fold-isolated imputation + complete-case sensitivity ($N=7,302$).
- **Experiment C (Multi-Temporal Phenological Trajectory):**
  - Features: 4 windows $\times$ 34 features (80 bands + 56 indices) + 4 phenological trajectory deltas (`ndvi_growth_delta`, `peak_to_early_ratio`, `ndvi_senescence_delta`, `moisture_loss_delta`) = **140 features**.
  - Completeness: 90.62% complete across all 4 windows ($N=7,107$). Evaluated with fold-isolated imputation for RF / native NaN for XGBoost + complete-case sensitivity ($N=7,107$).
- **Experiment D (Multimodal Comparison):**
  - Features: 140 multi-temporal satellite features + 3 non-leaking pre-sowing calendar features (`Season`, `sowing_month`, `sowing_doy`) + 3 non-leaking geometry shape metrics (`area_ha`, `perimeter_m`, `compactness`) = **146 features**.
  - Strictly excludes post-harvest calendar metrics (`duration_days`, `harvest_month`), coordinates, and survey proxies.

---

## 4. Models to Benchmark
1. `DummyClassifier(strategy='most_frequent')` (Majority class baseline, prior macro-F1 ~0.036).
2. `RandomForestClassifier` (`n_estimators=300`, `max_depth=15`, `class_weight='balanced'`, fold-pipeline median imputation).
3. `XGBClassifier` (`n_estimators=300`, `max_depth=7`, `learning_rate=0.05`, `subsample=0.8`, native NaN routing).

---

## 5. Stage 4B Execution Steps (Pending User Approval)
- **Step 1:** Implement `scripts/generate_phase6f_features.py` to create `outputs/data/s2_phase6f_features.csv`.
- **Step 2:** Implement `scripts/generate_phase6f_cv_folds.py` to create `outputs/data/phase6f_cv_folds.csv`.
- **Step 3:** Implement `scripts/benchmark_phase6f_models.py` to execute Experiments A, B, C, D across the 5 spatial folds.
- **Step 4:** Implement `tests/test_phase6f_stage4_benchmark.py` to assert fold isolation, score integrity, and reproducibility.
- **Step 5:** Generate `outputs/reports/phase6f_stage4_benchmark_report.json` and `.md`.

**STOP: Stage 4A complete. Awaiting user approval before initiating Stage 4B.**
