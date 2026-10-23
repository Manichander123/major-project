# Phase 7 — Test.csv Read-Only Preflight Audit Report

**Phase:** Phase 7, Plan 1 (Read-Only Preflight Audit)  
**Date:** October 23, 2026  
**Status:** COMPLETE (Read-Only Preflight Audit)  
**Evaluator:** Senior Remote Sensing & Machine Learning Reproducibility Auditor  
**Scope:** `datasss/Train.csv`, `datasss/Test.csv`, `datasss/SampleSubmission.csv`, `datasss/VariableDefinitions.docx`, Sentinel-2 extraction pipelines, and saved model artifacts.

---

## 1. Dataset Inventories & Schema Audit

### 1.1 File Summary & Record Counts

| Dataset File | Total Rows | Columns | Unique FarmIDs | Duplicates | Missing Values | Geometry Validity |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `datasss/Train.csv` | 7,888 | 20 | 7,888 | 0 | 0 | 7,888 valid WKT (after topological repair in Phase 1) |
| `datasss/Test.csv` | 2,718 | 19 | 2,718 | 0 | 0 | 2,717 strictly valid; 1 self-intersection (FarmID `1321446`, cleanly repairable via `make_valid`) |
| `datasss/SampleSubmission.csv` | 3,016 | 2 (`ID`, `Target`) | 3,016 | 0 | 0 | N/A (tabular submission template) |

*Note on Cohort Size:* In `Train.csv`, 45 records suffered 100% persistent cloud/shadow failure across all Sentinel-2 acquisition dates, yielding $N = 7,843$ eligible farms for spatial CV.

### 1.2 Column Comparison Between Train and Test

| Train.csv Column | Test.csv Column | Data Type | Notes / Role in Project |
| :--- | :--- | :--- | :--- |
| `FarmID` | `FarmID` | Integer / String | Unique parcel identifier; excluded from predictor matrix |
| `category` | **MISSING** | Categorical | Source competition target: Crop Health Status (*Healthy, Stressed, Diseased, Pests*). Not used in crop classification. |
| `Crop` | `Crop` | Categorical | **Primary Project Target**: 7 crop classes. Present in BOTH files. |
| `State` | `State` | Categorical | Constant (`Telangana`); excluded from predictor matrix |
| `District` | `District` | Categorical | Administrative District (14 in Train, 12 in Test); excluded from predictor matrix |
| `Sub-District` | `Sub-District` | Categorical | Administrative Sub-District / Mandal (116 in Train, 35 in Test); grouping key |
| `SDate` | `SDate` | Date String | Sowing date; range: 2023-01-05 to 2024-09-06 |
| `HDate` | `HDate` | Date String | Harvest date; range: 2023-01-09 to 2024-12-06 |
| `CropCoveredArea` | `CropCoveredArea` | Numeric (%) | Survey ground proxy; excluded from satellite models |
| `CHeight` | `CHeight` | Numeric (ft) | Survey ground proxy; excluded from satellite models |
| `CNext` | `CNext` | Categorical | Survey ground proxy; excluded from satellite models |
| `CLast` | `CLast` | Categorical | Survey ground proxy; excluded from satellite models |
| `CTransp` | `CTransp` | Categorical | Survey ground proxy; excluded from satellite models |
| `IrriType` | `IrriType` | Categorical | Survey ground proxy; excluded from satellite models |
| `IrriSource` | `IrriSource` | Categorical | Survey ground proxy; excluded from satellite models |
| `IrriCount` | `IrriCount` | Numeric | Survey ground proxy; excluded from satellite models |
| `WaterCov` | `WaterCov` | Numeric (%) | Survey ground proxy; excluded from satellite models |
| `ExpYield` | `ExpYield` | Numeric | Survey ground proxy; excluded from satellite models |
| `Season` | `Season` | Categorical | Kharif / Rabi; excluded from pure satellite S+T matrix |
| `geometry` | `geometry` | WKT Polygon | MultiPolygon / Polygon in WGS84 (EPSG:4326) |

---

## 2. Investigation of `Test.csv` `Crop` Column Provenance

### 2.1 Why Does `Test.csv` Contain `Crop`?
1. **Competition Origin**: The source dataset originates from the *Telangana Crop Health Challenge* hosted on Zindi (organized with GIZ and the Telangana Agricultural Data Exchange - ADeX).
2. **Hidden Competition Target**: In that competition, the challenge was to predict crop health status (`category`: *Healthy, Stressed, Diseased, Pests*).
3. **Provided Metadata**: Competitors were provided with the observed crop type (`Crop`) as an input feature in both `Train.csv` and `Test.csv` so their models knew what plant was being evaluated for disease.
4. **Research Objective Shift in This Project**: This major project explicitly redefined the research problem as **7-class satellite crop type identification** (`Crop`), treating satellite spectral-temporal signatures as the predictor space.
5. **Ground Truth Status**: `VariableDefinitions.docx` states: `Crop Description: Name of the crop corresponding to this observation. Data Schema: Text`. This confirms that `Crop` in `Test.csv` is not a placeholder or artifact; it is the ground-truth crop label recorded during agricultural field surveys.

### 2.2 Class Label Distributions: Train vs Test

| Crop Species | Train Count ($N=7,888$) | Train % | Test Count ($N=2,718$) | Test % | Distribution Shift |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Paddy** (Rice) | 2,268 | 28.75% | 714 | 26.27% | Mild decrease (-2.48 pp) |
| **Maize** | 1,384 | 17.55% | 699 | 25.72% | Significant increase (+8.17 pp) |
| **Cotton** | 1,692 | 21.45% | 569 | 20.93% | Consistent (-0.52 pp) |
| **Chillies** | 614 | 7.78% | 381 | 14.02% | Substantial increase (+6.24 pp) |
| **Groundnut** | 768 | 9.74% | 266 | 9.79% | Exactly matched (+0.05 pp) |
| **Bengal Gram** | 938 | 11.89% | 48 | 1.77% | Major drop (-10.12 pp) |
| **Red Gram** | 224 | 2.84% | 41 | 1.51% | Minority in both (-1.33 pp) |

*Finding:* All 7 classes exist in `Test.csv`. However, agro-climatic zoning causes natural regional shifts: Maize and Chillies are more prevalent in the test Mandals, whereas Bengal Gram is less frequent.

---

## 3. Spatial Isolation Audit: Zero Group Overlap

A critical discovery of this preflight audit is that the dataset creators partitioned `Train.csv` and `Test.csv` geographically:

* **Unique FarmIDs Overlap:** Exactly **0** (no farm appears in both Train and Test).
* **Train Mandals (Sub-Districts):** 116 unique Mandals.
* **Test Mandals (Sub-Districts):** 35 unique Mandals.
* **Mandal Overlap:** Exactly **0 Mandals in common**.
* **Implication:** `Test.csv` is a **100% geographically disjoint out-of-district test set**. The 2,718 test parcels reside entirely in 35 unseen Mandals that were never present in the training cohort. This makes `Test.csv` a genuine spatial generalization benchmark.

---

## 4. Feature Schema & Model Artifact Audit

### 4.1 Authoritative Model Artifact
- **File Path:** `outputs/models/phase6h/xgb_weighted_s+t.joblib` (SHA-256 verified)
- **Model Architecture:** `xgboost.sklearn.XGBClassifier`
- **Trained Cohort:** All 7,843 eligible training farms from `Train.csv`
- **Sample Weighting:** Inverse-frequency class weighting computed fold-internally and fitted on the 100% cohort
- **Hyperparameters:**
  - `max_depth: 7`
  - `learning_rate: 0.05`
  - `colsample_bytree: 0.75`
  - `subsample: 0.80`
  - `reg_lambda: 5.0`
  - `min_child_weight: 3`
  - `gamma: 0.2`
  - `reg_alpha: 0.2`
  - `n_estimators: 300`
  - `random_state: 42`
- **Expected Feature Count:** Exactly **140 features** (`n_features_in_ = 140`).
- **Target Encoding Mapping:**
  ```python
  EXPECTED_CLASSES = [
      "Bengal Gram", # 0
      "Chillies",    # 1
      "Cotton",      # 2
      "Groundnut",   # 3
      "Maize",       # 4
      "Paddy",       # 5
      "Red Gram",    # 6
  ]
  ```

### 4.2 Approved 140 Feature Schema (`groups['S+T']`)
From `outputs/reports/phase6h/phase6h_feature_schema_manifest.json`:
1. **Spectral (S) Full-Season Group (34 features):**
   - 10 bands $\times$ 2 statistics (mean, std) = 20 features: `B2, B3, B4, B8, B5, B6, B7, B8A, B11, B12` with `_full_mean` and `_full_std`.
   - 14 spectral indices for `full` window: `ndvi, evi, gndvi, savi, msavi, ndmi, ndwi, ndre1, ndre2, psri, bsi, msi, cire, cig` with `_full`.
2. **Temporal (T) Multi-Window Group (106 features):**
   - Early window (June–July): 10 bands $\times$ 2 stats (20) + 14 indices (14) = 34 features (`_early`).
   - Mid window (August–September): 10 bands $\times$ 2 stats (20) + 14 indices (14) = 34 features (`_mid`).
   - Late window (October–November): 10 bands $\times$ 2 stats (20) + 14 indices (14) = 34 features (`_late`).
   - 4 Dynamic Phenological Transition Deltas:
     - `ndvi_growth_delta`: $\text{NDVI}_{\text{mid}} - \text{NDVI}_{\text{early}}$
     - `peak_to_early_ratio`: $\text{NDVI}_{\text{mid}} / (\text{NDVI}_{\text{early}} + \epsilon)$
     - `ndvi_senescence_delta`: $\text{NDVI}_{\text{late}} - \text{NDVI}_{\text{mid}}$
     - `moisture_loss_delta`: $\text{NDMI}_{\text{late}} - \text{NDMI}_{\text{mid}}$
   - Total: $34 + 34 + 34 + 4 = 106$ features.
3. **Total S+T Features:** $34 + 106 = 140$ features.
4. **Strictly Excluded:** All 20 GLCM spatial texture features (`P`), all survey proxy attributes, all spatial coordinates, and FarmID.

---

## 5. Google Earth Engine (GEE) Prerequisites & Environment

1. **Python Package:** `earthengine-api` version `1.7.46` installed.
2. **Local Credentials:** Found at `C:\Users\Manichander\.config\earthengine\credentials` (size 374 bytes).
3. **Project Resolution:** Configured in `config/project_config.yaml`:
   - `earth_engine.project: "scientific-axle-446313-i6"`
4. **Authentication Check:** `EarthEngineManager.initialize()` executed with project resolution; returned **`True`** (Active connection confirmed).
5. **Sentinel-2 Collection:** `COPERNICUS/S2_SR_HARMONIZED` (Level-2A Bottom-of-Atmosphere surface reflectance).
6. **Cloud & Shadow Filtering:** Scene cloud threshold $\le 20\%$; pixel-level cloud/cirrus/shadow masking via Scene Classification Layer (SCL).
7. **Spatial Resolution & Reducer:** 
   - 10 m bands reduced at scale = 10 m; 20 m bands reduced at scale = 20 m.
   - Combined reducer: `ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True)`.

---

## 6. License, Data Governance, and Privacy

1. **Dataset Origin:** Government of Telangana / GIZ / ADeX Agricultural Data Exchange through the Zindi Telangana Crop Health Challenge.
2. **Permitted Use:** Academic and non-commercial research benchmarking.
3. **Data Privacy:** Farm geometries represent agricultural parcel boundaries in Telangana. No farmer PII (names, contact details, land ownership records) exists in the repository.
4. **Data Isolation Constraint:** All raw survey files remain strictly in `datasss/`. Extracted features will be stored in `outputs/data/phase7/`. No internal tokens, credentials, or private keys will be committed or exported.

---

## 7. Preflight Verification Status

| Checklist Item | Status | Verified Finding |
| :--- | :---: | :--- |
| Row counts & missing values | **VERIFIED** | 2,718 rows in Test.csv; 0 missing values; 1 geometry repaired. |
| Ground truth status of `Crop` | **VERIFIED** | Survey ground truth for crop type; verified by `VariableDefinitions.docx`. |
| Target alignment | **VERIFIED** | Target is `Crop` (7 classes). `category` was competition disease target. |
| Spatial independence | **VERIFIED** | 0 FarmID overlap, 0 Mandal overlap (35 unseen test Mandals). |
| 140-feature schema match | **VERIFIED** | Matched bit-for-bit with `phase6h_feature_schema_manifest.json`. |
| Saved model usability | **VERIFIED** | `outputs/models/phase6h/xgb_weighted_s+t.joblib` loaded; 140 features expected. |
| GEE authentication | **VERIFIED** | Session active via `scientific-axle-446313-i6`. |
| Licensing & data protection | **VERIFIED** | Non-commercial research permitted; raw data isolated. |

**Plan 1 Checkpoint Decision:** Ground-truth labels are confirmed as trustworthy observed crop types (`Crop`). Path A (supervised evaluation of frozen model) is scientifically justified and authorized upon approval.
