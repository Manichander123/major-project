# AI-Based Multi-Temporal Remote Sensing for Crop Monitoring and Irrigation Decision Support Using Sentinel-2 Imagery

An enterprise-grade, reproducible remote sensing and machine learning pipeline for agricultural monitoring across 10,606 smallholder farm parcels in Telangana, India.

---

## 📌 Table of Contents
1. [Executive Summary & Current Milestone](#-executive-summary--current-milestone)
2. [Dataset Summary & Evaluation Cohort Status](#-dataset-summary--evaluation-cohort-status)
3. [System Architecture](#-system-architecture)
4. [Project Structure](#-project-structure)
5. [Implementation Roadmap & Status](#-implementation-roadmap--status)
6. [Detailed Technical Walkthrough of Completed Work](#-detailed-technical-walkthrough-of-completed-work)
   - [Phase 1: Dataset Analysis, Validation & Exploratory Data Analysis](#phase-1-dataset-analysis-validation--exploratory-data-analysis)
   - [Phase 2: Farm Geometry Parsing, Topological Repair & Feature Engineering](#phase-2-farm-geometry-parsing-topological-repair--feature-engineering)
   - [Phase 3: Farm Spatial Visualisation & Cartography](#phase-3-farm-spatial-visualisation--cartography)
   - [Phase 4: Sentinel-2 Multi-Temporal Remote Sensing Extraction Engine](#phase-4-sentinel-2-multi-temporal-remote-sensing-extraction-engine)
   - [Phase 5: Spectral, Moisture & Vegetation Indices Engineering](#phase-5-spectral-moisture--vegetation-indices-engineering)
   - [Phase 6: Crop Classification Modeling & Robustness Evaluation (Phases 6A–6I)](#phase-6-crop-classification-modeling--robustness-evaluation-phases-6a6i)
7. [Status of Unimplemented Modules & Unprocessed Datasets](#-status-of-unimplemented-modules--unprocessed-datasets)
8. [Step-by-Step Execution & Verification Guide](#-step-by-step-execution--verification-guide)
9. [Summary of Generated Artifacts & Reports](#-summary-of-generated-artifacts--reports)
10. [References & Standards](#-references--standards)

---

## ⚡ Executive Summary & Current Milestone

The primary research objective of this project is automated, leakage-safe classification of 7 crop types (`Bengal Gram`, `Chillies`, `Cotton`, `Groundnut`, `Maize`, `Paddy`, `Red Gram`) in smallholder parcels using multi-temporal **Sentinel-2 MSI Level-2A surface reflectance** imagery.

- **Completed Primary Pipeline:** **Phase 6I Robustness Validation and Model Selection**.
- **Selected Primary Production Model:** **`Weighted S+T`** (140 Spectral + Temporal features, inverse-frequency weighted regularized XGBoost).
- **Evaluation Cohort:** **7,843 eligible training farms** across 14 districts and 116 Sub-Districts (Mandals) in Telangana.
- **Validation Protocol:** 5-Fold `StratifiedGroupKFold` grouped strictly by Sub-District (Mandal) with zero spatial group overlap (zero leakage between training and validation Mandals).
- **Authoritative Performance:**
  - **Pooled Out-of-Fold Macro-F1:** **`0.7226`**
  - **Balanced Accuracy:** **`0.7180`**
  - **Overall Accuracy:** **`0.7881`**
  - **Fold Macro-F1 (Mean ± Std):** **`0.7117 ± 0.0181`** (Fold 0: 0.7269, Fold 1: 0.7381, Fold 2: 0.6919, Fold 3: 0.6945, Fold 4: 0.7073).
  > [!IMPORTANT]
  > The **0.7226 Macro-F1** score is an **out-of-fold spatial cross-validation result** evaluated across unseen geographic Mandals. It is **NOT** a test-set evaluation score.
- **Test Dataset Status:** The test dataset (`datasss/Test.csv`, 2,718 parcels) has **NOT** yet been processed for Sentinel-2 satellite features and has **NOT** been evaluated.
- **Downstream Modules Status:** Crop-health classification (`category`), irrigation recommendations (FAO-56 ET, water stress), weather integration, and dashboard/agent interfaces are **NOT YET IMPLEMENTED** in code.

---

## 📊 Dataset Summary & Evaluation Cohort Status

### Raw Ingested Data
- **Raw Training Set (`datasss/Train.csv`):** 7,888 records.
- **Raw Test Set (`datasss/Test.csv`):** 2,718 records.
- **Geographic Coverage:** State of Telangana, India (14 administrative districts: Adilabad, Nalgonda, Mahabubabad, Nirmal, Medak, Mahabubnagar, etc.; 116 Sub-Districts / Mandals).
- **Temporal Span:** Sowing from `2023-01-05` to `2024-09-28`; Harvest up to `2024-12-06`.

### Verified Operational Training Cohort (N = 7,843)
During Phase 6F data audit and recovery, 7,843 out of 7,888 farms were verified as eligible and fully populated with multi-temporal Sentinel-2 features across all 4 phenological stages (45 candidate records lacked sufficient cloud-free observations or failed polygon intersection requirements).

#### Target Variable (`Crop` — 7 Crop Classes in N = 7,843 Cohort):
| Crop Class | Farm Count | Class Percentage | Agronomic Characteristics |
|---|:---:|:---:|---|
| **Paddy** | 2,260 | 28.81% | Wetland rice; high moisture absorption; long duration |
| **Cotton** | 1,679 | 21.41% | Kharif commercial fiber crop; prolonged vegetative and boll development (~210 days) |
| **Maize** | 1,376 | 17.54% | High biomass cereal; rapid vegetative peak (~120 days) |
| **Bengal Gram** | 938 | 11.96% | Rabi chickpea pulse; compact canopy; short duration (~90 days) |
| **Groundnut** | 768 | 9.79% | Oilseed legume; prostrate canopy; distinctive red-edge profile |
| **Chillies** | 601 | 7.66% | High-value spice; extended flowering and picking cycles (~180 days) |
| **Red Gram** | 221 | 2.82% | Minor pulse (pigeon pea); severe class minority; frequently intercropped |

#### Secondary Target Variable (`category` — Crop Health Status):
- **Distribution in Raw Train:** `Healthy`: 6,506 (82.5%), `Pests`: 485 (6.1%), `Diseased`: 474 (6.0%), `Stressed`: 423 (5.4%).
- **Current Status:** **NOT YET IMPLEMENTED**. Preserved exclusively for secondary diagnostic modeling in Phase 7; strictly excluded from primary crop-type classification to prevent target contamination.

---

## 🏗️ System Architecture

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                   RAW DATA INGESTION                                   │
│  Train.csv (7,888 farms) & Test.csv (2,718 farms) [WKT Polygon, Agronomic Metadata]   │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        PHASE 1 & 2: GEOMETRIC & DATA PIPELINE                         │
│  - Data Cleaning & 0-Leakage Verification (0 missing values, 0 duplicates)             │
│  - Topological Repair: shapely.make_valid (100% validity achieved)                     │
│  - Projected CRS: EPSG:4326 (WGS84) ──► EPSG:32644 (UTM 44N Metric)                   │
│  - Metric Features: Area (ha/m²), Perimeter, Polsby-Popper Compactness, Centroids      │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                 PHASE 3, 4 & 5: SATELLITE REMOTE SENSING ENGINE                        │
│  - Interactive Folium Cartography (telangana_farms_interactive_map.html)               │
│  - Sentinel-2 Level-2A Multi-Temporal Extraction (COPERNICUS/S2_SR_HARMONIZED)         │
│  - SCL Scene Classification & QA60 Cloud Masking; MGRS Tile Daily Mosaicking           │
│  - Phenological Window Partitioning: Early (0-25%), Mid (25-75%), Late (75-100%), Full │
│  - 14 Spectral Indices: NDVI, EVI, GNDVI, SAVI, MSAVI, NDMI, NDWI, NDRE1, NDRE2, PSRI │
│  - 140 S+T Features: 80 Band Stats + 56 Index Stats + 4 Trajectory Deltas              │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                     PHASE 6 (6A–6I): MACHINE LEARNING BENCHMARKING                     │
│  - Phase 6A: Tabular/Calendar Baseline (Archived: post-harvest leakage, Macro-F1 0.90) │
│  - Phase 6F: Statewide 5-Fold Spatial CV Benchmark (7,843 farms, Macro-F1 0.7038)      │
│  - Phase 6G: Hyperparameter Search & Sample Weighting Pilot (Outer Fold 0)             │
│  - Phase 6H: GLCM Spatial Texture & S/T/P Ablation (Texture non-significant, p=0.8600) │
│  - Phase 6I: Final Model Selection: Parsimonious Weighted S+T XGBoost (Macro-F1 0.7226)│
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                    DOWNSTREAM WORKSTREAMS (NOT YET IMPLEMENTED)                        │
│  - Test Cohort Processing: S2 Extraction & Inference on datasss/Test.csv (2,718 farms) │
│  - Phase 7: Crop Health & Stress Classification (secondary target: category)           │
│  - Phase 8-11: Weather Integration, FAO-56 ET0, & Prescriptive Irrigation Advisory    │
│  - Phase 12-13: Interactive Streamlit Web Dashboard & LangChain Agronomic Agents       │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
major_proj_final/
├── config/
│   └── project_config.yaml         ← Central pipeline configuration (CRS, bands, ML params)
├── datasss/                        ← Immutable competition data
│   ├── Train.csv                   ← 7,888 train records with target category & Crop
│   ├── Test.csv                    ← 2,718 test records (UNPROCESSED FOR SATELLITE FEATURES)
│   ├── SampleSubmission.csv        ← Submission format template
│   └── Starter_Notebook.ipynb      ← Official competition starter
├── notebooks/                      ← Production Jupyter analysis notebooks
│   ├── 01_dataset_analysis.ipynb   ← Phase 1 comprehensive data quality & distributions
│   └── 02_geometry_and_spatial_analysis.ipynb ← Phase 2 & 3 geometry repair & spatial mapping
├── outputs/
│   ├── data/                       ← Generated datasets & feature matrices
│   │   ├── train_enriched_geometry.csv  ← 7,888 rows × 32 columns with metric geometry
│   │   ├── test_enriched_geometry.csv   ← 2,718 rows × 31 columns with metric geometry
│   │   ├── phase6f_cv_folds.csv         ← Authoritative 5-Fold StratifiedGroupKFold splits
│   │   ├── phase6f_full_multimodal_features.csv ← 7,843 farms × 140 features
│   │   ├── phase6g/                     ← Phase 6G hyperparameter search artifacts
│   │   └── phase6h/                     ← Phase 6H ablation features and OOF predictions
│   ├── figures/                    ← Generated diagnostic plots & interactive maps
│   │   ├── telangana_farms_interactive_map.html  ← 3.59 MB interactive Folium web map
│   │   ├── telangana_spatial_overview.png        ← 4-panel high-res overview map
│   │   └── geometry_diagnostics.png
│   ├── models/                     ← Model weights and checkpoints
│   │   ├── archive/                ← Archived models (baseline_rf.joblib)
│   │   ├── phase6f/                ← Phase 6F 5-fold trained models
│   │   └── phase6g/                ← Phase 6G tuned model weights
│   └── reports/                    ← Formal audit reports & benchmark documentation
│       ├── project_methodology_complete.md       ← Comprehensive project methodology review
│       ├── project_evidence_and_discrepancies.md ← Resolved audit discrepancies log
│       ├── baseline_rf_metrics.json              ← Archived Phase 6A tabular baseline report
│       ├── phase6f_final_benchmark_report.md     ← Phase 6F 5-fold benchmark report
│       ├── phase6h/                              ← Phase 6H texture ablation reports
│       └── phase6i/
│           ├── phase6i_model_selection_report.md ← Authoritative model selection decision
│           └── phase6i_robustness_audit.json     ← Formal reproducibility verification log
├── scripts/                        ← Executable command-line pipelines
│   ├── analyze_dataset.py          ← Phase 1 analysis script
│   ├── validate_geometries.py      ← Phase 2 geometry repair & metric calculation
│   ├── visualize_spatial.py        ← Phase 3 interactive & static spatial visualizer
│   ├── test_sentinel2_small.py     ← Phase 4/5 integration test on 15 farms
│   └── audit_phase6i_robustness.py ← Phase 6I verification & statistical test runner
├── src/                            ← Reusable Python package modules
│   ├── __init__.py
│   ├── data/
│   │   ├── load_data.py            ← Safe loaders for train/test with type casting
│   │   └── validate_data.py        ← Schema and integrity validator
│   ├── features/
│   │   ├── __init__.py
│   │   ├── indices.py              ← 14 vegetation, moisture & red-edge spectral indices
│   │   └── spatial_texture.py      ← GLCM spatial texture extraction module
│   ├── geospatial/
│   │   ├── geometry_utils.py       ← WKT parsing, repair, metric projection, compactness
│   │   └── spatial_viz.py          ← Folium interactive map & matplotlib cartography
│   ├── satellite/
│   │   ├── s2_extractor.py         ← GEE Sentinel-2 multi-temporal extractor
│   │   └── s2_mosaic.py            ← Daily MGRS tile mosaicking engine
│   └── models/
│       ├── __init__.py             ← Package initialization
│       ├── baseline_rf.py          ← Deprecation shim pointing to archive/baseline_rf.py
│       ├── train_pilot_models.py   ← Deprecation shim pointing to archive/train_pilot_models.py
│       └── archive/                ← Archived historical baselines (with leakage warnings)
│           ├── __init__.py
│           ├── baseline_rf.py      ← Phase 6A tabular baseline (NO satellite imagery)
│           └── train_pilot_models.py ← Phase 6C 350-farm pilot benchmark
├── tests/                          ← Comprehensive unit and regression test suite (16 files)
├── requirements.txt                ← Python package dependencies
└── README.md                       ← Detailed project documentation
```

---

## 🚀 Implementation Roadmap & Status

| Phase | Milestone Description | Status | Verification & Deliverables |
|:---:|---|:---:|---|
| **1** | **Dataset Analysis & Cleaning** | ✅ **Done** | `scripts/analyze_dataset.py`, 7 figures, 2 reports, `01_dataset_analysis.ipynb` |
| **2** | **Farm Geometry Parsing & Validation** | ✅ **Done** | `scripts/validate_geometries.py`, 100% validity, enriched CSVs, `geometry_diagnostics.png` |
| **3** | **Farm Spatial Visualisation** | ✅ **Done** | `scripts/visualize_spatial.py`, `telangana_farms_interactive_map.html`, `02_geometry...ipynb` |
| **4** | **Sentinel-2 Multi-Temporal Extraction** | ✅ **Done** | `src/satellite/s2_extractor.py`, GEE L2A cloud masking, phenological window compositing |
| **5** | **Spectral & Vegetation-Index Features** | ✅ **Done** | `src/features/indices.py`, 14 spectral indices + multi-temporal growth/senescence dynamics |
| **6A** | **Tabular/Calendar Baseline (Archived)** | ⚠️ *Superseded* | `src/models/archive/baseline_rf.py` (0.9030 Macro-F1, contains post-harvest leakage) |
| **6C–E**| **Sentinel-2 Multi-Temporal Campaign** | ✅ **Done** | 15 chunk extractions, daily MGRS tile mosaic, SCL cloud masking |
| **6F** | **Full-Scale 5-Fold Spatial CV Benchmark** | ✅ **Done** | 7,843 eligible farms, 5-Fold StratifiedGroupKFold on Mandal, Macro-F1: 0.7038 |
| **6G** | **Hyperparameter & Sample Weighting Pilot** | ✅ **Done** | Trial 4 regularization + balanced inverse-frequency weighting on Outer Fold 0 |
| **6H** | **Spatial Texture & S/T/P Feature Ablation** | ✅ **Done** | 20 GLCM features; proven statistically non-significant globally ($p=0.8600$) |
| **6I** | **Robustness Validation & Final Model Selection** | ✅ **Done** | **Primary Winner: Weighted S+T** (140 features, Pooled Macro-F1 = **0.7226**) |
| **Test** | **Test Set Feature Extraction & Inference** | ⏳ *Not Yet Implemented* | `datasss/Test.csv` (2,718 parcels) not yet processed or evaluated |
| **7** | **Crop Health Classification (`category`)** | ⏳ *Not Yet Implemented* | Secondary target modeling (`Healthy`, `Diseased`, `Pests`, `Stressed`) |
| **8** | **Sentinel-1 SAR C-band Integration** | ⏳ *Planned* | Dual-pol backscatter (VV, VH) & Radar Vegetation Index (RVI) |
| **9** | **Weather & Agrometeorology Integration** | ⏳ *Not Yet Implemented* | CHIRPS rainfall & ERA5-Land temperature/solar radiation ingestion |
| **10** | **Moisture & Evapotranspiration (FAO-56)** | ⏳ *Not Yet Implemented* | CWSI, NDMI dynamics, and reference evapotranspiration ($ET_c = K_c \times ET_0$) |
| **11** | **Irrigation Decision-Support Engine** | ⏳ *Not Yet Implemented* | Prescriptive urgency score ($0$–$100$), volume deficit ($m^3/\text{ha}$), schedule generator |
| **12** | **Streamlit Interactive Dashboard** | ⏳ *Not Yet Implemented* | Web dashboard for farm inspection, map explorer, and advisory lookup |
| **13** | **LangChain Explanation & Advisory Agents**| ⏳ *Not Yet Implemented* | LLM agronomic agents for farmer-friendly advisory generation |

---

## 🔍 Detailed Technical Walkthrough of Completed Work

### Phase 1: Dataset Analysis, Validation & Exploratory Data Analysis
- Ingested 7,888 train and 2,718 test farm records. Verified **0 missing values**, **0 duplicate IDs**, and **0 train/test farm ID overlap**.
- Temporal timeline verification: Sowing strictly precedes harvest (`HDate > SDate`) for all farms. Mean growing duration: 193 days (Train), 191 days (Test).
- Identified severe class imbalance across the primary target (`Crop`), necessitating macro-F1 and balanced accuracy optimization.

### Phase 2: Farm Geometry Parsing, Topological Repair & Feature Engineering
- Ingested WKT boundary strings using `shapely`. Detected 2 invalid geometries with self-intersection anomalies (FarmID `1330965` in Train, FarmID `1326463` in Test).
- Repaired all anomalies using `shapely.make_valid()`, achieving **100% geometric validity** across all 10,606 parcels.
- Projected coordinates to **EPSG:32644 (UTM Zone 44N)** for metric area and distance calculations.
- Engineered 12 geometric features: `area_ha`, `area_m2`, `perimeter_m`, Polsby-Popper `compactness`, `centroid_lon`, `centroid_lat`, bounding box bounds, and `vertex_count`.

### Phase 3: Farm Spatial Visualisation & Cartography
- Built standalone interactive web map [`outputs/figures/telangana_farms_interactive_map.html`](file:///C:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/figures/telangana_farms_interactive_map.html) (3.59 MB) featuring Esri World Imagery basemaps, farm polygon layers, agronomic popup dossiers, and density heatmaps.
- Built 4-panel publication-grade static spatial overview [`outputs/figures/telangana_spatial_overview.png`](file:///C:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/figures/telangana_spatial_overview.png).

### Phase 4: Sentinel-2 Multi-Temporal Remote Sensing Extraction Engine
- Built Google Earth Engine extraction engine targeting **Sentinel-2 Level-2A Harmonized Surface Reflectance** (`COPERNICUS/S2_SR_HARMONIZED`).
- Implemented pixel-level cloud masking utilizing Scene Classification Layer (`SCL` values 3, 8, 9, 10, 11 masked) and `QA60` bitmask filtering.
- Implemented daily MGRS tile mosaicking (`src/satellite/s2_mosaic.py`) to collapse orbit overlaps and prevent duplicate sampling.
- Partitioned growth timelines into 4 phenological stages:
  1. Early Season (0% to 25% of growth cycle)
  2. Mid Season (25% to 75% of growth cycle, peak vegetative canopy)
  3. Late Season (75% to 100% of growth cycle, ripening & senescence)
  4. Full Season (complete lifecycle baseline)
- Reduced 10 spectral bands at native resolutions (10m for B2, B3, B4, B8; 20m for B5, B6, B7, B8A, B11, B12) via mean and standard deviation zonal reducers.

### Phase 5: Spectral, Moisture & Vegetation Indices Engineering
- Engineered 14 agricultural spectral indices in [`src/features/indices.py`](file:///C:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/src/features/indices.py):
  - **Photosynthetic greenness:** NDVI, EVI, GNDVI, SAVI, MSAVI.
  - **Canopy moisture & water stress:** NDMI, NDWI, MSI.
  - **Red-edge foliar chlorophyll & nitrogen:** NDRE1, NDRE2, CIRE, CIG.
  - **Senescence & soil background:** PSRI, BSI.
- Computed multi-temporal dynamic trajectory deltas: $\text{NDVI}_{\text{growth\_delta}}$, $\text{NDVI}_{\text{senescence\_delta}}$, $\text{NDMI}_{\text{loss\_delta}}$, and $\text{NDVI}_{\text{peak\_ratio}}$.

---

### Phase 6: Crop Classification Modeling & Robustness Evaluation (Phases 6A–6I)

#### Phase 6A: Tabular/Calendar Baseline (Archived Historical Artifact)
- **Script:** [`src/models/archive/baseline_rf.py`](file:///C:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/src/models/archive/baseline_rf.py)
- **Features Used:** 18 non-satellite tabular features (`duration_days`, `harvest_month`, `harvest_doy`, `sowing_month`, `sowing_doy`, `District`, `Sub-District`, `centroid_lon`, `centroid_lat`, geometric metrics).
- **Reported Metric:** Macro-F1 = `0.9030`, Accuracy = `0.9030` under an 80/20 random stratified split.
- **Deconstructive Audit Finding:** This metric is **artificially inflated and ungeneralizable**:
  1. *Calendar Data Leakage:* Harvest date and total growing duration directly encode crop identity (e.g., Cotton ~210 days vs Bengal Gram ~90 days), which is unavailable during mid-season prediction.
  2. *Spatial Autocorrelation:* Geographic coordinates and administrative names combined with random splitting allow trees to memorize spatial clusters rather than learning spectral signatures.
- **Resolution:** This experiment is strictly archived in `src/models/archive/baseline_rf.py` with explicit deprecation and leakage warnings.

#### Phase 6C–6E: Multi-Temporal Sentinel-2 Extraction Scaling
- Scaled cloud-free extraction across 15 parallel chunk campaigns covering candidate farms in Telangana, incorporating SCL cloud-shadow filtering and daily MGRS tile compositing.

#### Phase 6F: Full-Scale Dataset Audit & 5-Fold Spatial CV Benchmark
- Verified and harmonized the **7,843 eligible farm cohort**.
- Established the official **5-Fold StratifiedGroupKFold validation protocol grouped strictly by Sub-District (Mandal)** with zero group leakage (116 Mandals strictly partitioned).
- Benchmarked unweighted XGBoost, LightGBM, and Random Forest on 140 Sentinel-2 features. Baseline unweighted XGBoost achieved pooled Macro-F1 = **`0.7038`** (Fold mean: `0.6975 ± 0.0210`).

#### Phase 6G: Hyperparameter Tuning & Sample Weighting Pilot (Outer Fold 0)
- Conducted bounded nested grouped CV on Outer Fold 0 ($N=1,508$) across 10 hyperparameter trials and 7 class-weighting strategies.
- Identified optimal regularized architecture (Trial 4: `max_depth=7, colsample=0.75, subsample=0.8, reg_lambda=5.0, gamma=0.2, reg_alpha=0.2`).
- Discovered that balanced inverse-frequency sample weighting dramatically improved minority class recovery: Red Gram recall increased from 25.00% to 44.64%, lifting Fold 0 Macro-F1 to `0.7339`.

#### Phase 6H: Spatial Texture (GLCM) Integration & S/T/P Feature Ablation
- Implemented Gray-Level Co-occurrence Matrix (GLCM) spatial texture extraction on 10m B8 imagery (contrast, dissimilarity, homogeneity, energy, correlation) across 4 temporal windows (20 texture features).
- Conducted 7 feature-ablation experiments: S (Spectral), T (Temporal), P (Spatial Texture), S+T, S+P, T+P, S+T+P across all 5 spatial folds.
- **Key Empirical Finding:** Spatial texture added marginal accuracy (+0.0024) in large consolidated fields, but failed to yield a statistically significant improvement in statewide Macro-F1:
  $$\text{Weighted S+T (140 feats): } 0.7226 \quad \text{vs} \quad \text{Weighted S+T+P (160 feats): } 0.7228 \quad (p = 0.8600)$$
- **Boundary Contamination Diagnosis:** In Telangana's fragmented smallholder landscape, 35.04% of farms are smaller than 1,800 m² (less than twice the $30\text{ m} \times 30\text{ m}$ GLCM kernel footprint). The moving window inevitably bleeds edge pixels from adjoining bare soil, trees, and neighboring crops, injecting noise that degrades delicate minority crops like Red Gram (F1: 0.4420 → 0.4317) and Chillies (F1: 0.5343 → 0.5217).

#### Phase 6I: Robustness Validation, Parsimony Analysis & Final Model Selection
- Formally benchmarked all candidate configurations under 5-fold spatial cross-validation and executed group-aware Mandal cluster bootstrap hypothesis testing (B=1,000 resamples).
- **Four-Way Benchmark Matrix (N = 7,843 Eligible Farms):**

| Model Configuration | Feature Count | Sample Weighting | Pooled Macro-F1 | Balanced Accuracy | Overall Accuracy | Weighted-F1 | Fold Macro-F1 (Mean ± Std) | Status |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `unweighted_S+T` | 140 | Unweighted | 0.7069 | 0.6914 | 0.7873 | 0.7796 | 0.6997 ± 0.0228 | Baseline |
| **`weighted_S+T`** | **140** | **Inverse-Freq** | **`0.7226`** | **`0.7180`** | **`0.7881`** | **`0.7857`** | **`0.7117 ± 0.0181`** | 🏆 **PRIMARY WINNER** |
| `unweighted_S+T+P` | 160 | Unweighted | 0.7038 | 0.6895 | 0.7900 | 0.7818 | 0.6965 ± 0.0173 | Ablation Ref |
| `weighted_S+T+P` | 160 | Inverse-Freq | 0.7228 | 0.7167 | 0.7905 | 0.7879 | 0.7131 ± 0.0133 | Secondary Ref |

- **Paired Hypothesis Test:**
  - Mean Macro-F1 Delta (S+T+P - S+T): **`+0.00094`** ($SE = 0.00397$).
  - **95% Bootstrap Confidence Interval:** **`[-0.00663, +0.00917]`** ($p = 0.8600$).
  - **Paired $t$-test:** $t = 0.3037$, $p = 0.7765$.
- **Final Decision via Occam's Razor:**
  1. *Statistical Parity:* The +0.0002 difference is within random evaluation noise.
  2. *Minority Crop Protection:* Weighted S+T achieves higher F1 scores on vulnerable minority crops: Red Gram (**0.4420** vs 0.4317) and Chillies (**0.5343** vs 0.5217).
  3. *Operational Parsimony:* Eliminating 20 GLCM texture rasters halves cloud computation latency, reduces payload bandwidth, and avoids parcel-boundary edge corruption.
  4. **`Weighted S+T` (140 features) is selected as the primary operational production model.**

- **Per-Class Precision, Recall, and F1 Breakdown (Primary Weighted S+T Model):**

| Crop Class | Support (N) | Precision | Recall | F1-Score | Confusion Summary |
|:---|:---:|:---:|:---:|:---:|---|
| **Paddy** | 2,260 | 0.8664 | 0.8752 | **0.8708** | Dominant wetland signature; well separated from dryland crops |
| **Groundnut** | 768 | 0.8597 | 0.8698 | **0.8647** | Distinctive spectral canopy; low cross-confusion |
| **Bengal Gram** | 938 | 0.7916 | 0.8667 | **0.8275** | Robust Rabi phenological separation |
| **Cotton** | 1,679 | 0.7939 | 0.8261 | **0.8097** | Strong long-season vegetative trajectory |
| **Maize** | 1,376 | 0.7492 | 0.6730 | **0.7090** | Occasional confusion with Bengal Gram (134) and Paddy (126) |
| **Chillies** | 601 | 0.5249 | 0.5441 | **0.5343** | Moderate confusion with Paddy (91) and Cotton (82) |
| **Red Gram** | 221 | 0.5467 | 0.3710 | **0.4420** | Significant confusion with Cotton (76) due to similar phenological timelines |
| **Macro Average** | **7,843** | **0.7332** | **0.7180** | **`0.7226`** | Balanced, minority-sensitive evaluation |

---

## ⚠️ Status of Unimplemented Modules & Unprocessed Datasets

To ensure complete scientific and operational transparency, the following components are formally documented as **NOT YET IMPLEMENTED / UNPROCESSED**:

### 1. Test Dataset (`datasss/Test.csv`)
- **Status:** **UNPROCESSED / NOT EVALUATED**.
- The 2,718 test parcels have not been submitted to Google Earth Engine for Sentinel-2 feature extraction, and no inference or submission has been generated.

### 2. Secondary Target: Crop Health Classification (`category`)
- **Status:** **NOT YET IMPLEMENTED**.
- While exploratory statistics exist, no machine learning models have been trained or validated to predict the 4 crop health classes (`Healthy`, `Diseased`, `Pests`, `Stressed`).

### 3. Weather & Agro-Meteorological Integration
- **Status:** **NOT YET IMPLEMENTED**.
- Integration with ERA5-Land or CHIRPS precipitation and temperature datasets has not been implemented.

### 4. Crop Water Stress & Evapotranspiration Estimation
- **Status:** **NOT YET IMPLEMENTED**.
- FAO-56 Penman-Monteith crop evapotranspiration ($ET_c = K_c \times ET_0$) and satellite CWSI algorithms have not yet been coded into executable pipelines.

### 5. Irrigation Decision-Support Engine
- **Status:** **NOT YET IMPLEMENTED**.
- Urgency scoring, deficit volume calculations ($m^3/\text{ha}$), and watering schedule generators have not been developed.

### 6. Interactive Web Dashboard & LLM Advisory Agents
- **Status:** **NOT YET IMPLEMENTED**.
- The Streamlit web interface and LangChain conversational decision support agents remain planned for future milestones.

---

## 💻 Step-by-Step Execution & Verification Guide

### 1. Environment Setup
```powershell
# Navigate to the project root
cd "c:\Users\Manichander\OneDrive\Desktop\cbit\major project\major_proj_final"

# Install core dependencies
pip install -r requirements.txt
pip install folium xgboost lightgbm shap pyproj
```

### 2. Execute Phase 1: Data Analysis & Quality Audit
```powershell
python scripts/analyze_dataset.py
```
*Outputs: Generates 7 diagnostic figures in `outputs/figures/` and quality reports in `outputs/reports/`.*

### 3. Execute Phase 2: Geometry Parsing, Repair & Metric Calculation
```powershell
python scripts/validate_geometries.py
```
*Outputs: Validates geometries, reprojects to UTM Zone 44N, exports `train_enriched_geometry.csv` and `test_enriched_geometry.csv`.*

### 4. Execute Phase 3: Spatial Visualisation & Interactive Cartography
```powershell
python scripts/visualize_spatial.py
```
*Outputs: Generates `telangana_farms_interactive_map.html` (3.59 MB interactive map) and `telangana_spatial_overview.png`.*

### 5. Run Targeted Unit & Regression Test Suite
Run the 83 targeted offline unit and integrity tests (zero Earth Engine network dependency):
```powershell
python -m unittest tests/test_baseline.py tests/test_phase6c.py tests/test_phase6d_audit.py tests/test_phase6f_stage1_audit.py tests/test_phase6f_stage2_recovery.py tests/test_phase6f_stage3_harmonization.py tests/test_phase6f_stage4_benchmark.py tests/test_phase6f_stage4c_audit.py tests/test_phase6g_exp2_imbalance.py tests/test_phase6g_pilot.py tests/test_phase6h_stage3_benchmark.py tests/test_phase6i_robustness.py
```

### 6. Execute Phase 6I Verification & Statistical Audit
```powershell
python scripts/audit_phase6i_robustness.py
```
*Outputs: Evaluates out-of-fold predictions, runs cluster bootstrap hypothesis tests, verifies protected file hashes, and validates the selection of Weighted S+T.*

---

## 📑 Summary of Generated Artifacts & Reports

| Artifact Name | File Path | Scope & Status |
|---|---|---|
| **Phase 6I Model Selection Report** | [`outputs/reports/phase6i/phase6i_model_selection_report.md`](file:///c:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/reports/phase6i/phase6i_model_selection_report.md) | **Authoritative production model selection**: Weighted S+T (Macro-F1 = 0.7226). |
| **Phase 6I Robustness Audit** | [`outputs/reports/phase6i/phase6i_robustness_audit.json`](file:///c:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/reports/phase6i/phase6i_robustness_audit.json) | Reproducibility log, cluster bootstrap p-value ($p=0.8600$), fold metrics. |
| **Comprehensive Methodology Review**| [`outputs/reports/project_methodology_complete.md`](file:///c:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/reports/project_methodology_complete.md) | Full technical review of data engineering, modeling tracks, and audit resolutions. |
| **Audit Discrepancies Log** | [`outputs/reports/project_evidence_and_discrepancies.md`](file:///c:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/reports/project_evidence_and_discrepancies.md) | Resolution evidence for baseline leakage (DISC-05), fold 0 tuning (DISC-07), texture boundary effects (DISC-08). |
| **Archived Tabular Baseline** | [`outputs/reports/baseline_rf_metrics.json`](file:///c:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/reports/baseline_rf_metrics.json) | **Archived Phase 6A**: Tabular baseline with post-harvest calendar leakage (Macro-F1: 0.9030). |
| **Phase 6H Weighting & Boundary Report** | [`outputs/reports/phase6h/phase6h_weighting_boundary_report.md`](file:///c:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/reports/phase6h/phase6h_weighting_boundary_report.md) | Parcel-area quartile analysis and smallholder GLCM boundary contamination analysis. |
| **Authoritative CV Folds** | [`outputs/data/phase6f_cv_folds.csv`](file:///c:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/data/phase6f_cv_folds.csv) | 7,843 farms assigned to 5 spatial folds grouped by Sub-District (Mandal). |
| **Enriched Train Geometry** | [`outputs/data/train_enriched_geometry.csv`](file:///c:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/data/train_enriched_geometry.csv) | 7,888 farms × 32 columns with metric geometric features and repaired WKT. |
| **Enriched Test Geometry** | [`outputs/data/test_enriched_geometry.csv`](file:///c:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/data/test_enriched_geometry.csv) | 2,718 farms × 31 columns ready for satellite extraction. |
| **Interactive Map** | [`outputs/figures/telangana_farms_interactive_map.html`](file:///c:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/figures/telangana_farms_interactive_map.html) | Standalone 3.59 MB Folium map with satellite layer, vector farm polygons, and heatmaps. |

---

## 📚 References & Standards
- **Sentinel-2 MSI Level-2A:** European Space Agency (ESA) Copernicus Open Access Hub (`COPERNICUS/S2_SR_HARMONIZED`).
- **Spectral Indices:**
  - Rouse, J. W., et al. (1974). *Monitoring vegetation systems in the Great Plains with ERTS.* (NDVI).
  - Huete, A., et al. (2002). *Overview of the radiometric and biophysical performance of the MODIS vegetation indices.* (EVI).
  - Gao, B. C. (1996). *NDWI—A normalized difference water index for remote sensing of vegetation liquid water from space.* (NDMI).
- **Coordinate Reference Systems:** EPSG:4326 (WGS 84 Geographic) and EPSG:32644 (WGS 84 / UTM Zone 44N Metric Projection).
- **Spatial Validation:** Roberts, D. R., et al. (2017). *Cross-validation strategies for data with spatial, temporal, or phylogenetic structure.* Ecography, 40(8), 913-929.
