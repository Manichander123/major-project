# AI-Based Multi-Temporal Remote Sensing for Crop Monitoring and Irrigation Decision Support Using Sentinel-2 Imagery

An enterprise-grade, reproducible remote sensing and machine learning pipeline for 7-class crop type classification across **7,843 verified smallholder farm parcels** in Telangana, India — using multi-temporal **Sentinel-2 MSI Level-2A surface reflectance** and rigorous spatial cross-validation.

> **Data Source:** Telangana Crop Health Challenge (GIZ-ADEX / Data Foundation).  
> **Geographic Scope:** State of Telangana, India — 14 districts, 116 Sub-Districts (Mandals).

---

## 📌 Table of Contents

1. [Executive Summary & Completed Pipeline](#-executive-summary--completed-pipeline)
2. [Dataset Summary & Cohort Accounting](#-dataset-summary--cohort-accounting)
3. [System Architecture](#-system-architecture)
4. [Project Structure](#-project-structure)
5. [Full Implementation Roadmap & Audit Status](#-full-implementation-roadmap--audit-status)
6. [Phase-by-Phase Technical Walkthrough](#-phase-by-phase-technical-walkthrough)
   - [Phase 1: Dataset Analysis, Validation & EDA](#phase-1-dataset-analysis-validation--eda)
   - [Phase 2: Farm Geometry Parsing, Repair & Feature Engineering](#phase-2-farm-geometry-parsing-repair--feature-engineering)
   - [Phase 3: Spatial Visualisation & Cartography](#phase-3-spatial-visualisation--cartography)
   - [Phase 4: Sentinel-2 Multi-Temporal Extraction Engine](#phase-4-sentinel-2-multi-temporal-extraction-engine)
   - [Phase 5: Spectral, Moisture & Vegetation Indices](#phase-5-spectral-moisture--vegetation-indices)
   - [Phase 6A: Tabular Baseline (Archived — Leakage Documented)](#phase-6a-tabular-baseline-archived--leakage-documented)
   - [Phase 6C–6E: Satellite Extraction Scaling Campaign](#phase-6c6e-satellite-extraction-scaling-campaign)
   - [Phase 6F: Full-Scale 5-Fold Spatial CV Benchmark](#phase-6f-full-scale-5-fold-spatial-cv-benchmark)
   - [Phase 6G: Hyperparameter Tuning & Sample Weighting Pilot](#phase-6g-hyperparameter-tuning--sample-weighting-pilot)
   - [Phase 6H: GLCM Spatial Texture & S/T/P Feature Ablation](#phase-6h-glcm-spatial-texture--stp-feature-ablation)
   - [Phase 6I: Robustness Validation & Final Model Selection](#phase-6i-robustness-validation--final-model-selection)
7. [Scientific Integrity & Resolved Discrepancy Register](#-scientific-integrity--resolved-discrepancy-register)
8. [Leakage Prevention Architecture](#-leakage-prevention-architecture)
9. [Status of Unimplemented Modules & Unprocessed Datasets](#-status-of-unimplemented-modules--unprocessed-datasets)
10. [Execution & Verification Guide](#-execution--verification-guide)
11. [Complete Artifacts & Reports Index](#-complete-artifacts--reports-index)
12. [References & Standards](#-references--standards)

---

## ⚡ Executive Summary & Completed Pipeline

### Primary Research Objective
Automated leakage-safe classification of 7 crop types from multi-temporal Sentinel-2 MSI Level-2A surface reflectance across smallholder parcels in Telangana, India, using rigorous spatial cross-validation that prohibits any geographic group from appearing in both training and validation partitions.

### 🏆 Selected Primary Production Model — Phase 6I Result

| Attribute | Value |
|:---|:---|
| **Model** | Regularized XGBoost (`max_depth=7`, `colsample=0.75`, `subsample=0.8`, `reg_lambda=5.0`, `gamma=0.2`, `reg_alpha=0.2`, `lr=0.05`, `n_estimators=300`) |
| **Feature Configuration** | **`Weighted S+T`** — 140 Spectral + Temporal features with inverse-frequency sample weighting |
| **Evaluation Cohort** | **7,843 eligible training farms** across 116 Mandals (Sub-Districts) |
| **Validation Protocol** | 5-Fold `StratifiedGroupKFold` grouped strictly by Sub-District (Mandal) — **zero spatial group leakage** |
| **Pooled Out-of-Fold Macro-F1** | **`0.7226`** *(spatial cross-validation result, NOT a test-set score)* |
| **Balanced Accuracy** | **`0.7180`** |
| **Overall Accuracy** | **`0.7881`** |
| **Fold Macro-F1 (Mean ± Std)** | **`0.7117 ± 0.0181`** (Fold 0: 0.7269, Fold 1: 0.7381, Fold 2: 0.6919, Fold 3: 0.6945, Fold 4: 0.7073) |
| **Majority Baseline (DummyClassifier)** | Macro-F1 = 0.0639, Accuracy = 0.2882 |

> [!IMPORTANT]
> The **0.7226 Macro-F1** is an **out-of-fold spatial cross-validation result** on the training cohort. `datasss/Test.csv` (2,718 parcels) has **NOT been processed for satellite features** and has **NOT been evaluated**.

> [!NOTE]
> The project has deliberately chosen `Crop` (7-class) as the primary research target over the competition's `category` (crop health), because accurate crop-type identification is a prerequisite for crop-specific health and irrigation assessment. Crop health modeling remains a planned future phase.

---

## 📊 Dataset Summary & Cohort Accounting

### Raw Survey Data
- **Training Set (`datasss/Train.csv`):** 7,888 farm records × 20 columns with ground-truth crop labels.
- **Test Set (`datasss/Test.csv`):** 2,718 farm records × 19 columns — **unprocessed for satellite imagery**.
- **Submission Template (`datasss/SampleSubmission.csv`):** 3,016 rows (2,718 genuine + 298 synthetic anti-probing decoy IDs — see `DISC-03` in the Discrepancy Register).

### Target Variable: Primary (`Crop` — 7 Classes)
| Crop Class | Train.csv Total | Eligible Cohort (N=7,843) | Cohort % | Season Behavior |
|:---|:---:|:---:|:---:|:---|
| **Paddy** | 2,268 | 2,260 | 28.81% | Rabi + Kharif; wetland rice; distinctive waterlogged canopy |
| **Cotton** | 1,692 | 1,679 | 21.41% | Kharif; long-season (~210 days); prolonged boll-development |
| **Maize** | 1,384 | 1,376 | 17.54% | Kharif; rapid vegetative peak (~120 days); high biomass |
| **Bengal Gram** | 938 | 938 | 11.96% | Rabi; compact chickpea canopy; short duration (~90 days) |
| **Groundnut** | 768 | 768 | 9.79% | Rabi; oilseed legume; distinctive prostrate canopy |
| **Chillies** | 614 | 601 | 7.66% | Extended flowering cycle (~180 days); high agronomic value |
| **Red Gram** | 224 | 221 | 2.82% | Minor pulse (pigeon pea); severe class minority; intercropped |

### Target Variable: Secondary (`category` — Crop Health Status)
- `Healthy`: 6,506 (82.5%) | `Pests`: 485 (6.1%) | `Diseased`: 474 (6.0%) | `Stressed`: 423 (5.4%).
- **Status:** Preserved exclusively in raw data. **No classification model has been trained on this target.** Excluded from all primary feature sets to prevent target leakage.

### Cohort Accounting: 7,888 → 7,843 Eligible Farms
```
Total Survey Rows in Train.csv: 7,888
  ├── Phase 6C Pilot (175 attempted): 173 valid farms
  ├── Phase 6D Expansion (724 attempted): 714 valid farms, 10 cloud-failed
  ├── Phase 6E Full Campaign (7,164 attempted across 15 chunks):
  │     ├── Raw successful: 7,114 farms
  │     └── Cloud-obscured/timeout failures: 50 farms
  ├── Phase 6F Stage 2 Recovery (50 retried):
  │     ├── Successfully recovered: 5 farms (expanded composite window)
  │     └── Irrevocably failed (persistent cloud mask): 45 farms
  └── Final Authoritative Eligible Cohort (Phase 6F Stage 3): 7,843 farms (99.43%)

Extraction Coverage: 7,843 / 7,888 = 99.43%
Excluded (Honest Physical Missingness): 45 farms (0.57%) — 100% persistent monsoon cloud
```

### Phenological Window Completeness (N = 7,843 Eligible Farms)
| Window | Definition | Valid Farms | Valid % | Missing | Physical Cause |
|:---|:---|:---:|:---:|:---:|:---|
| `early` | 0–25% of crop growth cycle | 7,302 | 93.10% | 541 | July–August monsoon cloud peak |
| `mid` | 25–75% of crop growth cycle | 7,722 | 98.46% | 121 | Post-monsoon clearing |
| `late` | 75–100% of crop growth cycle | 7,651 | 97.55% | 192 | Mostly clear dry season |
| `full` | Complete lifecycle (100%) | **7,843** | **100.00%** | **0** | Always available |

### Additional Dataset Details
- **Crop Seasons:** Rabi (78.3%), Kharif (21.7%).
- **Irrigation Types:** Drip (36.4%), Sprinkler (28.9%), Flood (28.8%), Surface (5.9%).
- **Irrigation Sources:** Groundwater (42.3%), Canal (40.4%), Rainfed (15.9%), Well (0.7%), Tubewell (0.7%).
- **Mean Irrigation Rounds:** 3.54 (std = 1.41, min = 1, max = 6).
- **Temporal Span:** Sowing `2023-01-05` → `2024-09-28`; Harvest up to `2024-12-06`. Average growing duration: 193 days (Train), 191 days (Test).
- **Top Districts by Farm Count:** Adilabad (1,770), Nalgonda (1,386), Mahabubabad (1,314), Nirmal (1,244), Medak (1,210).

---

## 🏗️ System Architecture

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                           RAW DATA INGESTION & INTEGRITY AUDIT                         │
│  Train.csv (7,888 farms) & Test.csv (2,718 farms) [WKT Polygon, Agronomic Metadata]   │
│  ✓ 0 missing values  ✓ 0 duplicate FarmIDs  ✓ 0 temporal inversions  ✓ 0 leakage     │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                    PHASE 1-3: GEOMETRIC, EDA & SPATIAL PIPELINE                        │
│  - shapely.make_valid() Topological Repair: 1 invalid fixed in Train, 1 in Test       │
│  - Projected CRS: EPSG:4326 (WGS84) ──► EPSG:32644 (UTM Zone 44N Metric)             │
│  - 12 Geometric Features: Area (ha), Perimeter (m), Polsby-Popper Compactness...      │
│  - Interactive Folium Cartography: 3-basemap map, polygons, popups, heatmaps          │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│           PHASE 4-5: SENTINEL-2 MSI L2A EXTRACTION ENGINE & INDICES SUITE             │
│  - Target Dataset: COPERNICUS/S2_SR_HARMONIZED (Atmospheric Surface Reflectance)      │
│  - Cloud Masking: SCL pixel classification (bands 3,8,9,10,11) + QA60 bitmask        │
│  - Daily MGRS Tile Mosaicking: collapse orbit overlaps, prevent duplicate sampling    │
│  - 4 Phenological Windows: Early (0-25%), Mid (25-75%), Late (75-100%), Full         │
│  - 10 Spectral Bands: B2,B3,B4,B8 @ 10m; B5,B6,B7,B8A,B11,B12 @ 20m                │
│  - 14 Spectral Indices: NDVI,EVI,GNDVI,SAVI,MSAVI,NDMI,NDWI,NDRE1,NDRE2,PSRI,BSI... │
│  - Full S+T Feature Matrix: 140 features (80 band stats + 56 index stats + 4 deltas)  │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                PHASE 6 (6A-6I): MACHINE LEARNING PIPELINE & VALIDATION                 │
│                                                                                        │
│  Phase 6A: Tabular baseline [ARCHIVED — 0.9030 Macro-F1 from calendar + coord leakage]│
│  Phase 6C-6D: Pilot Benchmarks (350 → 714 farms, Macro-F1 0.4961 → 0.7047 XGB)       │
│  Phase 6E: Full-scale Extraction Campaign (7,164 farms in 15 parallel chunks)         │
│  Phase 6F: Statewide 5-Fold Spatial CV Benchmark (7,843 farms):                       │
│    Exp A (Full-Season):   XGB Macro-F1 0.6174  RF Macro-F1 0.5900                    │
│    Exp B (Early-Season):  XGB Macro-F1 0.5058  RF Macro-F1 0.5101                    │
│    Exp C (Multi-Temporal):XGB Macro-F1 0.7038  RF Macro-F1 0.6477  [140 S+T feats]   │
│    Exp D (Multimodal):    XGB Macro-F1 0.7954  RF Macro-F1 0.7422  [146 S+T+Cal feats]│
│  Phase 6G: Hyperparameter search (10 trials) + Sample weighting pilot (7 strategies)  │
│  Phase 6H: GLCM Spatial Texture (20 features) + S/T/P Ablation (7 experiments)       │
│  Phase 6I: Statistical model selection → Winner: Weighted S+T, Macro-F1 0.7226       │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                 DOWNSTREAM WORKSTREAMS — NOT YET IMPLEMENTED                           │
│  • Test Set (datasss/Test.csv): S2 feature extraction & model inference [UNPROCESSED] │
│  • Phase 7: Secondary target `category` (crop health) classification [PLANNED]        │
│  • Phase 8-11: SAR, weather, FAO-56 ET₀, prescriptive irrigation [PLANNED]            │
│  • Phase 12-13: Streamlit dashboard & LangChain advisory agents [PLANNED]             │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
major_proj_final/
├── config/
│   └── project_config.yaml              ← Central pipeline config (CRS, bands, ML params)
├── datasss/                             ← Immutable raw competition data (do not modify)
│   ├── Train.csv                        ← 7,888 farms × 20 cols (with Crop, category)
│   ├── Test.csv                         ← 2,718 farms × 19 cols (UNPROCESSED)
│   ├── SampleSubmission.csv             ← 3,016 rows (2,718 real + 298 decoy IDs)
│   └── Starter_Notebook.ipynb           ← Official competition starter notebook
├── notebooks/
│   ├── 01_dataset_analysis.ipynb        ← Phase 1: EDA, distributions, quality audit
│   └── 02_geometry_and_spatial_analysis.ipynb  ← Phase 2&3: geometry & cartography
├── outputs/
│   ├── data/                            ← Generated feature matrices & fold splits
│   │   ├── train_enriched_geometry.csv  ← 7,888 × 32 columns with metric geometry
│   │   ├── test_enriched_geometry.csv   ← 2,718 × 31 columns ready for S2 extraction
│   │   ├── s2_phase6f_harmonized_cohort.csv ← 7,888 × 100 raw bands + provenance
│   │   ├── s2_phase6f_features.csv      ← 7,888 × 165 full feature matrix (SHA256 verified)
│   │   ├── phase6f_cv_folds.csv         ← 7,843 farms × 5-fold Mandal assignments
│   │   ├── phase6f_full_multimodal_features.csv ← 7,843 × 146 S+T+Calendar features
│   │   ├── phase6f_oof_predictions.csv  ← OOF predictions for all 4 Phase 6F experiments
│   │   ├── phase6g/                     ← Phase 6G pilot OOF predictions & trial ledger
│   │   └── phase6h/                     ← Phase 6H ablation features & OOF predictions
│   ├── figures/
│   │   ├── telangana_farms_interactive_map.html ← 3.59 MB interactive Folium map
│   │   ├── telangana_spatial_overview.png       ← 4-panel high-res static overview
│   │   ├── crop_and_category_distribution.png
│   │   ├── date_range_analysis.png
│   │   ├── district_crop_heatmaps.png
│   │   ├── farm_area_and_centroids.png
│   │   └── geometry_diagnostics.png
│   ├── models/
│   │   ├── archive/                     ← Archived models (baseline_rf.joblib)
│   │   ├── phase6f/                     ← Phase 6F 5-fold trained model checkpoints
│   │   └── phase6g/                     ← Phase 6G tuned model checkpoints
│   ├── satellite_cache/                 ← Per-farm GEE extraction JSON checkpoints
│   └── reports/                         ← Formal audit reports & benchmark documentation
│       ├── data_quality_report.txt/.json         ← Phase 1 integrity audit
│       ├── geometry_validation_report.txt/.json  ← Phase 2 topological audit
│       ├── baseline_rf_metrics.json              ← Archived Phase 6A tabular baseline
│       ├── phase6d_research_baseline_report.md   ← Phase 6D verification & fold analysis
│       ├── phase6e_launch_readiness.md           ← Phase 6E gate report & campaign plan
│       ├── phase6f_stage3_harmonization_report.md ← Final cohort accounting (7,843 eligible)
│       ├── phase6f_stage4_benchmark_report.md    ← Phase 6F 4-experiment benchmark report
│       ├── phase6f_stage4c_integrity_audit.md    ← Scientific integrity audit with 6 addenda
│       ├── phase6f_feature_schema_manifest.json  ← SHA256-verified 165-column feature schema
│       ├── phase6g/
│       │   ├── phase6g_pilot_fold0_report.md     ← Trial 4 hyperparameter selection report
│       │   └── phase6g_exp2_imbalance_report.md  ← 7-strategy weighting pilot report
│       ├── phase6h/
│       │   ├── phase6h_stage2_extraction_report.md ← GLCM texture extraction report
│       │   ├── phase6h_stage3_benchmark_report.md  ← 7-experiment S/T/P ablation benchmark
│       │   └── phase6h_weighting_boundary_report.md ← Boundary contamination analysis
│       ├── phase6i/
│       │   ├── phase6i_model_selection_report.md   ← Authoritative model selection decision
│       │   └── phase6i_robustness_audit.json        ← Full reproducibility verification log
│       ├── project_methodology_complete.md          ← Comprehensive 70KB methodology review
│       └── project_evidence_and_discrepancies.md   ← 8-item discrepancy resolution register
├── scripts/
│   ├── analyze_dataset.py               ← Phase 1 analysis runner
│   ├── validate_geometries.py           ← Phase 2 geometry repair runner
│   ├── visualize_spatial.py             ← Phase 3 Folium map generator
│   ├── test_sentinel2_small.py          ← Phase 4/5 integration test (15 farms)
│   └── audit_phase6i_robustness.py      ← Phase 6I verification & bootstrap audit runner
├── src/
│   ├── data/
│   │   ├── load_data.py                 ← Safe train/test loaders with type casting
│   │   └── validate_data.py             ← Schema and integrity validator
│   ├── features/
│   │   ├── indices.py                   ← 14 spectral vegetation & moisture indices
│   │   └── spatial_texture.py           ← GLCM texture extraction module (Phase 6H)
│   ├── geospatial/
│   │   ├── geometry_utils.py            ← WKT parsing, repair, UTM reprojection
│   │   └── spatial_viz.py               ← Folium map & matplotlib cartography
│   ├── satellite/
│   │   ├── s2_extractor.py              ← GEE Sentinel-2 multi-temporal extractor
│   │   └── s2_mosaic.py                 ← Daily MGRS tile mosaicking engine
│   └── models/
│       ├── __init__.py                  ← Package routing to archive
│       ├── baseline_rf.py               ← Deprecation shim → archive/baseline_rf.py
│       ├── train_pilot_models.py        ← Deprecation shim → archive/train_pilot_models.py
│       └── archive/
│           ├── baseline_rf.py           ← Phase 6A tabular baseline (ARCHIVED, leakage documented)
│           └── train_pilot_models.py    ← Phase 6C pilot benchmark (ARCHIVED)
├── tests/                              ← 16-file unit & regression test suite
│   ├── test_baseline.py
│   ├── test_phase6c.py
│   ├── test_phase6d_audit.py
│   ├── test_phase6f_stage{1..4}*.py
│   ├── test_phase6g_{pilot,exp2}.py
│   ├── test_phase6h_stage{2,3}.py
│   ├── test_phase6i_robustness.py
│   └── test_pre_extraction_gate.py
└── requirements.txt
```

---

## 🚀 Full Implementation Roadmap & Audit Status

| Phase | Milestone | Status | Key Metric / Deliverable |
|:---:|:---|:---:|:---|
| **1** | Dataset Analysis, Validation & EDA | ✅ Done | 0 missing, 0 duplicates, 0 temporal inversions; 7 figures |
| **2** | Geometry Parsing, Topological Repair, Metric Features | ✅ Done | 100% validity; area/perimeter/compactness in UTM 44N |
| **3** | Spatial Visualisation & Interactive Cartography | ✅ Done | 3.59 MB Folium HTML map; 4-panel static overview |
| **4** | Sentinel-2 Multi-Temporal Extraction Engine | ✅ Done | GEE L2A; SCL + QA60 masking; daily mosaic; 4 windows |
| **5** | Spectral & Vegetation Indices Engineering | ✅ Done | 14 indices; 4 temporal trajectory deltas |
| **6A** | Tabular Calendar Baseline | ⚠️ *Archived* | 0.9030 Macro-F1 **inflated** by calendar leakage & spatial autocorrelation |
| **6C** | S2 Pilot Benchmark (175/350 farms) | ⚠️ *Archived Pilot* | RF/XGB on 173–350 farms; seed for Phase 6D |
| **6D** | S2 Expansion & Mosaic Refinement (714 farms) | ✅ Done | XGB_multimodal Macro-F1 **0.7047**; 714 eligible farms |
| **6E** | Full-Scale S2 Extraction (7,164 farms, 15 chunks) | ✅ Done | 7,124 successful; 50 cloud-failed; daily mosaic validated |
| **6F Stage 1-3** | Data Audit, Recovery & Cohort Harmonization | ✅ Done | 7,843 eligible farms (99.43%); 5 farms recovered; 45 cloud-failed |
| **6F Exp A** | Full-Season Satellite Retrospective (34 features) | ✅ Done | XGB Macro-F1 **0.6174**, RF **0.5900** (N=7,843) |
| **6F Exp B** | Early-Season In-Season (34 features, 541 monsoon gaps) | ✅ Done | XGB Macro-F1 **0.5058**, RF **0.5101** (81.9% of full-season retained) |
| **6F Exp C** | Multi-Temporal Trajectory Satellite (140 features) | ✅ Done | XGB Macro-F1 **0.7038**, RF **0.6477** |
| **6F Exp D** | Multimodal Benchmark (146 S+T+Calendar features) | ✅ Done | XGB Macro-F1 **0.7954**, RF **0.7422** *(pre-sowing calendar, not satellite-only)* |
| **6F Stage 4C** | Scientific Integrity Audit (15 model-runs) | ✅ Done | PASS with 6 addenda; metrics reproduced to 4dp; 24 baselines verified |
| **6G Pilot** | Hyperparameter Tuning (10 Trials, Outer Fold 0) | ✅ Done | Trial 4 selected (Inner Macro-F1 **0.6797**); `depth=7, colsample=0.75, reg_lambda=5.0` |
| **6G Exp 2** | Sample Weighting Pilot (7 Strategies, Outer Fold 0) | ✅ Done | `balanced_inv_freq` selected; Fold 0 Macro-F1 **0.7339**, Red Gram Recall **44.64%** |
| **6H Stage 2** | GLCM Spatial Texture Extraction (20 features, all 7,843 farms) | ✅ Done | 100% coverage; `full` window 100% finite; early 93.94% finite |
| **6H Stage 3** | 7-Experiment S/T/P Feature Ablation Benchmark | ✅ Done | S+T=**0.7069**, S+T+P=**0.7038** unweighted; texture adds noise to minority crops |
| **6H Weighting** | 4-Model Weighted Benchmark & Boundary Analysis | ✅ Done | Weighted S+T=**0.7226**, Weighted S+T+P=**0.7228**; GLCM kernel 900m² vs parcels |
| **6I** | Statistical Robustness Validation & Final Model Selection | ✅ Done | **Weighted S+T (140 features)** wins via Occam's razor; $p=0.8600$ |
| **Test Set** | Sentinel-2 Extraction & Inference on `datasss/Test.csv` | ⏳ *Not Yet Implemented* | 2,718 parcels unprocessed |
| **7** | Crop Health Classification (`category`) | ⏳ *Not Yet Implemented* | No models trained |
| **8** | Sentinel-1 SAR C-band Integration | ⏳ *Planned* | VV/VH backscatter, RVI |
| **9–10** | Weather, Moisture & FAO-56 ET₀ Estimation | ⏳ *Not Yet Implemented* | ERA5, CHIRPS, Penman-Monteith |
| **11** | Irrigation Decision-Support Engine | ⏳ *Not Yet Implemented* | Urgency score, volume deficit |
| **12** | Streamlit Interactive Dashboard | ⏳ *Not Yet Implemented* | Multi-page web app |
| **13** | LangChain Agronomic Advisory Agents | ⏳ *Not Yet Implemented* | Conversational LLM advisory |

---

## 🔍 Phase-by-Phase Technical Walkthrough

### Phase 1: Dataset Analysis, Validation & EDA

**Script:** [`scripts/analyze_dataset.py`](file:///C:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/scripts/analyze_dataset.py)  
**Report:** [`outputs/reports/data_quality_report.txt`](file:///C:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/reports/data_quality_report.txt)

#### Data Quality Verification (All from audit report)
```
Train: 7,888 rows × 20 columns | Test: 2,718 rows × 19 columns
  ✓ Missing values:         0 in Train, 0 in Test
  ✓ Duplicate FarmIDs:      0 in Train, 0 in Test
  ✓ Duplicate geometries:   0 in Train, 0 in Test
  ✓ Train-Test FarmID overlap: 0
  ✓ Temporal inversions (HDate ≤ SDate): 0
  ✓ 'Crop' in Test: TRUE (feature input, not prediction target)
  ✓ 'category' in Test: FALSE (expected — health labels not provided for test)
  ✓ CRITICAL ISSUES DETECTED: False
```

#### Column Schema
| Column | Type | Description |
|:---|:---:|:---|
| `FarmID` | int | Unique farm identifier |
| `State` | str | Constant: Telangana |
| `District` | str | 14 districts |
| `Sub-District` | str | 116 sub-districts (Mandals) |
| `geometry` | str | WKT POLYGON boundary string |
| `SDate` | date | Sowing date (DD-MM-YYYY) |
| `HDate` | date | Harvest date (DD-MM-YYYY) |
| `Season` | str | Rabi or Kharif |
| `CropCoveredArea` | float | Field coverage % (40–100%) |
| `CHeight` | float | Crop height cm (20–250) |
| `ExpYield` | float | Expected yield quintals/acre (5–30) |
| `IrriType` | str | Drip / Sprinkler / Flood / Surface |
| `IrriSource` | str | Groundwater / Canal / Rainfed / Well / Tubewell |
| `IrriCount` | int | Irrigation rounds (1–6) |
| `WaterCov` | float | Water coverage % (20–100%) |
| `CNext` / `CLast` / `CTransp` | str | Crop rotation & transplant survey fields |
| `Crop` | str | **Primary Target** — 7 crop classes |
| `category` | str | **Secondary Target** (Train only) — health status |

---

### Phase 2: Farm Geometry Parsing, Repair & Feature Engineering

**Script:** [`scripts/validate_geometries.py`](file:///C:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/scripts/validate_geometries.py)  
**Report:** [`outputs/reports/geometry_validation_report.txt`](file:///C:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/reports/geometry_validation_report.txt)

#### Topological Repair Results
- **Train:** 1 initially invalid polygon (FarmID `1330965`, self-intersecting boundary ring). Repaired with `shapely.make_valid()`. Post-repair: **7,888 / 7,888 valid (100%)**.
- **Test:** 1 initially invalid polygon (FarmID `1326463`, self-touching boundary). Repaired. Post-repair: **2,718 / 2,718 valid (100%)**.

#### Geometric Statistics (from audit report)
| Metric | Train | Test |
|:---|:---:|:---:|
| Total farms | 7,888 | 2,718 |
| Area (ha): min / mean / max | 0.0054 / 0.4231 / 6.9491 | 0.0069 / 0.3612 / 4.8004 |
| Area (m²): mean / median | 4,231 / 2,779 | 3,612 / 2,610 |
| Perimeter (m) mean | 249.9 m | 238.8 m |
| Polsby-Popper Compactness mean | 0.6885 | 0.6758 |
| Vertex count mean | 7.43 | 8.00 |
| Within Telangana bounding box | 7,888 ✓ | 2,718 ✓ |

All farms projected from **EPSG:4326 (WGS84)** → **EPSG:32644 (WGS84 / UTM Zone 44N)** for metric-correct area and perimeter computation.

---

### Phase 3: Spatial Visualisation & Cartography

**Script:** [`scripts/visualize_spatial.py`](file:///C:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/scripts/visualize_spatial.py)

- **[`outputs/figures/telangana_farms_interactive_map.html`](file:///C:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/figures/telangana_farms_interactive_map.html)** (3.59 MB Folium map):
  - 3 base layers: Esri World Imagery (satellite), OpenStreetMap, CartoDB Positron.
  - Farm polygon layer color-coded by crop health (`Healthy`=green, `Pests`=orange, `Diseased`=red, `Stressed`=yellow).
  - Interactive popup dossiers per farm: FarmID, Crop, Health, Season, District, Area, Irrigation details, Yield.
  - Density heatmap overlay, district cluster markers, Fullscreen and MeasureControl widgets.
- **[`outputs/figures/telangana_spatial_overview.png`](file:///C:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/figures/telangana_spatial_overview.png)** — 4-panel static overview: farm locations, health distribution, crop type distribution, irrigation sources.

---

### Phase 4: Sentinel-2 Multi-Temporal Extraction Engine

**Module:** [`src/satellite/s2_extractor.py`](file:///C:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/src/satellite/s2_extractor.py) + [`src/satellite/s2_mosaic.py`](file:///C:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/src/satellite/s2_mosaic.py)

#### Extraction Architecture
- **Target Collection:** `COPERNICUS/S2_SR_HARMONIZED` — Sentinel-2 Level-2A Harmonized Surface Reflectance.
- **Cloud-Cover Threshold Filter:** `ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', 20.0)` pre-filter on scene metadata.
- **Pixel-Level SCL Masking:** Scene Classification Layer values 3 (cloud shadow), 8 (medium cloud), 9 (high cloud), 10 (cirrus), 11 (snow/ice) masked out.
- **QA60 Bitmask Masking:** Opaque cloud (bit 10) and cirrus cloud (bit 11) independently masked.
- **MGRS Daily Mosaicking:** Same-date scenes from overlapping Sentinel-2 tiles are mosaicked before zonal reduction to eliminate inter-tile duplicate sampling. Unit-tested in `tests/test_s2_mosaic.py`.
- **Temporal Median Compositing:** Robust pixel-wise median across all cloud-free scenes within each phenological window.

#### Spectral Band Extraction
| Bands | Native Resolution | Reduction Scale | Spectral Range |
|:---|:---:|:---:|:---|
| B2 (Blue), B3 (Green), B4 (Red), B8 (NIR) | 10 m | `scale=10` | 490–842 nm |
| B5 (RedEdge1), B6 (RedEdge2), B7 (RedEdge3), B8A (NarrowNIR), B11 (SWIR1), B12 (SWIR2) | 20 m | `scale=20` | 705–2190 nm |

Per band per window: **mean** (central spectral response) + **standard deviation** (intra-field spatial heterogeneity).

#### Extraction Performance (Phase 6E Full Campaign)
- **Extraction throughput:** ~40.46 seconds/farm single-thread; ~5.5–6.0 farms/min with 4 parallel workers.
- **Campaign duration:** ~19–21 hours for 7,164 farms across 15 chunks.
- **Checkpoint safeguard:** Per-farm atomic JSON cache writes in `outputs/satellite_cache/`. Failed farms retain retryable status and are never written to cache.

---

### Phase 5: Spectral, Moisture & Vegetation Indices

**Module:** [`src/features/indices.py`](file:///C:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/src/features/indices.py)

All 14 indices computed per phenological window (early, mid, late, full) as vectorized, zero-division-safe operations:

| Index | Formula | Agronomic Proxy |
|:---|:---:|:---|
| **NDVI** | $(B8-B4)/(B8+B4)$ | Photosynthetic canopy greenness & vegetative vigor |
| **EVI** | $2.5 \cdot (B8-B4)/(B8+6B4-7.5B2+1)$ | Structural canopy proxy; reduces atmospheric & soil effects |
| **GNDVI** | $(B8-B3)/(B8+B3)$ | Leaf chlorophyll concentration & photosynthetic activity |
| **SAVI** | $1.5 \cdot (B8-B4)/(B8+B4+0.5)$ | Soil-adjusted canopy greenness (L=0.5) |
| **MSAVI** | $(2B8+1-\sqrt{(2B8+1)^2-8(B8-B4)})/2$ | Self-adjusting soil factor for emerging or sparse crops |
| **NDMI** | $(B8-B11)/(B8+B11)$ | Canopy foliage liquid water content & water stress |
| **NDWI** | $(B3-B8)/(B3+B8)$ | Surface water & open-water/soil moisture proxy |
| **MSI** | $B11/B8$ | Relative leaf water deficit (rises with stress) |
| **NDRE1** | $(B8-B5)/(B8+B5)$ | Early nitrogen deficiency & foliar stress (red-edge 1) |
| **NDRE2** | $(B8-B6)/(B8+B6)$ | Dense canopy chlorophyll dynamics (red-edge 2) |
| **PSRI** | $(B4-B2)/B6$ | Carotenoid-to-chlorophyll ratio; ripening & senescence |
| **BSI** | $((B11+B4)-(B8+B2))/((B11+B4)+(B8+B2))$ | Bare soil exposure & fallow ground detection |
| **CIRE** | $B8/B5-1$ | Foliar chlorophyll (red-edge 1) |
| **CIG** | $B8/B3-1$ | Total leaf chlorophyll mass (green reflectance) |

#### Multi-Temporal Trajectory Deltas (4 additional features)
| Feature | Formula | Phenological Meaning |
|:---|:---:|:---|
| `ndvi_growth_delta` | $\text{NDVI}_{mid} - \text{NDVI}_{early}$ | Rate of vegetative emergence & growth |
| `ndvi_senescence_delta` | $\text{NDVI}_{mid} - \text{NDVI}_{late}$ | Senescence intensity & maturation rate |
| `ndmi_loss_delta` | $\text{NDMI}_{mid} - \text{NDMI}_{late}$ | Canopy moisture depletion toward harvest |
| `peak_to_early_ratio` | $\text{NDVI}_{mid} / \text{NDVI}_{early}$ | Vigor amplification from emergence to peak |

---

### Phase 6A: Tabular Baseline (Archived — Leakage Documented)

**Module:** [`src/models/archive/baseline_rf.py`](file:///C:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/src/models/archive/baseline_rf.py)  
**Report:** [`outputs/reports/baseline_rf_metrics.json`](file:///C:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/reports/baseline_rf_metrics.json)

> [!WARNING]
> This model is archived. Its reported Macro-F1 of **0.9030** is artificially inflated due to two compounding leakage sources.

**Features used (18 total):**
- Calendar: `Season`, `duration_days`, `sowing_month`, `sowing_doy`, `harvest_month`, `harvest_doy`.
- Geographic coordinates: `centroid_lon`, `centroid_lat`, `bbox_min_lon`, `bbox_min_lat`, `bbox_max_lon`, `bbox_max_lat`.
- Geometric: `area_ha`, `perimeter_m`, `compactness`, `vertex_count`.
- Administrative: `District`, `Sub-District`.

**Documented Leakage Sources (see `DISC-05` in Discrepancy Register):**
1. **Post-Harvest Calendar Leakage:** `duration_days`, `harvest_month`, `harvest_doy` directly encode crop identity (Cotton ~210 days vs Bengal Gram ~90 days) but are unknown during the active growing season.
2. **Spatial Autocorrelation:** Geographic coordinates + administrative names + random 80/20 stratified split allows neighboring farms (same village) to appear in both train and validation, memorizing geographic crop zones rather than learning spectral signatures.

**Top Feature Importances from `baseline_rf_metrics.json`:**
```
1. bbox_min_lon:   0.1004  ←  spatial coordinate leakage
2. centroid_lat:   0.0957  ←  spatial coordinate leakage
3. bbox_max_lat:   0.0948  ←  spatial coordinate leakage
4. bbox_min_lat:   0.0938  ←  spatial coordinate leakage
5. bbox_max_lon:   0.0934  ←  spatial coordinate leakage
6. centroid_lon:   0.0911  ←  spatial coordinate leakage
7. Season:         0.0808  (legitimate)
8. harvest_doy:    0.0779  ←  post-harvest temporal leakage
```
*The top 6 features are all spatial coordinates — confirming geographic memorization, not satellite-based learning.*

**Phase 6A Results (train/val split 6,310/1,578 farms):**
- Majority Baseline: Accuracy = 0.2877, Macro-F1 = 0.0638.
- Random Forest: **Accuracy = 0.9030, Macro-F1 = 0.9030, Weighted-F1 = 0.9021**.
- *All values inflated — do not cite as operational performance.*

---

### Phase 6C–6E: Satellite Extraction Scaling Campaign

**Phase 6C (175 farms, pilot):** Initial multi-crop Sentinel-2 expansion to 173 valid farms. Established the multi-temporal extraction engine.

**Phase 6D (714 farms, 10 failed):**  
**Report:** [`outputs/reports/phase6d_research_baseline_report.md`](file:///C:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/reports/phase6d_research_baseline_report.md)

Expanded to 714 valid farms (10 cloud-failed). Key Phase 6D benchmark results under 5-fold StratifiedGroupKFold by Mandal:

| Model | OOF Acc | Macro-F1 | Configuration |
|:---|:---:|:---:|:---|
| RF_satellite_only | 0.5014 | 0.4961 | Spectral indices only |
| XGB_satellite_only | 0.5518 | 0.5511 | Spectral indices only |
| RF_multimodal | 0.6611 | 0.6577 | Satellite + geometric |
| **XGB_multimodal** | **0.7073** | **0.7047** | Satellite + geometric |

Phase 6D fold breakdown identified performance degradation in Folds 3 & 4 (Mandals in northern Adilabad like Utnur, Inderavelly), attributed to regional agroclimatic variation — confirming spatial validation is revealing genuine generalization boundaries, not artifacts.

**Phase 6E (7,164 farms, 15 parallel extraction chunks):**  
**Report:** [`outputs/reports/phase6e_launch_readiness.md`](file:///C:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/reports/phase6e_launch_readiness.md)  
Full-scale campaign with 4 parallel GEE worker threads, daily MGRS mosaicking, SCL + QA60 cloud filtering. 7,124 successful, 50 cloud-failed.

---

### Phase 6F: Full-Scale 5-Fold Spatial CV Benchmark

**Report:** [`outputs/reports/phase6f_stage4_benchmark_report.md`](file:///C:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/reports/phase6f_stage4_benchmark_report.md)  
**Integrity Audit:** [`outputs/reports/phase6f_stage4c_integrity_audit.md`](file:///C:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/reports/phase6f_stage4c_integrity_audit.md)

#### Feature Schema — 165-Column Feature Matrix (SHA256-verified)
**`outputs/reports/phase6f_feature_schema_manifest.json`** documents the complete feature schema:
- **Experiment A (34 features):** Full-season band means + stds + 14 indices for the `full` window.
- **Experiment B (34 features):** Same 34 features for the `early` window.
- **Experiment C (140 features):** 10 bands × 2 stats × 4 windows (80) + 14 indices × 4 windows (56) = 136 + 4 trajectory deltas.
- **Experiment D (146 features):** Exp C + 6 non-leaking calendar/geometry features (`Season`, `sowing_month`, `sowing_doy`, `area_ha`, `perimeter_m`, `compactness`).

#### Phase 6F Complete Benchmark Results (N = 7,843 Eligible Farms)

| Experiment | Model | Features | Pooled Macro-F1 | Pooled Accuracy | Balanced Acc | Fold Mean ± Std |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Full-Season Retrospective (Exp A) | Dummy | 34 | 0.0639 | 0.2882 | 0.1429 | — |
| Full-Season Retrospective (Exp A) | RF | 34 | 0.5900 | 0.7106 | 0.5977 | 0.5879 ± 0.0361 |
| **Full-Season Retrospective (Exp A)** | **XGB** | **34** | **0.6174** | **0.7366** | **0.6178** | **0.6164 ± 0.0316** |
| Early-Season In-Season (Exp B) | RF | 34 | 0.5101 | 0.5977 | 0.5244 | 0.5051 ± 0.0247 |
| Early-Season In-Season (Exp B) | XGB | 34 | 0.5058 | 0.6245 | 0.5064 | 0.5040 ± 0.0260 |
| Multi-Temporal Trajectory (Exp C) | RF | 140 | 0.6477 | 0.7493 | 0.6432 | 0.6417 ± 0.0257 |
| **Multi-Temporal Trajectory (Exp C)** | **XGB** | **140** | **0.7038** | **0.7877** | **0.6884** | **0.6961 ± 0.0230** |
| Multimodal Benchmark (Exp D)† | RF | 146 | 0.7422 | 0.7959 | 0.7220 | 0.7332 ± 0.0155 |
| **Multimodal Benchmark (Exp D)†** | **XGB** | **146** | **0.7954** | **0.8307** | **0.7750** | **0.7848 ± 0.0112** |

*† Exp D adds pre-sowing calendar features (`sowing_month`, `sowing_doy`, `Season`) from farmer registration records — multimodal, not satellite-only.*

#### Key Research Questions Answered by Phase 6F
- **Multi-temporal vs full-season-only:** +8.64 Macro-F1 points (0.6174→0.7038). Multi-temporal trajectory captures emergence rate (`ndvi_growth_delta`), peak vigor (`peak_to_early_ratio`), and senescence rate (`ndvi_senescence_delta`), separating crops with distinct phenological lifecycles.
- **Early-season capability:** Early-season model retains **81.9%** of full-season performance (0.5058 vs 0.6174) with first-25% of growth data — operationally valuable for mid-season irrigation planning.
- **Missingness robustness:** Complete-case sensitivity (7,099 farms, 0 NaN) vs primary cohort (7,843, 541 monsoon gaps) shows only +2.27 Macro-F1 gap — XGBoost native sparsity routing handles cloud gaps without distorting rankings.
- **Red Gram fold variability:** Validation support ranges from 21 to 59 Red Gram farms per fold due to northern Telangana geographic clustering. Actual per-fold Red Gram F1 std deviation: **±0.1309** (Exp C) and **±0.1918** (Exp D) — far exceeding overall fold Macro-F1 std dev of ±0.0230.

---

### Phase 6G: Hyperparameter Tuning & Sample Weighting Pilot

**Reports:** [`outputs/reports/phase6g/phase6g_pilot_fold0_report.md`](file:///C:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/reports/phase6g/phase6g_pilot_fold0_report.md) | [`outputs/reports/phase6g/phase6g_exp2_imbalance_report.md`](file:///C:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/reports/phase6g/phase6g_exp2_imbalance_report.md)

> [!NOTE]
> Phase 6G evaluated **Outer Fold 0 only** (N=1,508 farms). These results are **exploratory pilot findings**, not final 5-fold generalization estimates. The full 5-fold evaluation was completed in Phase 6I.

#### Pilot Stage: Bounded Nested Grouped CV (10 Hyperparameter Trials)
- **Inner tuning:** 3-Fold StratifiedGroupKFold on 93 Mandals (N=6,335) — zero group overlap.
- **Selected Trial 4:** `max_depth=7, lr=0.05, colsample=0.75, subsample=0.8, reg_lambda=5.0, gamma=0.2, reg_alpha=0.2` — Inner Macro-F1 = **0.6797** (vs 0.6649 baseline ref, +0.0148).
- **Outer Fold 0 result (Trial 4 vs baseline):** Macro-F1 0.7024 vs 0.7095 (-0.0071 — Outer Fold 0 happened to slightly favor the baseline config; the full 5-fold evaluation in 6I tells the complete story).

#### Experiment 2: 7-Strategy Imbalance-Aware Sample Weighting (Outer Fold 0)
All strategies evaluated via 3-fold inner CV on 93 Mandals:

| Strategy | Inner Macro-F1 | Red Gram F1 (Inner) | Selected? |
|:---|:---:|:---:|:---:|
| `unweighted_ref` (Trial 4 reference) | 0.6797 | 0.3022 | — |
| **`balanced_inv_freq`** ($w_c = N/(7 \times N_c)$) | **0.7020** | **0.4211** | ✅ **SELECTED** |
| `sqrt_balanced` (square-root smoothed) | 0.6858 | 0.3291 | — |
| `red_gram_2.0x` (targeted 2x) | 0.6861 | 0.3605 | — |
| `targeted_2.0x_1.3x` (RG+Chillies) | 0.6779 | 0.2882 | — |

**Selected Inverse-Frequency Weights (fit on 6,335 training farms):**
```json
{
  "Paddy": 0.5011, "Cotton": 0.6744, "Maize": 0.8168,
  "Bengal Gram": 1.2019, "Groundnut": 1.3568,
  "Chillies": 1.832, "Red Gram": 5.4848
}
```

**Outer Fold 0 Final Results (`balanced_inv_freq` weighted, N=1,508):**
- Macro-F1 = **0.7339** (+0.0315 vs unweighted Trial 4, +0.0244 vs Phase 6F baseline).
- Red Gram Recall = **44.64%** (25/56 farms) vs 25.00% unweighted.
- Red Gram → Cotton confusions: **12/56** (down from 36/56), a reduction of two-thirds.
- Overall Accuracy preserved at **78.45%** (+0.47% vs baseline).

---

### Phase 6H: GLCM Spatial Texture & S/T/P Feature Ablation

**Reports:** [`outputs/reports/phase6h/phase6h_stage3_benchmark_report.md`](file:///C:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/reports/phase6h/phase6h_stage3_benchmark_report.md) | [`outputs/reports/phase6h/phase6h_weighting_boundary_report.md`](file:///C:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/reports/phase6h/phase6h_weighting_boundary_report.md)

#### Stage 2: GLCM Spatial Texture Extraction
**Module:** [`src/features/spatial_texture.py`](file:///C:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/src/features/spatial_texture.py)  
**Report:** [`outputs/reports/phase6h/phase6h_stage2_extraction_report.md`](file:///C:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/reports/phase6h/phase6h_stage2_extraction_report.md)

- **Kernel:** GLCM 3×3 moving window on Sentinel-2 B8 (NIR) raster at 10m native resolution.
- **Physical kernel footprint:** 30m × 30m = 900 m².
- **5 GLCM properties × 4 windows = 20 spatial texture features:** Contrast, Dissimilarity, Homogeneity, Energy, Correlation × {early, mid, late, full}.
- **Extraction coverage:** 7,843/7,843 farms (100%). `full` window: 100.0% finite; `early` window: 93.94% finite (matching monsoon cloud gap pattern).
- **Sample GLCM statistics:** `glcm_contrast_full` mean=0.52; `glcm_homogeneity_full` mean=0.81; `glcm_correlation_full` mean=0.45.

#### Stage 3: 7-Experiment S/T/P Feature Ablation (Unweighted, N=7,843)

| Experiment | Feature Description | Count | Pooled Macro-F1 | Overall Acc | Fold Mean ± Std | Runtime |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **S** | Spectral-only full-season | 34 | 0.6227 | 0.7365 | 0.6212 ± 0.0322 | 27.6s |
| **T** | Temporal-only (3 windows, 4 deltas) | 106 | 0.6948 | 0.7815 | 0.6851 ± 0.0194 | 111.3s |
| **P** | Spatial GLCM texture-only | 20 | 0.2945 | 0.3899 | 0.2905 ± 0.0204 | 44.0s |
| **S+T** | Spectral + Temporal | 140 | **0.7069** | 0.7873 | 0.6997 ± 0.0228 | 155.0s |
| **S+P** | Spectral + Spatial | 54 | 0.6282 | 0.7379 | 0.6250 ± 0.0334 | 57.8s |
| **T+P** | Temporal + Spatial | 126 | 0.6935 | 0.7820 | 0.6853 ± 0.0143 | 125.6s |
| **S+T+P** | Full fusion | 160 | **0.7038** | 0.7900 | 0.6965 ± 0.0173 | 164.8s |

**Key finding:** S+T (140 feats) **beats** S+T+P (160 feats) by 0.0031 Macro-F1 — spatial texture degrades minority crops.

#### Boundary Contamination Analysis (from Phase 6H audit report)
The GLCM 3×3 kernel has a 30m × 30m (900 m²) physical footprint in a smallholder landscape:

| Metric | Value |
|:---|:---:|
| Parcels smaller than GLCM kernel footprint (<900 m²) | 1,353 (17.25%) |
| Parcels smaller than 2× kernel footprint (<1,800 m²) | 2,748 (35.04%) |
| Parcels with equivalent radius ≤ 15m (100% boundary overlap) | 1,024 (13.06%) |
| Mean estimated boundary contamination fraction | **64.9%** |

**Per-crop boundary contamination risk:**
| Crop | Median Area (m²) | % Parcels < 900 m² | Mean Edge Fraction |
|:---|:---:|:---:|:---:|
| Paddy | 1,374.9 | 30.62% | 81.2% |
| Chillies | 3,080.1 | 23.46% | 67.3% |
| Maize | 3,008.0 | 22.53% | 63.6% |
| Groundnut | 3,557.1 | 10.29% | 59.7% |
| Cotton | 4,166.8 | 5.48% | 54.4% |
| Red Gram | 4,159.1 | 1.81% | 51.0% |

The GLCM kernel frequently bleeds edge pixels from adjoining bare soil, trees, or neighboring crops, injecting noise that disproportionately degrades minority crops.

#### Weighted Benchmark (4-Model Matrix with Inverse-Frequency Weighting)
| Model | Features | Pooled Macro-F1 | Balanced Acc | Overall Acc | Red Gram F1 | Chillies F1 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| S+T Unweighted | 140 | 0.7069 | 0.6914 | 0.7873 | 0.3683 | 0.5062 |
| **S+T Weighted** | **140** | **0.7226** | **0.7180** | **0.7881** | **0.4420** | **0.5343** |
| S+T+P Unweighted | 160 | 0.7038 | 0.6895 | 0.7900 | 0.3333 | 0.4930 |
| S+T+P Weighted | 160 | 0.7228 | 0.7167 | 0.7905 | 0.4317 | 0.5217 |

---

### Phase 6I: Robustness Validation & Final Model Selection

**Report:** [`outputs/reports/phase6i/phase6i_model_selection_report.md`](file:///C:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/reports/phase6i/phase6i_model_selection_report.md)  
**Verification Log:** [`outputs/reports/phase6i/phase6i_robustness_audit.json`](file:///C:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/reports/phase6i/phase6i_robustness_audit.json)

#### Four-Way Benchmark Matrix (N = 7,843, 5-Fold Mandal Spatial CV)
| Model | Features | Weighting | Pooled Macro-F1 | Balanced Acc | Overall Acc | Weighted-F1 | Fold Mean ± Std | Status |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Unweighted S+T | 140 | None | 0.7069 | 0.6914 | 0.7873 | 0.7796 | 0.6997 ± 0.0228 | Reference |
| **Weighted S+T** | **140** | **Inv-Freq** | **0.7226** | **0.7180** | **0.7881** | **0.7857** | **0.7117 ± 0.0181** | 🏆 **WINNER** |
| Unweighted S+T+P | 160 | None | 0.7038 | 0.6895 | 0.7900 | 0.7818 | 0.6965 ± 0.0173 | Ablation Ref |
| Weighted S+T+P | 160 | Inv-Freq | 0.7228 | 0.7167 | 0.7905 | 0.7879 | 0.7131 ± 0.0133 | Secondary Ref |

#### Paired Statistical Comparison: Weighted S+T vs Weighted S+T+P
| Fold | Val N | Weighted S+T Macro-F1 | Weighted S+T+P Macro-F1 | Delta (S+T+P − S+T) |
|:---:|:---:|:---:|:---:|:---:|
| Fold 0 | 1,508 | 0.7269 | 0.7203 | **-0.0066** |
| Fold 1 | 1,600 | 0.7381 | 0.7357 | **-0.0024** |
| Fold 2 | 1,595 | 0.6919 | 0.7048 | **+0.0129** |
| Fold 3 | 1,530 | 0.6945 | 0.7064 | **+0.0119** |
| Fold 4 | 1,610 | 0.7073 | 0.6985 | **-0.0088** |

**Hypothesis Tests:**
- **Mandal Cluster Bootstrap** (B=1,000, resampling unit = Sub-District): Mean delta = **+0.00094**, SE = 0.00397, **95% CI: [-0.00663, +0.00917]**, **p = 0.8600**.
- **Paired t-test (5 folds):** t = 0.3037, **p = 0.7765**.
- **Conclusion:** The +0.0002 Macro-F1 difference is centered at zero and within random evaluation noise.

#### Final Decision — Weighted S+T (140 Features) Selected via Occam's Razor
1. **Statistical Parity:** +0.0002 difference is statistically indistinguishable from zero (p=0.8600).
2. **Minority Crop Protection:** Red Gram F1 **0.4420 > 0.4317** and Chillies F1 **0.5343 > 0.5217** — Weighted S+T better protects vulnerable minority crops.
3. **Parsimony:** Eliminates 20 GLCM texture rasters — halves computation latency, reduces network payload and memory footprint.
4. **Boundary Robustness:** Avoids texture edge-pixel bleeding in smallholder parcels (<1,250 m²) where GLCM kernels exceed parcel interior radii.
5. **Parcel-Area Subgroup Analysis:** Weighted S+T wins or ties in 3/4 area quartiles. S+T+P only advantageous in Q3 medium-large farms (2,784–5,517 m²): Macro-F1 0.7530 vs 0.7400.

#### 🏆 Final Per-Class Results (Primary Weighted S+T Model, N=7,843)
| Crop Class | Support | Precision | Recall | **F1-Score** | Key Error Pattern |
|:---|:---:|:---:|:---:|:---:|:---|
| **Paddy** | 2,260 | 0.8664 | 0.8752 | **0.8708** | Well-separated; 91 confused with Chillies |
| **Groundnut** | 768 | 0.8597 | 0.8698 | **0.8647** | Highly distinctive spectral profile |
| **Bengal Gram** | 938 | 0.7916 | 0.8667 | **0.8275** | Robust Rabi phenological separation |
| **Cotton** | 1,679 | 0.7939 | 0.8261 | **0.8097** | Strong long-season vegetative trajectory |
| **Maize** | 1,376 | 0.7492 | 0.6730 | **0.7090** | 134 confused with Bengal Gram; 126 with Paddy |
| **Chillies** | 601 | 0.5249 | 0.5441 | **0.5343** | 91 confused with Paddy; 82 with Cotton |
| **Red Gram** | 221 | 0.5467 | 0.3710 | **0.4420** | 76 confused with Cotton; 29 with Bengal Gram |
| **Macro Average** | **7,843** | **0.7332** | **0.7180** | **`0.7226`** | Balanced, minority-sensitive evaluation |

**Confusion Matrix (Weighted S+T, OOF Pooled N=7,843):**
```
Predicted ->    Bengal  Chilli  Cotton  Ground   Maize   Paddy  Red Gr
Bengal Gram  [    813       9      24       2      73       4      13 ]
Chillies     [      3     327      82      23      70      91       5 ]
Cotton       [     39      81    1387      37      49      56      30 ]
Groundnut    [      0      29      36     668      13      20       2 ]
Maize        [    134      81      71      23     926     126      15 ]
Paddy        [      9      91      71      21      87    1978       3 ]
Red Gram     [     29       5      76       3      18       8      82 ]
```

---

## 🔬 Scientific Integrity & Resolved Discrepancy Register

From [`outputs/reports/project_evidence_and_discrepancies.md`](file:///C:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/reports/project_evidence_and_discrepancies.md):

| ID | Issue | Canonical Value | Resolution |
|:---|:---|:---:|:---|
| **DISC-01** | Exp A Macro-F1 discrepancy (0.6174 vs 0.6483 in Phase 6H audit doc) | **0.6174** | 0.6483 was a copy-paste transcription error in `phase6h_stage1_audit.md`. Traced to `phase6f_stage4_benchmark_report.json`. |
| **DISC-02** | Exp D RF accuracy uncertainty (0.7493? vs 0.7959) | **0.7959** | 0.7493 was Exp C RF accuracy erroneously duplicated. Actual Exp D RF: Acc=0.7959, Macro-F1=0.7422. |
| **DISC-03** | SampleSubmission.csv 3,016 rows vs Test.csv 2,718 rows | **2,718 real** | 298 synthetic anti-probing decoy IDs in submission template (Zindi/Kaggle competition standard). |
| **DISC-04** | Competition targets `category`; project targets `Crop` | **`Crop` (7-class)** | Crop-type identification is the prerequisite for health assessment. Health modeling is a planned secondary phase. |
| **DISC-05** | Tabular baseline 0.9030 Macro-F1 vs satellite models 0.70–0.79 | **0.9030 inflated** | Calendar leakage + spatial autocorrelation. Top features are geographic coordinates. Not satellite generalization. |
| **DISC-06** | 7,888 survey farms vs 7,843 benchmark cohort | **7,843 eligible** | 45 farms irrevocably cloud-obscured (100% NaN). Phase 6F Stage 2 recovered 5 additional farms. |
| **DISC-07** | Phase 6G Fold 0 Macro-F1 0.7339 vs Phase 6I pooled 0.7226 | **0.7226 authoritative** | Phase 6G was Outer Fold 0 only (exploratory pilot). Full 5-fold evaluation in Phase 6I gives 0.7226. |
| **DISC-08** | S+T+P expected to outperform S+T; actual S+T+P=0.7228 vs S+T=0.7226 | **Not significant (p=0.8600)** | GLCM 900m² kernel contaminates 35% of smallholder parcels <1,800m². Texture hurts minority crops. |

---

## 🔒 Leakage Prevention Architecture

From [`outputs/reports/project_evidence_and_discrepancies.md`](file:///C:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/reports/project_evidence_and_discrepancies.md) (Section 4):

1. **Target and Identity Leakage:** Strictly excluded from all model inputs: `FarmID`, `category`, `Crop`, `ExpYield`, `CHeight`, `CropCoveredArea`, `CTransp`, `CLast`, `CNext`, `duration_days`, `HDate`, `State`, `District`, `Sub-District`.

2. **Spatial Autocorrelation Leakage:** 5-Fold `StratifiedGroupKFold` grouped by `Sub-District` (Mandal). **Zero Mandal overlap across all 5 folds** (verified by `tests/test_phase6i_robustness.py` and `scripts/audit_phase6i_robustness.py`). No coordinates or bounding box features in satellite model inputs.

3. **Class Weighting Leakage:** Inverse-frequency weights $w_c = N_{\text{train}} / (K \times N_{c,\text{train}})$ computed strictly inside fold-internal training partitions. Validation fold labels never accessed during weight calculation.

4. **Preprocessing and Imputation Leakage:** `SimpleImputer(strategy='median')` for RF models fitted strictly on training folds inside `sklearn.Pipeline`. XGBoost models use native sparsity-aware default direction routing — zero imputation artifacts.

---

## ⚠️ Status of Unimplemented Modules & Unprocessed Datasets

| Component | Status | Notes |
|:---|:---:|:---|
| **`datasss/Test.csv` — Satellite Feature Extraction** | ❌ NOT PROCESSED | 2,718 parcels have not been submitted to GEE for extraction. No inference executed. |
| **Crop Health Classification (`category`)** | ❌ NOT IMPLEMENTED | No `Healthy/Pests/Diseased/Stressed` model trained. Feature set not designed. |
| **Weather & Agro-Meteorological Integration** | ❌ NOT IMPLEMENTED | ERA5-Land, CHIRPS, or station data not ingested. |
| **FAO-56 Penman-Monteith ET₀ & CWSI** | ❌ NOT IMPLEMENTED | Crop water stress and reference evapotranspiration algorithms not coded. |
| **Prescriptive Irrigation Advisory Engine** | ❌ NOT IMPLEMENTED | Urgency scoring, deficit volume ($m^3/\text{ha}$), schedule generator not built. |
| **Sentinel-1 SAR Integration** | ❌ NOT IMPLEMENTED | VV/VH backscatter, RVI not extracted. |
| **Streamlit Dashboard** | ❌ NOT IMPLEMENTED | No web interface exists. |
| **LangChain Advisory Agents** | ❌ NOT IMPLEMENTED | No LLM integration built. |

---

## 💻 Execution & Verification Guide

### Environment Setup
```powershell
cd "c:\Users\Manichander\OneDrive\Desktop\cbit\major project\major_proj_final"
pip install -r requirements.txt
pip install folium xgboost lightgbm shap pyproj earthengine-api scikit-image
```

### Run Phase 1: Data Quality Audit
```powershell
python scripts/analyze_dataset.py
# → outputs/reports/data_quality_report.txt (.json)
# → outputs/figures/ (7 diagnostic figures)
```

### Run Phase 2: Geometry Repair & Metric Calculation
```powershell
python scripts/validate_geometries.py
# → outputs/data/train_enriched_geometry.csv (7,888 × 32)
# → outputs/data/test_enriched_geometry.csv (2,718 × 31)
# → outputs/reports/geometry_validation_report.txt (.json)
```

### Run Phase 3: Spatial Visualisation & Interactive Map
```powershell
python scripts/visualize_spatial.py
# → outputs/figures/telangana_farms_interactive_map.html (3.59 MB)
# → outputs/figures/telangana_spatial_overview.png
```

### Run Phase 6I Audit & Statistical Verification
```powershell
python scripts/audit_phase6i_robustness.py
# → Verifies 24 protected baseline SHA-256 hashes
# → Reproduces pooled Macro-F1 = 0.7226 from OOF predictions
# → Runs Mandal cluster bootstrap (p=0.8600)
# → Zero Mandal leakage confirmed across all 5 folds
```

### Run Targeted Unit & Regression Test Suite (83 tests, no GEE required)
```powershell
python -m unittest tests/test_baseline.py tests/test_phase6c.py ^
    tests/test_phase6d_audit.py tests/test_phase6f_stage1_audit.py ^
    tests/test_phase6f_stage2_recovery.py tests/test_phase6f_stage3_harmonization.py ^
    tests/test_phase6f_stage4_benchmark.py tests/test_phase6f_stage4c_audit.py ^
    tests/test_phase6g_exp2_imbalance.py tests/test_phase6g_pilot.py ^
    tests/test_phase6h_stage3_benchmark.py tests/test_phase6i_robustness.py
# → 83 tests, all PASS, ~4-5 seconds
```

---

## 📑 Complete Artifacts & Reports Index

### Primary Decision Reports
| Report | Path | Description |
|:---|:---|:---|
| **Phase 6I Model Selection** | [`outputs/reports/phase6i/phase6i_model_selection_report.md`](file:///C:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/reports/phase6i/phase6i_model_selection_report.md) | **Authoritative**: Weighted S+T selected; 4-way benchmark; paired tests; per-class breakdown |
| **Phase 6I Robustness Audit** | [`outputs/reports/phase6i/phase6i_robustness_audit.json`](file:///C:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/reports/phase6i/phase6i_robustness_audit.json) | Reproducibility log; bootstrap CI; SHA-256 verification |
| **Methodology Complete** | [`outputs/reports/project_methodology_complete.md`](file:///C:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/reports/project_methodology_complete.md) | 70KB comprehensive methodology, Q&A, and experiment registry |
| **Discrepancy Register** | [`outputs/reports/project_evidence_and_discrepancies.md`](file:///C:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/reports/project_evidence_and_discrepancies.md) | 8-item resolved discrepancy ledger with root causes and canonical values |

### Phase Reports
| Report | Path | Key Content |
|:---|:---|:---|
| Data Quality Audit | [`outputs/reports/data_quality_report.txt`](file:///C:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/reports/data_quality_report.txt) | 0 missing, 0 duplicates, 0 inversions |
| Geometry Validation | [`outputs/reports/geometry_validation_report.txt`](file:///C:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/reports/geometry_validation_report.txt) | 100% validity; area/perimeter/compactness stats |
| Phase 6D Research Baseline | [`outputs/reports/phase6d_research_baseline_report.md`](file:///C:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/reports/phase6d_research_baseline_report.md) | 714-farm verification; fold performance; fold 3&4 degradation analysis |
| Phase 6E Launch Readiness | [`outputs/reports/phase6e_launch_readiness.md`](file:///C:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/reports/phase6e_launch_readiness.md) | Gate report; 7,164-farm campaign plan; throughput measurements |
| Phase 6F Cohort Harmonization | [`outputs/reports/phase6f_stage3_harmonization_report.md`](file:///C:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/reports/phase6f_stage3_harmonization_report.md) | Final 7,843-farm cohort; window completeness; SHA-256 |
| Phase 6F Benchmark | [`outputs/reports/phase6f_stage4_benchmark_report.md`](file:///C:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/reports/phase6f_stage4_benchmark_report.md) | All 15 model-run results; 4 experiments; research Q&A |
| Phase 6F Integrity Audit | [`outputs/reports/phase6f_stage4c_integrity_audit.md`](file:///C:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/reports/phase6f_stage4c_integrity_audit.md) | PASS; 6 scientific addenda; Red Gram true variance documented |
| Feature Schema Manifest | [`outputs/reports/phase6f_feature_schema_manifest.json`](file:///C:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/reports/phase6f_feature_schema_manifest.json) | SHA256-verified; 165-column schema for all 4 experiments |
| Phase 6G Pilot Report | [`outputs/reports/phase6g/phase6g_pilot_fold0_report.md`](file:///C:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/reports/phase6g/phase6g_pilot_fold0_report.md) | 10-trial optimization ledger; Trial 4 selected |
| Phase 6G Weighting Report | [`outputs/reports/phase6g/phase6g_exp2_imbalance_report.md`](file:///C:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/reports/phase6g/phase6g_exp2_imbalance_report.md) | 7-strategy weighting ledger; Red Gram recovery (+0.1627 F1) |
| Phase 6H Texture Extraction | [`outputs/reports/phase6h/phase6h_stage2_extraction_report.md`](file:///C:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/reports/phase6h/phase6h_stage2_extraction_report.md) | 20 GLCM features; 100% coverage; finite rate statistics |
| Phase 6H S/T/P Ablation | [`outputs/reports/phase6h/phase6h_stage3_benchmark_report.md`](file:///C:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/reports/phase6h/phase6h_stage3_benchmark_report.md) | 7-experiment ablation; S+T wins unweighted; texture hurts minority |
| Phase 6H Boundary Analysis | [`outputs/reports/phase6h/phase6h_weighting_boundary_report.md`](file:///C:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/reports/phase6h/phase6h_weighting_boundary_report.md) | 4-model weighted matrix; parcel-area quartile analysis; 64.9% contamination |
| Archived Tabular Baseline | [`outputs/reports/baseline_rf_metrics.json`](file:///C:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/reports/baseline_rf_metrics.json) | 0.9030 inflated metric; coordinate feature importances |

### Key Data Artifacts
| Artifact | Path | Dimensions |
|:---|:---|:---|
| Authoritative CV Folds | [`outputs/data/phase6f_cv_folds.csv`](file:///C:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/data/phase6f_cv_folds.csv) | 7,843 × 5-fold assignments |
| Full Feature Matrix | `outputs/data/s2_phase6f_features.csv` | 7,888 × 165 columns (SHA-256 verified) |
| Harmonized Raw Bands | `outputs/data/s2_phase6f_harmonized_cohort.csv` | 7,888 × 100 (SHA-256 verified) |
| Phase 6H OOF Predictions | `outputs/data/phase6h/phase6h_weighting_boundary_oof_predictions.csv` | 7,843 × pred columns |
| Enriched Train Data | [`outputs/data/train_enriched_geometry.csv`](file:///C:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/data/train_enriched_geometry.csv) | 7,888 × 32 |
| Enriched Test Data | [`outputs/data/test_enriched_geometry.csv`](file:///C:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/data/test_enriched_geometry.csv) | 2,718 × 31 |
| Interactive Farm Map | [`outputs/figures/telangana_farms_interactive_map.html`](file:///C:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/figures/telangana_farms_interactive_map.html) | 3.59 MB |

---

## 📚 References & Standards

- **Sentinel-2 MSI Level-2A:** European Space Agency (ESA) Copernicus Open Access Hub. `COPERNICUS/S2_SR_HARMONIZED`.
- **Spectral Indices:**
  - Rouse, J. W., et al. (1974). Monitoring vegetation systems in the Great Plains with ERTS. (NDVI).
  - Huete, A., et al. (2002). Overview of the radiometric and biophysical performance of the MODIS vegetation indices. (EVI).
  - Gao, B. C. (1996). NDWI — A normalized difference water index for remote sensing of vegetation liquid water from space. (NDMI).
  - Gitelson, A. A., & Merzlyak, M. N. (1996). Signature analysis of leaf reflectance spectra. (GNDVI).
  - Barnes, E. M., et al. (2000). Coincident detection of crop water stress, nitrogen status and canopy density. (NDRE).
- **Gray-Level Co-occurrence Matrix:** Haralick, R. M., Shanmugam, K., & Dinstein, I. H. (1973). Textural features for image classification.
- **Spatial Cross-Validation:** Roberts, D. R., et al. (2017). Cross-validation strategies for data with spatial, temporal, or phylogenetic structure. *Ecography*, 40(8), 913-929.
- **Coordinate Reference Systems:** EPSG:4326 (WGS 84 Geographic) and EPSG:32644 (WGS 84 / UTM Zone 44N Metric Projection).
- **FAO-56 Penman-Monteith:** Allen, R. G., et al. (1998). Crop evapotranspiration — Guidelines for computing crop water requirements. FAO Irrigation and Drainage Paper 56. *(Planned: Phase 10)*.
