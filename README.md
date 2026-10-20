# AI-Based Multi-Temporal Remote Sensing for Crop Monitoring and Irrigation Decision Support Using Sentinel-2 Imagery

An enterprise-grade, reproducible remote sensing and machine learning pipeline for agricultural monitoring, crop health diagnosis, and irrigation decision support across 10,606 farms in Telangana, India.

---

## 📌 Table of Contents
1. [Project Overview](#-project-overview)
2. [Dataset Summary](#-dataset-summary)
3. [System Architecture](#-system-architecture)
4. [Project Structure](#-project-structure)
5. [Implementation Roadmap & Status](#-implementation-roadmap--status)
6. [Detailed Phase-by-Phase Walkthrough of Completed Work](#-detailed-phase-by-phase-walkthrough-of-completed-work)
   - [Phase 1: Dataset Analysis, Validation & Exploratory Data Analysis](#phase-1-dataset-analysis-validation--exploratory-data-analysis)
   - [Phase 2: Farm Geometry Parsing, Topological Repair & Feature Engineering](#phase-2-farm-geometry-parsing-topological-repair--feature-engineering)
   - [Phase 3: Farm Spatial Visualisation & Cartography](#phase-3-farm-spatial-visualisation--cartography)
   - [Phase 4: Sentinel-2 Multi-Temporal Remote Sensing Extraction Engine](#phase-4-sentinel-2-multi-temporal-remote-sensing-extraction-engine)
   - [Phase 5: Spectral, Moisture & Vegetation Indices Engineering](#phase-5-spectral-moisture--vegetation-indices-engineering)
7. [Step-by-Step Execution Guide](#-step-by-step-execution-guide)
8. [Summary of Generated Artifacts & Reports](#-summary-of-generated-artifacts--reports)
9. [Next Steps (Phases 6–13)](#-next-steps-phases-613)

---

## 📖 Project Overview
This project combines **multi-temporal satellite remote sensing (Sentinel-2 MSI, and Sentinel-1 C-band SAR)** with advanced **gradient boosting and tree-based machine learning (LightGBM, XGBoost, Random Forest)**, crop phenology modeling, and automated irrigation advisory logic.

- **Primary Goal:** Accurately classify crop type (`Paddy`, `Cotton`, `Maize`, `Bengal Gram`, `Groundnut`, `Chillies`, `Red Gram`) using multi-temporal Sentinel-2 remote sensing and agronomic metadata, track phenological vegetation indices across growth stages, assess crop health, and provide data-driven irrigation decision support for smallholder farms.
- **Geographic Focus:** State of Telangana, India (14 districts including Adilabad, Nalgonda, Mahabubabad, Nirmal, Medak, Mahabubnagar, etc.).
- **Data Source:** Telangana Crop Health Challenge (GIZ-ADEX / Data Foundation).

---

## 📊 Dataset Summary
- **Training Set:** `7,888` farm records with ground-truth labels and field boundaries.
- **Testing Set:** `2,718` farm records.
- **Primary Target Variable (`Crop` — 7 Crop Classes):**
  - `Paddy`: 2,272 farms (28.8%)
  - `Cotton`: 1,697 farms (21.5%)
  - `Maize`: 1,382 farms (17.5%)
  - `Bengal Gram`: 936 farms (11.9%)
  - `Groundnut`: 763 farms (9.7%)
  - `Chillies`: 616 farms (7.8%)
  - `Red Gram`: 222 farms (2.8%)
- **Secondary Target Variable (`category` — Crop Health Status):**
  - `Healthy`: 6,506 farms (82.5%)
  - `Pests`: 485 farms (6.1%)
  - `Diseased`: 474 farms (6.0%)
  - `Stressed`: 423 farms (5.4%)
- **Crop Seasons:** Rabi (78.3%), Kharif (21.7%).
- **Irrigation Types:** Drip (36.4%), Sprinkler (28.9%), Flood (28.8%), Surface (5.9%).
- **Irrigation Sources:** Groundwater (42.3%), Canal (40.4%), Rainfed (15.9%), Well (0.7%), Tubewell (0.7%).
- **Temporal Horizon:** Sowing from `2023-01-05` to `2024-09-28`; Harvest up to `2024-12-06`.

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
│  - Data Cleaning & 0-Leakage Verification (0 missing, 0 duplicates)                    │
│  - Topological Repair: shapely.make_valid (100% validity achieved)                     │
│  - Projected CRS: EPSG:4326 (WGS84) ──► EPSG:32644 (UTM 44N Metric)                   │
│  - Metric Features: Area (ha/m²), Perimeter, Polsby-Popper Compactness, Centroids      │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                 PHASE 3 & 4: SPATIAL MAPPING & SATELLITE ENGINE                       │
│  - Interactive Folium Cartography (Satellite/Street, Polygons, Heatmaps, Clusters)    │
│  - Sentinel-2 Multi-Temporal Query Engine (COPERNICUS/S2_SR_HARMONIZED)               │
│  - Cloud Masking (SCL & QA60 Bitmasks)                                                 │
│  - Phenological Partitioning: Early (0-25%), Mid (25-75%), Late (75-100%)              │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                   PHASE 5: SPECTRAL & VEGETATION INDICES SUITE                         │
│  - 14 Indices: NDVI, EVI, GNDVI, SAVI, MSAVI, NDMI, NDWI, NDRE1, NDRE2, PSRI, BSI...   │
│  - Growth & Senescence Dynamics (ΔNDVI_growth, ΔNDVI_senescence, ΔNDMI_loss)           │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                          PHASE 6 - 13: ML & DECISION SUPPORT                           │
│  - Class-Imbalance Aware Modeling (LightGBM, XGBoost, Stratified K-Fold)               │
│  - Crop Water Stress (NDMI/MSI) & FAO-56 Penman-Monteith Evapotranspiration            │
│  - Irrigation Advisory Engine (Urgency, Quantity m³/ha, Schedule)                      │
│  - Interactive Streamlit Dashboard & LangChain Advisory Agents                         │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
major_proj_final/
├── config/
│   └── project_config.yaml         ← Central pipeline configuration (CRS, bands, ML params)
├── datasss/                        ← Immutable raw competition data
│   ├── Train.csv                   ← 7,888 train records with target category
│   ├── Test.csv                    ← 2,718 test records
│   ├── SampleSubmission.csv        ← Submission format
│   ├── Starter_Notebook.ipynb      ← Official benchmark notebook
│   └── VariableDefinitions.docx    ← Field definitions
├── notebooks/                      ← Production Jupyter analysis notebooks
│   ├── 01_dataset_analysis.ipynb   ← Phase 1 comprehensive data quality & distributions
│   └── 02_geometry_and_spatial_analysis.ipynb ← Phase 2 & 3 geometry repair & spatial mapping
├── outputs/
│   ├── data/                       ← Enriched datasets with computed features
│   │   ├── train_enriched_geometry.csv  ← 7,888 rows × 32 columns with metric geometry
│   │   └── test_enriched_geometry.csv   ← 2,718 rows × 31 columns with metric geometry
│   ├── figures/                    ← Generated diagnostic plots & interactive maps
│   │   ├── crop_and_category_distribution.png
│   │   ├── date_range_analysis.png
│   │   ├── district_crop_heatmaps.png
│   │   ├── farm_area_and_centroids.png
│   │   ├── farm_boundaries_map.png
│   │   ├── geometry_diagnostics.png
│   │   ├── irrigation_analysis.png
│   │   ├── numeric_feature_distributions.png
│   │   ├── telangana_farms_interactive_map.html  ← 3.59 MB interactive Folium web map
│   │   └── telangana_spatial_overview.png        ← 4-panel high-res overview map
│   ├── models/                     ← Model weights and checkpoints
│   └── reports/                    ← Formal audit reports
│       ├── data_quality_report.json
│       ├── data_quality_report.txt
│       ├── geometry_validation_report.json
│       └── geometry_validation_report.txt
├── scripts/                        ← Executable command-line pipelines
│   ├── analyze_dataset.py          ← Phase 1 analysis script
│   ├── validate_geometries.py      ← Phase 2 geometry repair & metric calculation
│   └── visualize_spatial.py        ← Phase 3 interactive & static spatial visualizer
├── src/                            ← Reusable Python package modules
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── load_data.py            ← Safe loaders for train/test with type casting
│   │   └── validate_data.py        ← Schema and integrity validator
│   ├── features/
│   │   ├── __init__.py
│   │   └── indices.py              ← 14 vegetation, moisture & red-edge indices
│   ├── geospatial/
│   │   ├── __init__.py
│   │   ├── geometry_utils.py       ← WKT parsing, repair, metric projection, compactness
│   │   └── spatial_viz.py          ← Folium interactive map & matplotlib cartography
│   ├── satellite/
│   │   ├── __init__.py
│   │   └── s2_extractor.py         ← GEE Sentinel-2 multi-temporal extractor
│   ├── models/                     ← ML training & evaluation
│   ├── phenology/                  ← Time-series phenological stage detection
│   ├── water/                      ← Water requirement & irrigation logic
│   └── agents/                     ← LangChain decision support agents
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
| **6** | **Crop Classification Modeling (RF, XGBoost, LightGBM)** | 🔜 *Next* | Multi-class Stratified K-Fold (7 crops), Macro-F1 / Accuracy optimization, SHAP feature importance *(Phase 6b: Secondary crop health module)* |
| **7** | **Temporal Monitoring & Phenology Curves** | 🔜 *Planned* | NDVI/EVI curve smoothing, double-logistic fitting, phenology metrics |
| **8** | **Sentinel-1 SAR C-band Integration** | 🔜 *Planned* | VV/VH backscatter, Dual-Pol Radar Vegetation Index (RVI), cloud-free radar metrics |
| **9** | **Moisture & Water-Stress Assessment** | 🔜 *Planned* | NDMI / MSI anomaly tracking, root-zone moisture deficit mapping |
| **10** | **Water-Requirement Estimation** | 🔜 *Planned* | FAO-56 Penman-Monteith reference ET0 and crop-specific coefficient (Kc) |
| **11** | **Irrigation Decision-Support Engine** | 🔜 *Planned* | Rule-based irrigation urgency score, volume estimation ($m^3/\text{ha}$), schedule generator |
| **12** | **Streamlit Interactive Dashboard** | 🔜 *Planned* | Multi-page web dashboard for farm inspection, map explorer, and advisory output |
| **13** | **LangChain Explanation & Support Agents** | 🔜 *Planned* | LLM agent providing agronomic recommendations and farmer-friendly advice |

---

## 🔍 Detailed Phase-by-Phase Walkthrough of Completed Work

### Phase 1: Dataset Analysis, Validation & Exploratory Data Analysis

#### Step 1.1: Raw Schema Audit & Data Ingestion
- Inspected `datasss/Train.csv` (7,888 rows × 20 columns) and `datasss/Test.csv` (2,718 rows × 19 columns).
- Confirmed column presence and data types:
  - Identifiers: `FarmID` (Unique integer).
  - Spatial attributes: `State` ("Telangana"), `District` (14 districts), `Sub-District` (116 sub-districts), `geometry` (WKT POLYGON string).
  - Agronomic timeline: `SDate` (Sowing date, DD-MM-YYYY), `HDate` (Harvest date, DD-MM-YYYY), `Season` (Rabi, Kharif).
  - Field metrics: `CropCoveredArea` (40%–100%), `CHeight` (20–250 cm), `ExpYield` (5–30 quintals/acre).
  - Irrigation parameters: `IrriType` (Drip, Sprinkler, Flood, Surface), `IrriSource` (Groundwater, Canal, Rainfed, Well, Tubewell), `IrriCount` (1–6 rounds), `WaterCov` (20%–100%).
  - Succession crops: `CNext`, `CLast`, `CTransp`.
  - Target: `category` (`Healthy`, `Diseased`, `Pests`, `Stressed`) — present only in Train.

#### Step 1.2: Integrity & Quality Verification
- **Missing Values:** Exactly **0 missing values** across all columns in both Train and Test.
- **Duplicate Farms:** Exactly **0 duplicate `FarmID` values** in Train and Test.
- **Train-Test Leakage:** Exactly **0 overlapping `FarmID` values** between Train and Test.
- **Temporal Consistency:** Sowing always strictly precedes Harvest (`HDate > SDate` for all 10,606 farms).
  - Average crop duration: **193 days** in Train (1 to 693 days), **191 days** in Test (1 to 691 days).

#### Step 1.3: Statistical Findings & Imbalance Analysis
- **Target Distribution:** Extreme class imbalance requiring macro-F1 evaluation and class-weighted loss:
  - `Healthy`: $82.48\%$ ($6,506$ farms)
  - `Pests`: $6.15\%$ ($485$ farms)
  - `Diseased`: $6.01\%$ ($474$ farms)
  - `Stressed`: $5.36\%$ ($423$ farms)
- **Top Districts:** Adilabad ($1,770$), Nalgonda ($1,386$), Mahabubabad ($1,314$), Nirmal ($1,244$), Medak ($1,210$), Mahabubnagar ($655$).
- **Irrigation Breakdown:**
  - Water sources: Groundwater ($42.3\%$), Canal ($40.4\%$), Rainfed ($15.9\%$), Well ($0.7\%$), Tubewell ($0.7\%$).
  - Systems: Drip ($36.4\%$), Sprinkler ($28.9\%$), Flood ($28.8\%$), Surface ($5.9\%$).
  - Mean irrigation rounds: $3.54$ ($\text{std} = 1.41$, $\min=1$, $\max=6$).

---

### Phase 2: Farm Geometry Parsing, Topological Repair & Feature Engineering

#### Step 2.1: Safe WKT Geometry Ingestion
- Created [`src/geospatial/geometry_utils.py`](file:///C:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/src/geospatial/geometry_utils.py) with safe `shapely.wkt.loads` parsers handling any malformed strings.
- Geometry coordinate reference system confirmed as **EPSG:4326 (WGS84)** in decimal degrees longitude and latitude.

#### Step 2.2: Topological Anomaly Detection & Automated Repair
- An exhaustive topological check (`geom.is_valid`) revealed:
  - `Train`: **1 invalid polygon** (FarmID `1330965` at index 4389 had a self-intersecting boundary ring).
  - `Test`: **1 invalid polygon** (FarmID `1326463` had a self-touching boundary anomaly).
- Implemented `repair_invalid_geometries()` using `shapely.make_valid()`.
- Result: **100% geometry validity achieved across all 7,888 Train and 2,718 Test farms**.

#### Step 2.3: Metric Reprojection & High-Precision Geometric Calculations
- Standard geographic degrees ($\Delta^\circ$) cannot accurately compute metric area or perimeter without distortion.
- Defined **EPSG:32644 (WGS84 / UTM Zone 44N)** as the official metric projected CRS suitable for local area and distance calculations covering Telangana ($78^\circ\text{E}$ to $84^\circ\text{E}$).
- Engineered 12 spatial feature columns:
  1. `area_m2`: Exact polygon area in square metres ($m^2$).
  2. `area_ha`: Polygon area in hectares ($1\text{ ha} = 10,000\text{ m}^2$).
     - Train: $\min=0.0054\text{ ha}$, $\text{mean}=0.4231\text{ ha}$ ($4,231\text{ m}^2$), $\text{median}=0.2779\text{ ha}$, $\max=6.9491\text{ ha}$.
     - Test: $\min=0.0069\text{ ha}$, $\text{mean}=0.3612\text{ ha}$ ($3,612\text{ m}^2$), $\text{median}=0.2610\text{ ha}$, $\max=4.8004\text{ ha}$.
  3. `perimeter_m`: Boundary perimeter in metres.
     - Train: $\min=29.7\text{ m}$, $\text{mean}=249.9\text{ m}$, $\max=1,179.2\text{ m}$.
  4. `compactness`: Polsby-Popper isoperimetric compactness index:
     $$\text{Compactness} = \frac{4\pi \cdot \text{Area}}{\text{Perimeter}^2}$$
     *(Values range from $0.0$ for highly irregular, narrow plots to $1.0$ for circular plots. Train mean: $0.6885$, median: $0.7186$).*
  5. `centroid_lon` & `centroid_lat`: True planar centroid computed in UTM Zone 44N and reprojected to WGS84 (preventing geographic curvature distortion).
  6. `bbox_min_lon`, `bbox_min_lat`, `bbox_max_lon`, `bbox_max_lat`: Bounding box extents for spatial indexing and satellite tile clipping.
  7. `vertex_count`: Number of boundary coordinate nodes ($\text{mean}=7.43$, range $4$ to $48$).
  8. `is_valid_geom`: Boolean confirmation flag (`True`).

#### Step 2.4: Enriched Geospatial Dataset Export
- Generated and saved clean, feature-rich datasets ready for machine learning into `outputs/data/`:
  - [`outputs/data/train_enriched_geometry.csv`](file:///C:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/data/train_enriched_geometry.csv) (7,888 rows × 32 columns).
  - [`outputs/data/test_enriched_geometry.csv`](file:///C:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/data/test_enriched_geometry.csv) (2,718 rows × 31 columns).
- Generated [`outputs/figures/geometry_diagnostics.png`](file:///C:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/figures/geometry_diagnostics.png) (6-panel diagnostic visualization) and formal audit reports [`outputs/reports/geometry_validation_report.txt`](file:///C:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/reports/geometry_validation_report.txt) and `.json`.

---

### Phase 3: Farm Spatial Visualisation & Cartography

#### Step 3.1: Interactive Folium Web Cartography
- Created [`src/geospatial/spatial_viz.py`](file:///C:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/src/geospatial/spatial_viz.py) and runner [`scripts/visualize_spatial.py`](file:///C:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/scripts/visualize_spatial.py).
- Produced a self-contained interactive web map: [`outputs/figures/telangana_farms_interactive_map.html`](file:///C:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/figures/telangana_farms_interactive_map.html) (3.59 MB):
  - **Base Maps:** Switch between *Esri World Imagery (High-Resolution Satellite)*, *OpenStreetMap*, and *CartoDB Positron*.
  - **Vector Farm Polygons:** Sampled farms rendered as true boundary polygons color-coded by health category:
    - 🟢 `Healthy`: `#2ecc71`
    - 🟠 `Pests`: `#e67e22`
    - 🔴 `Diseased`: `#e74c3c`
    - 🟡 `Stressed`: `#f1c40f`
  - **Interactive Popups:** Clicking any farm displays an agronomic dossier: FarmID, Crop, Health Status, Season, District & Sub-District, Area in ha and $m^2$, Sowing and Harvest dates, Irrigation system, source, rounds, water coverage %, and expected yield.
  - **Density Heatmap Overlay:** Folium `HeatMap` layer showing regional farm concentration.
  - **District Clusters:** Scaled circle markers indicating farm counts per district.
  - **Controls:** Embedded Fullscreen toggle, MeasureControl (dynamic length/area measuring tool), and LayerControl.

#### Step 3.2: High-Resolution Static Overview Map
- Generated [`outputs/figures/telangana_spatial_overview.png`](file:///C:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/figures/telangana_spatial_overview.png) (4-panel high-res figure):
  1. *Regional Farm Locations across Telangana* (Train vs Test scatter).
  2. *Crop Health Category Spatial Distribution* (Healthy, Pests, Diseased, Stressed clusters).
  3. *Geographic Distribution by Crop Type* (Paddy, Cotton, Maize, Bengal Gram, Groundnut, Chillies, Red Gram).
  4. *Irrigation Water Source Distribution* (Groundwater, Canal, Rainfed, Well, Tubewell).

---

### Phase 4: Sentinel-2 Multi-Temporal Remote Sensing Extraction Engine

#### Step 4.1: Google Earth Engine (GEE) Session Architecture & Robust Error Handling
- Installed and integrated `earthengine-api` (v1.7.46).
- Created [`src/satellite/s2_extractor.py`](file:///C:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/src/satellite/s2_extractor.py) with `EarthEngineManager` handling session authentication, initialization with project ID fallback, and non-blocking connectivity checks.
- If GEE credentials are not found, extractions gracefully record `status="failed"`, populate `error_message`, and remain retryable (failed farms are **never** treated as successful or written to persistent cache).

#### Step 4.2: Atmospheric Correction, Cloud Filtering & Safe Reduction
- **Target Dataset:** Sentinel-2 Level-2A Harmonized Surface Reflectance (`COPERNICUS/S2_SR_HARMONIZED`).
- **Cloud-Cover Threshold Filter:** Applies an explicit collection filter (`ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', 20.0)`) before scene retrieval.
- **Pixel-Level Masking:**
  - Evaluates Scene Classification Layer (`SCL`) to eliminate cloud shadows (value 3), medium/high cloud probability (values 8 & 9), cirrus (value 10), and snow/ice (value 11).
  - Evaluates `QA60` bitmask filtering (opaque and cirrus cloud bits).
- **Empty Collection Safe Handling:** Tests collection size prior to reduction; empty windows return NaN metrics rather than causing server compute crashes.

#### Step 4.3: Phenological Window Partitioning & Median Compositing
Single-date imagery fails to capture dynamic crop cycles. The engine divides each farm's verified lifecycle (`SDate` $\rightarrow$ `HDate`) into 4 distinct phenological stages:
1. **Early Season (Emergence & Tillering):** Sowing date to $25\%$ of growth cycle.
2. **Mid Season (Peak Vegetative & Flowering):** $25\%$ to $75\%$ of growth cycle (peak LAI and biomass).
3. **Late Season (Maturation & Pre-Harvest):** $75\%$ to $100\%$ of growth cycle (senescence and ripening).
4. **Full Season:** Complete duration for seasonal median baseline.
- **Compositing Method:** Robust pixel-wise **temporal median compositing** across cloud-free pixels in each window.

#### Step 4.4: Multi-Resolution Zonal Reduction Strategy
- Captures 10 Sentinel-2 multispectral bands at their native spatial resolutions:
  - **10m Native Bands (reduced at `scale=10`):** `B2` (Blue, 490 nm), `B3` (Green, 560 nm), `B4` (Red, 665 nm), `B8` (NIR Broad, 842 nm).
  - **20m Native Bands (reduced at `scale=20`):** `B5` (RedEdge 1, 705 nm), `B6` (RedEdge 2, 740 nm), `B7` (RedEdge 3, 783 nm), `B8A` (NIR Narrow, 865 nm), `B11` (SWIR 1, 1610 nm), `B12` (SWIR 2, 2190 nm).
- **Zonal Reducers:** Computes both **mean** (central spectral response) and **standard deviation** (intra-field spatial heterogeneity) clipped to polygon boundaries.
- **Batch Processing & Caching:** `BatchSentinelExtractor` maintains individual per-farm JSON checkpoints in `outputs/satellite_cache/`. Successful farms are cached; failed farms remain un-cached and eligible for automatic retry.

---

### Phase 5: Spectral, Moisture & Vegetation Indices Engineering

#### Step 5.1: 14 Agricultural Spectral Indices
Created [`src/features/indices.py`](file:///C:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/src/features/indices.py) providing vectorized, zero-division protected index calculations documented as proxy indicators:

| Index Name | Acronym | Authoritative Formulation | Primary Agronomic Significance |
|---|:---:|:---:|---|
| **Normalized Difference Vegetation Index** | **NDVI** | $\frac{B8 - B4}{B8 + B4}$ (Rouse et al., 1974) | Proxy indicator for photosynthetic canopy greenness and vegetative vigor |
| **Enhanced Vegetation Index** | **EVI** | $2.5 \cdot \frac{B8 - B4}{B8 + 6B4 - 7.5B2 + 1}$ (Huete et al., 2002) | Structural proxy reducing atmospheric haze and canopy background reflectance |
| **Green NDVI** | **GNDVI** | $\frac{B8 - B3}{B8 + B3}$ (Gitelson & Merzlyak, 1996) | Proxy indicator for leaf chlorophyll concentration and photosynthetic activity |
| **Soil Adjusted Vegetation Index** | **SAVI** | $\frac{B8 - B4}{B8 + B4 + 0.5} \cdot 1.5$ (Huete, 1988) | Canopy greenness adjusted for bright soil background reflectance ($L=0.5$) |
| **Modified SAVI** | **MSAVI** | $\frac{2B8 + 1 - \sqrt{(2B8 + 1)^2 - 8(B8 - B4)}}{2}$ (Qi et al., 1994) | Iterative self-adjusting soil factor for emerging or sparse crops |
| **Normalized Difference Moisture Index** | **NDMI** | $\frac{B8 - B11}{B8 + B11}$ (Gao, 1996) | Proxy indicator for canopy foliage liquid water content & water stress |
| **Normalized Difference Water Index** | **NDWI** | $\frac{B3 - B8}{B3 + B8}$ (McFeeters, 1996) | Surface water delineation and open-water/soil moisture proxy |
| **Moisture Stress Index** | **MSI** | $\frac{B11}{B8}$ (Rock et al., 1986) | Relative water stress indicator (rises with leaf water deficit) |
| **Normalized Difference Red Edge 1** | **NDRE1** | $\frac{B8 - B5}{B8 + B5}$ (Barnes et al., 2000) | Red-edge proxy sensitive to early nitrogen deficiency and foliar stress |
| **Normalized Difference Red Edge 2** | **NDRE2** | $\frac{B8 - B6}{B8 + B6}$ (Barnes et al., 2000) | Mid red-edge transition proxy for dense canopy chlorophyll dynamics |
| **Plant Senescence Reflectance Index** | **PSRI** | $\frac{B4 - B2}{B6}$ (Merzlyak et al., 1999) | Carotenoid-to-chlorophyll proxy associated with ripening, stress & senescence |
| **Bare Soil Index** | **BSI** | $\frac{(B11 + B4) - (B8 + B2)}{(B11 + B4) + (B8 + B2)}$ (Rikimaru et al., 2002) | Distinguishes exposed bare soil and fallow ground from green vegetation |
| **Chlorophyll Index Red Edge** | **CIRE** | $\frac{B8}{B5} - 1$ (Gitelson et al., 2003) | Proxy indicator for foliar chlorophyll content using red-edge 1 |
| **Chlorophyll Index Green** | **CIG** | $\frac{B8}{B3} - 1$ (Gitelson et al., 2005) | Proxy indicator for total chlorophyll mass using green reflectance |

#### Step 5.2: Multi-Temporal Dynamic Trajectories (Delta Features)
Captures temporal changes across phenological stages:
- **Vegetative Growth Delta:** $\text{NDVI}_{\text{growth\_delta}} = \text{NDVI}_{\text{mid}} - \text{NDVI}_{\text{early}}$
- **Senescence Delta:** $\text{NDVI}_{\text{senescence\_delta}} = \text{NDVI}_{\text{mid}} - \text{NDVI}_{\text{late}}$
- **Moisture Depletion Delta:** $\text{NDMI}_{\text{depletion\_delta}} = \text{NDMI}_{\text{mid}} - \text{NDMI}_{\text{late}}$
- **Peak-to-Early Vigor Ratio:** $\text{NDVI}_{\text{peak\_ratio}} = \frac{\text{NDVI}_{\text{mid}}}{\text{NDVI}_{\text{early}}}$

---

## 💻 Step-by-Step Execution Guide

### 1. Environment Setup
```powershell
# Clone or navigate to the repository
cd "C:\Users\Manichander\OneDrive\Desktop\cbit\major project\major_proj_final"

# Install required dependencies
pip install -r requirements.txt
pip install folium earthengine-api
```

### 2. Execute Phase 1: Data Analysis & Quality Audit
```powershell
python scripts/analyze_dataset.py
```
*Outputs: Generates 7 PNG figures in `outputs/figures/` and 2 data quality audit files in `outputs/reports/`.*

### 3. Execute Phase 2: Geometry Parsing, Repair & Feature Enrichment
```powershell
python scripts/validate_geometries.py
```
*Outputs: Repaired geometries, computes area/perimeter/compactness in UTM 44N, produces `geometry_diagnostics.png`, `geometry_validation_report.txt`, and exports `train_enriched_geometry.csv` & `test_enriched_geometry.csv`.*

### 4. Execute Phase 3: Spatial Visualisation & Interactive Cartography
```powershell
python scripts/visualize_spatial.py
```
*Outputs: Generates `telangana_farms_interactive_map.html` (3.59 MB interactive Folium map) and `telangana_spatial_overview.png` (4-panel publication overview).*

### 5. Open Jupyter Notebooks for Visual Exploration
```powershell
jupyter notebook notebooks/01_dataset_analysis.ipynb
jupyter notebook notebooks/02_geometry_and_spatial_analysis.ipynb
```

### 6. Execute Phase 4 & 5 Audit & Small Sentinel-2 Test
```powershell
python scripts/test_sentinel2_small.py
```
*Outputs: Evaluates 15 test farms on Sentinel-2 Level-2A Harmonized scenes across Early, Mid, Late, and Full phenological windows, computes scene counts before and after cloud cover threshold (<= 20%), tests cloud-masking/compositing pipelines, computes 14 spectral indices, tracks failures with retryable status, and exports `outputs/s2_small_test_features.csv`.*

---

## 📑 Summary of Generated Artifacts & Reports

| Artifact Name | File Path | Description |
|---|---|---|
| **Interactive Map** | [`outputs/figures/telangana_farms_interactive_map.html`](file:///C:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/figures/telangana_farms_interactive_map.html) | Standalone 3.59 MB multi-layer Folium web map with satellite basemap, vector farm polygons, popups, and heatmap. |
| **Spatial Overview Map** | [`outputs/figures/telangana_spatial_overview.png`](file:///C:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/figures/telangana_spatial_overview.png) | 4-panel static high-res overview of farm locations, health categories, crops, and irrigation. |
| **Geometry Diagnostics** | [`outputs/figures/geometry_diagnostics.png`](file:///C:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/figures/geometry_diagnostics.png) | 6-panel chart of area (ha), perimeter, Polsby-Popper compactness, vertices, and centroids. |
| **Data Quality Report** | [`outputs/reports/data_quality_report.txt`](file:///C:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/reports/data_quality_report.txt) | Integrity audit: 0 missing values, 0 duplicate IDs, 0 temporal inversions. |
| **Geometry Report** | [`outputs/reports/geometry_validation_report.txt`](file:///C:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/reports/geometry_validation_report.txt) | Validation audit: 100% validity post-repair, area and compactness distributions. |
| **Enriched Train Data** | [`outputs/data/train_enriched_geometry.csv`](file:///C:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/data/train_enriched_geometry.csv) | 7,888 farms × 32 columns with metric geometric features and repaired WKT geometries. |
| **Enriched Test Data** | [`outputs/data/test_enriched_geometry.csv`](file:///C:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/data/test_enriched_geometry.csv) | 2,718 farms × 31 columns ready for satellite extraction and ML inference. |
| **Sentinel-2 Test Features** | [`outputs/s2_small_test_features.csv`](file:///C:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/s2_small_test_features.csv) | 15 test farms × 109 feature columns (windowed scene counts, band mean/std stats, 14 spectral indices, status & error tracking). |

---

## 🔮 Next Steps (Phases 6–13)
1. **Phase 6 — Crop Classification Modeling:** Implement LightGBM, XGBoost, and Random Forest with Stratified 5-Fold Cross Validation to classify the 7 crop types (Paddy, Cotton, Maize, Bengal Gram, Groundnut, Chillies, Red Gram) from multi-temporal Sentinel-2 spectral indices and agro-climatic features, with SHAP explainability. *(Phase 6b: Secondary crop health diagnosis module using `category`).*
2. **Phase 7 — Temporal Phenology Modeling:** Double-logistic envelope fitting over NDVI time-series to track green-up, peak growth, and senescence dates.
3. **Phase 8 — Sentinel-1 SAR Integration:** Extract dual-polarization (VV, VH) backscatter and Radar Vegetation Index (RVI) for cloud-penetrating biomass evaluation.
4. **Phases 9 & 10 — Water Stress & Evapotranspiration Estimation:** Compute Crop Water Stress Index (CWSI) and FAO-56 Penman-Monteith crop evapotranspiration ($ET_c = K_c \times ET_0$).
5. **Phase 11 — Irrigation Decision Support Logic:** Prescriptive engine calculating irrigation urgency score ($0$–$100$), volume deficit ($m^3/\text{ha}$), and optimal next watering date.
6. **Phase 12 — Streamlit Web Dashboard:** Interactive dashboard for interactive farm exploration, time-series inspection, and irrigation decision lookup.
7. **Phase 13 — LangChain Advisory Agents:** Conversational AI agent generating natural language agronomic advisories based on farm diagnostics.
