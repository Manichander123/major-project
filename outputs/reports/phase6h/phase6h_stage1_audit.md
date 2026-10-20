# Phase 6H Stage 1: Read-Only Audit & Feasibility Preflight Report

**Project:** AI-Based Remote Sensing Framework for Crop Monitoring and Irrigation Decision Support Using Multi-Temporal Sentinel-2 Satellite Imagery  
**Phase:** Phase 6H — Spatial Texture Integration and S/T/P Feature Ablation  
**Stage:** Stage 1 (Read-Only Audit)  
**Status:** AUDIT COMPLETE — AWAITING STAGE 2 APPROVAL  
**Scope Boundary:** **READ-ONLY AUDIT. NO MODEL RETRAINING OR EARTH ENGINE MUTATION PERFORMED.**  

---

## Executive Summary

This preflight audit investigates the integration of spatial texture features (Gray-Level Co-occurrence Matrix, GLCM) into the multi-temporal Sentinel-2 crop classification framework and specifies the methodology for seven feature-ablation experiments: S (Spectral), T (Temporal), P (Spatial Texture), S+T, S+P, T+P, and S+T+P.

### Key Findings:
1. **Zero Local Raster Arrays:** All previous phases extracted tabular zonal statistics (`mean`, `stdDev`) directly on Google Earth Engine servers. No GeoTIFF image patches or raw pixel grids exist on disk. GLCM computation requires GEE extraction.
2. **The Smallholder Geometry Bottleneck:** 18.00% of farms in Telangana ($N=1,420$) have an area $< 900\text{ m}^2$ (smaller than a single $3\times 3$ pixel window at 10m resolution). A $3\times 3$ window at 20m resolution covers $3,600\text{ m}^2$ (exceeding 59.88% of all farms). Computing GLCM strictly within the interior of small parcels would produce 100% NaN values. Therefore, GLCM must be computed over the continuous Sentinel-2 scene prior to parcel zonal reduction.
3. **Recommended Spatial Configuration:** 10m native `B8` (NIR) band, 32 gray-level quantization, $3\times 3$ moving window, 4 omnidirectional directions, extracting 5 standard Haralick metrics: Contrast, Dissimilarity, Homogeneity, Energy, and Correlation.
4. **Exact Feature Schema:** Spectral ($S=34$), Temporal ($T=106$), Spatial Texture ($P=20$). Combined $S+T = 140$ features reproduces the Phase 6F Experiment C baseline; full fusion $S+T+P = 160$ features.

---

## 1. Existing Spectral Feature Definitions & Source Bands

- **Source Imagery:** `COPERNICUS/S2_SR_HARMONIZED` (Sentinel-2 Level-2A Surface Reflectance).
- **10m Bands:** `B2` (Blue, 490 nm), `B3` (Green, 560 nm), `B4` (Red, 665 nm), `B8` (NIR Broad, 842 nm).
- **20m Bands:** `B5` (RedEdge 1, 705 nm), `B6` (RedEdge 2, 740 nm), `B7` (RedEdge 3, 783 nm), `B8A` (NIR Narrow, 865 nm), `B11` (SWIR 1, 1610 nm), `B12` (SWIR 2, 2190 nm).
- **Raw Band Statistics:** Mean and Standard Deviation extracted across 4 phenological windows (10 bands $\times$ 4 windows $\times$ 2 stats = **80 features**).
- **Spectral Indices:** 14 vegetation, red-edge, moisture, and soil indices computed across 4 windows (14 indices $\times$ 4 windows = **56 features**):
  - `ndvi`, `evi`, `gndvi`, `savi`, `msavi`, `ndmi`, `ndwi`, `ndre1`, `ndre2`, `psri`, `bsi`, `msi`, `cire`, `cig`.
- **Total Spectral Features:** 80 raw band stats + 56 spectral indices = **136 spectral features**.

---

## 2. Existing Temporal Windows & Trajectory Dynamics

- **Phenological Partitioning (relative to sowing date `SDate` and harvest date `HDate`):**
  - `early`: 0% to 25% duration (emergence / early vegetative)
  - `mid`: 25% to 75% duration (peak canopy growth / flowering)
  - `late`: 75% to 100% duration (maturation / senescence)
  - `full`: 0% to 100% duration (entire season composite)
- **Temporal Trajectory Deltas (4 features):**
  1. `ndvi_growth_delta` = $\text{ndvi}_{mid} - \text{ndvi}_{early}$ (canopy expansion rate)
  2. `peak_to_early_ratio` = $\text{ndvi}_{mid} / (\text{ndvi}_{early} + 10^{-4})$ (vegetative vigor surge)
  3. `ndvi_senescence_delta` = $\text{ndvi}_{mid} - \text{ndvi}_{late}$ (canopy dry-down rate)
  4. `moisture_loss_delta` = $\text{ndmi}_{mid} - \text{ndmi}_{late}$ (foliage moisture loss)
- **Total Multi-Temporal Trajectory Features (Exp C):** 136 spectral + 4 deltas = **140 features**.

---

## 3. Existing Geometry Descriptors vs. Spatial Texture

| Feature | Definition | Scale / Units | Role in Framework |
| :--- | :--- | :--- | :--- |
| `area_ha` | Projected surface area in metric CRS (EPSG:32644) | Hectares | Multimodal Parcel Size |
| `perimeter_m` | Projected parcel boundary perimeter (EPSG:32644) | Meters | Boundary Length |
| `compactness` | Polsby-Popper isoperimetric ratio: $4 \pi \cdot \text{area} / \text{perimeter}^2$ | Unitless $(0, 1]$ | Parcel Geometric Shape |

> **Methodological Clarification:** Parcel geometry descriptors measure vector polygon boundary properties, NOT raster pixel-level canopy texture. In Phase 6F, they were strictly restricted to the multimodal experiment (Experiment D, 146 features) and excluded from the primary satellite-only benchmark (Experiment C, 140 features). Spatial texture (GLCM) represents the spatial arrangement of pixel reflectances within the canopy and is an optical remote sensing descriptor.

---

## 4. Local Raster Availability & Extraction History

- **Status:** **Zero local raster pixel arrays (`.tif`, `.npy`, `.h5`) exist in the repository.**
- **Audit Trail:** Prior extraction scripts (`extract_s2_expansion_phase6d.py`, `extract_s2_full_campaign_phase6e.py`) used Google Earth Engine's server-side `reduceRegion()` API, streaming aggregated scalar tables directly to CSV.
- **Implication:** GLCM spatial texture must be extracted via Google Earth Engine or downloaded as pixel patches. Downloading full rasters for 7,888 farms would consume ~10 GB and risk GEE rate limits. Server-side GEE extraction using `ee.Image.glcmTexture()` is the only scalable and reproducible approach.

---

## 5. Candidate GLCM Spatial Texture Parameters

| Parameter | Candidate Options | Selected / Recommended | Scientific Justification |
| :--- | :--- | :---: | :--- |
| **Band Selection** | B8 (NIR), B4 (Red), NDVI | **B8 (NIR 10m)** | NIR canopy reflectance responds directly to leaf area index, foliage clumpiness, and crop row structure without 20m coarse-pixel artifacts. |
| **Pixel Resolution** | 10m vs 20m | **10m** | 20m pixels ($400\text{ m}^2$) are coarser than 462 farms in the cohort. 10m native grid preserves smallholder structure. |
| **Kernel Size** | $3\times 3$ (30m), $5\times 5$ (50m), $7\times 7$ (70m) | **$3\times 3$ (30m)** | $5\times 5$ covers $2,500\text{ m}^2$ (exceeding 46% of all farms), causing severe boundary spillover. $3\times 3$ minimizes edge contamination. |
| **Quantization** | 16, 32, 64 levels | **32 levels** | Bins 0–10,000 reflectance into 32 discrete levels, balancing texture contrast sensitivity against sparsity. |
| **Directions** | $0^\circ, 45^\circ, 90^\circ, 135^\circ$ | **Directional Average (Isotropic)** | Rotational invariance; parcel orientations vary widely across farms. |
| **Texture Metrics** | 14 Haralick metrics | **5 Core Metrics** | **Contrast, Dissimilarity, Homogeneity, Energy, Correlation**. Uncorrelated, robust descriptors of canopy roughness. |
| **Polygon Aggregation** | Zonal Mean vs StdDev | **Zonal Mean** | Summarizes representative parcel texture without sensitivity to extreme boundary outlier pixels. |

---

## 6. Smallholder Geometry Distribution & Boundary Constraints

Empirical analysis of the 7,888 farm parcels in `train_enriched_geometry.csv` reveals extreme smallholder land fragmentation:

| Statistic | Parcel Area ($m^2$) | Area (Hectares) | Equivalent 10m Pixels | Equivalent 20m Pixels |
| :--- | :---: | :---: | :---: | :---: |
| **Minimum** | 53.75 | 0.0054 | **0.54 pixels** | 0.13 pixels |
| **1st Percentile** | 188.77 | 0.0189 | 1.89 pixels | 0.47 pixels |
| **5th Percentile** | 370.06 | 0.0370 | 3.70 pixels | 0.93 pixels |
| **10th Percentile** | 580.30 | 0.0580 | 5.80 pixels | 1.45 pixels |
| **25th Percentile** | 1,251.14 | 0.1251 | 12.51 pixels | 3.13 pixels |
| **Median (50th)** | 2,779.46 | 0.2779 | 27.79 pixels | 6.95 pixels |
| **75th Percentile** | 5,516.39 | 0.5516 | 55.16 pixels | 13.79 pixels |
| **90th Percentile** | 9,520.63 | 0.9521 | 95.21 pixels | 23.80 pixels |
| **99th Percentile** | 21,483.86 | 2.1484 | 214.84 pixels | 53.71 pixels |
| **Maximum** | 69,490.99 | 6.9491 | 694.91 pixels | 173.73 pixels |

### Operational Constraint & Boundary Handling Rule:
- **1,420 farms (18.00%)** are smaller than a $3\times 3$ window ($900\text{ m}^2$).
- **4,723 farms (59.88%)** are smaller than a $3\times 3$ window at 20m resolution ($3,600\text{ m}^2$).
- **Rule:** If GLCM were computed exclusively on pixels interior to the farm polygon, 18% of parcels would have zero complete $3\times 3$ windows and produce 100% NaN texture values. Therefore, **GLCM texture must be computed at the continuous image level** across the Sentinel-2 scene, followed by zonal mean reduction across the parcel polygon. This guarantees 100% non-NaN coverage for all cloud-free farms.

---

## 7. Computational Cost & Dependency Evaluation

| Criterion | Option A: GEE Server-Side (`glcmTexture()`) | Option B: Local Patch Download & `skimage` |
| :--- | :--- | :--- |
| **Dependencies** | `earthengine-api` (already installed and validated) | `scikit-image`, `rasterio`, `geopandas`, `earthengine-api` |
| **Compute Architecture** | Google Cloud Earth Engine distributed workers | Local CPU (multiprocessing) |
| **Network Transfer** | ~15 MB tabular data | 5–10 GB raster image patches |
| **Estimated Time** | **2–3 hours** for full cohort | 8–15 hours (high probability of GEE 429 quota failure) |
| **Failure Recovery** | Chunked, atomic checkpointing (Phase 6E architecture) | Complex raster file caching and retry logic |
| **Verdict** | **ACCEPTED (Standard Method)** | **REJECTED (High complexity & failure risk)** |

---

## 8. Proposed Feature Schema & Seven-Experiment Ablation Matrix

### Feature Group Definitions:
1. **Spectral (S, 34 features):** 20 full-season raw band stats (`B{b}_full_{mean,std}`) + 14 full-season spectral indices (`{idx}_full`). (Identical to Phase 6F Experiment A).
2. **Temporal (T, 106 features):** 102 multi-window spectral features across early, mid, late (`B{b}_{w}_{mean,std}` and `{idx}_{w}`) + 4 trajectory dynamics deltas. Note: $S + T = 34 + 106 = 140$ features, exactly reproducing Phase 6F Experiment C.
3. **Spatial Texture (P, 20 features):** 5 GLCM texture metrics (contrast, dissimilarity, homogeneity, energy, correlation) computed on `B8` (NIR 10m) across 4 phenological windows (early, mid, late, full): $5 \times 4 = 20$ features.

### Seven-Experiment Ablation Matrix:

| Experiment ID | Feature Group Configuration | Feature Count | Primary Hypothesis | Reference Baseline |
| :---: | :--- | :---: | :--- | :---: |
| `Exp 1 (S)` | **S** (Spectral Only) | **34** | Measures baseline full-season spectral separability without phenology or texture. | Phase 6F Exp A (0.6483) |
| `Exp 2 (T)` | **T** (Temporal Only) | **106** | Evaluates multi-temporal dynamics without full-season baseline summary. | New |
| `Exp 3 (P)` | **P** (Spatial Texture Only) | **20** | Tests standalone crop discriminatory power of canopy spatial texture. | New |
| `Exp 4 (S+T)` | **S + T** (Spectral + Temporal) | **140** | Established state-of-the-art multi-temporal benchmark. | Phase 6F Exp C (0.7038) |
| `Exp 5 (S+P)` | **S + P** (Spectral + Spatial) | **54** | Evaluates spatial texture enhancement over static single-window spectral data. | New |
| `Exp 6 (T+P)` | **T + P** (Temporal + Spatial) | **126** | Combines phenology and spatial texture without full-season static features. | New |
| `Exp 7 (S+T+P)` | **S + T + P** (Full Satellite Fusion) | **160** | Comprehensive tri-modal remote sensing fusion benchmark. | **Primary Research Target** |

### Common Evaluation Protocol:
- **Common Classifier:** XGBoost with Trial 4 regularized parameters (`depth=7, lr=0.05, colsample=0.75, subsample=0.80, reg_lambda=5.0`).
- **Spatial Grouping:** Identical 5-Fold `StratifiedGroupKFold` grouped by Sub-District ($N=7,843$ eligible farms, zero Mandal overlap).
- **Preprocessing:** Learned median imputation strictly inside each training fold.
- **Class Weighting:** Initially unweighted for fair feature ablation; class-weighting tested as a subsequent optimization on the best configuration.

---

## 9. Protected Baseline Hash Verification

All 24 historical protected baseline files were re-verified bit-for-bit identical via SHA-256 before and after Stage 1 audit. Zero historical files were modified.

---

## 10. Stage 1 Verification Checklist & Stop Condition

- [x] Existing spectral feature definitions and source bands documented.
- [x] Existing temporal windows and trajectory deltas documented.
- [x] Geometry descriptors vs. raster spatial texture distinguished.
- [x] Raster pixel availability verified (zero local rasters).
- [x] GLCM candidate parameters and smallholder constraints evaluated.
- [x] Decoupling from crop labels verified.
- [x] Small-polygon area distribution analyzed with exact percentiles.
- [x] Computational cost and GEE dependencies compared.
- [x] Identical cohort and spatial fold protocol defined.
- [x] Seven-experiment ablation matrix and feature schema established.
- [x] All 76 project unit tests passing.
- [x] All 24 protected baseline hashes verified intact.

**Stop Condition Enforced:** Stage 1 audit complete. Awaiting explicit user authorization before proceeding to Stage 2 implementation.
