# Phase 6F Stage 4A: Preflight Addendum & Final Clarifications

**Document:** Stage 4A Preflight Clarification Addendum  
**Date:** 2026-10-18  
**Context:** Final reconciliation before Stage 4B model benchmarking  

---

## 1. Feature Dimension Reconciliations (Experiments A, B, C, D)

This addendum reconciles the proposed feature counts against the codebase implementation in [`src/features/indices.py`](file:///c:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/src/features/indices.py) and historical schema [`outputs/data/s2_phase6d_724_features.csv`](file:///c:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/data/s2_phase6d_724_features.csv).

### Summary Dimension Table
| Experiment | Scope | Band Stats | Spectral Indices | Temporal Deltas | Calendar Features | Geometry Features | **Reconciled Total Features** | Earlier Estimate | Reconciliation Explanation |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Exp A** | Full-Season Retrospective | 20 | 14 | 0 | 0 | 0 | **34** | 34 | Exact match. 10 bands $\times$ (mean, std) + 14 indices. |
| **Exp B** | In-Season Early Vegetative | 20 | 14 | 0 | 0 | 0 | **34** | 34 | Exact match. 10 bands $\times$ (mean, std) + 14 indices. |
| **Exp C** | Multi-Temporal Trajectory | 80 | 56 | 4 | 0 | 0 | **140** | 148 | Reconciled from 148 $\rightarrow$ **140**. Codebase implements 4 phenological deltas, not 12. |
| **Exp D** | Multimodal Benchmark | 80 | 56 | 4 | 3 | 3 | **146** | 154 | Reconciled from 154 $\rightarrow$ **146**. Excludes post-harvest calendar & bounding box coordinates. |

---

### Detailed Column Listings

#### Experiment A: Retrospective Full-Season Satellite-Only (34 Columns)
- **20 Raw Band Statistics (`full` window):**
  `B2_full_mean`, `B2_full_std`, `B3_full_mean`, `B3_full_std`, `B4_full_mean`, `B4_full_std`, `B8_full_mean`, `B8_full_std`, `B5_full_mean`, `B5_full_std`, `B6_full_mean`, `B6_full_std`, `B7_full_mean`, `B7_full_std`, `B8A_full_mean`, `B8A_full_std`, `B11_full_mean`, `B11_full_std`, `B12_full_mean`, `B12_full_std`
- **14 Computed Spectral Indices (`full` window):**
  `ndvi_full`, `evi_full`, `gndvi_full`, `savi_full`, `msavi_full`, `ndmi_full`, `ndwi_full`, `ndre1_full`, `ndre2_full`, `psri_full`, `bsi_full`, `msi_full`, `cire_full`, `cig_full`

#### Experiment B: In-Season Early Satellite-Only (34 Columns)
- **20 Raw Band Statistics (`early` window):**
  `B2_early_mean`, `B2_early_std`, `B3_early_mean`, `B3_early_std`, `B4_early_mean`, `B4_early_std`, `B8_early_mean`, `B8_early_std`, `B5_early_mean`, `B5_early_std`, `B6_early_mean`, `B6_early_std`, `B7_early_mean`, `B7_early_std`, `B8A_early_mean`, `B8A_early_std`, `B11_early_mean`, `B11_early_std`, `B12_early_mean`, `B12_early_std`
- **14 Computed Spectral Indices (`early` window):**
  `ndvi_early`, `evi_early`, `gndvi_early`, `savi_early`, `msavi_early`, `ndmi_early`, `ndwi_early`, `ndre1_early`, `ndre2_early`, `psri_early`, `bsi_early`, `msi_early`, `cire_early`, `cig_early`

#### Experiment C: Multi-Temporal Phenological Trajectory (140 Columns)
- **80 Raw Band Statistics across 4 windows:**
  `B{2,3,4,8,5,6,7,8A,11,12}_{early,mid,late,full}_{mean,std}` (80 features)
- **56 Computed Spectral Indices across 4 windows:**
  `{ndvi,evi,gndvi,savi,msavi,ndmi,ndwi,ndre1,ndre2,psri,bsi,msi,cire,cig}_{early,mid,late,full}` (56 features)
- **4 Phenological Trajectory Dynamics Features ([indices.py:335-356](file:///c:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/src/features/indices.py#L335-L356)):**
  1. `ndvi_growth_delta` ($= \text{ndvi\_mid} - \text{ndvi\_early}$)
  2. `peak_to_early_ratio` ($= \text{ndvi\_mid} / \max(\text{ndvi\_early}, 0.05)$)
  3. `ndvi_senescence_delta` ($= \text{ndvi\_mid} - \text{ndvi\_late}$)
  4. `moisture_loss_delta` ($= \text{ndmi\_mid} - \text{ndmi\_late}$)
- **Total: $80 + 56 + 4 = 140$ features.**

#### Experiment D: Multimodal Benchmark (146 Columns)
- **140 Multi-Temporal Satellite Features** (from Experiment C).
- **3 Non-Leaking Calendar Features (Available at Sowing):**
  1. `Season` (`Kharif` vs `Rabi` encoded via `OrdinalEncoder`)
  2. `sowing_month` (integer 1–12 derived from `SDate`)
  3. `sowing_doy` (Day of Year 1–365 derived from `SDate`)
- **3 Non-Leaking Geometry Shape Descriptors (Scale-Invariant & Coordinate-Free):**
  1. `area_ha` (Field polygon area in hectares, projected EPSG:32644)
  2. `perimeter_m` (Field polygon perimeter in meters, projected EPSG:32644)
  3. `compactness` (Dimensionless shape complexity: $\text{perimeter}^2 / (4\pi \times \text{area})$)
- **Total: $140 + 3 + 3 = 146$ features.**

---

## 2. Missingness Semantics & Trajectory Counts

### Granular Missingness Invariants
Empirical inspection of the 7,843 eligible satellite farms demonstrates:
- **Zero-Valid Image Windows (`n_images_valid == 0`):**
  - `early`: **541 farms**
  - `mid`: **121 farms**
  - `late`: **192 farms**
  - `full`: **0 farms**
- **All-Band NaN Windows (all 20 band statistics are NaN):**
  - `early`: **542 farms** (541 zero-valid + 1 farm [ID 1326265] where 1 image passed cloud filter but returned no unmasked pixels over the farm polygon)
  - `mid`: **121 farms** (exact match to zero-valid)
  - `late`: **192 farms** (exact match to zero-valid)
  - `full`: **0 farms** (exact match)
- **Any-Band NaN Windows (at least 1 band statistic is NaN):**
  - `early`: **545 farms** (542 all-NaN + 3 farms with sub-pixel resolution differences between 10m and 20m bands)
  - `mid`: **123 farms** (121 all-NaN + 2 farms)
  - `late`: **195 farms** (192 all-NaN + 3 farms)
  - `full`: **0 farms**

### Trajectory Coverage Counts Verified ($N=7,843$)
- **Complete Trajectory (All 4 Windows Valid):** Exactly **7,107 farms (90.62%)**.
- **Incomplete Trajectory ($\ge 1$ Missing Sub-Window):** Exactly **736 farms (9.38%)**.
- **Full Season Missing:** Exactly **0 farms (0.00%)**.

### Leakage and All-NaN Invariants for Trajectory Deltas
1. **Zero Data Leakage:** Trajectory deltas (e.g. $\text{ndvi\_mid} - \text{ndvi\_early}$) operate row-by-row on a single farm's own temporal observations. No target labels, group aggregates, or cross-farm statistics are used.
2. **No All-NaN Columns:** For all 7,107 complete-trajectory farms, all 4 trajectory deltas evaluate to valid finite numbers. For the 736 partial farms, missing components evaluate to NaN, which is safely imputed by fold-isolated median imputers or processed natively by XGBoost.

---

## 3. Justification of Multimodal Features (Experiment D)

### Permitted Calendar Features (3 Features)
1. `Season`: Known prior to sowing. Sowing in Kharif (monsoon) vs Rabi (winter) dictates cropping patterns in Telangana.
2. `sowing_month` & `sowing_doy`: Known at planting time. Early sowing vs delayed sowing is a strong agronomic indicator.
- **Strictly Excluded Post-Harvest Calendar Fields:** `duration_days`, `harvest_month`, and `harvest_doy` are **excluded** because they require knowledge of the harvest date, which is unknown in operational in-season monitoring.

### Permitted Geometry Shape Features (3 Features)
1. `area_ha`, `perimeter_m`, and `compactness`:
   - Delineation of the farm polygon is already required to query and clip Sentinel-2 pixels in Earth Engine.
   - These 3 metrics capture parcel size and shape complexity without encoding spatial location.
   - Different crops exhibit distinct parcel size signatures in Telangana (e.g. Paddy plots are typically small, leveled, and clustered near irrigation tanks; Cotton and Maize often occupy larger rainfed fields).
- **Strictly Excluded Geometric Features:** `lat`, `lon`, `centroid_lat`, `centroid_lon`, `bbox_min_lat`, `bbox_max_lat`, `bbox_min_lon`, `bbox_max_lon`, and `vertex_count` are **strictly prohibited** because they directly or indirectly encode geographic location, which would cause models to memorize mandal/district geography.

---

## 4. Qualified Spatial Cross-Validation Interpretation

The primary evaluation strategy is **5-Fold `StratifiedGroupKFold` grouped by `Sub-District` (Mandal)** across the 7,843 eligible cohort.

### Red Gram Class Support & Metric Sensitivity Warning
Red Gram is the rarest crop class ($N=221$ in the eligible cohort, 2.82% class share). Because Red Gram is geographically concentrated in northern Telangana (Gadiguda, Inderavelly, Utnur account for 169 of the 221 farms), grouping by Mandal forces entire concentrations into specific folds:
- Fold 0: 56 validation farms (165 training)
- Fold 1: 26 validation farms (195 training)
- Fold 2: 21 validation farms (200 training)
- Fold 3: 59 validation farms (162 training)
- Fold 4: 59 validation farms (162 training)

> [!WARNING]
> **Metric Variability Notice:**  
> Because Red Gram validation support ranges from 21 to 59 instances across folds, per-fold Red Gram recall and fold-wise macro-F1 will exhibit higher variance ($\sigma \approx 0.03\text{--}0.06$). Overall pooled out-of-fold metrics across all 7,843 farms provide the primary stable benchmark.

### Geographic Independence Qualification
Grouping by Sub-District ensures that **no sub-district appears in both training and validation folds**. However:
- This does **not** guarantee absolute agro-climatic isolation. Farms located near mandal boundaries may share micro-climatic or soil conditions with neighboring mandals.
- Mandal grouping is the most granular, defensible spatial cross-validation strategy available in this survey, but residual spatial autocorrelation across boundaries is an acknowledged scientific limitation.

---

## 5. Clarification on `Test.csv` Suitability

[`datasss/Test.csv`](file:///c:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/datasss/Test.csv) ($N=2,718$) is **unsuitable for satellite model evaluation** for one specific, objective reason:
- **Sentinel-2 satellite imagery was never extracted for `Test.csv`**. The entire Phase 6D and Phase 6E extraction campaigns were scoped and budgeted strictly for `Train.csv` ($724 + 7,164 = 7,888$ farms).
- Zero FarmID overlap between `Train.csv` and `Test.csv` is the standard design of partitioned machine learning datasets, **not a defect**.
- Because satellite features cannot be generated without running a new 2,718-farm Earth Engine extraction (strictly prohibited in Phase 6F), all model benchmarking must rely on 5-fold spatial cross-validation on `Train.csv`.

---

## 6. Preflight Checklist Summary

- [x] Exact feature columns listed and reconciled for Experiments A (34), B (34), C (140), and D (146).
- [x] Missingness semantics clarified: 541 early, 121 mid, 192 late zero-valid image windows verified.
- [x] Trajectory coverage verified: 7,107 complete (90.62%), 736 partial (9.38%).
- [x] Experiment D scoped strictly to non-leaking pre-sowing calendar (3) and shape geometry (3).
- [x] Spatial-CV variance and boundary limitations qualified.
- [x] `Test.csv` lack of satellite features clarified without mislabeling ID disjointness.
- [x] Zero feature matrices or fold files generated.
- [x] Zero model training or hyperparameter tuning executed.
- [x] All 24 baseline protected files remain 100% unmodified.
