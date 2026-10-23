# Technical Methodology: Multi-Temporal Remote Sensing and Machine Learning Pipeline

**Document Role:** Comprehensive Technical Methodology for Academic Paper Preparation  
**Project Title:** AI-Based Remote Sensing Framework for Crop Monitoring and Irrigation Decision Support Using Multi-Temporal Sentinel-2 Satellite Imagery  
**Geographic Domain:** Telangana State, Semi-Arid Tropics, India (14 Districts, 116 Mandals / Sub-Districts)  
**Evaluated Cohort:** $N = 7,843$ validated smallholder farm parcels across 7 crop species  
**Primary Model Architecture:** Regularized XGBoost with Fold-Internal Inverse-Frequency Sample Weighting  
**Feature Space:** 140 Multi-Temporal Spectral Features (S+T)  

---

## 1. Study Area and Dataset Characteristics

### 1.1 Geographic Setting
The study region encompasses 14 administrative districts in the state of Telangana, located in the semi-arid Deccan Plateau of south-central India ($15^\circ 50' \text{N}$ to $19^\circ 55' \text{N}$, $77^\circ 14' \text{E}$ to $81^\circ 19' \text{E}$). The agricultural landscape is characterized by:
- Severe monsoon dependence (Southwest monsoon, June–October).
- Extreme land fragmentation: median farm parcel area is **$2,779\text{ m}^2$** (~0.28 hectares; IQR: $1,257\text{ m}^2$ to $5,528\text{ m}^2$), with a minimum valid parcel area of $54\text{ m}^2$.
- Complex polyculture and mixed cropping systems where adjacent parcels cultivate crops with overlapping vegetative seasons.

### 1.2 Ground-Truth Survey Cohort
The ground truth dataset originates from field surveys conducted during the Kharif/Rabi agricultural cycles (Telangana Crop Health Challenge, GIZ-ADEX / Data Foundation). Farm parcels were digitized as vector polygons with verified ground-truth crop identity across seven major economic crop species:
1. **Bengal Gram** (*Cicer arietinum* / Chickpea, Rabi legume)
2. **Chillies** (*Capsicum annuum*, Cash crop / horticultural spice)
3. **Cotton** (*Gossypium hirsutum*, Long-duration commercial fiber)
4. **Groundnut** (*Arachis hypogaea*, Oilseed legume)
5. **Maize** (*Zea mays*, Coarse grain cereal)
6. **Paddy** (*Oryza sativa*, Wetland staple grain)
7. **Red Gram** (*Cajanus cajan* / Pigeon pea, Long-duration pulse)

### 1.3 Polygon Verification and Integrity Pipeline
All raw vector geometries underwent a rigorous automated geometric validation pipeline:
1. **Coordinate Reference System (CRS) Transformation:** Polygons were validated in WGS 84 (EPSG:4326) and projected to UTM Zone 44N (EPSG:32644) for exact metric area and perimeter calculation.
2. **Topological Cleaning:** Polygons were checked for self-intersections (`is_valid`), coordinate inversions, duplicate coordinates, and sliver geometries. Polygons with inverted vertex windings were corrected using `shapely.validation.make_valid`.
3. **Boundary Contamination Filter:** Extremely degraded geometries were flagged. Minimum bounding boxes and centroids were calculated for Earth Engine point-in-polygon queries.
4. **Cohort Reconciliation:** Out of 7,888 initial training records, 45 records experienced unrecoverable cloud contamination across all acquisition windows and were quarantined, leaving an authoritative training cohort of **$N = 7,843$** eligible farms.

| Crop Class | Training Parcels ($N$) | Percentage of Cohort | Growth Duration | Water Demand Profile |
| :--- | :---: | :---: | :---: | :---: |
| **Bengal Gram** | 938 | 11.96% | 90–110 days (Rabi) | Low (residual moisture) |
| **Chillies** | 601 | 7.66% | 150–180 days | High (frequent irrigation) |
| **Cotton** | 1,679 | 21.41% | 150–210 days | Moderate–High |
| **Groundnut** | 768 | 9.79% | 100–120 days | Moderate |
| **Maize** | 1,376 | 17.54% | 95–115 days | High (critical tasseling stage) |
| **Paddy** | 2,260 | 28.82% | 110–140 days | Very High (standing water) |
| **Red Gram** | 221 | 2.82% | 180–240 days | Low–Moderate (deep taproot) |
| **Total** | **7,843** | **100.00%** | — | — |

---

## 2. Multi-Temporal Sentinel-2 Earth Observation Pipeline

### 2.1 Satellite Ingestion and Cloud Masking
Multispectral Level-2A Bottom-Of-Atmosphere (BOA) surface reflectance imagery was ingested from the European Space Agency (ESA) Copernicus Sentinel-2 constellation via Google Earth Engine (`COPERNICUS/S2_SR_HARMONIZED`):
- **Bands Acquired:**
  - 10 m resolution: Band 2 (Blue, 490 nm), Band 3 (Green, 560 nm), Band 4 (Red, 665 nm), Band 8 (Broad NIR, 842 nm).
  - 20 m resolution: Band 5 (Red Edge 1, 705 nm), Band 6 (Red Edge 2, 740 nm), Band 7 (Red Edge 3, 783 nm), Band 8A (Narrow NIR, 865 nm), Band 11 (SWIR 1, 1610 nm), Band 12 (SWIR 2, 2190 nm).
- **Atmospheric Quality Masking:**
  - The Scene Classification Layer (SCL) was applied to mask thick clouds (SCL=9), thin cirrus (SCL=10), cloud shadows (SCL=3), and defective or saturated pixels (SCL=1).
  - Scenes with total cloud cover $>60\%$ were discarded before zonal aggregation.
  - Multi-granule MGRS tile mosaics were stitched to prevent boundary seam artifacts across tile edges.

### 2.2 Phenological Temporal Windows
Agricultural phenology in Telangana spans distinct growth phases. Because single-date acquisitions fail to discriminate crops with similar peak Leaf Area Index (LAI) (e.g., Cotton vs Maize), observations were partitioned into four chronologically targeted phenological windows:
1. **Early Season ($W_{\text{early}}$: June 1 – July 31):** Land preparation, monsoon onset, crop emergence, and early vegetative growth.
2. **Mid Season ($W_{\text{mid}}$: August 1 – September 30):** Peak vegetative expansion, canopy closure, flowering, and reproductive phase.
3. **Late Season ($W_{\text{late}}$: October 1 – November 30):** Grain filling, maturation, senescence, and early harvesting.
4. **Full Season ($W_{\text{full}}$: June 1 – December 31):** Integrated composite capturing the complete phenological trajectory, post-monsoon senescence, and Rabi transition.

Within each window, cloud-free pixels were aggregated using temporal median compositing ($\text{median}(R_\lambda(t))$), providing clean, outlier-robust surface reflectance measurements across each parcel.

---

## 3. Feature Engineering: Spectral and Phenological Indices

### 3.1 Spectral Indices
For each temporal window, 10 primary spectral reflectance bands and 14 biophysical and water indices were computed:
1. **Normalized Difference Vegetation Index (NDVI):**
   $$\text{NDVI} = \frac{B_8 - B_4}{B_8 + B_4}$$
2. **Enhanced Vegetation Index (EVI):**
   $$\text{EVI} = 2.5 \times \frac{B_8 - B_4}{B_8 + 6.0 B_4 - 7.5 B_2 + 1.0}$$
3. **Normalized Difference Moisture Index (NDMI):**
   $$\text{NDMI} = \frac{B_8 - B_{11}}{B_8 + B_{11}}$$
4. **Normalized Difference Water Index (NDWI):**
   $$\text{NDWI} = \frac{B_3 - B_8}{B_3 + B_8}$$
5. **Green Normalized Difference Vegetation Index (GNDVI):**
   $$\text{GNDVI} = \frac{B_8 - B_3}{B_8 + B_3}$$
6. **Normalized Difference Red Edge Index (NDRE):**
   $$\text{NDRE} = \frac{B_8 - B_5}{B_8 + B_5}$$
7. **Modified Soil Adjusted Vegetation Index (MSAVI):**
   $$\text{MSAVI} = \frac{2 B_8 + 1 - \sqrt{(2 B_8 + 1)^2 - 8 (B_8 - B_4)}}{2}$$
8. **Chlorophyll Index Green (CI Green):**
   $$\text{CI}_{\text{green}} = \frac{B_8}{B_3} - 1$$
9. **Chlorophyll Index Red Edge (CI Red Edge):**
   $$\text{CI}_{\text{re}} = \frac{B_8}{B_5} - 1$$
10. **Moisture Stress Index (MSI):**
    $$\text{MSI} = \frac{B_{11}}{B_8}$$
11. **Normalized Difference Built-Up/Bare Soil Index (NDBI):**
    $$\text{NDBI} = \frac{B_{11} - B_8}{B_{11} + B_8}$$
12. **Soil Adjusted Vegetation Index (SAVI, $L=0.5$):**
    $$\text{SAVI} = 1.5 \times \frac{B_8 - B_4}{B_8 + B_4 + 0.5}$$
13. **Structure Insensitive Pigment Index (SIPI):**
    $$\text{SIPI} = \frac{B_8 - B_2}{B_8 - B_4}$$
14. **Normalized Multi-Band Drought Index (NMDI):**
    $$\text{NMDI} = \frac{B_8 - (B_{11} - B_{12})}{B_8 + (B_{11} - B_{12})}$$

### 3.2 Zonal Aggregation
For each parcel polygon $P_i$, spatial zonal statistics were extracted across the interior raster grid:
$$\bar{F}_{k}(P_i, W_j) = \frac{1}{|P_i|} \sum_{p \in P_i} f_k(p, W_j)$$
where $f_k(p, W_j)$ represents index or band $k$ at pixel $p$ within temporal composite $W_j$. Both mean and standard deviation were computed for key indices.

### 3.3 Dynamic Phenological Deltas
To explicitly model crop developmental trajectories and senescence velocity, differential features were constructed between phenological windows:
$$\Delta_{\text{veg}} = \text{NDVI}(W_{\text{mid}}) - \text{NDVI}(W_{\text{early}})$$
$$\Delta_{\text{sen}} = \text{NDVI}(W_{\text{late}}) - \text{NDVI}(W_{\text{mid}})$$
$$\Delta_{\text{water}} = \text{NDMI}(W_{\text{mid}}) - \text{NDMI}(W_{\text{early}})$$
$$\Delta_{\text{stress}} = \text{MSI}(W_{\text{late}}) - \text{MSI}(W_{\text{mid}})$$

### 3.4 Feature Set Architectures
- **Baseline Feature Set (34 features):** 10 bands + 14 indices + 10 aggregations computed exclusively over the retrospective full-season window ($W_{\text{full}}$).
- **Multi-Temporal S+T Feature Set (140 features):** 
  - $4 \text{ windows} \times 34 \text{ spectral features} = 136 \text{ features}$.
  - $4 \text{ dynamic cross-window phenological deltas} (\Delta_{\text{veg}}, \Delta_{\text{sen}}, \Delta_{\text{water}}, \Delta_{\text{stress}})$.
  - Total: **140 Spectral + Temporal (S+T) features**.
- **Spatial Texture S+T+P Feature Set (160 features):**
  - 140 S+T features + 20 Gray-Level Co-occurrence Matrix (GLCM) spatial texture metrics (contrast, dissimilarity, homogeneity, energy, correlation) derived from $3 \times 3$ and $5 \times 5$ spatial kernels on Sentinel-2 NIR band ($B_8$).

---

## 4. Validation Protocol: Spatial Autocorrelation Mitigation

### 4.1 Danger of Standard Random Splitting
Standard random cross-validation ($k$-fold) in agricultural remote sensing introduces severe **spatial autocorrelation leakage**. Farm parcels located within hundreds of meters share:
- Identical soil type, micro-climate, rainfall events, and topography.
- Synchronized sowing calendars and pest outbreaks.
- Co-located satellite footprint calibration.

A model evaluated under random splitting memorizes spatial proximity rather than learning crop-specific phenology, yielding artificially inflated F1 scores ($>0.90$) that collapse when deployed on new regions.

### 4.2 Strict Five-Fold Stratified Group Cross-Validation
To guarantee true spatial generalizability, we implemented **5-Fold `StratifiedGroupKFold`** grouped strictly by administrative Sub-District (Mandal):
- **Grouping Variable:** `Sub-District` (116 distinct Mandals).
- **Stratification Goal:** Maintain global 7-class crop proportions inside each fold.
- **Leakage Invariant:** $0\%$ Mandal overlap between training and validation sets:
  $$\text{Mandals}(\text{Train}_k) \cap \text{Mandals}(\text{Val}_k) = \emptyset, \quad \forall k \in \{0, 1, 2, 3, 4\}$$
- Each fold withheld 21 to 26 entire Mandals (mean 23.2 Mandals; 1,508 to 1,610 unseen parcels) for out-of-fold evaluation.

```
Total Cohort (N = 7,843 farms, 116 Mandals)
 ├── Fold 0: 23 Mandals (N = 1,508) held out | Train on 93 Mandals (N = 6,335)
 ├── Fold 1: 21 Mandals (N = 1,600) held out | Train on 95 Mandals (N = 6,243)
 ├── Fold 2: 22 Mandals (N = 1,595) held out | Train on 94 Mandals (N = 6,248)
 ├── Fold 3: 26 Mandals (N = 1,530) held out | Train on 90 Mandals (N = 6,313)
 └── Fold 4: 24 Mandals (N = 1,610) held out | Train on 92 Mandals (N = 6,233)
```

---

## 5. Machine Learning Modeling and Optimization

### 5.1 Classifier Choice: Regularized Gradient Boosted Decision Trees
Extreme Gradient Boosting (XGBoost) was selected as the core classifier based on:
1. Native handling of tabular feature interactions and non-linear spectral responses.
2. Invariance to monotonic feature scaling (no normalization artifacts).
3. Robustness to multi-collinearity among correlated spectral bands.
4. Fast inference speed for regional-scale mapping.

### 5.2 Hyperparameter Configuration
Hyperparameters were optimized via nested cross-validation on Fold 0 to prevent test-leakage and prevent tree over-specialization:

```python
xgb_params = {
    'objective': 'multi:softprob',
    'num_class': 7,
    'learning_rate': 0.05,
    'max_depth': 6,
    'n_estimators': 350,
    'subsample': 0.80,
    'colsample_bytree': 0.80,
    'reg_alpha': 0.10,          # L1 Regularization (Lasso)
    'reg_lambda': 1.00,         # L2 Regularization (Ridge)
    'min_child_weight': 3,
    'eval_metric': 'mlogloss',
    'random_state': 42,
    'tree_method': 'hist'       # Histogram-based binning for efficiency
}
```

### 5.3 Class Imbalance Strategy: Fold-Internal Inverse-Frequency Weighting
The dataset exhibits severe class imbalance: Paddy ($N=2,260$) and Cotton ($N=1,679$) comprise $50.2\%$ of all samples, while Red Gram ($N=221$, $2.82\%$) and Chillies ($N=601$, $7.66\%$) represent fragile minority populations.

To prevent minority crop erasure, we implemented **Fold-Internal Sample Weighting**:
$$\omega_c^{(k)} = \frac{N_{\text{train}}^{(k)}}{C \cdot N_{c, \text{train}}^{(k)}}$$
where $C = 7$, $N_{\text{train}}^{(k)}$ is the training set size in fold $k$, and $N_{c, \text{train}}^{(k)}$ is the number of samples in class $c$.

**Critical Integrity Constraint:** Sample weights were derived strictly from the training partition of fold $k$. Zero validation frequency data leaked into weight computation.
- Paddy received weight $\approx 0.495\times$.
- Cotton received weight $\approx 0.667\times$.
- Red Gram received weight $\approx 5.07\times$ to $5.48\times$.

---

## 6. Evaluation Metrics and Statistical Significance Framework

### 6.1 Authoritative Metrics
For multi-class evaluation with class imbalance, Overall Accuracy is uninformative because high accuracy can be achieved while entirely missing minority classes. We report:

1. **Pooled Out-of-Fold Macro-F1 ($F_{1, \text{macro}}$):**
   $$F_{1, \text{macro}} = \frac{1}{C} \sum_{c=1}^C \frac{2 \cdot P_c \cdot R_c}{P_c + R_c}$$
   where $P_c$ and $R_c$ are the precision and recall computed across the aggregated out-of-fold predictions ($N=7,843$).
2. **Balanced Accuracy ($\text{BA}$):**
   $$\text{BA} = \frac{1}{C} \sum_{c=1}^C R_c$$
3. **Overall Accuracy ($\text{OA}$):**
   $$\text{OA} = \frac{\sum_{c=1}^C \text{TP}_c}{N}$$
4. **Weighted-F1:**
   $$F_{1, \text{weighted}} = \sum_{c=1}^C \left(\frac{N_c}{N}\right) F_{1, c}$$

### 6.2 Statistical Testing: Paired Differences and Cluster Bootstrapping
To rigorously compare model configurations (e.g., Weighted S+T vs Weighted S+T+P):
1. **Paired Student's $t$-test and Wilcoxon Signed-Rank Test:** Conducted across the 5 paired spatial fold deltas.
2. **Mandal Cluster Bootstrap ($B=1,000$ iterations):** Resampling entire Mandals with replacement to construct empirical $95\%$ confidence intervals around the Macro-F1 difference ($\Delta = \text{Macro-F1}_{\text{STP}} - \text{Macro-F1}_{\text{ST}}$), accounting directly for within-Mandal spatial correlation.
