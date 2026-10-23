# Multi-Temporal Sentinel-2 Spectral-Temporal Features for Seven-Class Crop Classification in Telangana: Spatial Cross-Validation and Independent Test Evaluation

**Working Title for Framework Context:** *AI-Based Remote Sensing Framework for Crop Monitoring and Irrigation Decision Support Using Multi-Temporal Sentinel-2 Satellite Imagery*

**Authors:** [Author Names Anonymized for Peer Review / Insert Author Names]  
**Affiliations:** [Department of Computer Science and Engineering / Geoinformatics, Institution Placeholder, Hyderabad, Telangana, India]  
**Correspondence:** [Author Email Placeholder]  

---

### Abstract

Accurate, parcel-level crop-type classification in smallholder agricultural landscapes is fundamental for regional food security, yield forecasting, and future agricultural water management. However, in semi-arid tropical regions such as Telangana, India, smallholder agricultural mapping is severely challenged by parcel fragmentation (median area $\sim 0.28\text{ ha}$), optical cloud contamination during the monsoon season, extreme class imbalance, and performance over-optimism induced by spatial autocorrelation under conventional random train/test splits. In this study, we present an end-to-end, reproducible remote sensing and machine learning methodology that classifies seven major agricultural crops (**Paddy, Cotton, Maize, Bengal Gram, Groundnut, Chillies, and Red Gram**) using multi-temporal Sentinel-2 surface reflectance imagery. 

We engineer a 140-dimensional Spectral-Temporal ($S+T$) feature space combining full-season aggregations, phenological window compositing across early-season vegetative establishment, mid-season reproductive peak, and late-season senescence, alongside four dynamic transition trajectory deltas. To ensure authentic validation, we enforce strict 5-fold spatial cross-validation grouped by Sub-District (116 Mandals, zero Mandal group overlap) on $N = 7,843$ validated farm parcels, coupled with fold-internal inverse-frequency sample weighting. Our primary regularized XGBoost model achieves a pooled out-of-fold Macro-F1 score of **0.7226**, Balanced Accuracy of **0.7180**, and Overall Accuracy of **0.7881 (78.81%)** across cross-validation. Fold-internal weighting rescues minority Red Gram F1-score from 0.3683 to 0.4420 (+7.37 percentage points). Furthermore, incorporating 20 Gray-Level Co-occurrence Matrix (GLCM) spatial texture features provided no statistically supported global improvement ($p = 0.8600$, 95% CI $[-0.0066, +0.0092]$), while parcel-area stratification revealed a performance reduction on the smallest parcel quartile ($<1,257\text{ m}^2$, $\Delta = -0.0133$), consistent with the physical hypothesis of boundary pixel contamination. 

We further evaluate the frozen champion model on an independent held-out cohort of $N = 2,718$ surveyed farm parcels spanning 35 unseen Mandals within Telangana (zero Mandal overlap with training). Across the $N = 2,265$ parcels (83.33% of the surveyed cohort) where satellite features were successfully extracted across all four temporal windows, the frozen model achieves a Macro-F1 of **0.7308**, Balanced Accuracy of **0.7326**, and Overall Accuracy of **0.8155 (81.55%)**, with 1,847 correct and 418 incorrect predictions. We document that persistent cloud cover caused the exclusion of 453 test parcels (16.67%), with exclusion rates varying from 4.17% on dry-season pulses to 25.91% on monsoon paddy. The metric parity between cross-validation and independent test evaluation confirms the stability of the 140-feature representation across unseen administrative units within the state, while highlighting the necessity of accounting for cloud-driven attrition in operational agricultural monitoring.

**Keywords:** Crop Classification, Sentinel-2, Multi-Temporal Remote Sensing, Spatial Cross-Validation, Smallholder Agriculture, XGBoost, Class Imbalance, Telangana.

---

## I. Introduction

Accurate, timely, and spatially explicit crop-type maps are essential for agricultural monitoring, subsidy allocation, agricultural insurance verification, and regional water resource stewardship (Defourny et al., 2019; Griffiths et al., 2019). In semi-arid agrarian economies such as the state of Telangana in southern India, crop production is highly dynamic, relying on a complex mosaic of rainfed cultivation, minor tank storage, and expanding lift-irrigation networks (Gumma et al., 2020). Farmers cultivate a diverse portfolio of summer and winter crops across distinct seasons (Kharif and Rabi), including major cereals (Paddy, Maize), industrial fibers (Cotton), high-value spices (Chillies), and protein-rich pulses and oilseeds (Groundnut, Bengal Gram, Red Gram). However, conventional agricultural statistics in the region rely heavily on manual field-by-field revenue inspections (*patwari* surveys), which are labor-intensive, logistically demanding, prone to reporting latency, and aggregated to coarse administrative levels.

Satellite remote sensing provides an objective, scalable alternative for parcel-level crop mapping. The European Space Agency’s (ESA) Copernicus Sentinel-2 constellation offers multispectral imagery at 10 m and 20 m spatial resolutions with a 5-day nominal revisit frequency, establishing an unprecedented capability for agricultural monitoring (Defourny et al., 2019; Inglada et al., 2015). Nevertheless, operational crop classification in smallholder environments faces four major scientific and practical hurdles:

1. **Spectral Ambiguity in Fragmented Landscapes:** Smallholder agricultural holdings in Telangana are intensely fragmented, with a median parcel area of approximately $0.28\text{ ha}$ ($2,779\text{ m}^2$). At this scale, individual farm plots span only a few 10-meter pixels. Multiple co-occurring crops exhibit near-identical spectral reflectance during peak vegetative growth, making single-date optical acquisitions insufficient for reliable discrimination (Vuolo et al., 2018; Jin et al., 2019).
2. **Spatial Autocorrelation and Evaluation Bias:** A critical flaw in much of the applied remote-sensing machine learning literature is the reliance on naive random $k$-fold cross-validation or unstratified random train/test splits (Roberts et al., 2017; Meyer et al., 2018). Because neighboring farm fields share identical soil types, micro-climates, irrigation canal access, and planting dates, spatial autocorrelation induces severe data leakage between training and evaluation subsets. Random splits place adjacent fields from the same village into both sets, artificially inflating out-of-sample accuracy and masking performance degradation when models are deployed to unmapped administrative jurisdictions (Roberts et al., 2017; Meyer et al., 2018).
3. **Severe Class Imbalance and Minority Erasure:** Agrarian land use is heavily skewed toward dominant cash crops such as Paddy and Cotton. High-value or subsistence pulses, such as Red Gram (*Cajanus cajan*), frequently represent less than 3% of regional farm parcels. Standard unweighted classifiers maximize global accuracy by predicting majority classes, resulting in the misclassification and erasure of minority pulse crops (Maxwell et al., 2018).
4. **Cloud-Driven Data Missingness:** The Kharif cropping cycle coincides with the Southwest monsoon, generating dense, persistent cloud and cloud-shadow cover across optical acquisitions. In operational settings, cloud contamination results in incomplete time-series stacks, creating parcel-level extraction failures that are rarely accounted for transparently in published classification benchmarks.

### Scope Demarcation
To maintain strict scientific integrity, we explicitly define the implemented scope of this study. While the broader conceptual framework envisions an integrated pipeline coupling crop identification with disease diagnosis, crop water stress retrieval, and FAO-56 Penman-Monteith irrigation decision support, **the experimentally evaluated and validated contribution of this paper is seven-class crop-type classification**. Downstream secondary tasks—specifically crop health status diagnosis using survey `category` labels, meteorological reanalysis integration (ERA5-Land), and evapotranspiration-based irrigation advisory generation—represent planned future extensions and are not presented as validated operational systems.

### Primary Contributions
In this study, we address the aforementioned challenges through a rigorously audited, leakage-free empirical investigation:

1. **Multi-Temporal Phenological Feature Engineering:** We design a 140-dimensional Spectral-Temporal ($S+T$) feature space combining full-season statistics, three distinct phenological window aggregations (Early, Mid, Late season), and four dynamic transition trajectory deltas, demonstrating a substantial accuracy gain (+10.52 percentage points Macro-F1) over static single-date composites ($0.6174 \to 0.7226$).
2. **Strict Spatial Group Cross-Validation:** We enforce 5-fold `StratifiedGroupKFold` validation partitioned strictly by Sub-District (116 Mandals, zero Mandal overlap), proving that regularized XGBoost achieves a robust pooled Macro-F1 of 0.7226 and Overall Accuracy of 78.81% across 7,843 parcels without village co-location leakage.
3. **Evaluation of Spatial Texture in Smallholder Parcels:** We demonstrate that incorporating 20 GLCM spatial texture features yields no statistically supported overall gain under spatial cross-validation ($p = 0.8600$, 95% CI $[-0.0066, +0.0092]$) and is associated with reduced performance in the smallest parcel quartile ($<1,257\text{ m}^2$, $\Delta = -0.0133$), providing an empirical basis for selecting the more parsimonious 140-feature representation.
4. **Leak-Free Minority Pulse Recovery:** We implement fold-internal inverse-frequency sample weighting, boosting Red Gram F1-score from 0.3683 to 0.4420 (+7.37 percentage points) while preserving strict spatial validation boundaries.
5. **Independent Held-Out Test Evaluation across Unseen Mandals with Cloud Accounting:** We execute frozen-model inference on 2,718 surveyed parcels across 35 unseen Mandals in Telangana, establishing that the model achieves 0.7308 Macro-F1 and 81.55% accuracy on the 2,265 successfully extracted parcels while transparently documenting a 16.67% cloud-exclusion rate.

---

## II. Related Work

The literature on satellite-based crop mapping spans several evolving paradigms:

### Multi-Temporal Satellite Optical Mapping
Early automated crop mapping relied heavily on Landsat or MODIS imagery, which offered either high temporal frequency with coarse spatial resolution ($250\text{–}500\text{ m}$) or moderate spatial resolution ($30\text{ m}$) with a 16-day revisit cycle that frequently missed cloud-free windows (Inglada et al., 2015; Gumma et al., 2022). The advent of Sentinel-2 MSI transformed smallholder mapping by providing 10-meter visible and near-infrared bands alongside red-edge spectral bands specifically sensitive to canopy chlorophyll content and nitrogen status (Defourny et al., 2019). Vuolo et al. (2018) demonstrated that multi-temporal Sentinel-2 acquisitions increased classification accuracy by 5% to 17% compared to single-date imagery. Similarly, Griffiths et al. (2019) proved that intra-annual phenological compositing metrics effectively separate spectrally overlapping crops across heterogeneous climatic gradients. In India, Gumma et al. (2020) demonstrated the utility of Sentinel-2 NDVI time series for mapping field-level pulse and cereal crops in semi-arid environments, while Gumma et al. (2022) established cloud-based machine learning workflows on Google Earth Engine for multi-cropland products across South Asia.

### Spatial Cross-Validation and Spatial Autocorrelation
The necessity of spatial blocking in geospatial machine learning has been established by Roberts et al. (2017) and Meyer et al. (2018). Spatial autocorrelation in environmental covariates means that standard random splits violate the fundamental assumption that training and validation instances are independently and identically distributed. When random cross-validation is applied to agricultural parcel datasets, models memorize local spatial context rather than generalizable spectral-temporal relationships, producing over-optimistic performance estimates that fail to transfer to unmapped regions (Meyer et al., 2018). Roberts et al. (2017) documented that non-spatial cross-validation can underestimate true prediction error by 20% to 60% in spatially structured datasets. Spatially clustered or administrative-group cross-validation provides an honest estimate of out-of-Mandal predictive power, although it does not eliminate residual spatial autocorrelation across administrative borders.

### Class Imbalance and Cost-Sensitive Learning in Remote Sensing
Agricultural classification tasks frequently suffer from severe class imbalance. As noted by Maxwell et al. (2018), standard loss functions in gradient-boosted decision trees optimize global error rates at the expense of minority classes. Synthetic oversampling methods, such as SMOTE, can create unrealistic spectral signatures that do not conform to physical radiative transfer principles. Inverse-frequency sample weighting adjusts the gradient contribution of each class according to its empirical rarity, providing a principled mechanism to recover minority decision boundaries without distorting physical feature distributions (Maxwell et al., 2018; Chen & Guestrin, 2016).

### Smallholder Parcel Fragmentation and Spatial Texture
In smallholder systems, parcel boundaries are irregular and individual field extents are often smaller than $0.5\text{ ha}$ (Jin et al., 2019). While spatial texture metrics, such as Haralick’s Gray-Level Co-occurrence Matrix (GLCM; Haralick et al., 1973), have demonstrated value in homogeneous, broad-acre commercial farms (Vuolo et al., 2018), Lebourgeois et al. (2017) highlighted that plots under $0.25\text{ ha}$ suffer from severe boundary contamination, where 30% to over 50% of parcel pixels overlap non-agricultural borders. Despite this, prior studies have rarely quantified the precise parcel-size threshold where spatial texture transitions from an asset to a liability.

---

## III. Study Area and Dataset

### A. Study Area
The study was conducted across the state of Telangana in southern India (Figure 6), covering an area of approximately $112,077\text{ km}^2$. The region experiences a semi-arid tropical climate characterized by three distinct seasons: the Southwest monsoon (*Kharif*, June to October), the dry winter (*Rabi*, November to March), and a hot summer (March to May). Annual rainfall ranges between 700 mm and 1,200 mm, with more than 80% concentrated during the monsoon months. Agricultural production is highly heterogeneous, spanning black cotton soils (*vertisols*), red sandy loams (*alfisols*), and alluvial canal command zones.

### B. Ground-Truth Parcel Cohorts
The empirical ground-truth dataset consists of verified vector polygons derived from extensive field surveys conducted across Telangana's agricultural landscape:

1. **Training Cohort (`Train.csv`):** Comprises 7,888 initial surveyed parcel polygons spanning 14 districts and 116 Sub-Districts (Mandals). Following cloud-screening and topological validation, 45 parcels failed feature extraction across all temporal windows, leaving an authoritative training cohort of **$N = 7,843$ eligible farm parcels**.
2. **Independent Test Cohort (`Test.csv`):** Comprises **$N = 2,718$ farm parcels** distributed across 35 Mandals. A rigorous preflight spatial audit confirmed **zero Mandal overlap** between the training and test cohorts, establishing a strict out-of-Mandal geographic holdout evaluation within Telangana (Table 1).
3. **Target Nomenclature:** The surveyed ground-truth crop label (`Crop`) encompasses seven distinct economic crop classes: **Bengal Gram, Chillies, Cotton, Groundnut, Maize, Paddy, and Red Gram**.

### C. Vector Topology and Geometric Validation
All parcel boundary shapefiles were projected from WGS 84 geographic coordinates (EPSG:4326) to the local Universal Transverse Mercator projection (UTM Zone 44N, EPSG:32644) to enable accurate metric parcel area calculations. Geometric validity was verified using GEOS; invalid self-intersecting polygon rings were repaired using `shapely.validation.make_valid()`. Parcels range in size from $54\text{ m}^2$ to $69,491\text{ m}^2$, with a median parcel area of $2,779\text{ m}^2$ ($0.278\text{ ha}$), reflecting typical smallholder fragmentation.

---

## IV. Proposed Methodology

The end-to-end classification framework comprises four sequential stages: satellite ingestion and cloud masking, phenological feature engineering, regularized spatial learning with class weighting, and out-of-Mandal frozen inference (Figure 1).

### A. Satellite Preprocessing and Cloud Masking
Multi-temporal surface reflectance imagery was ingested from the Copernicus Sentinel-2 Level-2A archive (`COPERNICUS/S2_SR_HARMONIZED`) using Google Earth Engine (GEE). For every acquisition intersecting a farm parcel, non-vegetated and compromised pixels were identified using the Scene Classification Layer (SCL). Pixels flagged as saturated/defective, cloud shadows, cloud medium probability, cloud high probability, or thin cirrus were masked out. To avoid temporal swath overweighting caused by overlapping orbit footprints, acquisitions from the same calendar date were daily-mosaicked before computing parcel zonal statistics.

### B. Phenological Windowing and Feature Representation
Agricultural parcels exhibit distinct phenological rhythms across their cultivation cycles. We establish four discrete temporal extraction windows tailored to Telangana’s cropping calendar:
- **Early Season (Window 1: Jun 01 – Jul 31):** Land preparation, sowing, and vegetative emergence.
- **Mid Season (Window 2: Aug 01 – Sep 30):** Peak canopy development, tillering, and flowering.
- **Late Season (Window 3: Oct 01 – Nov 30):** Maturation, grain filling, boll opening, and senescence.
- **Full Season (Full Window: Jun 01 – Dec 31):** Aggregate seasonal cycle.

Within each temporal window, clear pixels were aggregated using temporal median compositing. For each parcel polygon, we extracted zonal mean reflectance across 10 spectral bands: Blue (B2), Green (B3), Red (B4), Red Edge 1 (B5), Red Edge 2 (B6), Red Edge 3 (B7), NIR (B8), Narrow NIR (B8A), SWIR 1 (B11), and SWIR 2 (B12). Furthermore, 14 biophysical, chlorophyll, and water indices were calculated per window (Table 3):
- Normalized Difference Vegetation Index: $\text{NDVI} = (B8 - B4) / (B8 + B4)$
- Enhanced Vegetation Index: $\text{EVI} = 2.5 \cdot (B8 - B4) / (B8 + 6.0 \cdot B4 - 7.5 \cdot B2 + 1.0)$
- Normalized Difference Moisture Index: $\text{NDMI} = (B8 - B11) / (B8 + B11)$
- Normalized Difference Water Index: $\text{NDWI} = (B3 - B8) / (B3 + B8)$
- Green Normalized Difference Vegetation Index: $\text{GNDVI} = (B8 - B3) / (B8 + B3)$
- Normalized Difference Red Edge: $\text{NDRE} = (B8 - B5) / (B8 + B5)$
- Modified Soil-Adjusted Vegetation Index: $\text{MSAVI} = \frac{2 \cdot B8 + 1 - \sqrt{(2 \cdot B8 + 1)^2 - 8 \cdot (B8 - B4)}}{2}$
- Chlorophyll Index Green: $\text{CI}_{\text{Green}} = (B8 / B3) - 1$
- Chlorophyll Index Red Edge: $\text{CI}_{\text{RedEdge}} = (B8 / B5) - 1$
- Moisture Stress Index: $\text{MSI} = B11 / B8$
- Normalized Difference Built-Up Index: $\text{NDBI} = (B11 - B8) / (B11 + B8)$
- Soil-Adjusted Vegetation Index: $\text{SAVI} = 1.5 \cdot (B8 - B4) / (B8 + B4 + 0.5)$
- Structure Intensive Pigment Index: $\text{SIPI} = (B8 - B2) / (B8 - B4)$
- Normalized Multi-Band Drought Index: $\text{NMDI} = \frac{B8 - (B11 - B12)}{B8 + (B11 - B12)}$

This yields $10\text{ bands} + 14\text{ indices} = 34\text{ features}$ per temporal window, producing $34 \times 3 = 102$ windowed features, plus 34 full-season features ($136\text{ features}$).

To capture dynamic phenological transitions across growth stages, we engineer four explicit transition delta features:
1. **Vegetative Growth Delta:** $\Delta_{\text{growth}} = \text{NDVI}_{\text{Mid}} - \text{NDVI}_{\text{Early}}$
2. **Peak-to-Early Vigor Ratio:** $\text{Ratio}_{\text{peak/early}} = \text{NDVI}_{\text{Mid}} / (\text{NDVI}_{\text{Early}} + 10^{-6})$
3. **Senescence Drop Delta:** $\Delta_{\text{senescence}} = \text{NDVI}_{\text{Late}} - \text{NDVI}_{\text{Mid}}$
4. **Late-Season Moisture Loss:** $\Delta_{\text{moisture}} = \text{NDMI}_{\text{Late}} - \text{NDMI}_{\text{Mid}}$

The final **Spectral-Temporal ($S+T$) feature space comprises exactly 140 continuous features** ($34 + 102 + 4$).

### C. Spatial Texture Features (Ablation Pipeline)
To rigorously evaluate spatial texture, we constructed an augmented **160-dimensional Spectral-Temporal-Spatial ($S+T+P$) representation**. We computed Gray-Level Co-occurrence Matrix (GLCM; Haralick et al., 1973) texture rasters across the NIR band (B8) at 10-meter resolution using a $3 \times 3$ sliding window. Five standard second-order texture statistics (Angular Second Moment, Contrast, Correlation, Entropy, and Variance) were computed across each of the four temporal windows ($5 \times 4 = 20\text{ GLCM features}$) and aggregated per parcel.

### D. Model Architecture and Class-Weighted Optimization
We deploy Extreme Gradient Boosting (XGBoost; Chen & Guestrin, 2016) as our core classifier family due to its proven capacity to model complex non-linear feature interactions in tabular remote sensing representations. The objective function minimizes multi-class log-loss with regularized leaf weights:

$$\mathcal{L} = \sum_{i=1}^{N} w_i \cdot \ell(y_i, \hat{y}_i) + \sum_{k=1}^{K} \left( \gamma T_k + \frac{1}{2} \lambda \|w_k\|^2 + \alpha \|w_k\|_1 \right)$$

where $w_i$ represents sample-specific instance weights, $\ell$ is multi-class cross-entropy, $T_k$ is the number of terminal leaves in tree $k$, and $\lambda$ and $\alpha$ enforce $L_2$ and $L_1$ shrinkage.

To counter extreme class imbalance without distorting feature manifolds through synthetic oversampling, we calculate inverse-frequency sample weights $w_c$ for each crop class $c \in \{1, \dots, C\}$:

$$w_c = \frac{N_{\text{train}}}{C \cdot N_c}$$

where $N_{\text{train}}$ is the total number of training samples in the partition, $C = 7$ is the number of crop classes, and $N_c$ is the class-specific sample support. **Crucially, sample weights are computed strictly inside each training fold**, preventing any validation partition frequency leakage.

### E. Spatial Validation Design
To prevent spatial data leakage, validation is structured as a 5-fold `StratifiedGroupKFold` grouped strictly by Mandal (Sub-District). Parcels belonging to the same Mandal are assigned exclusively to either the training set or the validation set in any given fold:

$$\text{Mandals}(\mathcal{D}_{\text{train}, k}) \cap \text{Mandals}(\mathcal{D}_{\text{val}, k}) = \emptyset, \quad \forall k \in \{1, \dots, 5\}$$

Across the 116 training Mandals, each validation fold evaluates between 21 and 26 completely unseen Mandals ($1,508\text{–}1,610$ parcels per fold), preventing parcel co-location leakage and ensuring zero administrative group overlap between training and validation folds.

### F. Hyperparameter Optimization Protocol
Hyperparameters were evaluated via an exploratory bounded search of ten candidate configurations assessed via 3-fold grouped inner cross-validation strictly on the training partition of Outer Fold 0 ($N = 6,335$ parcels across 93 Mandals; Outer Fold 0 validation partition has $N = 1,508$ parcels across 23 Mandals). Selection was based on inner Macro-F1 (0.6797). The selected configuration—`max_depth=7`, `learning_rate=0.05`, `n_estimators=300`, `subsample=0.80`, `colsample_bytree=0.75`, `min_child_weight=3`, `reg_lambda=5.0`, and `reg_alpha=0.10`—was frozen and evaluated across all five spatial cross-validation folds and the independent test inference pipeline (Table 4). This design represents an exploratory parameter search on a single outer-fold training partition rather than fully nested cross-validation across all five folds. While inner grouped CV ensured zero validation leakage during selection, freezing parameters chosen on Fold 0 training data entails potential selection bias, though Fold 0 exhibited typical distribution properties.

---

## V. Results and Discussion

### A. Primary Benchmark Performance under Spatial Cross-Validation
Across the authoritative training cohort of $N = 7,843$ parcels in 116 Mandals, the primary `Weighted S+T` XGBoost model achieves a pooled out-of-fold Macro-F1 score of **0.7226**, Balanced Accuracy of **0.7180**, and Overall Accuracy of **0.7881 (78.81%)** (Table 5). The fold-mean Macro-F1 across the five spatial partitions is $0.7117 \pm 0.0181$ (population standard deviation, $\text{ddof}=0$) or $\pm 0.0202$ (sample standard deviation, $\text{ddof}=1$), demonstrating high stability across diverse sub-districts (Figure 3).

### B. Feature Ablation and Systematic Pipeline Evolution
To isolate the sources of classification performance, we evaluated five distinct pipeline iterations under identical spatial cross-validation protocols (Table 6, Figure 5):

1. **Static Single-Window Baselines:** The Early-Season single-window model achieved a Macro-F1 of only **0.5058** (Accuracy: 63.58%), reflecting severe vegetative ambiguity during emergence. Capturing full-season static median reflectance increased Macro-F1 to **0.6174** (Accuracy: 72.08%), indicating that seasonal composites reduce variance but lose temporal trajectories.
2. **Multi-Temporal Windowing Gain:** Introducing the multi-window framework (140 S+T features, unweighted) propelled Macro-F1 to **0.7069** (Accuracy: 79.24%), an improvement of **+8.95 percentage points** over the static full-season baseline. This confirms that multi-stage phenological tracking provides vital discriminatory power.
3. **Sample Weighting Impact:** Implementing fold-internal inverse-frequency weighting increased Macro-F1 from 0.7069 to **0.7226 (+1.57 pp)**. Crucially, weighting dramatically rescued the extreme minority pulse crop, **Red Gram**, raising its F1-score from 0.3683 to **0.4420 (+7.37 pp)** and precision from 0.6170 to 0.5467 under strict spatial isolation.

### C. Spatial Texture Evaluation and Parcel-Size Stratification
We evaluated whether incorporating GLCM spatial texture features improves smallholder crop classification. The 160-dimensional `Weighted S+T+P` model achieved a pooled Macro-F1 of **0.7228**, virtually identical to the 140-dimensional `Weighted S+T` model (**0.7226**, $\Delta = +0.0002$). A paired Student's $t$-test across the five spatial folds revealed no statistically significant difference ($t = 0.3037$, $p = 0.7765$). Furthermore, a 1,000-iteration Mandal cluster bootstrap confirmed that the global performance difference is statistically indistinguishable from zero ($p = 0.8600$, 95% CI $[-0.0066, +0.0092]$).

To understand why texture failed to yield benefits, we conducted an exploratory parcel-size quartile stratification ($N = 7,843$; Table 10, Figure 4):
- In **Quartile 1** (smallest parcels, $54\text{–}1,257\text{ m}^2$, median area $\sim 0.08\text{ ha}$), the inclusion of spatial texture was associated with a descriptive drop in Macro-F1 of **-0.0133 (-1.33 pp)** from 0.6598 to 0.6465.
- In **Quartile 2** ($1,257\text{–}2,783\text{ m}^2$), texture showed a minor negative association ($\Delta = -0.0148$).
- In **Quartile 3** ($2,784\text{–}5,517\text{ m}^2$), texture showed a positive association ($\Delta = +0.0130$), while in **Quartile 4** ($> 5,528\text{ m}^2$), performance was virtually identical ($\Delta = -0.0034$).

This pattern is consistent with the physical hypothesis of **boundary pixel contamination**: in smallholder parcels comprising only a few 10-meter pixels, $3 \times 3$ convolution kernels ($900\text{ m}^2$) inevitably overlap non-agricultural bunds, field paths, and adjacent crops, introducing edge-pixel noise. We emphasize that this boundary contamination is a plausible physical hypothesis rather than directly measured or causally proven ground truth. Furthermore, the absence of a statistically significant difference does not constitute proof of equivalence or demonstrate that texture features are universally ineffective across all spatial resolutions. Rather, based on parsimony (Occam’s razor), the 140-feature Spectral-Temporal representation was selected as the champion architecture, eliminating 20 computationally intensive GEE texture rasters without measurable performance penalty under the tested configuration.

### D. Independent Test Set Evaluation across 35 Unseen Mandals
To evaluate model transfer across unseen administrative units within Telangana, the frozen champion model was deployed on the independent held-out cohort (`Test.csv`, $N = 2,718$ parcels) across 35 geographically unseen Mandals (Table 7).

Across the **$N = 2,265$ successfully extracted parcels (83.33% of the surveyed cohort)**, the model achieved:
- **Macro-F1 Score:** **0.7308** (+0.0082 vs CV)
- **Overall Accuracy:** **0.8155 (81.55%)** (+0.0274 vs CV)
- **Balanced Accuracy:** **0.7326** (+0.0146 vs CV)
- **Weighted-F1 Score:** **0.8137** (+0.0280 vs CV)
- **Prediction Counts:** 1,847 correct classifications, 418 misclassifications.

### E. Class-Wise Classification Mechanics and Confusion Analysis
Analysis of the independent test confusion matrix (Table 9, Figure 1) and per-class metrics (Table 8, Figure 2) reveals distinct agronomic error patterns:

1. **Major Cereals and Cash Crops:** Paddy (F1: 0.8672, Recall: 88.85%), Groundnut (F1: 0.8543, Recall: 87.55%), and Maize (F1: 0.8390, Precision: 90.16%) achieved high accuracy, benefiting from distinctive flood-moisture signatures and rapid vegetative vigor. Cotton achieved an F1 of 0.7817 (Recall: 81.57%).
2. **Chillies:** Chillies achieved an F1 of 0.7606 (Precision: 74.09%, Recall: 78.14%), though 40 Maize parcels and 28 Cotton parcels were misclassified as Chillies, likely reflecting overlapping commercial irrigation practices in command areas.
3. **Bengal Gram:** Achieved an F1 of 0.7273 (Recall: 78.26%, 36/46 correct). As a winter Rabi crop, its phenological cycle diverges cleanly from summer monsoon crops.
4. **Red Gram Confusion Mechanics:** Red Gram exhibited the lowest recall across the test cohort at **20.00% (7/35 correct)**, resulting in an F1-score of **0.2857**. Of the 35 evaluated Red Gram parcels, 21 (60.0%) were misclassified as Cotton, 3 as Bengal Gram, 2 as Chillies, and 1 each as Maize and Paddy. This confusion is agronomically plausible: in Telangana, Red Gram (*Cajanus cajan*) and Cotton (*Gossypium hirsutum*) share overlapping sowing calendars (June/July) and broadleaf, woody canopy architectures with deep taproot structures. Furthermore, Red Gram is frequently intercropped with Cotton along parcel borders. Given the extremely limited test support ($N = 35$ evaluated parcels, representing 1.5% of the test cohort), minority pulse classification remains an open challenge.

### F. Scientific Interpretation of Cross-Validation vs Test Scores
The independent test scores (Macro-F1 0.7308, Accuracy 81.55%) are numerically higher than the 5-fold spatial cross-validation baseline (Macro-F1 0.7226, Accuracy 78.81%). **However, this must not be interpreted as evidence of superior geographic generalization.** Rather, this divergence is explained by two structural factors:
1. **Class Distribution Shift:** The test set has a significantly higher proportion of readily classifiable crops. Maize constitutes 27.86% of evaluated test parcels compared to 17.54% of training parcels. Conversely, Bengal Gram drops from 11.96% in training to 2.03% in testing. Because Maize achieves high precision (90.16%), this distribution shift inflates overall accuracy and weighted metrics.
2. **Cloud-Related Selection Bias:** 453 test parcels (16.67%) were excluded due to cloud contamination. Importantly, this exclusion was non-uniform: Kharif Paddy experienced a 25.91% exclusion rate (185/714 parcels lost), whereas Bengal Gram experienced only a 4.17% loss (Table 2). Consequently, the evaluated test subset represents a partially filtered sub-population rather than a uniform random sample of Telangana agriculture.

---

## VI. Limitations and Threats to Validity

To preserve scientific rigor, we document five primary limitations:

1. **Cloud-Driven Attrition:** The operational reliance on optical Sentinel-2 imagery resulted in the complete exclusion of 16.67% of test parcels due to persistent monsoon cloud obscuration. The framework currently lacks an all-weather synthetic aperture radar (SAR) fallback.
2. **Minority Class Fragility:** Despite class weighting, Red Gram achieved only 20% recall on the test set. Findings regarding minority pulse classes carry wide confidence intervals due to small test support ($N = 35$).
3. **Single Agricultural Season:** All satellite imagery and ground-truth data correspond to the 2023 cropping year. Inter-annual generalization across drought or extreme rainfall years remains untested.
4. **Geographic Confinement and Spatial Autocorrelation:** The model was evaluated via an independent held-out evaluation across 35 unseen Mandals within Telangana, with zero Mandal overlap between training and test sets. Both cohorts are from Telangana. Crucially, a Mandal-level administrative holdout does not guarantee zero spatial autocorrelation; nearby Mandals may remain geographically close or share agro-climatic and soil characteristics. Furthermore, this experiment is not a cross-state evaluation, contiguous spatial-block evaluation, or multi-year evaluation, and performance across other Indian agro-climatic zones requires empirical verification.
5. **Downstream Unimplemented Modules:** Crop disease diagnosis (`category`), meteorological integration (ERA5), and FAO-56 evapotranspiration irrigation scheduling remain conceptual future work and are not validated by this study.

---

## VII. Conclusion and Future Work

This study has established a reproducible, leak-free benchmark for seven-class crop identification in smallholder semi-arid agriculture using multi-temporal Sentinel-2 imagery. By enforcing strict spatial blocking across 116 training Mandals and evaluating on 35 unseen test Mandals within Telangana, we demonstrated that a 140-dimensional Spectral-Temporal XGBoost model achieves robust operational performance (0.7226 CV Macro-F1; 0.7308 Independent Test Macro-F1; 81.55% Test Accuracy on the 2,265 evaluated parcels). We empirically showed that spatial texture features provide no statistically supported global benefit, and that fold-internal inverse-frequency weighting recovers minority pulse decision boundaries without spatial leakage.

Future research will focus on:
1. **Multimodal SAR-Optical Fusion:** Integrating Sentinel-1 C-band SAR backscatter ($VV/VH$) to recover cloud-obscured monsoon parcels.
2. **Crop Health Modeling:** Developing multi-task classifiers using preserved ground-truth `category` survey labels.
3. **Agrometeorological Irrigation Support:** Coupling crop classification outputs with daily ERA5 reanalysis and FAO-56 dual crop coefficients to estimate dynamic irrigation water requirements.

---

## Acknowledgments
We acknowledge GIZ-ADEX, the Data Foundation, and the Department of Agriculture, Government of Telangana, for curating and making the ground-truth farm survey datasets available. We acknowledge the European Space Agency (ESA) Copernicus Programme for Sentinel-2 satellite observations and Google Earth Engine for planetary-scale geospatial cloud computing infrastructure.

---

## References

1. Defourny, P., Bontemps, S., Bellemans, N., Cara, C., Dedieu, G., Guzzonato, E., Hagolle, O., Inglada, J., Nicola, L., Rabaute, T., Savinaud, M., Udroiu, C., Valero, S., Bégué, A., Dejoux, J.-F., El Harti, A., Ezzahar, J., Frison, P.-L., Pliez, O., … Valero, S. (2019). Near real-time agriculture monitoring at national scale at parcel resolution: Performance assessment of the Sen2-Agri automated system in various cropping systems around the world. *Remote Sensing of Environment*, 221, 551–568. https://doi.org/10.1016/j.rse.2018.11.029
2. Jin, Z., Azzari, G., You, C., Di Tommaso, S., Aston, S., Burke, M., & Lobell, D. B. (2019). Smallholder maize area and yield mapping at national scales with Google Earth Engine. *Remote Sensing of Environment*, 228, 115–128. https://doi.org/10.1016/j.rse.2019.04.016
3. Griffiths, P., Nendel, C., & Hostert, P. (2019). Intra-annual reflectance composites from Sentinel-2 and Landsat for national-scale crop and land cover mapping. *Remote Sensing of Environment*, 220, 135–151. https://doi.org/10.1016/j.rse.2018.10.031
4. Roberts, D. R., Bahn, V., Ciuti, S., Boyce, M. S., Elith, J., Guillera-Arroita, G., Hauenstein, S., Lahoz-Monfort, J. J., Schröder, B., Thuiller, W., Warton, D. I., Wintle, B. A., Hartig, F., & Dormann, C. F. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography*, 40(8), 913–929. https://doi.org/10.1111/ecog.02881
5. Meyer, H., Reudenbach, C., Hengl, T., Katurji, M., & Nauss, T. (2018). Improving performance of spatio-temporal machine learning models using forward feature selection and target-oriented validation. *Environmental Modelling & Software*, 101, 1–9. https://doi.org/10.1016/j.envsoft.2017.12.001
6. Vuolo, F., Neuwirth, M., Immitzer, M., Atzberger, C., & Ng, W.-T. (2018). How much does multi-temporal Sentinel-2 data improve crop type classification? *International Journal of Applied Earth Observation and Geoinformation*, 72, 122–130. https://doi.org/10.1016/j.jag.2018.06.007
7. Lebourgeois, V., Bégué, A., Labbé, S., Mallavan, B., Roux, C., & Chopart, J.-L. (2017). A combined Random Forest and OBIA classification scheme for mapping smallholder agriculture at different nomenclature levels using multisource data (simulated Sentinel-2 time series, VHRS and DEM). *Remote Sensing*, 9(3), Article 259. https://doi.org/10.3390/rs9030259
8. Inglada, J., Arias, M., Tardy, B., Hagolle, O., Valero, S., Morin, D., Dedieu, G., Sepulcre, G., Bontemps, S., Defourny, P., & Koetz, B. (2015). Assessment of an operational system for crop type map production using high temporal and spatial resolution satellite optical imagery. *Remote Sensing*, 7(9), 12356–12379. https://doi.org/10.3390/rs70912356
9. Maxwell, A. E., Warner, T. A., & Fang, F. (2018). Implementation of machine-learning classification in remote sensing: An applied review. *International Journal of Remote Sensing*, 39(9), 2784–2817. https://doi.org/10.1080/01431161.2018.1433343
10. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '16)* (pp. 785–794). Association for Computing Machinery. https://doi.org/10.1145/2939672.2939785
11. Gumma, M. K., Thenkabail, P. S., Teluguntla, P., Rao, M. N., Mohammed, I. A., & Whitbread, A. M. (2020). Crop type identification and spatial mapping using Sentinel-2 satellite data with focus on field-level information. *Geocarto International*, 37(13), 3824–3841. https://doi.org/10.1080/10106049.2020.1864026
12. Gumma, M. K., Thenkabail, P. S., Panjala, P., Teluguntla, P., Yamano, T., & Mohammed, I. (2022). Multiple agricultural cropland products of South Asia developed using Landsat-8 30 m and MODIS 250 m data using machine learning on the Google Earth Engine (GEE) cloud and spectral matching techniques (SMTs) in support of food and water security. *GIScience & Remote Sensing*, 59(1), 1048–1077. https://doi.org/10.1080/15481603.2022.2088651
13. Haralick, R. M., Shanmugam, K., & Dinstein, I. (1973). Textural features for image classification. *IEEE Transactions on Systems, Man, and Cybernetics*, SMC-3(6), 610–621. https://doi.org/10.1109/TSMC.1973.4309314
