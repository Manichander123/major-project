# Scientific Figure Inventory and Manuscript Placement Plan

**Project Title:** Multi-Temporal Sentinel-2 Spectral-Temporal Features for Seven-Class Crop Classification in Telangana: Spatial Cross-Validation and Independent Test Evaluation  
**Auditor:** Senior Remote Sensing Researcher & Visualization Editor  
**Status:** RECONCILED AGAINST PRIMARY REPOSITORY ARTIFACTS (STAGE E REMEDIATION)  

---

## 1. Master Figure Inventory

| Figure ID | Title / Concept | Artifact File Path | Generating Script / Source Data | Intended Section | Verification Status |
| :---: | :--- | :--- | :--- | :---: | :---: |
| **Figure 1** | **Spatial Cross-Validation vs Independent Test Confusion Matrices** | `outputs/reports/paper_materials/figures/fig1_confusion_matrices.png` & `outputs/reports/phase7/phase7_confusion_matrix.png` | `outputs/data/phase7/test_predictions.csv` & `phase6h_weighting_boundary_oof_predictions.csv` | Section V-C (Class Mechanics) | **VERIFIED (300 DPI)** |
| **Figure 2** | **Class-Wise Precision, Recall, and F1 Performance Across Seven Crops** | `outputs/reports/paper_materials/figures/fig2_class_wise_performance.png` | `outputs/reports/phase6i/phase6i_robustness_audit.json` & `phase7_test_metrics.json` | Section V-C (Class Mechanics) | **VERIFIED (300 DPI)** |
| **Figure 3** | **Inter-Fold Stability and Metric Dispersion Across 5 Spatial Folds** | `outputs/reports/paper_materials/figures/fig3_fold_variability.png` | 5-Fold validation scores in `phase6i_robustness_audit.json` | Section V-B (Fold Stability) | **VERIFIED (300 DPI)** |
| **Figure 4** | **Parcel-Size Quartile Stratification and Texture Evaluation** | `outputs/reports/paper_materials/figures/fig4_parcel_size_quartiles.png` | `reconciled_models.weighted_S+T.area_quartile_metrics` vs `weighted_S+T+P` | Section V-D (Texture Ablation) | **VERIFIED (300 DPI)** |
| **Figure 5** | **Systematic Pipeline Evolution and Feature Ablation Progression** | `outputs/reports/paper_materials/figures/fig5_pipeline_evolution.png` | `scripts/generate_paper_fig5.py` | Section V-E (Ablation Progression) | **VERIFIED & REGENERATED (300 DPI)** |
| **Figure 6** | **Telangana Study Area, Agro-Climatic Context, and Parcel Locations** | `outputs/figures/telangana_spatial_overview.png` | `outputs/figures/farm_boundaries_map.png` & GEE parcel boundary shapefiles | Section III-A (Study Area) | **VERIFIED** |

---

## 2. Detailed Figure Descriptions and Formal Captions

### Figure 1: Confusion Matrices Under Spatial Partitioning
- **Path:** `outputs/reports/paper_materials/figures/fig1_confusion_matrices.png` (and `outputs/reports/phase7/phase7_confusion_matrix.png`)
- **Caption:** *Normalized and absolute confusion matrices for seven-class crop identification. (A) Pooled out-of-fold predictions under 5-fold spatial cross-validation across 116 training Mandals ($N = 7,843$). (B) Independent held-out evaluation across 35 unseen Mandals within Telangana ($N = 2,265$). Matrix rows represent surveyed ground-truth crop identity; columns represent predicted class labels. Off-diagonal dispersion highlights asymmetric confusion between Red Gram and Cotton (canopy architecture overlap) and between Chillies and irrigated Paddy/Maize.*
- **Manuscript Placement:** Section V, Subsection C ("Per-Class Classification Mechanics and Confusion Analysis").

### Figure 2: Class-Wise Discrimination Profile
- **Path:** `outputs/reports/paper_materials/figures/fig2_class_wise_performance.png`
- **Caption:** *Class-wise Precision, Recall, and F1-Scores for the seven target crops under spatial cross-validation ($N = 7,843$) and independent held-out evaluation ($N = 2,265$). Major summer cereals and cash crops (Paddy, Groundnut, Cotton, Maize) achieve high operational F1-scores ($\ge 0.78$–$0.87$). Minor pulse crops, particularly Red Gram (35 evaluated test parcels), exhibit severe recall suppression (0.2000), illustrating persistent spectral ambiguity and sample rarity.*
- **Manuscript Placement:** Section V, Subsection C.

### Figure 3: Fold-to-Fold Metric Stability Across Spatial Partitions
- **Path:** `outputs/reports/paper_materials/figures/fig3_fold_variability.png`
- **Caption:** *Performance consistency of the Weighted S+T regularized XGBoost model across the five spatial validation folds. Each fold evaluates between 21 and 26 completely isolated Mandals (zero inter-fold Mandal group overlap). The standard deviation ($\sigma_{\text{pop}} = \pm 0.0181$, $\sigma_{\text{samp}} = \pm 0.0202$ around mean Macro-F1 = 0.7117) illustrates metric dispersion across distinct held-out administrative clusters in Telangana.*
- **Manuscript Placement:** Section V, Subsection B ("Spatial Stability and Inter-Fold Variance").

### Figure 4: GLCM Spatial Texture Stratification by Parcel Area
- **Path:** `outputs/reports/paper_materials/figures/fig4_parcel_size_quartiles.png`
- **Caption:** *Macro-F1 delta ($\Delta_{\text{P - ST}} = \text{Macro-F1}_{\text{S+T+P}} - \text{Macro-F1}_{\text{S+T}}$) stratified across parcel area quartiles ($N = 7,843$). For smallholder parcels in Quartile 1 ($< 1,257\text{ m}^2$), the inclusion of 20 GLCM spatial texture rasters is associated with a descriptive drop in Macro-F1 ($\Delta = -0.0133$), consistent with the physical hypothesis of $3 \times 3$ convolution kernels overlapping non-agricultural field boundaries. In larger commercial holdings ($> 2,784\text{ m}^2$, Quartile 3), spatial texture shows modest gains ($+0.0130$), balancing to zero statistically supported difference globally ($p = 0.8600$, 95% CI $[-0.0066, +0.0092]$).*
- **Manuscript Placement:** Section V, Subsection D ("The Spatial Texture Dilemma in Smallholder Parcels").

### Figure 5: Systematic Pipeline Progression
- **Path:** `outputs/reports/paper_materials/figures/fig5_pipeline_evolution.png`
- **Caption:** *Stepwise classification accuracy and Macro-F1 progression across experimental development stages. Moving from single-window static composites (Early Season: 0.5058; Full Season: 0.6174) to multi-temporal phenological windowing (+140 S+T: 0.7069) delivers an +8.95 pp Macro-F1 improvement. Incorporating fold-internal inverse-frequency sample weighting delivers an additional +1.57 pp gain (0.7226), primarily through recovering minority pulse decision boundaries. Note that early exploratory pilot results ($N = 714$) are demarcated from full-cohort experiments ($N = 7,843$).*
- **Manuscript Placement:** Section V, Subsection E ("Ablation Analysis and Pipeline Evolution").

### Figure 6: Spatial Context and Geographic Distribution
- **Path:** `outputs/figures/telangana_spatial_overview.png`
- **Caption:** *Geographic distribution of surveyed farm parcels across the 14 surveyed districts and 116 training Mandals in Telangana, India. Inset maps illustrate parcel fragmentation, topological validation, and the distinct spatial clustering of test Mandals ensuring zero Mandal overlap.*
- **Manuscript Placement:** Section III, Subsection A ("Study Area and Geospatial Data Ingestion").
