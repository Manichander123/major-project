# Project: AI-Based Remote Sensing Framework for Crop Monitoring & Irrigation Decision Support

## Overview
Automated satellite remote sensing and machine learning framework for 7-class crop classification and irrigation decision support in Telangana, India using multi-temporal Sentinel-2 imagery.

## Objectives
- **Primary Research Target**: 7-class crop classification using the `Crop` target variable:
  - Paddy, Cotton, Maize, Bengal Gram, Groundnut, Chillies, Red Gram.
- **Secondary Target**: Crop Health / Stress category (`category` column: Healthy, Diseased, Pests, Stressed) evaluated independently without replacing or leaking into the primary crop classification objective.
- **Spatial Evaluation**: Rigorous spatial generalization testing via 5-fold StratifiedGroupKFold grouped strictly by Sub-District (Mandal), with zero group leakage.

## Milestones & Evolution
- **Phase 1 - 5**: Foundational EDA, geometry validation, CRS standardization (EPSG:4326 geographic, EPSG:32644 metric), baseline Random Forest on survey tabular data.
- **Phase 6C**: Initial multi-crop Sentinel-2 expansion (350 farms) with temporal median compositing.
- **Phase 6D**: Research audit, MGRS duplicate acquisition collapse via daily mosaicking, spatial CV benchmarking (714 farms, XGBoost multimodal macro-F1: 0.7047).
- **Phase 6E**: Full-scale extraction campaign on remaining 7,164 candidate farms in 15 chunks with daily mosaicking and SCL cloud masking.
- **Phase 6F**: Full-Scale Dataset Audit, Recovery, Feature Engineering & Model Benchmark across the full Telangana cohort (7,843 eligible farms).
- **Phase 6G**: Scientific Model Improvement — Hyperparameter Optimization & Imbalance-Aware Sample Weighting Pilot (Outer Fold 0 Macro-F1: 0.7339).
- **Phase 6H (Current)**: Spatial Texture Integration and S/T/P Feature Ablation.
