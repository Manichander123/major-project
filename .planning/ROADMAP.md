# Project Roadmap

## Phase History
- [x] **Phase 1-5**: Ingestion, geometry checks, baseline tabular models.
- [x] **Phase 6C**: Initial S2 Pilot (350 farms).
- [x] **Phase 6D**: Multi-temporal S2 Mosaic Refinement & Benchmark (714 farms, Macro-F1: 0.7047).
- [x] **Phase 6E**: Full-scale extraction campaign on remaining 7,164 candidate farms in 15 chunks.

---

- [x] **Phase 6F**: Full-Scale Dataset Audit, Recovery, Feature Engineering & 5-Fold Benchmark (7,843 eligible farms, Macro-F1: 0.7038).
- [x] **Phase 6G**: Scientific Model Improvement — Pilot & Imbalance-Aware Sample Weighting on Outer Fold 0 (Macro-F1: 0.7339, Red Gram Recall: 44.64%).

---

## Active Phase: Phase 6H — Spatial Texture Integration and S/T/P Feature Ablation

### Stage 1: Read-Only Audit & Feasibility Preflight (Active)
- **Goal**: Full read-only audit of existing spectral definitions, temporal windows, geometry descriptors, raster pixel availability, GLCM candidate parameters, smallholder boundary constraints, cloud masking, and computational costs. Define exact feature schema and ablation matrix.
- **Deliverables**:
  - `outputs/reports/phase6h/phase6h_stage1_audit.json`
  - `outputs/reports/phase6h/phase6h_stage1_audit.md`
  - GSD Stage 1 Plan & Verification Artifacts.
- **Status**: Authorized & In Progress.

### Stage 2: Spatial Feature Implementation (Pending Approval)
- **Goal**: Implement isolated spatial-texture extraction module (GLCM contrast, dissimilarity, homogeneity, energy, correlation) via Earth Engine / native raster pipeline. Validate on smallholder test parcels with missingness/validity checks and unit tests.
- **Deliverables**:
  - `src/features/spatial_texture.py`
  - `outputs/data/phase6h/s2_phase6h_spatial_texture.csv`
  - `outputs/reports/phase6h/phase6h_spatial_extraction_report.json`
  - Unit tests in `tests/test_phase6h_spatial_features.py`.
- **Status**: Awaiting Stage 1 Review & User Approval.

### Stage 3: Seven Feature-Ablation Experiments (Pending Approval)
- **Goal**: Systematically benchmark Spectral (S), Temporal (T), Spatial Texture (P), and combined feature sets (S, T, P, S+T, S+P, T+P, S+T+P) under common 5-fold StratifiedGroupKFold (zero Mandal overlap) using XGBoost common classifier. Matched-cohort analysis and fold-level metrics reporting.
- **Deliverables**:
  - `outputs/data/phase6h/phase6h_ablation_oof_predictions.csv`
  - `outputs/reports/phase6h/phase6h_ablation_report.json`
  - `outputs/reports/phase6h/phase6h_ablation_report.md`
  - Comparison against Phase 6F/6G baselines.
- **Status**: Awaiting Stage 2 Completion & User Approval.
