# Project Roadmap

## Phase History & Status Overview
- [x] **Phase 1-5**: Ingestion, geometry checks, topological repair, and baseline tabular models (Phase 6A tabular baseline archived due to calendar leakage & spatial autocorrelation).
- [x] **Phase 6C**: Initial S2 Pilot (350 farms, archived exploratory pilot).
- [x] **Phase 6D**: Multi-temporal S2 Mosaic Refinement & Benchmark (714 farms, Macro-F1: 0.7047).
- [x] **Phase 6E**: Full-scale extraction campaign on remaining 7,164 candidate farms in 15 chunks.
- [x] **Phase 6F**: Full-Scale Dataset Audit, Recovery, Feature Engineering & 5-Fold Benchmark (7,843 eligible farms, unweighted Macro-F1: 0.7038).
- [x] **Phase 6G**: Scientific Model Improvement — Pilot & Imbalance-Aware Sample Weighting on Outer Fold 0 (Macro-F1: 0.7339, Red Gram Recall: 44.64% on Fold 0).
- [x] **Phase 6H**: Spatial Texture Integration & S/T/P Feature Ablation (evaluated 20 GLCM spatial texture features; confirmed texture fails to yield statistically significant improvement over S+T globally due to smallholder boundary pixel contamination: 0.7228 vs 0.7226, bootstrap p=0.8600).
- [x] **Phase 6I**: Robustness Validation, Parsimony Analysis & Final Model Selection:
  - **Selected Production Winner**: **`Weighted S+T`** (140 Spectral + Temporal features, inverse-frequency weighted regularized XGBoost).
  - **Cohort**: 7,843 eligible training farms across Telangana.
  - **Validation**: 5-Fold StratifiedGroupKFold grouped strictly by Sub-District (Mandal) with zero spatial group leakage.
  - **Performance**: Pooled Out-of-Fold Macro-F1 = **0.7226**, Balanced Accuracy = **0.7180**, Overall Accuracy = **0.7881** (Mean fold Macro-F1 = 0.7117 ± 0.0181).
  - *Note on Evaluation*: The 0.7226 Macro-F1 score is an out-of-fold spatial cross-validation metric, NOT a test-set score.
  - *Secondary Comparison Model*: `Weighted S+T+P` (160 features, pooled Macro-F1 = 0.7228, overall accuracy = 0.7905).

---

---

## Phase 7: Test.csv Satellite Feature Extraction, Model Inference & Independent Evaluation Audit

- [x] **Plan 1**: Read-only preflight audit & leakage-safe evaluation protocol (`phase7_test_preflight_audit.md`, `phase7_evaluation_protocol.md`) — **COMPLETE**
- [x] **Plan 2**: Test.csv Sentinel-2 feature extraction via GEE (2,718 parcels, 140 S+T features, 2,265 successful, 453 cloud failures, 100% schema matched) — **COMPLETE**
- [x] **Plan 3**: Model inference with frozen Phase 6I Weighted S+T model & supervised evaluation against verified `Crop` ground truth (N=2,265, Macro-F1: 0.7308, Accuracy: 81.55%) — **COMPLETE**
- [x] **Plan 4**: Quality control, reproducibility checks, offline test suite (`test_phase7_inference.py`), and `PHASE7_FINAL_SUMMARY.md` — **COMPLETE**

---

## ⚠️ Future Workstreams & Downstream Modules

### Phase 8: Crop Health & Stress Classification (Planned / Secondary Target)
- **Goal**: Model secondary target `category` (`Healthy`, `Diseased`, `Pests`, `Stressed`) independently without leaking into primary crop-type classification.
- **Status**: **NOT YET IMPLEMENTED** (No crop-health models trained).

### Phase 9: Weather & Agro-Meteorological Integration (Planned / Not Yet Implemented)
- **Goal**: Ingest ERA5-Land or CHIRPS precipitation, temperature, and solar radiation metrics.
- **Status**: **NOT YET IMPLEMENTED**

### Phase 10: Moisture & Water-Stress / Evapotranspiration (Planned / Not Yet Implemented)
- **Goal**: FAO-56 Penman-Monteith reference ET0 and crop coefficient (Kc) water requirement estimation.
- **Status**: **NOT YET IMPLEMENTED**

### Phase 11: Irrigation Decision-Support Engine (Planned / Not Yet Implemented)
- **Goal**: Rule-based irrigation urgency score, deficit volume calculation ($m^3/\text{ha}$), and irrigation schedule generator.
- **Status**: **NOT YET IMPLEMENTED**

### Phase 12: Streamlit Interactive Dashboard (Planned / Not Yet Implemented)
- **Goal**: Multi-page web dashboard for farm inspection, map explorer, and advisory output.
- **Status**: **NOT YET IMPLEMENTED**

### Phase 13: LangChain Agronomic Advisory Agents (Planned / Not Yet Implemented)
- **Goal**: Conversational LLM agent providing farmer-friendly advisories based on farm diagnostics.
- **Status**: **NOT YET IMPLEMENTED**
