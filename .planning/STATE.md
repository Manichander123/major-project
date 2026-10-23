## Current State
- **Active Milestone**: Phase 7 (Test.csv Satellite Feature Extraction, Model Inference & Independent Evaluation Audit) — **COMPLETE**
- **Active Plan**: **Phase 7 Complete (All Plans 1-4 Delivered & Audited)**
- **Primary Operational Model**: **`Weighted S+T`** (140 Spectral + Temporal features, inverse-frequency weighted regularized XGBoost)
- **Cohort**: 7,843 eligible training farms across 14 districts and 116 Mandals in Telangana
- **Test Cohort Audit**: `datasss/Test.csv` (2,718 parcels across 35 unseen Mandals; 0 FarmID overlap, 0 Mandal overlap). `Crop` confirmed as surveyed ground-truth crop type.
- **Validation**: 5-Fold StratifiedGroupKFold grouped strictly by Sub-District (Mandal) with zero spatial group overlap
- **Authoritative Performance**:
  - **Pooled Out-of-Fold Macro-F1 (CV)**: **0.7226**
  - **Overall Accuracy (CV)**: **0.7881**
  - **Fold Macro-F1 (CV)**: **0.7117 ± 0.0181** (population SD; sample SD = 0.0203)
- **Independent Test Set Performance (N=2,265 across 35 unseen Mandals)**:
  - **Test Macro-F1**: **0.7308** (+0.0082 vs CV)
  - **Test Overall Accuracy**: **0.8155 (81.55%)** (+0.0274 vs CV)
  - **Test Balanced Accuracy**: **0.7326**
  - **Test Weighted-F1**: **0.8137**
  - **Cloud-Excluded Cohort**: 453 parcels (16.67%)
- **Status of Downstream Modules**: Crop-health classification (`category`), irrigation recommendations, weather integration, and dashboard/agent functionality are **NOT YET IMPLEMENTED**.

## Progress Summary
| Stage | Description | Gate / Checkpoint | Status | Completed Date |
| :--- | :--- | :---: | :---: | :---: |
| **Phase 1-5** | Geometry Repair, EDA, Indices Suite | Milestone | **PASS** | 2026-10-16 |
| **Phase 6A** | Tabular/Calendar Baseline (Archived: post-harvest leakage & spatial autocorrelation) | Archived | **SUPERSEDED** | 2026-10-16 |
| **Phase 6C-6E** | Multi-temporal Sentinel-2 Extraction Campaign (15 chunks, daily mosaic) | Checkpoint | **PASS** | 2026-10-17 |
| **Phase 6F** | Full-Scale Dataset Audit & 5-Fold Benchmark (7,843 farms, unweighted) | Gate B / Final Review | **PASS (Macro-F1: 0.7038)** | 2026-10-18 |
| **Phase 6G** | Bounded Hyperparameter Tuning & Sample Weighting Pilot (Outer Fold 0) | Pilot Gate | **PASS (Trial 4 + Inv-Freq Selected)** | 2026-10-18 |
| **Phase 6H** | GLCM Spatial Texture Extraction & 7-Way S/T/P Ablation Benchmark | Checkpoint | **PASS (S+T Parity Established, p=0.8600)** | 2026-10-19 |
| **Phase 6I** | Robustness Validation, Parsimony Analysis & Final Model Selection | Final Decision Gate | **PASS (Weighted S+T Selected, Macro-F1: 0.7226)** | 2026-10-20 |
| **Phase 7 Plan 1** | Test.csv Read-Only Preflight Audit & Leak-Safe Evaluation Protocol | Preflight Gate | **PASS (Ground truth verified; 0 overlap)** | 2026-10-23 |
| **Phase 7 Plan 2** | Sentinel-2 GEE Feature Extraction for Test Cohort (2,718 farms) | Extraction Gate | **PASS (2,265 success, 453 cloud, 140 feats)** | 2026-10-23 |
| **Phase 7 Plan 3** | Frozen Model Inference & Supervised Evaluation on Test Cohort | Evaluation Gate | **PASS (Macro-F1: 0.7308, Acc: 81.55%)** | 2026-10-23 |
| **Phase 7 Plan 4** | Quality Control & Final Synthesis | Delivery Gate | **PASS (All Audits & 120 Tests Pass)** | 2026-10-23 |
| **Crop Health Module** | Secondary Target (`category`) Modeling & Diagnosis | Gate | **NOT YET IMPLEMENTED** | — |
| **Weather / Agromet** | ERA5 / CHIRPS Meteorological Integration | Gate | **NOT YET IMPLEMENTED** | — |
| **Water / Irrigation** | Crop Water Stress & FAO-56 Penman-Monteith Irrigation Advisory | Gate | **NOT YET IMPLEMENTED** | — |
| **Dashboard & Agents** | Streamlit Web App & LangChain Advisory Agents | Delivery Gate | **NOT YET IMPLEMENTED** | — |

## Authoritative Phase 6I Benchmark Findings (N=7,843 Eligible Farms)
- **Model Decision**: Weighted S+T (140 features) selected over Weighted S+T+P (160 features) via Occam's razor:
  - Both achieve statistically indistinguishable Macro-F1 (0.7226 vs 0.7228; cluster bootstrap 95% CI [-0.0066, +0.0092], $p=0.8600$).
  - S+T achieves superior minority class recall on Red Gram (F1: **0.4420** vs 0.4317) and Chillies (F1: **0.5343** vs 0.5217).
  - Eliminates 20 GLCM texture rasters, avoiding boundary pixel contamination in smallholder parcels (35% of farms < 1,800 m²) and reducing GEE computational overhead.
- **Per-Class F1 Performance (Weighted S+T, 5-Fold Spatial CV)**:
  - `Bengal Gram` (N=938): F1 = **0.8275** (Prec: 0.7916, Rec: 0.8667)
  - `Chillies` (N=601): F1 = **0.5343** (Prec: 0.5249, Rec: 0.5441)
  - `Cotton` (N=1679): F1 = **0.8097** (Prec: 0.7939, Rec: 0.8261)
  - `Groundnut` (N=768): F1 = **0.8647** (Prec: 0.8597, Rec: 0.8698)
  - `Maize` (N=1376): F1 = **0.7090** (Prec: 0.7492, Rec: 0.6730)
  - `Paddy` (N=2260): F1 = **0.8708** (Prec: 0.8664, Rec: 0.8752)
  - `Red Gram` (N=221): F1 = **0.4420** (Prec: 0.5467, Rec: 0.3710)

## Integrity & Verification
- Unit test suite: 83 targeted offline unit tests passing (`python -m unittest tests/test_*.py`).
- Zero spatial group leakage across Mandals (0 Mandal overlap across all 5 folds).
- Archived experiments (`src/models/archive/baseline_rf.py` and `train_pilot_models.py`) isolated with explicit leakage disclaimers.
