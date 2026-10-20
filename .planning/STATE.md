## Current State
- **Active Milestone**: Phase 6 Completed (Phase 6I: Robustness Validation and Final Model Selection Complete & Audited)
- **Primary Operational Model**: **`Weighted S+T`** (140 Spectral + Temporal features, inverse-frequency weighted regularized XGBoost)
- **Cohort**: 7,843 eligible training farms across 14 districts and 116 Mandals in Telangana
- **Validation**: 5-Fold StratifiedGroupKFold grouped strictly by Sub-District (Mandal) with zero spatial group overlap
- **Authoritative Performance**:
  - **Pooled Out-of-Fold Macro-F1**: **0.7226** *(Existing spatial cross-validation result, NOT a test-set score)*
  - **Balanced Accuracy**: **0.7180**
  - **Overall Accuracy**: **0.7881**
  - **Fold Macro-F1**: **0.7117 ± 0.0181** (Range: 0.6919 to 0.7381)
- **Status of Test Dataset (`datasss/Test.csv`)**: **UNPROCESSED / NOT EVALUATED**. The 2,718 test parcels have not undergone Sentinel-2 satellite feature extraction or model inference.
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
| **Test Set Ingestion** | Sentinel-2 Feature Extraction & Inference on `datasss/Test.csv` (2,718 farms) | Execution Gate | **NOT YET PROCESSED** | — |
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
