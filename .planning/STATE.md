## Current State
- **Active Phase**: Phase 6H — Spatial Texture Integration and S/T/P Feature Ablation
- **Current Stage**: Stage 1: Read-Only Audit & Feasibility Preflight (COMPLETE — Awaiting Stage 2 Approval)
- **Status**: Stage 1 audit complete. 10 core preflight items answered with empirical evidence. Zero local rasters identified; GEE server-side glcmTexture() recommended. Awaiting explicit authorization before Stage 2.

## Progress Summary
| Stage | Description | Gate / Checkpoint | Status | Completed Date |
| :--- | :--- | :---: | :---: | :---: |
| **Phase 6F Stage 1** | Dataset Inventory, Independent Audit & Reconciliation | **Gate A** | **PASS** | 2026-10-18 |
| **Phase 6F Stage 2** | Failure Recovery & Schema Validation | Checkpoint | **PASS (5/5 Recovered)** | 2026-10-18 |
| **Phase 6F Stage 3** | Final Cohort Harmonization & Feature Readiness | **Gate B** | **PASS (7,888 Harmonized)** | 2026-10-18 |
| **Phase 6F Stage 4A** | Read-Only Modeling Preflight & Verification | Preflight | **READY WITH CAVEATS** | 2026-10-18 |
| **Phase 6F Stage 4B** | Controlled Model Benchmarking & Scientific Evaluation | Final Review | **PASS (All 4 Experiments Complete)** | 2026-10-18 |
| **Phase 6F Stage 4C** | Read-Only Scientific Benchmark Integrity Audit | Audit Checkpoint | **PASS WITH SCIENTIFIC ADDENDA** | 2026-10-18 |
| **Phase 6G Pilot** | Bounded Nested Grouped CV Pilot (Outer Fold 0) | Pilot Gate | **PASS (Trial 4 Selected, 10 Trials)** | 2026-10-18 |
| **Phase 6G Exp 2** | Imbalance-Aware Sample Weighting Pilot (Outer Fold 0) | Exp 2 Gate | **PASS (Balanced Inv-Freq Selected, 7 Strategies)** | 2026-10-18 |
| **Phase 6H Stage 1** | Spatial Texture Preflight & S/T/P Ablation Audit | Preflight Gate | **PASS (Audit Complete & Verified)** | 2026-10-18 |
| **Phase 6H Stage 2** | Spatial Texture Extraction Module Implementation | Checkpoint | **PENDING USER APPROVAL** | — |
| **Phase 6H Stage 3** | Seven-Experiment S/T/P Feature Ablation Benchmark | Final Review | **PENDING STAGE 2** | — |

## Phase 6G Pilot Findings (Outer Fold 0 Exploratory Pilot)
- **Inner Tuning Search (3-Fold Grouped on 93 Mandals, N=6,335):**
  - Trial 0 (Baseline ref): Inner Macro-F1 = 0.6649 (Acc: 0.7631, Red Gram F1: 0.2559)
  - Selected Trial 4 (`depth=7, colsample=0.75, subsample=0.8, reg_lambda=5.0, gamma=0.2, reg_alpha=0.2`): Inner Macro-F1 = **0.6797** (+0.0148 over baseline; Red Gram F1: 0.3022)
- **Outer Fold 0 Refit & Untouched Evaluation (N=1,508):**
  - Tuned Model: Macro-F1 = **0.7024**, Accuracy = **0.7798** (Baseline: Macro-F1 = 0.7095, Acc = 0.7818)
  - Red Gram Outer F1 = **0.3636** (Recall: 25.00%, Precision: 66.67%)
  - Total pilot runtime: 577.8s (~9.6 minutes) for 30 inner fits + 1 outer refit.

## Phase 6G Experiment 2 Findings (Imbalance-Aware Pilot on Outer Fold 0)
- **Inner Grouped Weighting Search (7 Strategies across 3 Folds, N=6,335):**
  - `unweighted_ref`: Inner Macro-F1 = 0.6797 (Red Gram F1: 0.3022, Recall: 20.61%, Confused w/ Cotton: 70/165)
  - `balanced_inv_freq` **(SELECTED)**: Inner Macro-F1 = **0.7020** (Red Gram F1: 0.4211, Recall: 33.94%, Chillies Recall: 50.81%)
  - Modest Red Gram upweighting (`red_gram_2.0x`): Inner Macro-F1 = 0.6861 (Red Gram F1: 0.3605)
- **Outer Fold 0 Refit & Untouched Evaluation (N=1,508):**
  - Outer Macro-F1 = **0.7339** (+0.0315 over unweighted Trial 4, +0.0244 over Phase 6F baseline)
  - Balanced Accuracy = **0.7336** (+0.0386 over unweighted Trial 4)
  - Red Gram F1 = **0.5263** (+0.1627; Recall: **44.64%** vs 25.00%, Precision: 64.10%)
  - Red Gram misclassification into Cotton dropped from 36/56 down to 12/56.
  - Overall accuracy preserved at **78.45%** (vs 77.98% unweighted, 78.18% baseline).

## Integrity & Regression Verification
- All 76 project unit tests pass (`python -m unittest discover tests`).
- All 24 protected baseline files verified 100% bit-for-bit identical before and after execution.
- Zero spatial data leakage across Mandals (0 Mandal overlap across inner and outer splits).
- All Phase 6G Experiment 2 outputs strictly isolated in `outputs/data/phase6g/`, `outputs/reports/phase6g/`, and `outputs/models/phase6g/`.
