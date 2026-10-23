# Research Paper Remediation: Baseline Repository Inventory & Audit

**Date:** 2026-10-23  
**Auditor Roles:** Senior Research Engineer, ML Experiment Auditor, Scientific Editor, Reproducibility Reviewer  
**Repository Root:** `c:\Users\Manichander\OneDrive\Desktop\cbit\major project\major_proj_final`  
**Git Branch:** `main` (clean tracking `origin/main` at commit `442e043`)  
**Working Tree State:**  
- Modified: `.planning/ROADMAP.md`, `.planning/STATE.md`  
- Staged: 20 paper material documents and figures in `outputs/reports/paper_materials/`  
- Untracked: `outputs/paper/`, `outputs/reports/phase7/`, `scripts/audit_phase7_metrics.py`, `scripts/extract_s2_test_phase7.py`, `scripts/run_phase7_test_inference.py`, `tests/test_phase7_inference.py`  

---

## 1. Inventory of Primary Evidence Artifacts

### 1.1 Datasets and Vector Geometries
- `datasss/Train.csv`: 7,888 original surveyed parcels across 116 Mandals and 14 districts in Telangana.
- `datasss/Test.csv`: 2,718 surveyed parcels across 35 Mandals in Telangana (zero Mandal overlap with training).
- `outputs/data/s2_phase6f_features.csv`: 7,843 eligible training farms (45 parcels failed GEE extraction due to persistent cloud cover).
- `outputs/data/phase6f_cv_folds.csv`: Exact 5-fold spatial split definition grouped by Sub-District (Mandal).

### 1.2 Model Artifacts
- `outputs/models/phase6h/xgb_weighted_s+t.joblib`: Frozen production model (`XGBClassifier`, 140 features, inverse-frequency class weights, `max_depth=7`, `colsample_bytree=0.75`, `reg_lambda=5.0`).
- `outputs/models/phase6g/xgb_pilot_fold0_selected.joblib`: Exploratory pilot model trained on Outer Fold 0 training partition.

### 1.3 Feature Manifests
- `phase6h_feature_schema_manifest.json`: Exact 140-feature Spectral-Temporal schema (34 full season + 102 across Early/Mid/Late windows + 4 transition deltas) and 20 GLCM spatial texture rasters.

### 1.4 Spatial Cross-Validation Primary Records ($N = 7,843$)
- `outputs/reports/phase6i/phase6i_robustness_audit.json`: Authoritative source for 5-fold CV metrics:
  - Pooled Out-of-Fold Macro-F1: **0.7226**
  - Pooled Out-of-Fold Accuracy: **0.7881 (78.81%)**
  - Pooled Out-of-Fold Balanced Accuracy: **0.7180**
  - Pooled Out-of-Fold Weighted-F1: **0.7857**
  - Fold-mean Macro-F1: **0.7117 ± 0.0181** (population SD, ddof=0) / **0.0202** (sample SD, ddof=1)
  - Unweighted S+T Macro-F1: **0.7069**
  - Unweighted S+T+P Macro-F1: **0.7038**
  - Weighted S+T+P Macro-F1: **0.7228** ($p = 0.8600$, 95% CI $[-0.0066, +0.0092]$)
  - Red Gram F1: unweighted 0.3683 $\to$ weighted 0.4420 (+7.37 pp)
  - Area Quartile F1: Q1 ($<1,257\text{ m}^2$) S+T 0.6598 vs S+T+P 0.6465 ($\Delta = -0.0133$)

### 1.5 Phase 6G Hyperparameter Search Primary Records
- `outputs/reports/phase6g/phase6g_pilot_fold0_report.json`:
  - `outer_fold`: 0
  - `outer_train_samples`: **6,335**
  - `outer_val_samples`: **1,508** (Sum = 7,843)
  - `inner_cv_folds`: 3-fold grouped CV on the 6,335 training samples
  - `total_trials`: 10 distinct configurations
  - `selected_trial`: Trial 4 (`inner_macro_f1 = 0.6797`, `outer_val_macro_f1 = 0.7024`, baseline Phase 6F = 0.7095)
- `outputs/reports/phase6g/pilot_fold0_trial_ledger.csv`: Full trial-by-trial ledger across the 10 inner-CV trials.

### 1.6 Independent Test Set Primary Records ($N = 2,718$)
- `outputs/data/phase7/test_predictions.csv`: 2,718 total rows, 2,265 non-null predictions, 453 null (cloud-excluded) rows.
- `outputs/reports/phase7/phase7_test_metrics.json`:
  - Evaluated farms: 2,265 (83.33%)
  - Cloud-excluded farms: 453 (16.67%)
  - Test Mandals: 35 (0 overlap with training)
  - Accuracy: **0.8155 (81.55%)** (1,847 correct / 418 incorrect)
  - Macro-F1: **0.7308**
  - Balanced Accuracy: **0.7326**
  - Weighted-F1: **0.8137**
  - Red Gram Recall: **0.2000 (7/35)**
  - Cloud exclusion rates: Paddy 25.91% (185/714) vs Bengal Gram 4.17% (2/48)
- `scripts/audit_phase7_metrics.py`: Script independently verifying test prediction matrix and metric calculations.
- `tests/test_phase7_inference.py`: 9 test assertions passing.

---

## 2. Inventory of Current Documentation & Drafts

### 2.1 Newly Created Paper Package (`outputs/paper/`)
1. `outputs/paper/RESEARCH_EVIDENCE_AUDIT.md`
2. `outputs/paper/RELATED_WORK_AND_REFERENCES.md`
3. `outputs/paper/MANUSCRIPT_DRAFT.md`
4. `outputs/paper/RESULTS_TABLES.md`
5. `outputs/paper/FIGURE_INVENTORY.md`
6. `outputs/paper/CLAIM_AND_CITATION_AUDIT.md`
7. `outputs/paper/DELIVERABLES_MANIFEST.md`

### 2.2 Earlier Intermediate Paper Materials (`outputs/reports/paper_materials/`)
- `outputs/reports/paper_materials/paper_outline.md`
- `outputs/reports/paper_materials/methodology.md`
- `outputs/reports/paper_materials/results_and_discussion.md`
- `outputs/reports/paper_materials/related_work_matrix.md`
- `outputs/reports/paper_materials/research_gaps_and_contributions.md`
- `outputs/reports/paper_materials/figures_and_tables_plan.md`
- `outputs/reports/paper_materials/audit_summary.md`
- `outputs/reports/paper_materials/evidence_audit/FINAL_EVIDENCE_LEDGER.md`
- `outputs/reports/paper_materials/evidence_audit/feature_schema_audit.md`
- `outputs/reports/paper_materials/evidence_audit/figures_and_tables_audit.md`
- `outputs/reports/paper_materials/evidence_audit/metric_reconciliation.md`
- `outputs/reports/paper_materials/evidence_audit/quartile_and_spatial_claims_audit.md`
- `outputs/reports/paper_materials/evidence_audit/research_integrity_and_citations_audit.md`
- `outputs/reports/paper_materials/evidence_audit/validation_and_tuning_audit.md`

---

## 3. Discrepancies and Contradictions Identified

### Contradiction 1: Outer Fold 0 Partition Counts (Phase 1)
- `MANUSCRIPT_DRAFT.md` Section IV-F and `RESULTS_TABLES.md` Table 4 state: Outer Fold 0 training partition has $N = 6,274$.
- Primary Evidence in `phase6f_cv_folds.csv` and `phase6g_pilot_fold0_report.json` proves:
  - Fold 0 Training: **6,335 parcels**
  - Fold 0 Validation: **1,508 parcels**
  - Sum: $6,335 + 1,508 = 7,843$.
- *Root Cause:* $6,274$ was an idealized calculation ($7,843 - 1,569 = 6,274$) based on dividing 7,843 equally into 5 folds ($1,568.6 \approx 1,569$), ignoring the actual Mandal group cluster sizes.

### Contradiction 2: Conflicting Phase 7 Status Records (Phase 6)
- `outputs/reports/paper_materials/evidence_audit/research_integrity_and_citations_audit.md` line 29 states: "Satellite feature extraction for Test.csv (2,718 farms) is NOT done; Blind test submission / test set score is NOT done."
- In contrast, `outputs/data/phase7/test_predictions.csv`, `outputs/reports/phase7/phase7_test_metrics.json`, and `outputs/reports/phase7/PHASE7_FINAL_SUMMARY.md` provide full, reproducible proof that Phase 7 was executed and audited.
- *Root Cause:* `research_integrity_and_citations_audit.md` is an older, stale pre-Phase-7 audit document written before Phase 7 execution.

### Contradiction 3: Overstated Geographic Generalization Claims (Phase 3)
- `MANUSCRIPT_DRAFT.md` and `FIGURE_INVENTORY.md` use phrases like:
  - "true out-of-region independent evaluation"
  - "strict out-of-region geographic transfer"
  - "zero spatial autocorrelation"
- *Fact:* The 35 test Mandals are in Telangana, the same state as the 116 training Mandals. While there is zero Mandal overlap, adjacent Mandals share geographic proximity. Zero Mandal overlap does not equal zero spatial autocorrelation or out-of-state geographic transfer.

### Contradiction 4: Overstated Causal Claims on GLCM Boundary Contamination (Phase 4)
- `MANUSCRIPT_DRAFT.md` Section V-C states that the study "proves that $3 \times 3$ convolution kernels overlapping non-agricultural field boundaries caused performance degradation."
- *Fact:* The study measures performance degradation stratified by parcel area quartiles, but did not directly measure the percentage of border pixels or run synthetic edge-masking experiments. It is a plausible physical hypothesis/explanation, not a directly measured causal proof.

### Contradiction 5: Unmatched In-Text Citation Gumma et al. (2022) (Phase 5)
- `MANUSCRIPT_DRAFT.md` Section II cites Gumma et al. (2022).
- `RELATED_WORK_AND_REFERENCES.md` only lists Gumma et al. (2020) (*Geocarto International*).
- Gumma et al. (2022) refers to a different multi-cropland product paper (*GIScience & Remote Sensing*).

### Contradiction 6: Arithmetic Representation in `CLAIM_AND_CITATION_AUDIT.md` (Phase 2)
- In `CLAIM_AND_CITATION_AUDIT.md`, the arithmetic check string is a long concatenated string of row additions that needs to be presented cleanly as verified row sums matching Table 9 and `phase7_test_metrics.json`.
