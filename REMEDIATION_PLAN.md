# Research Paper Remediation: Structured Execution Plan (`/gsd`)

**Project Title:** Multi-Temporal Sentinel-2 Spectral-Temporal Features for Seven-Class Crop Classification in Telangana: Spatial Cross-Validation and Independent Test Evaluation  
**Auditor Roles:** Senior Research Engineer, ML Experiment Auditor, Scientific Editor, Reproducibility Reviewer  
**Date:** 2026-10-23  
**Status:** PHASE 0 COMPLETED — PLAN AWAITING EXECUTION  

---

## 1. Prioritized Phase Structure and Dependencies

```
[Phase 0: Inventory & Baseline] (COMPLETE)
        │
        ├──> [Phase 1: Reconcile Tuning & Partition Counts] ──────┐
        │                                                        │
        ├──> [Phase 2: Recompute Test Confusion Matrix] ─────────┤
        │                                                        │
        ├──> [Phase 6: Reconcile Conflicting Phase 7 Audits] ────┤
        │                                                        │
        └──> [Phase 5: Citation & Claim Audit (Gumma 2022)] ─────┤
                                                                 ▼
[Phase 3: Correct Geographic Claims] <── [Phases 1, 2, 5, 6 Reconciled]
[Phase 4: Review Texture & Boundary Claims]
[Phase 7: Reconcile Feature Counts & Ablation Labels]
[Phase 8: Cloud Attrition & Denial-of-Service Limitations]
        │
        ▼
[Phase 9: Full Manuscript & Tables Remediation]
        │
        ▼
[Phase 10: Automated Python Verification Suite Execution]
        │
        ▼
[Phase 11: Final Reports & Audit Deliverables (`REMEDIATION_CHANGELOG.md`, `REMAINING_BLOCKERS.md`, etc.)]
```

---

## 2. Phase-by-Phase Remediation Breakdown

### Phase 1 — Reconcile Hyperparameter Tuning and Data Partitions
- **Objective:** Eliminate discrepancy between idealized approximation ($N=6,274$) and true partition count ($N=6,335$ train, $1,508$ val).
- **Primary Source:** `outputs/data/phase6f_cv_folds.csv`, `outputs/reports/phase6f_cv_folds_audit.json`, `outputs/reports/phase6g/phase6g_pilot_fold0_report.json`.
- **Action:** Update `MANUSCRIPT_DRAFT.md`, `RESULTS_TABLES.md`, and audit files with exact counts: Outer Fold 0 training partition is $N = 6,335$ parcels across 93 Mandals; validation partition is $N = 1,508$ parcels across 23 Mandals.
- **Acceptance Criteria:** Every partition number matches `phase6f_cv_folds.csv` exactly. Disclose that tuning was exploratory on Fold 0 training partition (not full nested CV).

### Phase 2 — Recompute and Reconcile the Test Confusion Matrix
- **Objective:** Ensure 100% numerical and structural parity across all documents displaying or analyzing the test confusion matrix.
- **Primary Source:** `outputs/data/phase7/test_predictions.csv`, `outputs/reports/phase7/phase7_test_metrics.json`.
- **Action:** Verify row sums (support), column sums (total predictions), diagonal sum (1,847 correct), off-diagonal sum (418 incorrect), and exact per-class precision/recall formulas. Clean up arithmetic presentation in `CLAIM_AND_CITATION_AUDIT.md`.
- **Acceptance Criteria:** Script-verified exact match across `test_predictions.csv`, `phase7_test_metrics.json`, `RESULTS_TABLES.md`, `MANUSCRIPT_DRAFT.md`, and `CLAIM_AND_CITATION_AUDIT.md`.

### Phase 3 — Correct Geographic Generalization Claims
- **Objective:** Eliminate scientific over-claiming regarding out-of-region transfer and spatial independence.
- **Action:** Replace "out-of-region transfer", "true out-of-region independent evaluation", and "zero spatial autocorrelation" with exact, scientifically defensible phrasing: "Independent held-out evaluation across 35 geographically unseen Mandals within Telangana (zero Mandal overlap with training)."
- **Acceptance Criteria:** Acknowledge potential cross-border spatial autocorrelation between adjacent Mandals; restrict geographic claims to Telangana. Update Figure 1 and Figure 3 captions.

### Phase 4 — Review Texture Features and Parcel-Boundary Interpretation
- **Objective:** Reframe boundary contamination from an asserted direct measurement to a well-grounded physical hypothesis/interpretation.
- **Action:** Soften language in Section V-C, Figure 4, and Abstract. Report that incorporating GLCM texture features resulted in no statistically significant global improvement ($p = 0.8600$) and that the observed performance drop in smallholder parcels ($<1,257\text{ m}^2$, $\Delta = -0.0133$) is consistent with boundary pixel contamination, rather than directly measured boundary overlap.
- **Acceptance Criteria:** Distinguish measured area-stratified performance metrics from the proposed physical mechanism.

### Phase 5 — Citation and Claim Audit
- **Objective:** Ensure two-way parity between in-text citations and bibliography entries; resolve the Gumma et al. (2022) citation.
- **Action:** 
  1. Add verified entry for Gumma et al. (2022) (*GIScience & Remote Sensing*) or reconcile with Gumma et al. (2020) depending on context.
  2. Verify all quantitative literature claims (e.g., Roberts et al. 2017, Lebourgeois et al. 2017) and ensure wording accurately reflects source papers.
- **Acceptance Criteria:** 100% two-way match between citations and bibliography; all DOIs active and verified.

### Phase 6 — Resolve Conflicting Phase 7 Status Records
- **Objective:** Reconcile stale pre-Phase-7 audit documents that claimed test set extraction and scoring was "NOT done".
- **Primary Evidence:** Phase 7 extraction scripts, test predictions CSV, metrics JSON, test suite.
- **Action:** Update `outputs/reports/paper_materials/evidence_audit/research_integrity_and_citations_audit.md` to reflect that Phase 7 was executed, audited, and verified subsequent to the earlier audit date, superseding the stale entry.
- **Acceptance Criteria:** Clear provenance documented explaining the chronology of Phase 6 pre-testing vs Phase 7 post-testing execution.

### Phase 7 — Reconcile Feature Counts and Ablation Terminology
- **Objective:** Ensure unambiguous naming and attribution for all experimental configurations.
- **Action:** Consistently label 140 features as "Spectral-Temporal ($S+T$)" and 160 features as "Spectral-Temporal-Spatial ($S+T+P$)". Distinguish unweighted S+T (0.7069) from unweighted S+T+P (0.7038). Confirm Red Gram F1 gain (+7.37 pp, 0.3683 $\to$ 0.4420).
- **Acceptance Criteria:** Every ablation table and text citation uses identical feature counts, configuration names, and exact arithmetic differences.

### Phase 8 — Cloud Attrition, Selection Bias, and Generalization Limits
- **Objective:** Ensure cloud-driven parcel loss is prominently disclosed and accurately analyzed.
- **Action:** Emphasize in the Abstract, Results, and Limitations that the test set metrics apply to the 2,265 evaluated parcels (83.33%) and not the entire 2,718 surveyed cohort. Highlight crop-dependent selection bias (25.91% loss on Paddy vs 4.17% on Bengal Gram). Explicitly state that downstream modules (crop health, ERA5 weather, FAO-56 irrigation) are future extensions.
- **Acceptance Criteria:** Denominators explicitly stated for all metrics; single-year limitation (2023) clearly disclosed.

### Phase 9 — Manuscript and Figure Remediation
- **Objective:** Implement all scientific, numerical, and structural corrections in `MANUSCRIPT_DRAFT.md`, `RESULTS_TABLES.md`, `FIGURE_INVENTORY.md`, and related documents.
- **Acceptance Criteria:** Full document consistency across text, tables, figure captions, and citations.

### Phase 10 — Reproducibility and Automated Checks
- **Objective:** Build and execute an automated Python validation script (`scripts/validate_remediation_integrity.py`) that checks:
  - Dataset sample counts (7,843 train, 2,718 test, 2,265 evaluated, 453 cloud).
  - Exact fold split counts (Fold 0: 6,335 / 1,508).
  - 0 Mandal overlap between train and test.
  - Confusion matrix sums, diagonal, accuracy, and per-class metrics.
  - Absence of forbidden phrases ("out-of-region transfer", "zero spatial autocorrelation").
  - Two-way citation parity.
- **Acceptance Criteria:** Script executes cleanly with 100% assertions passing.

### Phase 11 — Final Audit and Deliverables
- **Deliverables:**
  1. `REMEDIATION_PLAN.md` (updated with final execution status)
  2. `REMEDIATION_BASELINE.md`
  3. `REMEDIATION_CHANGELOG.md`
  4. `REMAINING_BLOCKERS.md`
  5. `FINAL_VALIDATION_REPORT.md`
  6. Remediated `outputs/paper/` suite and updated reproducibility test suite.

---

## 3. Decisions & Questions for User / Author Approval

1. **Gumma et al. (2022) Citation Choice:**  
   In `MANUSCRIPT_DRAFT.md` line 98, Gumma et al. (2022) is cited alongside Gumma et al. (2020) for mapping crops in India.  
   *Option A (Recommended):* Add the full, verified citation for Gumma et al. (2022) (*GIScience & Remote Sensing*, DOI: 10.1080/15481603.2022.2088651) to `RELATED_WORK_AND_REFERENCES.md` and `MANUSCRIPT_DRAFT.md`.  
   *Option B:* Consolidate to Gumma et al. (2020) (*Geocarto International*, DOI: 10.1080/10106049.2020.1864026) which specifically evaluates Sentinel-2 10-meter NDVI time-series in smallholder fields in India.

2. **Fold Dispersion Standard Deviation Convention:**  
   Across the 5 spatial cross-validation folds, mean Macro-F1 is 0.7117.  
   - Population SD (`ddof=0`): $\pm 0.0181$  
   - Sample SD (`ddof=1`): $\pm 0.0202$  
   *Recommendation:* Report both explicitly: "0.7117 ± 0.0181 (population SD) / ± 0.0202 (sample SD, ddof=1)".

3. **Status of Downstream Irrigation & Crop-Health Vision:**  
   *Recommendation:* Retain the secondary framework title (*"AI-Based Remote Sensing Framework for Crop Monitoring and Irrigation Decision Support Using Multi-Temporal Sentinel-2 Satellite Imagery"*) as the broader research vision, while keeping the primary, tightly scoped title (*"Multi-Temporal Sentinel-2 Spectral-Temporal Features for Seven-Class Crop Classification in Telangana: Spatial Cross-Validation and Independent Test Evaluation"*) as the authoritative manuscript title.
