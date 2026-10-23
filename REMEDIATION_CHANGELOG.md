# Research Paper Remediation: Audit Changelog

**Project Title:** Multi-Temporal Sentinel-2 Spectral-Temporal Features for Seven-Class Crop Classification in Telangana: Spatial Cross-Validation and Independent Test Evaluation  
**Auditor Roles:** Senior Research Engineer, ML Experiment Auditor, Scientific Editor, Reproducibility Reviewer  
**Date:** 2026-10-23  

---

## Stage A & B: Phase 0 — Baseline Inventory and Remediation Planning

- **Created:** `REMEDIATION_BASELINE.md`
  - *Reason:* Document complete initial state of Git repository, working tree, primary evidence artifacts, feature schema manifests, split definitions, and contradictory secondary reports.
- **Created:** `REMEDIATION_PLAN.md`
  - *Reason:* Outline comprehensive 11-phase prioritized remediation workflow with clear dependencies, acceptance criteria, and stopping conditions.

---

## Stage C: Phase 1 — Hyperparameter Tuning and Partition Reconciliation

- **Created:** `PHASE1_TUNING_RECONCILIATION.md`
  - *Reason:* Authoritative audit of Outer Fold 0 training partition size ($N = 6,335$ parcels across 93 Mandals) and validation partition size ($N = 1,508$ parcels across 23 Mandals). Documented that $N = 6,274$ in previous drafts was an unverified idealized 80% approximation ($7,843 \times 0.8 = 6,274.4$). Verified that Trial 4 was selected solely on inner 3-fold grouped CV Macro-F1 ($0.6797$) with zero outer-validation leakage, and that tuning was exploratory on one outer-fold partition rather than full nested CV.

---

## Stage C: Phase 2 — Test Confusion Matrix and Metrics Reconciliation

- **Created:** `scripts/validate_stage_c_reconciliation.py`
  - *Reason:* Automated, reproducible Python validation script executing deterministic assertions on partition counts, zero Mandal overlap, 7×7 confusion matrix dimensions, row/column sums, diagonal trace (1,847), and headline metrics.
- **Created:** `outputs/remediation/regenerated_test_metrics.json`
  - *Reason:* Machine-readable regenerated output containing exact confusion matrix, row/column sums, and classification metrics directly derived from `test_predictions.csv` and `phase6f_cv_folds.csv`.
- **Created:** `PHASE2_TEST_METRICS_RECONCILIATION.md`
  - *Reason:* Authoritative audit report verifying the 7×7 test confusion matrix on the 2,265 evaluated parcels, documenting column sum arithmetic (Red Gram col sum = 14, Paddy col sum = 555), and establishing zero discrepancies with `phase7_test_metrics.json`.

---

## Stage D: Phases 3–8 — Scientific Claims, Citations, and Limitations Remediation

- **Created:** `PHASE3_GEOGRAPHIC_CLAIMS_AUDIT.md`
  - *Reason:* Audit and reframe geographic generalization claims. Removed unsupported claims of "out-of-region transfer" and "zero spatial autocorrelation." Reframed evaluation strictly as an out-of-Mandal holdout across 35 unseen Mandals within Telangana. Acknowledged residual cross-border spatial autocorrelation.
- **Created:** `PHASE4_TEXTURE_INTERPRETATION_AUDIT.md`
  - *Reason:* Reframe GLCM spatial texture findings. Confirmed global statistical non-significance ($p = 0.8600$, 95% CI $[-0.0066, +0.0092]$). Softened causal claims: reframed boundary contamination from an asserted proof to a well-grounded physical hypothesis consistent with observed parcel-size stratification. Clarified that quartile deltas are descriptive without independent subgroup hypothesis tests.
- **Created:** `PHASE5_CITATION_AUDIT.md`
  - *Reason:* Reconciled citation discrepancy for Gumma et al. (2022). Verified full bibliographic metadata and active DOI (`10.1080/15481603.2022.2088651`, *GIScience & Remote Sensing*) and established 100% two-way parity with zero orphan citations. Contextualized quantitative literature claims from Roberts et al. (2017), Lebourgeois et al. (2017), and Vuolo et al. (2018).
- **Created:** `PHASE6_PHASE7_STATUS_RECONCILIATION.md`
  - *Reason:* Investigated conflicting earlier audit stating Phase 7 test extraction was "NOT done." Proved via file timestamps that the conflicting document was authored at 13:06, prior to the user-authorized execution of Phase 7 (14:51–15:41). Officially classified the stale entry as superseded by the completed, verified Phase 7 artifacts.
- **Created:** `PHASE7_FEATURE_ABLATION_AUDIT.md`
  - *Reason:* Reconciled feature counts (140 S+T vs 160 S+T+P) and controlled ablation comparisons. Disclosed exact deltas: temporal windowing gain (+8.95 pp), class-weighting gain (+1.57 pp), Red Gram minority rescue (+7.37 pp, 0.3683 $\to$ 0.4420), and texture contribution under weighting (+0.02 pp, $p = 0.8600$). Prohibited cross-weighting attribution to texture.
- **Created:** `PHASE8_CLOUD_ATTRITION_AND_SCOPE_AUDIT.md`
  - *Reason:* Formalized explicit test denominators ($N = 2,265$ evaluated; 453 cloud-excluded). Documented crop-dependent attrition rates (Paddy 25.91% exclusion vs Bengal Gram 4.17% exclusion) and selection bias. Enforced strict scope governance: designated crop health (`category`), weather integration (ERA5), and FAO-56 irrigation scheduling as unvalidated future work.

---

## Stage E: Phase 9 — Manuscript, Tables, Figures, and Audit Remediation

- **Updated:** `outputs/paper/MANUSCRIPT_DRAFT.md` (and synced `./MANUSCRIPT_DRAFT.md`)
  - *Reason:* Systematic remediation of manuscript text. Reframed geographic evaluation to disjoint Mandal holdout within Telangana and acknowledged spatial autocorrelation limits; clarified exploratory tuning on Outer Fold 0 training partition ($N = 6,335$ train, $N = 1,508$ val); reconciled headline test metrics (81.55% accuracy, 0.7308 Macro-F1 on $N = 2,265$ evaluated parcels); qualified GLCM boundary contamination hypothesis; documented 453 cloud-excluded parcels; explicitly demarcated crop health and irrigation modules as future work; verified 13/13 citations.
- **Updated:** `outputs/paper/RESULTS_TABLES.md` (and synced `./RESULTS_TABLES.md`)
  - *Reason:* Reconciled Tables 1 through 10. Table 1 incorporates spatial autocorrelation note; Table 2 gives class-by-class cloud exclusion rates; Table 4 updates tuning partition sizes ($6,335$ / $1,508$); Tables 5–10 embed reconciled CV, ablation, parity, per-class, confusion matrix ($7\times7$), and quartile metrics.
- **Updated:** `outputs/paper/FIGURE_INVENTORY.md` (and synced `./FIGURE_INVENTORY.md`)
  - *Reason:* Updated formal publication captions for Figures 1 to 6, aligning terminology with verified findings and removing causal assertions.
- **Updated:** `outputs/paper/RELATED_WORK_AND_REFERENCES.md` (and synced `./RELATED_WORK_AND_REFERENCES.md`)
  - *Reason:* Updated Themes 1–5 and Gaps 1–4 with verified empirical resolutions. Retained both Gumma et al. (2020) and Gumma et al. (2022) with confirmed DOIs.
- **Updated:** `outputs/paper/CLAIM_AND_CITATION_AUDIT.md` (and synced `./CLAIM_AND_CITATION_AUDIT.md`)
  - *Reason:* Comprehensive 24-point claim audit matrix, 13/13 citation parity table, and structured confusion matrix arithmetic verification table.
- **Created:** `PHASE9_MANUSCRIPT_VALIDATION_REPORT.md` (and `outputs/paper/PHASE9_MANUSCRIPT_VALIDATION_REPORT.md`)
  - *Reason:* Document all material corrections, source evidence, modified files, automated validation runs, and compliance checklist.

---

## Stage F: Phase 10 — Automated Checks and Reproducibility

- **Created:** `scripts/validate_remediation_integrity.py`
  - *Reason:* Comprehensive automated validation script asserting dataset sample counts, 0 Mandal overlap, test confusion matrix sums, feature manifest counts, absence of forbidden overgeneralizations, two-way citation parity, and figure/table asset integrity. All 7 checks passed with 100% assertions satisfied.
- **Created:** `PHASE10_AUTOMATED_VALIDATION_REPORT.md`
  - *Reason:* Comprehensive validation report documenting execution of `validate_stage_c_reconciliation.py`, `validate_remediation_integrity.py`, `audit_phase7_metrics.py`, and `unittest test_phase7_inference.py`. Reconciled all 10 target audit domains against primary artifacts.

---

## Stage G: Phase 11 — Independent Adversarial Ponytail Final Audit

- **Created:** `PHASE11_INDEPENDENT_FINAL_AUDIT.md`
  - *Reason:* Adversarial, independent audit evaluating Git diff, primary source artifacts, reproducibility of headline metrics, scientific claim framing, and scope boundaries.
  - *Outcome:* Concluded with final verdict: **`PASS — READY FOR SUBMISSION`** (with standard note for authors to populate author attribution placeholders prior to final camera-ready/single-blind submission). Zero high-severity blockers remain.


