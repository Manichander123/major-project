# Phase 9: Manuscript and Tables Remediation Validation Report

**Project Title:** Multi-Temporal Sentinel-2 Spectral-Temporal Features for Seven-Class Crop Classification in Telangana: Spatial Cross-Validation and Independent Test Evaluation  
**Auditor Roles:** Senior Research Engineer, ML Experiment Auditor, Scientific Editor, Reproducibility Reviewer  
**Date:** 2026-10-23  
**Status:** PHASE 9 REMEDIATION COMPLETE — ALL DELIVERABLES SYNCHRONIZED AND VERIFIED  

---

## 1. Executive Summary

Phase 9 (Stage E) of the research paper remediation workflow has been executed in strict accordance with the approved Stage E authorization. All scientific corrections, empirical findings, and methodological qualifications established across Stages C and D (Phases 1 through 8) have been systematically embedded into the manuscript text, results tables, figure inventory, related work, and audit matrices.

Zero models were retrained, no datasets or predictions were altered, and all reported numbers are directly verified against authoritative primary repository artifacts.

---

## 2. Inventory of Remediated Deliverables

The following core publication files were updated, synchronized, and verified across both `outputs/paper/` and repository root:

| # | Deliverable | Primary Workspace Location | Root Synced Location | Status |
| :-: | :--- | :--- | :--- | :---: |
| **1** | `MANUSCRIPT_DRAFT.md` | `outputs/paper/MANUSCRIPT_DRAFT.md` | `./MANUSCRIPT_DRAFT.md` | **Fully Remediated** |
| **2** | `RESULTS_TABLES.md` | `outputs/paper/RESULTS_TABLES.md` | `./RESULTS_TABLES.md` | **Fully Remediated** |
| **3** | `FIGURE_INVENTORY.md` | `outputs/paper/FIGURE_INVENTORY.md` | `./FIGURE_INVENTORY.md` | **Fully Remediated** |
| **4** | `RELATED_WORK_AND_REFERENCES.md` | `outputs/paper/RELATED_WORK_AND_REFERENCES.md` | `./RELATED_WORK_AND_REFERENCES.md` | **Fully Remediated** |
| **5** | `CLAIM_AND_CITATION_AUDIT.md` | `outputs/paper/CLAIM_AND_CITATION_AUDIT.md` | `./CLAIM_AND_CITATION_AUDIT.md` | **Fully Remediated** |
| **6** | `PHASE9_MANUSCRIPT_VALIDATION_REPORT.md` | `outputs/paper/PHASE9_MANUSCRIPT_VALIDATION_REPORT.md` | `./PHASE9_MANUSCRIPT_VALIDATION_REPORT.md` | **Generated** |
| **7** | `REMEDIATION_CHANGELOG.md` | `./REMEDIATION_CHANGELOG.md` | `./REMEDIATION_CHANGELOG.md` | **Updated** |
| **8** | `REMAINING_BLOCKERS.md` | `./REMAINING_BLOCKERS.md` | `./REMAINING_BLOCKERS.md` | **Updated** |

---

## 3. Detailed Audit of Applied Scientific Corrections

### 3.1 Geographic Generalization Reframing (Phase 3 Audit)
- **Previous Language:** Erroneous or exaggerated claims of "out-of-region transfer", "true external geographic generalization", and "zero spatial autocorrelation".
- **Remediated Language:** Reframed consistently throughout the manuscript (Abstract, Sections I, III.B, IV.E, V.D, VI, Table 1, Figure 1 caption) as:  
  *"Independent held-out evaluation across 35 unseen Mandals within Telangana, with zero Mandal overlap between training and test sets."*
- **Explicit Limitations Disclosed:**
  - Both cohorts originate from the state of Telangana, India.
  - An administrative Mandal-level holdout does not eliminate spatial autocorrelation; nearby Mandals may remain geographically close or share agro-climatic, soil, and management conditions.
  - The experiment is strictly an intra-state administrative holdout, not a cross-state evaluation, contiguous spatial-block evaluation, or multi-year evaluation.
- **Verification Rule:** Full grep search confirms zero instances of "out-of-region" or "zero spatial autocorrelation" across all remediated documents.

### 3.2 Hyperparameter Tuning Protocol & Partition Reconciliation (Phase 1 Audit)
- **Previous Language:** Unverified claim of "nested cross-validation" and partition count of $N = 6,274$.
- **Remediated Language:** Accurately disclosed in Section IV.F, Table 4, and AUD-10:
  - Outer Fold 0 training partition contains exactly **$N = 6,335$ parcels** across 93 Mandals.
  - Outer Fold 0 validation partition contains exactly **$N = 1,508$ parcels** across 23 Mandals ($6,335 + 1,508 = 7,843$).
  - Clarified that $6,274$ in earlier drafts was an unverified 80% calculation ($7,843 \times 0.8 = 6,274.4$).
  - Described tuning as an *exploratory bounded search* of 10 candidate configurations assessed via 3-fold grouped inner cross-validation strictly on the training partition of Outer Fold 0 (Trial 4 selected on inner Macro-F1: 0.6797).
  - Explicitly acknowledged that this represents exploratory tuning on a single outer fold rather than fully nested CV across all five folds, disclosing the potential for selection bias while noting Fold 0 exhibited representative distribution properties.

### 3.3 Test Metrics and Confusion Matrix Arithmetic (Phase 2 Audit)
- **Reconciled Primary Results on $N = 2,265$ Evaluated Parcels:**
  - Overall Accuracy: **0.815453** ($\sim 81.55\%$)
  - Macro-F1: **0.730810** ($\sim 0.7308$)
  - Balanced Accuracy: **0.732585** ($\sim 0.7326$)
  - Weighted-F1: **0.813671** ($\sim 0.8137$)
  - Correct Predictions: **1,847**
  - Misclassifications: **418**
- **Confusion Matrix Consistency:**
  - True Support (Row Sums): $[46, 311, 472, 241, 631, 529, 35]$ (Sum = $2,265$)
  - Predicted Totals (Column Sums): $[53, 328, 513, 253, 549, 555, 14]$ (Sum = $2,265$)
  - Red Gram Precision verified: $7 / 14 = \mathbf{0.5000}$ (50.0%)
  - Paddy Precision verified: $470 / 555 = \mathbf{0.8468}$ (84.68%)
  - Maize Precision verified: $495 / 549 = \mathbf{0.9016}$ (90.16%)
- **Verification Rule:** `scripts/validate_stage_c_reconciliation.py` passed with 100% assertions satisfied.

### 3.4 Feature Configurations and Ablation Deltas (Phase 7 Audit)
- **Schema Demarcation:**
  - Spectral-Temporal ($S+T$): **140 features** (34 Full-Season + $3 \times 34$ Seasonal Windows + 4 Dynamic Transition Deltas).
  - Spectral-Temporal-Spatial ($S+T+P$): **160 features** (140 S+T + 20 GLCM texture rasters across NIR B8).
- **Controlled Ablation Progression (Table 6, Figure 5):**
  - Full-Season static composite to Unweighted Multi-Window S+T: $0.6174 \to 0.7069$ (**+8.95 percentage points**).
  - Unweighted S+T to Weighted S+T: $0.7069 \to 0.7226$ (**+1.57 percentage points**).
  - Red Gram minority pulse recovery: $0.3683 \to 0.4420$ (**+7.37 percentage points**).
  - Weighted S+T to Weighted S+T+P (+20 GLCM): $0.7226 \to 0.7228$ (**+0.02 percentage points**, $p = 0.8600$).
- **Integrity Rule:** The effect of inverse-frequency class weighting is strictly isolated and never attributed to spatial texture features.

### 3.5 GLCM Texture and Parcel Boundary Contamination (Phase 4 Audit)
- **Statistical Results:**
  - Paired $t$-test across 5 spatial folds: $t = 0.3037$, $p = 0.7765$.
  - 1,000-iteration cluster bootstrap: $p = 0.8600$, 95% CI $[-0.0066, +0.0092]$.
- **Nuanced Scientific Interpretation:**
  - Reframed boundary contamination from an asserted causal proof to a *plausible physical hypothesis consistent with observed parcel-size stratification*.
  - Quartile breakdown ($N = 7,843$):
    - Q1 ($<1,257\text{ m}^2$): $\Delta = -0.0133$ (-1.33 pp)
    - Q2 ($1,257\text{–}2,783\text{ m}^2$): $\Delta = -0.0148$ (-1.48 pp)
    - Q3 ($2,784\text{–}5,517\text{ m}^2$): $\Delta = +0.0130$ (+1.30 pp)
    - Q4 ($>5,528\text{ m}^2$): $\Delta = -0.0034$ (-0.34 pp)
  - Explicitly stated that non-significant difference does not constitute proof of equivalence or prove texture is universally ineffective across all spatial resolutions.
  - Selected 140 S+T on the basis of model parsimony (Occam's razor).

### 3.6 Cloud Attrition and Selection Bias Accounting (Phase 8 Audit)
- **Explicit Test Denominators Enforced Everywhere:**
  - Total Surveyed Test Cohort: **$N = 2,718$ parcels**.
  - Successfully Evaluated Parcels: **$N = 2,265$ parcels (83.33%)**.
  - Cloud-Excluded Parcels: **$N = 453$ parcels (16.67%)**.
- **Crop-Dependent Loss Analysis (Table 2):**
  - Kharif Paddy experienced 25.91% loss (185/714 parcels excluded).
  - Rabi Bengal Gram experienced 4.17% loss (2/48 parcels excluded).
  - Explicitly noted that evaluated test parcels constitute a partially filtered sub-population, preventing synthetic imputation bias while acknowledging selection bias.

### 3.7 Temporal Scope & Downstream Framework Governance (Phase 8 Audit)
- **Single-Year Boundary:** Evaluated experiments are strictly confined to the 2023 cropping year.
- **Future Work Demarcation:**
  - Crop health status categorization (`category` labels) preserved but marked as future work.
  - ERA5-Land agrometeorological integration marked as future work.
  - FAO-56 Penman-Monteith dynamic irrigation advisory marked as future work.
  - Multimodal Sentinel-1 C-band SAR integration marked as future work.
  - Manuscript retains broader context title as framework vision while anchoring manuscript title and claims strictly to crop classification.

### 3.8 Bibliography and In-Text Citation Two-Way Parity (Phase 5 Audit)
- **Reconciliation of Gumma et al. (2022) vs Gumma et al. (2020):**
  - Both references retained for distinct, legitimate scientific claims:
    - Gumma et al. (2020) (*Geocarto International*, DOI: `10.1080/10106049.2020.1864026`): Field-level Sentinel-2 NDVI time series for crops in Telangana.
    - Gumma et al. (2022) (*GIScience & Remote Sensing*, DOI: `10.1080/15481603.2022.2088651`): Cloud-based GEE machine learning workflows for South Asian croplands.
- **Two-Way Parity:** Exactly 13 references in the bibliography, all 13 cited in text. Zero orphan citations; zero uncited bibliography entries. All 13 DOIs active and verified.

---

## 4. Verification Script Execution

The automated reconciliation test script `scripts/validate_stage_c_reconciliation.py` was executed:

```
$ python -X utf8 scripts/validate_stage_c_reconciliation.py
Running Stage C Automated Validation...
Phase 1 Validation: PASS
Phase 2 Validation: PASS
Saved machine-readable audit outputs to: .../outputs/remediation/regenerated_test_metrics.json
```

---

## 5. Summary of Phase 9 Deliverable Status

| Requirement Item | Description | Status |
| :--- | :--- | :---: |
| **Geographic Wording** | Disjoint Mandal holdout within Telangana, spatial autocorrelation qualified | **COMPLIANT** |
| **Tuning Counts** | Outer Fold 0 Train=$6,335$, Val=$1,508$; exploratory inner CV | **COMPLIANT** |
| **Test Metrics** | 81.55% Acc, 0.7308 Macro-F1, 1,847 correct, 418 errors on $N=2,265$ | **COMPLIANT** |
| **Confusion Matrix** | $7\times7$ exact match; Red Gram precision $0.5000$, Paddy $0.8468$ | **COMPLIANT** |
| **Feature Schema** | 140 S+T vs 160 S+T+P; deltas +8.95 pp, +1.57 pp, +7.37 pp, +0.02 pp | **COMPLIANT** |
| **Texture Interpretation** | Non-significant ($p=0.8600$); boundary contamination hypothesis qualified | **COMPLIANT** |
| **Cloud Denominators** | 2,718 total, 2,265 evaluated (83.33%), 453 excluded (16.67%) | **COMPLIANT** |
| **Scope Governance** | Crop classification validated; health and irrigation marked future work | **COMPLIANT** |
| **Citations** | 13/13 verified with DOIs; 100% two-way match | **COMPLIANT** |
| **Working Tree** | Zero Git commits/pushes; source datasets/models untouched | **COMPLIANT** |

**Conclusion:** Phase 9 is complete. The remediated document package is internally consistent, scientifically rigorous, and ready for Stage F (Phase 10: Automated Checks) and Stage G (Phase 11: Final Validation Report and independent `/ponytail` audit).
