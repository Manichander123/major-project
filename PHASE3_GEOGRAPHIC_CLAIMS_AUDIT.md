# Phase 3: Geographic Generalization and Spatial Claims Audit

**Project Title:** Multi-Temporal Sentinel-2 Spectral-Temporal Features for Seven-Class Crop Classification in Telangana: Spatial Cross-Validation and Independent Test Evaluation  
**Auditor Roles:** Senior Research Engineer, ML Experiment Auditor, Scientific Editor  
**Date:** 2026-10-23  
**Status:** COMPLETED & SCIENTIFICALLY RECONCILED  

---

## 1. Executive Summary

This audit evaluates all geographic generalization claims, spatial validation terminology, and spatial autocorrelation assertions across the research manuscript, tables, figures, and supporting audits.

**Core Findings & Required Scientific Revisions:**
1. **Both Cohorts are Confined to Telangana:**  
   Both the training cohort (116 Mandals) and the test cohort (35 Mandals) reside entirely within the state of Telangana, India. While there is **zero Mandal overlap** between the training and test cohorts, the test set does not constitute an "out-of-region" or "cross-state" transfer experiment.
2. **"Zero Spatial Autocorrelation" is an Overstatement:**  
   Enforcing `StratifiedGroupKFold` grouped by Mandal guarantees **zero Mandal group overlap** (no parcel from the same Mandal appears in both training and validation). However, adjacent Mandals share physical boundaries, common agro-climatic zones, and river basins. Spatial autocorrelation decays with geographic distance; it is not reduced to zero simply by administrative grouping.
3. **Mandal Holdout vs Spatial Block vs External State:**  
   The experimental design is an **administrative-unit group cross-validation (Mandal holdout)** within Telangana. It must be clearly distinguished from:
   - Contiguous distance-buffered spatial block validation (e.g., spatial checkerboard blocks with buffer zones).
   - External-state geographic transfer (evaluating in Maharashtra, Andhra Pradesh, or Karnataka).
   - Multi-year temporal transfer (evaluating on subsequent cropping years).

---

## 2. Inventory of Substantive Revisions

| Document Location | Original Unreconciled Text / Claim | Evidence-Supported Reconciled Text | Scientific Justification |
| :--- | :--- | :--- | :--- |
| **`MANUSCRIPT_DRAFT.md` (Abstract, line 16)** | "...evaluating across 35 geographically unseen Mandals (zero overlap with training)... confirms the spatial robustness... across unseen administrative units..." | Retain "held-out evaluation across 35 unseen Mandals within Telangana (zero Mandal overlap with training)". Ensure no "out-of-region" phrasing is used. | Accurately describes administrative holdout without implying cross-state transfer. |
| **`MANUSCRIPT_DRAFT.md` (Sec. I, Contribution 5)** | "5. **Out-of-Region Independent Test Evaluation with Cloud Accounting**" | "5. **Independent Held-Out Test Evaluation across Unseen Mandals with Cloud Accounting**" | Both cohorts are in Telangana; evaluation is out-of-Mandal, not out-of-region. |
| **`MANUSCRIPT_DRAFT.md` (Sec. III-B, line 75)** | "...establishing a strict out-of-region geographic transfer test." | "...establishing a strict out-of-Mandal geographic holdout evaluation within Telangana." | Avoids overstating geographic distance of test cohort. |
| **`MANUSCRIPT_DRAFT.md` (Sec. IV-E, line 144)** | "...ensuring zero spatial autocorrelation across the validation boundary." | "...preventing parcel co-location leakage and ensuring zero administrative group overlap between training and validation folds." | Zero group overlap does not eliminate cross-border spatial autocorrelation. |
| **`MANUSCRIPT_DRAFT.md` (Sec. V-D, line 185)** | "Independent Test Set Evaluation across 35 Unseen Mandals: To evaluate true geographic generalization..." | "Independent Test Set Evaluation across 35 Unseen Mandals: To evaluate model transfer across unseen administrative units within Telangana..." | Accurately describes domain of generalization. |
| **`MANUSCRIPT_DRAFT.md` (Sec. V-F, line 200)** | "...must not be interpreted as evidence of superior out-of-region generalization." | "...must not be interpreted as evidence of superior geographic generalization." | Scope is within Telangana. |
| **`RESULTS_TABLES.md` (Table 1, line 18)** | `0 Mandals (Strict Out-of-Region Transfer)` | `0 Mandals (Disjoint Mandal Holdout within Telangana)` | Removes "out-of-region" label. |
| **`RESULTS_TABLES.md` (Table 1, line 19)** | `Frozen Model Out-of-Region Inference` | `Frozen Model Out-of-Mandal Inference` | Precision in split description. |
| **`FIGURE_INVENTORY.md` (Fig. 1 caption, line 26)** | "(B) True out-of-region independent evaluation across 35 geographically unseen Mandals ($N = 2,265$)." | "(B) Independent held-out evaluation across 35 unseen Mandals within Telangana ($N = 2,265$)." | Replaces overreaching claim with verified split design. |
| **`FIGURE_INVENTORY.md` (Fig. 3 caption, line 37)** | "...demonstrates that the multi-temporal 140-feature representation is robust to regional micro-climatic and edaphic shifts within Telangana." | "...illustrates performance dispersion across distinct held-out Mandal clusters within Telangana ($\sigma_{\text{pop}} = \pm 0.0181$, $\sigma_{\text{samp}} = \pm 0.0202$ around mean Macro-F1 = 0.7117)." | Softens unmeasured causal claims regarding edaphic/micro-climatic robustness. |

---

## 3. Explicit Acknowledgments of Geographic Boundaries

In Section VI (*Limitations and Threats to Validity*) of `MANUSCRIPT_DRAFT.md`:
1. **Geographic Confinement to Semi-Arid Telangana:**  
   The model was trained on 116 Mandals and evaluated on 35 unseen Mandals, all located within Telangana’s semi-arid tropical zone. Performance in sub-humid, humid, or high-altitude agro-ecological regions across India remains unverified.
2. **Residual Spatial Autocorrelation:**  
   While Mandal grouping eliminates direct farm-to-farm co-location leakage within villages, neighboring Mandals share common weather systems, irrigation canal commands, and soil associations. Spatial autocorrelation across Mandal boundaries cannot be assumed to be zero.
