# Phase 4: Spatial Texture Features and Parcel-Boundary Interpretation Audit

**Project Title:** Multi-Temporal Sentinel-2 Spectral-Temporal Features for Seven-Class Crop Classification in Telangana: Spatial Cross-Validation and Independent Test Evaluation  
**Auditor Roles:** Senior Research Engineer, ML Experiment Auditor, Scientific Editor  
**Date:** 2026-10-23  
**Status:** COMPLETED & EMPIRICALLY RECONCILED  

---

## 1. Executive Summary

This audit reviews all numerical values, statistical significance claims, and physical interpretations regarding the 20 Gray-Level Co-occurrence Matrix (GLCM) spatial texture features and their parcel-size quartile ablation.

**Core Findings & Scientific Reconciliations:**
1. **Verified Global Parity:**  
   The 160-feature `Weighted S+T+P` model achieves a pooled Macro-F1 of **0.7228**, compared to **0.7226** for the 140-feature `Weighted S+T` model (a difference of $+0.0002$ or $+0.02\text{ percentage points}$).
2. **Statistical Significance Testing:**  
   - Paired Student's $t$-test across the 5 spatial folds: $t = 0.3037$, $p = 0.7765$.
   - 1,000-iteration Mandal cluster bootstrap: 95% Confidence Interval $[-0.0066, +0.0092]$, bootstrap $p = 0.8600$.
   - **Scientific interpretation:** The $+0.0002$ global difference is statistically indistinguishable from evaluation noise. However, failing to find a statistically significant gain *does not prove mathematical equivalence* across all conditions, nor does it imply that spatial texture is ineffective under different sensor spatial resolutions or field configurations.
3. **Parcel Area Quartile Breakdown:**  
   Stratification across parcel area quartiles ($N = 7,843$) shows:
   - **Q1 (Smallest, $54\text{–}1,257\text{ m}^2$, $N=1,961$):** S+T = 0.6598, S+T+P = 0.6465 ($\Delta = \mathbf{-0.0133}$).
   - **Q2 (Med-Low, $1,257\text{–}2,783\text{ m}^2$, $N=1,961$):** S+T = 0.7332, S+T+P = 0.7184 ($\Delta = \mathbf{-0.0148}$).
   - **Q3 (Med-High, $2,784\text{–}5,517\text{ m}^2$, $N=1,960$):** S+T = 0.7400, S+T+P = 0.7530 ($\Delta = \mathbf{+0.0130}$).
   - **Q4 (Largest, $5,528\text{–}69,491\text{ m}^2$, $N=1,961$):** S+T = 0.7253, S+T+P = 0.7219 ($\Delta = \mathbf{-0.0034}$).
4. **Boundary Contamination is a Plausible Explanation, Not a Directly Measured Causal Proof:**  
   The experiments directly measured size-stratified classification metrics, but did *not* directly measure out-of-boundary pixel overlap or edge-contamination percentages at the pixel level. Describing boundary contamination as an established causal proof or "proven physical mechanism" is scientifically unwarranted. It must be framed as a *plausible physical explanation consistent with the observed size-dependent degradation*.
5. **Binning Artifact Clarification:**  
   The 11 $\text{m}^2$ gap between Q3's upper bound ($5,517\text{ m}^2$) and Q4's lower bound ($5,528\text{ m}^2$) is a verified discrete binning artifact of `pandas.qcut` on discrete parcel measurements, not missing data.

---

## 2. Inventory of Substantive Revisions

| Location | Original Claim / Wording | Revised Scientific Wording | Scientific Justification |
| :--- | :--- | :--- | :--- |
| **`MANUSCRIPT_DRAFT.md` (Abstract, line 16)** | "...we empirically prove that adding 20 GLCM spatial texture features provides zero statistically significant global benefit... and actively degrades classification on smallholder parcels... due to boundary contamination." | "...incorporating 20 GLCM spatial texture features provided no statistically supported global improvement ($p = 0.8600$, 95% CI $[-0.0066, +0.0092]$), while parcel-area stratification revealed performance drops on smallholder parcels ($<1,257\text{ m}^2$, $\Delta = -0.0133$), consistent with boundary pixel contamination." | Replaces causal assertion with measured outcome and plausible interpretation. |
| **`MANUSCRIPT_DRAFT.md` (Sec. I, Contribution 3)** | "3. **Empirical Refutation of Spatial Texture in Smallholder Plots:** Through parcel-size quartile ablation, we show that incorporating 20 GLCM spatial texture rasters yields zero statistically significant global improvement... and actively degrades classification on smallholder parcels... due to boundary pixel contamination." | "3. **Evaluation of Spatial Texture in Smallholder Parcels:** We demonstrate that adding 20 GLCM spatial texture features yields no statistically supported overall gain under spatial cross-validation ($p = 0.8600$) and is associated with reduced performance in the smallest parcel quartile ($<1,257\text{ m}^2$, $\Delta = -0.0133$), providing an empirical basis for selecting the more parsimonious 140-feature representation." | Prevents universal over-generalization beyond the tested configuration. |
| **`MANUSCRIPT_DRAFT.md` (Sec. V-C, line 175)** | "...we conducted a parcel-size quartile ablation... establishes empirical proof of the boundary contamination effect..." | "...we conducted an exploratory parcel-size quartile stratification... This pattern is consistent with the physical hypothesis of boundary pixel contamination, where $3 \times 3$ convolution kernels ($900\text{ m}^2$) in smallholder plots capture non-crop boundaries." | Clarifies that quartile stratification is an exploratory post-hoc analysis without separate confidence intervals. |
| **`RESULTS_TABLES.md` (Table 10, line 133)** | "Physical Mechanism: Boundary Contamination: $3\times3$ GLCM convolution spills over field boundaries, sampling uncropped borders." | "Interpretation / Hypothesis: Consistent with boundary contamination: the $30\text{ m} \times 30\text{ m}$ kernel footprint constitutes a large fraction of smallholder fields, potentially sampling uncropped borders." | Frames mechanism as a hypothesis supported by physical dimensions rather than direct measurement. |
| **`FIGURE_INVENTORY.md` (Fig. 4 caption, line 43)** | "...caused by $3 \times 3$ convolution kernels overlapping non-agricultural field boundaries... balancing to zero net statistical significance globally..." | "...consistent with edge-pixel mixing under $3 \times 3$ convolution kernels in small plots... resulting in no statistically supported overall gain globally ($p = 0.8600$)." | Removes unmeasured causal assertion. |

---

## 3. Methodological Disclosures & Guidance

1. **Parsimony Justification:**  
   The primary justification for discarding the 20 GLCM features is **parsimony (Occam's razor)**: adding 20 computationally intensive spatial texture layers in Google Earth Engine yielded no statistically verifiable performance gain ($+0.0002$ Macro-F1, $p = 0.8600$), while introducing negative performance associations in the smallest parcel cohort.
2. **Absence of Subgroup Hypothesis Testing:**  
   The manuscript must explicitly disclose that no separate hypothesis tests or confidence intervals were computed for the individual quartile deltas.
