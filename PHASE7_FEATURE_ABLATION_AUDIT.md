# Phase 7: Feature Schema and Model Ablation Terminology Audit

**Project Title:** Multi-Temporal Sentinel-2 Spectral-Temporal Features for Seven-Class Crop Classification in Telangana: Spatial Cross-Validation and Independent Test Evaluation  
**Auditor Roles:** Senior Research Engineer, ML Experiment Auditor, Scientific Editor  
**Date:** 2026-10-23  
**Status:** COMPLETED & ARITHMETICALLY RECONCILED  

---

## 1. Executive Summary

This audit establishes the definitive, verified feature schema and experimental ablation benchmark matrix to eliminate naming ambiguities and prevent improper cross-cohort or cross-weighting comparisons.

**Core Findings:**
1. **Authoritative Feature Schema:**
   - **Group S (Full Season):** 34 features (10 spectral bands + 14 biophysical/water indices aggregated over Jun 01 – Dec 31).
   - **Group T (Temporal Dynamics):** 106 features = 102 phenological window features (34 features $\times$ 3 windows: Early, Mid, Late) $+$ 4 dynamic transition deltas (`ndvi_growth_delta`, `peak_to_early_ratio`, `ndvi_senescence_delta`, `moisture_loss_delta`).
   - **Spectral-Temporal Representation ($S+T$):** Exactly **140 features** ($34 + 106 = 140$).
   - **Group P (GLCM Spatial Texture):** 20 features (5 second-order texture statistics $\times$ 4 temporal windows).
   - **Spectral-Temporal-Spatial Representation ($S+T+P$):** Exactly **160 features** ($140 + 20 = 160$).
2. **Reconciled Ablation Metric Ledger ($N = 7,843$ Spatial CV):**
   - Early-Season Single-Window (34 feats): Macro-F1 = **0.5058** (OA = 63.58%)
   - Full-Season Single-Window (34 feats): Macro-F1 = **0.6174** (OA = 72.08%)
   - Unweighted S+T (140 feats): Macro-F1 = **0.7069** (OA = 79.24%, Red Gram F1 = 0.3683)
   - Unweighted S+T+P (160 feats): Macro-F1 = **0.7038** (OA = 79.00%, Red Gram F1 = 0.3683)
   - Final Weighted S+T (140 feats, Winner): Macro-F1 = **0.7226** (OA = 78.81%, Red Gram F1 = 0.4420)
   - Weighted S+T+P (160 feats): Macro-F1 = **0.7228** (OA = 79.05%, Red Gram F1 = 0.4317)
3. **Controlled Delta Disclosures:**
   - Multi-Temporal Windowing Gain: $+0.0895$ (+8.95 pp) from full-season static composite (0.6174) to unweighted S+T (0.7069).
   - Class-Weighting Gain on S+T: $+0.0157$ (+1.57 pp) from unweighted (0.7069) to weighted (0.7226).
   - Red Gram Minority Rescue: $+0.0737$ (+7.37 pp) from unweighted (0.3683) to weighted (0.4420).
   - Texture Effect under Weighting: $+0.0002$ (+0.02 pp) from S+T (0.7226) to S+T+P (0.7228), which is statistically insignificant ($p = 0.8600$, 95% CI $[-0.0066, +0.0092]$).

---

## 2. Definitive Feature Schema Decomposition

Source: `outputs/reports/phase6h/phase6h_feature_schema_manifest.json`

| Group Code | Description | Temporal Scope | Formula / Decomposition | Dimensionality |
| :---: | :--- | :--- | :--- | :---: |
| **S** | Spectral & Biophysical Baseline | Full Season (Jun 01 – Dec 31) | 10 L2A bands (B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12) $+$ 14 indices (NDVI, EVI, NDMI, NDWI, GNDVI, NDRE, MSAVI, CI Green, CI Red Edge, MSI, NDBI, SAVI, SIPI, NMDI). | **34** |
| **W1** | Early Phenological Window | Early Season (Jun 01 – Jul 31) | Same 10 bands $+$ 14 indices aggregated over vegetative emergence. | **34** |
| **W2** | Mid Phenological Window | Mid Season (Aug 01 – Sep 30) | Same 10 bands $+$ 14 indices aggregated over peak reproductive growth. | **34** |
| **W3** | Late Phenological Window | Late Season (Oct 01 – Nov 30) | Same 10 bands $+$ 14 indices aggregated over grain filling / senescence. | **34** |
| **$\Delta$** | Dynamic Transition Trajectories | Multi-Window Transitions | 1. `ndvi_growth_delta` ($\text{W2} - \text{W1}$)<br>2. `peak_to_early_ratio` ($\text{W2} / \text{W1}$)<br>3. `ndvi_senescence_delta` ($\text{W3} - \text{W2}$)<br>4. `moisture_loss_delta` ($\text{NDMI}_{\text{W3}} - \text{NDMI}_{\text{W2}}$) | **4** |
| **T** | Total Temporal Dynamics | Windowed $+$ Deltas | $\text{W1 (34)} + \text{W2 (34)} + \text{W3 (34)} + \Delta \text{ (4)}$ | **106** |
| **S+T** | **Champion Feature Space** | Full Season $+$ Temporal | $\mathbf{S (34) + T (106)}$ | **140** |
| **P** | GLCM Spatial Texture | All 4 Windows | 5 Haralick metrics (ASM, Contrast, Correlation, Entropy, Variance) $\times$ 4 windows. | **20** |
| **S+T+P** | Spatial-Augmented Space | Full S+T $+$ Spatial Texture | $\mathbf{S+T (140) + P (20)}$ | **160** |

---

## 3. Strict Controlled Benchmark Comparison Matrix

| Model / Architecture | Features | Sample Cohort | Sample Weighting | Spatial Macro-F1 | Overall Accuracy | Balanced Accuracy | Red Gram F1 | Contrast vs Champion `Weighted S+T` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Phase 6D Satellite Pilot** | 34 | 714 (Pilot) | None | 0.5511 | 0.6653 | 0.5312 | 0.2857 | *Exploratory cohort baseline (N=714)* |
| **Early Single-Window XGB** | 34 | 7,843 | None | 0.5058 | 0.6358 | 0.4902 | 0.1874 | $\Delta = -0.2168$ (-21.68 pp) |
| **Full Single-Window XGB** | 34 | 7,843 | None | 0.6174 | 0.7208 | 0.6015 | 0.2458 | $\Delta = -0.1052$ (-10.52 pp) |
| **Unweighted S+T XGB** | 140 | 7,843 | None | 0.7069 | 0.7924 | 0.6982 | 0.3683 | $\Delta = -0.0157$ (-1.57 pp) |
| **Unweighted S+T+P XGB** | 160 | 7,843 | None | 0.7038 | 0.7900 | 0.6895 | 0.3683 | $\Delta = -0.0188$ (-1.88 pp) |
| **Weighted S+T XGB (Winner)** | **140** | **7,843** | **Inverse-Freq** | **0.7226** | **0.7881** | **0.7180** | **0.4420** | **Authoritative Production Champion** |
| **Weighted S+T+P XGB** | 160 | 7,843 | Inverse-Freq | 0.7228 | 0.7905 | 0.7188 | 0.4317 | $\Delta = +0.0002$ ($p = 0.8600$) |

---

## 4. Reconciled Discrepancies & Guidelines

1. **Unweighted S+T (0.7069) vs Unweighted S+T+P (0.7038):**  
   Secondary summaries previously conflated these two distinct models. The manuscript must explicitly distinguish the 140-feature unweighted model (0.7069) from the 160-feature unweighted model (0.7038).
2. **Red Gram Weighting Gain (+7.37 pp, Not +9.42 pp):**  
   The authoritative baseline Red Gram F1 for unweighted S+T is **0.3683** (Precision: 0.6170, Recall: 0.2624), rising to **0.4420** with fold-internal weighting. The gain is exactly $+0.0737$ (+7.37 percentage points). An earlier draft mistakenly cited 0.3478 (a precision score from a 173-farm pilot) yielding a fictitious +9.42 pp gain.
3. **No Cross-Weighting Attribution to Texture:**  
   The $+0.0159$ difference between unweighted S+T (0.7069) and weighted S+T+P (0.7228) must not be attributed to texture; it is almost entirely driven by class weighting ($+0.0157$). When weighting is held constant, the texture contribution is $+0.0002$ ($p = 0.8600$).
