# Phase 6I: Robustness Validation and Final Model Selection Report

**Timestamp:** 2026-10-20T08:01:20.861129  
**Cohort:** N = 7843 eligible farms across 7 crop classes  
**Cross-Validation:** 5-Fold StratifiedGroupKFold grouped strictly by Sub-District (Mandal)  
**Audit Status:** **PASS** (Protected hashes verified, zero Mandal overlap, zero leakage)  

---

## 1. Executive Summary & Final Recommendation

### Final Model Decision:
- **Primary Recommended Model:** **`Weighted S+T`** (140 Spectral + Temporal features, inverse-frequency weighted)
- **Secondary Comparison Model:** **`Weighted S+T+P`** (160 Spectral + Temporal + Spatial features, inverse-frequency weighted)

### Decision Rationale:
- 1. Statistical Parity: The +0.0002 Macro-F1 difference (0.7228 vs 0.7226) is statistically indistinguishable from zero (Mandal cluster bootstrap 95% CI [-0.0066, +0.0092], p=0.8600; paired t-test p=0.7777).
- 2. Minority Class Sensitivity: Weighted S+T achieves superior Red Gram F1 (0.4420 vs 0.4317) and Chillies F1 (0.5343 vs 0.5217), minimizing minority crop erasure.
- 3. Parsimony & Operational Cost: S+T relies on 140 spectral-temporal indices and eliminates the 20 GLCM spatial texture rasters, reducing Earth Engine processing latency, network payload, and memory footprint.
- 4. Boundary Robustness: Avoids spatial texture edge-pixel bleeding in smallholder parcels (<1,250 m2) where GLCM kernels exceed parcel interior radii.
- 5. Secondary Model Role: Weighted S+T+P is retained as the secondary comparison model for applications prioritizing raw overall accuracy (0.7905 vs 0.7881) or consolidated large parcels (Q3 Macro-F1: 0.7530).

---

## 2. Re-Calculated Four-Way Benchmark Matrix

| Model Configuration | Feature Count | Sample Weighting | Pooled Macro-F1 | Balanced Accuracy | Overall Accuracy | Weighted-F1 | Fold Macro-F1 (Mean ± Std) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`unweighted_S+T`** | 140 | Unweighted | **0.7069** | 0.6914 | 0.7873 | 0.7796 | 0.6997 ± 0.0228 |
| **`weighted_S+T`** | 140 | Inverse-Freq | **0.7226** | 0.718 | 0.7881 | 0.7857 | 0.7117 ± 0.0181 |
| **`unweighted_S+T+P`** | 160 | Unweighted | **0.7038** | 0.6895 | 0.79 | 0.7818 | 0.6965 ± 0.0173 |
| **`weighted_S+T+P`** | 160 | Inverse-Freq | **0.7228** | 0.7167 | 0.7905 | 0.7879 | 0.7131 ± 0.0133 |

---

## 3. Paired Statistical Comparison: Weighted S+T vs Weighted S+T+P

### Fold-Level Paired Performance:
| Spatial Fold | Val Mandal Count | Val N | Weighted S+T Macro-F1 | Weighted S+T+P Macro-F1 | Paired Delta (S+T+P - S+T) | Weighted S+T Accuracy | Weighted S+T+P Accuracy |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 0** | 23-24 | 1508 | 0.7269 | 0.7203 | **-0.0066** | 0.7812 | 0.7752 |
| **Fold 1** | 23-24 | 1600 | 0.7381 | 0.7357 | **-0.0024** | 0.8512 | 0.8506 |
| **Fold 2** | 23-24 | 1595 | 0.6919 | 0.7048 | **+0.0129** | 0.7994 | 0.8094 |
| **Fold 3** | 23-24 | 1530 | 0.6945 | 0.7064 | **+0.0119** | 0.7614 | 0.7725 |
| **Fold 4** | 23-24 | 1610 | 0.7073 | 0.6985 | **-0.0088** | 0.746 | 0.7435 |

### Group-Aware Uncertainty Analysis:
- **Paired t-test over 5 folds:** $t = 0.3037$, $p = 0.7765$ (Not statistically significant)
- **Mandal Cluster Bootstrap (B=1,000 resamples):**
  - Resampling Unit: Sub-District / Mandal (116 geographic clusters)
  - Mean Macro-F1 Delta: **+0.00094**
  - Standard Error: **0.00397**
  - **95% Bootstrap Confidence Interval:** **[-0.00663, +0.00917]**
  - Bootstrap Two-Sided $p$-value: **0.8600**
- **Conclusion:** The +0.0002 Macro-F1 difference is centered at zero and within random evaluation noise.

---

## 4. Per-Class Precision, Recall, and F1 Breakdown

| Crop Class | Support | Weighted S+T Precision | Weighted S+T Recall | Weighted S+T F1 | Weighted S+T+P Precision | Weighted S+T+P Recall | Weighted S+T+P F1 | F1 Delta (S+T+P - S+T) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bengal Gram** | 938 | 0.7916 | 0.8667 | **0.8275** | 0.8021 | 0.8774 | **0.8381** | **+0.0106** |
| **Chillies** | 601 | 0.5249 | 0.5441 | **0.5343** | 0.5146 | 0.5291 | **0.5217** | **-0.0126** |
| **Cotton** | 1679 | 0.7939 | 0.8261 | **0.8097** | 0.7947 | 0.8320 | **0.8129** | **+0.0032** |
| **Groundnut** | 768 | 0.8597 | 0.8698 | **0.8647** | 0.8795 | 0.8646 | **0.8720** | **+0.0073** |
| **Maize** | 1376 | 0.7492 | 0.6730 | **0.7090** | 0.7456 | 0.6773 | **0.7098** | **+0.0008** |
| **Paddy** | 2260 | 0.8664 | 0.8752 | **0.8708** | 0.8673 | 0.8792 | **0.8732** | **+0.0024** |
| **Red Gram** | 221 | 0.5467 | 0.3710 | **0.4420** | 0.5448 | 0.3575 | **0.4317** | **-0.0103** |

---

## 5. Parcel-Area Subgroup Analysis

| Area Quartile | Area Range (m²) | N | Weighted S+T Macro-F1 | Weighted S+T+P Macro-F1 | Area Macro-F1 Delta | Best Model |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Q1_small** | 54 - 1,257 m² | 1961 | 0.6598 | 0.6465 | **-0.0133** | **Weighted S+T** |
| **Q2_med_low** | 1,257 - 2,783 m² | 1961 | 0.7332 | 0.7184 | **-0.0148** | **Weighted S+T** |
| **Q3_med_high** | 2,784 - 5,517 m² | 1960 | 0.7400 | 0.7530 | **+0.0130** | **Weighted S+T+P** |
| **Q4_large** | 5,528 - 69,491 m² | 1961 | 0.7253 | 0.7219 | **-0.0034** | **Weighted S+T** |

---

## 6. Confusion Matrices (N=7843)

### Primary Model: Weighted S+T (140 Features):
```
Predicted ->    Bengal  Chilli  Cotton  Ground   Maize   Paddy  Red Gr
Bengal Gram  [    813       9      24       2      73       4      13 ]
Chillies     [      3     327      82      23      70      91       5 ]
Cotton       [     39      81    1387      37      49      56      30 ]
Groundnut    [      0      29      36     668      13      20       2 ]
Maize        [    134      81      71      23     926     126      15 ]
Paddy        [      9      91      71      21      87    1978       3 ]
Red Gram     [     29       5      76       3      18       8      82 ]
```

### Secondary Comparison Model: Weighted S+T+P (160 Features):
```
Predicted ->    Bengal  Chilli  Cotton  Ground   Maize   Paddy  Red Gr
Bengal Gram  [    823       8      19       2      69       5      12 ]
Chillies     [      2     318      84      23      73      96       5 ]
Cotton       [     40      80    1397      29      53      52      28 ]
Groundnut    [      0      31      42     664      12      18       1 ]
Maize        [    125      83      71      21     932     127      17 ]
Paddy        [      9      93      66      13      89    1987       3 ]
Red Gram     [     27       5      79       3      22       6      79 ]
```

---

## 7. Audit Verification & Leakage Guarantee

- **Historical Baseline SHA-256:** 24 baseline files + Phase 6F + Phase 6G files verified bit-for-bit intact.
- **Zero Mandal Overlap:** 100% spatial group separation across all 5 folds confirmed.
- **Zero Metadata Leakage:** Verified that zero FarmID, coordinate, administrative, or target-derived columns are present in feature sets.
- **Training-Partition-Only Weights:** All class weight factors calculated strictly inside fold-internal training splits.
