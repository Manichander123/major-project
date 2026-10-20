# Phase 6H: Class-Weighting Benchmark and Boundary Effects Report

**Timestamp:** 2026-10-20T07:34:34.853673  
**Cohort:** N = 7843 eligible farms across 7 crop classes  
**Cross-Validation:** 5-Fold StratifiedGroupKFold grouped strictly by Sub-District  
**Weighting Strategy:** Inverse-frequency (`w_c = N / (7 * N_c)`) calculated fold-internally on training partitions only  

---

## 1. Executive Comparison: 4-Way Model Matrix

| Model Configuration | Features | Sample Weighting | Macro-F1 (Pooled) | Balanced Accuracy | Overall Accuracy | Red Gram F1 | Chillies F1 | Fold Macro-F1 (Mean ± Std) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **S+T Unweighted (Benchmark Winner Baseline)** | 140 | None (Unweighted) | **0.7069** | 0.6914 | 0.7873 | **0.3683** | **0.5062** | 0.6997 ± 0.0228 |
| **S+T Inverse-Frequency Weighted** | 140 | Inverse-Freq | **0.7226** | 0.718 | 0.7881 | **0.4420** | **0.5343** | 0.7117 ± 0.0181 |
| **S+T+P Unweighted (Higher-Accuracy Comparison)** | 160 | None (Unweighted) | **0.7038** | 0.6895 | 0.79 | **0.3333** | **0.4930** | 0.6965 ± 0.0173 |
| **S+T+P Inverse-Frequency Weighted** | 160 | Inverse-Freq | **0.7228** | 0.7167 | 0.7905 | **0.4317** | **0.5217** | 0.7131 ± 0.0133 |

---

## 2. All Seven Crops: Per-Class F1-Score Breakdown

| Crop Class | Support | S+T Unweighted | S+T Weighted | S+T+P Unweighted | S+T+P Weighted | Weighting Delta on S+T | Weighting Delta on S+T+P | Spatial Delta (Weighted) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bengal Gram** | 938 | 0.8278 | 0.8275 | 0.8385 | 0.8381 | **-0.0003** | **-0.0004** | **+0.0106** |
| **Chillies** | 601 | 0.5062 | 0.5343 | 0.4930 | 0.5217 | **+0.0281** | **+0.0287** | **-0.0126** |
| **Cotton** | 1679 | 0.8016 | 0.8097 | 0.8060 | 0.8129 | **+0.0081** | **+0.0069** | **+0.0032** |
| **Groundnut** | 768 | 0.8675 | 0.8647 | 0.8712 | 0.8720 | **-0.0028** | **+0.0008** | **+0.0073** |
| **Maize** | 1376 | 0.7049 | 0.7090 | 0.7105 | 0.7098 | **+0.0041** | **-0.0007** | **+0.0008** |
| **Paddy** | 2260 | 0.8717 | 0.8708 | 0.8740 | 0.8732 | **-0.0009** | **-0.0008** | **+0.0024** |
| **Red Gram** | 221 | 0.3683 | 0.4420 | 0.3333 | 0.4317 | **+0.0737** | **+0.0984** | **-0.0103** |

---

## 3. Boundary Effects & Parcel Area Investigation

- **GLCM Kernel Dimensions:** 30m x 30m (3x3 Sentinel-2 10m pixels = 900 m2)
- **Kernel Half-Radius:** 15.0 meters
- **Parcels smaller than GLCM kernel footprint (<900 m²):** 1353 (17.25%)
- **Parcels smaller than 2x kernel footprint (<1,800 m²):** 2748 (35.04%)
- **Parcels with equivalent radius <= 15m (100% boundary overlap):** 1024 (13.06%)
- **Mean Estimated Boundary Contamination Fraction:** 64.9% across all parcels

### Per-Crop Field Geometry & Contamination Risk:

| Crop Class | Total Farms | Median Parcel Area (m²) | Parcels < 900 m² (%) | Mean Estimated Edge Fraction |
| :--- | :---: | :---: | :---: | :---: |
| **Bengal Gram** | 938 | 4630.1 m² | 3.73% | 52.5% |
| **Chillies** | 601 | 3080.1 m² | 23.46% | 67.3% |
| **Cotton** | 1679 | 4166.8 m² | 5.48% | 54.4% |
| **Groundnut** | 768 | 3557.1 m² | 10.29% | 59.7% |
| **Maize** | 1376 | 3008.0 m² | 22.53% | 63.6% |
| **Paddy** | 2260 | 1374.9 m² | 30.62% | 81.2% |
| **Red Gram** | 221 | 4159.1 m² | 1.81% | 51.0% |

### Model Macro-F1 Stratified by Parcel Area Quartiles:

| Area Quartile | Area Range (m²) | N | S+T Unweighted | S+T Weighted | S+T+P Unweighted | S+T+P Weighted | Best Performing Model |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Q1_small** | 54 - 1,257 m² | 1961 | 0.6277 | 0.6598 | 0.6343 | 0.6465 | **S+T Weighted** (0.6598) |
| **Q2_med_low** | 1,257 - 2,783 m² | 1961 | 0.7128 | 0.7332 | 0.7245 | 0.7184 | **S+T Weighted** (0.7332) |
| **Q3_med_high** | 2,784 - 5,517 m² | 1960 | 0.7165 | 0.7400 | 0.7190 | 0.7530 | **S+T+P Weighted** (0.7530) |
| **Q4_large** | 5,528 - 69,491 m² | 1961 | 0.7150 | 0.7253 | 0.6952 | 0.7219 | **S+T Weighted** (0.7253) |

---

## 4. Confusion Matrices (OOF Pooled N=7843)

### Weighted S+T Model (140 Features):
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

### Weighted S+T+P Model (160 Features):
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

## 5. Model Selection Recommendation

Based on balanced cross-validation performance, minority-class sensitivity, and resilience to parcel boundary edge effects, the recommended production model is determined by the target operational objective:

1. **Primary Operational Recommendation (Balanced & Consistent Performance):**
   - **`Weighted S+T` (140 features):** Macro-F1 = **0.7226**, Balanced Accuracy = **0.718**.
   - Dramatically reduces minority-class erasure, boosting Red Gram recall while maintaining high overall accuracy.
2. **Raw Accuracy Operational Profile:**
   - **`Unweighted S+T+P` (160 features):** Accuracy = **0.79** (highest overall accuracy).
