# Phase 6H Stage 2: Spatial Texture Extraction Report

**Timestamp:** 2026-10-19T20:39:40.355837  
**Expected Cohort:** 7843 farms  
**Extracted:** 7843 farms (100.0% coverage)  
**Successful:** 7843 (100.0%)  
**Failed:** 0  
**Unique FarmIDs:** 7843  
**GLCM Features:** 20  

## Feature Statistics (Successful Farms)

| Feature | Finite Rate (%) | Mean | Std | Min | Max |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `glcm_contrast_early` | 93.94 | 0.648466 | 0.527584 | 0.0 | 8.383934 |
| `glcm_contrast_mid` | 98.69 | 0.676318 | 0.539548 | 0.002236 | 8.209371 |
| `glcm_contrast_late` | 97.86 | 0.697573 | 0.572018 | 0.0 | 8.220182 |
| `glcm_contrast_full` | 100.0 | 0.519543 | 0.360587 | 0.0 | 4.711233 |
| `glcm_dissimilarity_early` | 93.94 | 0.46153 | 0.221365 | 0.0 | 2.247735 |
| `glcm_dissimilarity_mid` | 98.69 | 0.476561 | 0.2261 | 0.002236 | 2.168723 |
| `glcm_dissimilarity_late` | 97.86 | 0.484572 | 0.228471 | 0.0 | 2.080761 |
| `glcm_dissimilarity_full` | 100.0 | 0.40875 | 0.180102 | 0.0 | 1.5897 |
| `glcm_homogeneity_early` | 93.94 | 0.787214 | 0.086848 | 0.353188 | 1.0 |
| `glcm_homogeneity_mid` | 98.69 | 0.780986 | 0.087549 | 0.395697 | 0.998882 |
| `glcm_homogeneity_late` | 97.86 | 0.778137 | 0.087308 | 0.382129 | 1.0 |
| `glcm_homogeneity_full` | 100.0 | 0.806505 | 0.075129 | 0.468145 | 1.0 |
| `glcm_energy_early` | 93.94 | 0.511303 | 0.145128 | 0.162386 | 1.0 |
| `glcm_energy_mid` | 98.69 | 0.501075 | 0.141432 | 0.176752 | 0.997522 |
| `glcm_energy_late` | 97.86 | 0.497204 | 0.140488 | 0.164913 | 1.0 |
| `glcm_energy_full` | 100.0 | 0.537671 | 0.136199 | 0.205994 | 1.0 |
| `glcm_correlation_early` | 93.94 | 0.487013 | 0.127195 | -0.015102 | 1.0 |
| `glcm_correlation_mid` | 98.69 | 0.483526 | 0.128627 | -0.161443 | 0.948526 |
| `glcm_correlation_late` | 97.86 | 0.488463 | 0.125517 | -0.258477 | 1.0 |
| `glcm_correlation_full` | 100.0 | 0.452528 | 0.128816 | -0.161443 | 1.0 |

## Next Steps

- [ ] Verify cohort coverage matches 7,843 eligible farms
- [ ] Confirm all 20 GLCM features present and finite-value rates acceptable
- [ ] Run `test_phase6h_spatial_texture.py` and `test_phase6h_stage2_extraction.py`
- [ ] Await authorization for Stage 3: Seven model-ablation experiments
