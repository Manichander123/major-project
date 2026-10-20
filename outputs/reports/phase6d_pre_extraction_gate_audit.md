# Phase 6D Pre-Extraction Gate Audit & Verification Report

**Date:** 2026-10-15  
**Audit Status:** COMPLETE & VERIFIED  
**Gate Verdict:** **READY FOR APPROVAL TO EXTRACT (Batch Hold Maintained)**

---

## 1. Executive Summary

This pre-extraction gate report audits the data integrity, source-label distribution, daily mosaicking implementation, and model feature boundaries prior to launching the Phase 6D Sentinel-2 expansion batch (372 farms).

### Key Findings
1. **Source-Label Discrepancy Resolved:** The actual physical file on disk [`datasss/Train.csv`](file:///c:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/datasss/Train.csv) (SHA256: `29d31920f5b9aaea8213d7993dedc50c5d19458906f85642d360d785bb07aa06`) contains 7,888 rows with exact counts: Paddy (2,268), Cotton (1,692), Maize (1,384), Bengal Gram (938), Groundnut (768), Chillies (614), and Red Gram (224). The prior Phase 6D narrative summary suffered from a transcription transposition error (e.g. reporting Red Gram as 1,266 and Bengal Gram as 244). Red Gram is indeed the minority class with 224 farms; after 50 were extracted in Phase 6C, 174 unextracted candidate farms remain. Expanding by 53 farms requires only 30.5% of available candidates, confirming 100% feasibility.
2. **Manifest and OOF Reconciliation:** All 352 manifest records, 348 clean feature rows, 4 cloud-obscured failed records, and 348 saved OOF predictions agree 100% with zero FarmID or crop label discrepancies.
3. **Daily Mosaicking Fully Implemented & Guarded:** [`src/satellite/s2_extractor.py`](file:///c:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/src/satellite/s2_extractor.py) includes `mosaic_same_date_scenes`, collapsing multiple same-date scenes into a single mosaic before temporal median reduction. Missing observations preserve strict `np.nan` (zero fabricated zeros). Clear pixels shine through cloudy scenes within same-date mosaics.
4. **Automated Gate Tests Passed (7/7):** [`tests/test_pre_extraction_gate.py`](file:///c:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/tests/test_pre_extraction_gate.py) passed all 7 tests covering same-date collapse, swath overweighting elimination, SCL cloud/shadow masking, clear-pixel passthrough, strict NaN preservation, checkpoint/resume safeguards, and manifest alignment.
5. **Representative Pilot Passed (7/7):** Across 7 farms representing all crops, seasons, and tile overlap conditions:
   - For single-tile farms (1.00x overlap), daily mosaic is mathematically identical (delta = 0.00000).
   - For heavy 3–4 tile overlaps (3.42x–3.67x), daily mosaicking eliminates swath overweighting distortions of up to +0.028 in NDVI and +0.035 in NDMI.
   - For monsoon windows with zero valid images, NaN is preserved without pipeline failure.
6. **Model Feature Boundaries Verified:** Satellite-only models use exactly 60 Sentinel-2 features. Multimodal models explicitly add 4 polygon geometry and 6 calendar features. Retrospective census classification is strictly separated from operational in-season early prediction, which forbids harvest date leakage (`HDate`, `duration_days`, `harvest_month`, `harvest_doy`).

---

## 2. Source-Label Reconciliation

| Crop | Train.csv Actual | Prior Claim | Delta | Status | Available Unextracted | Phase 6D Need | Draw Ratio | Feasibility |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bengal Gram** | 938 | 244 | +694 | Resolved | 888 | 53 | 6.0% | **Feasible** |
| **Chillies** | 614 | 948 | -334 | Resolved | 563 | 53 | 9.4% | **Feasible** |
| **Cotton** | 1,692 | 1,624 | +68 | Resolved | 1,642 | 53 | 3.2% | **Feasible** |
| **Groundnut** | 768 | 201 | +567 | Resolved | 718 | 53 | 7.4% | **Feasible** |
| **Maize** | 1,384 | 1,475 | -91 | Resolved | 1,333 | 53 | 4.0% | **Feasible** |
| **Paddy** | 2,268 | 2,130 | +138 | Resolved | 2,218 | 54 | 2.4% | **Feasible** |
| **Red Gram** | 224 | 1,266 | -1,042 | Resolved | 174 | 53 | 30.5% | **Feasible** |
| **Total** | **7,888** | **7,888** | **0** | **100% Agreement** | **7,536** | **372** | **4.9%** | **Feasible** |

*Artifact on disk:* [`outputs/data/phase6d_source_label_reconciliation.csv`](file:///c:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/data/phase6d_source_label_reconciliation.csv)

### Manifest vs Features vs OOF Agreement
- Manifest ([`outputs/data/s2_phase6c_manifest_350.csv`](file:///c:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/data/s2_phase6c_manifest_350.csv)): 352 rows (51 Chillies, 51 Maize, 50 Bengal Gram, 50 Cotton, 50 Groundnut, 50 Paddy, 50 Red Gram).
- Clean Features ([`outputs/data/s2_phase6c_350_features.csv`](file:///c:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/data/s2_phase6c_350_features.csv)): 348 success, 4 failed (2 Chillies: 1319251, 1322898; 1 Maize: 1298973; 1 Paddy: 1298212).
- OOF Predictions ([`outputs/data/phase6c_oof_predictions.csv`](file:///c:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/data/phase6c_oof_predictions.csv)): 348 rows.
- Alignment: 100% agreement, 0 label mismatches, 0 FarmID discrepancies.

---

## 3. Pilot Verification Results

*Artifact on disk:* [`outputs/data/phase6d_pilot_comparison.csv`](file:///c:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/data/phase6d_pilot_comparison.csv)

| FarmID | Crop | Season | Overlap Ratio | NDVI Raw | NDVI Daily | NDVI Delta | NDMI Delta | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1327181** | Bengal Gram | Rabi | 3.67x | 0.45350 | 0.47746 | +0.02397 | +0.00772 | **PASSED** |
| **1331695** | Red Gram | Kharif | 1.96x | 0.39395 | 0.39143 | -0.00252 | -0.00568 | **PASSED** |
| **1327617** | Maize | Rabi | 3.42x | 0.38378 | 0.41170 | +0.02792 | +0.03516 | **PASSED** |
| **1324368** | Paddy | Rabi | 1.92x | 0.65269 | 0.66134 | +0.00865 | +0.00076 | **PASSED** |
| **1307790** | Cotton | Kharif | 2.00x | 0.23634 | 0.23074 | -0.00560 | -0.00417 | **PASSED** |
| **1322087** | Groundnut | Rabi | 1.00x | 0.64542 | 0.64542 | +0.00000 | +0.00000 | **PASSED** |
| **1330504** | Chillies | Rabi | 1.81x | 0.41094 | 0.40561 | -0.00533 | -0.00160 | **PASSED** |

---

## 4. Model Feature Boundaries

1. **Satellite-Only Track (60 features):**
   - 14 indices across 4 windows = 56 features (`ndvi`, `evi`, `gndvi`, `savi`, `msavi`, `ndmi`, `ndwi`, `msi`, `ndre1`, `ndre2`, `psri`, `bsi`, `cire`, `cig` for `early`, `mid`, `late`, `full`).
   - 4 seasonal deltas: `ndvi_growth_delta`, `peak_to_early_ratio`, `ndvi_senescence_delta`, `moisture_loss_delta`.
   - Guaranteed zero inclusion of coordinates, administrative boundaries, or calendar labels.
2. **Multimodal Track (70 features):**
   - 60 satellite features + 4 geometry features (`area_ha`, `perimeter_m`, `compactness`, `vertex_count`) + 6 calendar features (`sowing_month`, `sowing_doy`, `harvest_month`, `harvest_doy`, `duration_days`, `Season`).
3. **Retrospective vs Operational Boundary:**
   - *Retrospective Classification:* Used for seasonal agricultural statistics, insurance claim validation, and land records post-harvest. Can utilize harvest dates and full-season temporal features.
   - *Operational Early-Season Prediction:* Conducted while crop is standing. Models trained for this setting must strictly omit harvest dates (`HDate`, `harvest_month`, `harvest_doy`, `duration_days`) and late/full-season composites to prevent future leakage.

---

## 5. Workload, Safeguards & Next Steps

- **Batch Size:** 372 farms selected across districts in [`outputs/data/s2_phase6d_recommended_manifest_372.csv`](file:///c:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/data/s2_phase6d_recommended_manifest_372.csv).
- **Execution Script:** [`scripts/extract_s2_expansion_phase6d.py`](file:///c:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/scripts/extract_s2_expansion_phase6d.py).
- **Phase 6C Protection:** All Phase 6C artifacts remain untouched.
- **Estimated Runtime:** ~3.5 to 4.5 minutes with 4 worker threads. Checkpointed every 5 farms.
- **Full batch extraction is paused awaiting approval.**
