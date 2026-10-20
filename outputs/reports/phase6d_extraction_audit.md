# Phase 6D Dataset Expansion & Extraction Verification Report

**Execution Date:** 2026-10-15  
**Execution Pipeline:** [`scripts/extract_s2_expansion_phase6d.py`](file:///c:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/scripts/extract_s2_expansion_phase6d.py)  
**Status:** **EXTRACTION COMPLETE & VERIFIED — STOPPED BEFORE MODEL TRAINING**

---

## 1. Execution Summary

The 372-farm Phase 6D extraction batch was executed with 4 parallel worker threads in Google Earth Engine using the corrected same-date daily mosaicking pipeline and checkpoint safeguards.

- **Phase 6D Manifest:** [`outputs/data/s2_phase6d_recommended_manifest_372.csv`](file:///c:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/data/s2_phase6d_recommended_manifest_372.csv) (372 farms, 0 overlap with Phase 6C).
- **Phase 6D Batch 3 Output:** [`outputs/data/s2_phase6d_batch3_raw_bands.csv`](file:///c:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/data/s2_phase6d_batch3_raw_bands.csv) (366 success, 6 failed due to monsoon cloud cover).
- **Combined Raw Bands:** [`outputs/data/s2_phase6d_724_raw_bands.csv`](file:///c:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/data/s2_phase6d_724_raw_bands.csv) (724 farms).
- **Combined Manifest:** [`outputs/data/s2_phase6d_manifest_724.csv`](file:///c:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/data/s2_phase6d_manifest_724.csv) (724 farms).
- **Combined Features:** [`outputs/data/s2_phase6d_724_features.csv`](file:///c:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/data/s2_phase6d_724_features.csv) (724 rows, 168 columns: 56 multi-temporal indices + 4 dynamics + 10 raw bands/windows + polygon geometry).
- **Quality Report:** [`outputs/reports/phase6d_data_quality.json`](file:///c:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/reports/phase6d_data_quality.json).
- **Phase 6C Artifacts:** All Phase 6C artifacts (`s2_phase6c_350_*`, `phase6c_oof_predictions.csv`, `phase6c_model_benchmark.json`) remain 100% untouched.

---

## 2. Sample Counts & Valid Retained Observations

| Crop | Manifest Total | Failed Extractions | Clean Valid Farms | % of Clean Dataset |
| :--- | :---: | :---: | :---: | :---: |
| **Bengal Gram** | 103 | 0 | **103** | 14.4% |
| **Paddy** | 104 | 1 | **103** | 14.4% |
| **Groundnut** | 103 | 0 | **103** | 14.4% |
| **Maize** | 104 | 2 | **102** | 14.3% |
| **Cotton** | 103 | 1 | **102** | 14.3% |
| **Red Gram** | 103 | 1 | **102** | 14.3% |
| **Chillies** | 104 | 5 | **99** | 13.9% |
| **Total** | **724** | **10** | **714** | **100.0%** |

*Note on Failures (10 total: 4 from Phase 6C, 6 from Phase 6D):* All failed farms correspond to very short duration Kharif monsoon sowings (e.g. 1 to 4 days) where 100% cloud cover obscured all available satellite scenes.

---

## 3. Data Integrity & Verification Audit

1. **Duplicate FarmIDs:** 0 duplicates in manifest; 0 duplicates in features table.
2. **Crop Label Alignment:** 100% match against [`datasss/Train.csv`](file:///c:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/datasss/Train.csv) (0 mismatches across all 724 records).
3. **No Fabricated Zeros:** All 10 failed farms contain `np.nan` for band means and indices (0.0 was never substituted for missing data).
4. **Spectral Value Ranges (Clean Rows):**
   - `ndvi_full`: Min = 0.1055, Max = 0.8743, Mean = 0.4900 (0 infs)
   - `evi_full`: Min = 0.0506, Max = 0.7199, Mean = 0.3277 (0 infs)
   - `ndmi_full`: Min = -0.2492, Max = 0.4776, Mean = 0.0649 (0 infs)
   - `savi_full`: Min = 0.0594, Max = 0.6053, Mean = 0.3110 (0 infs)
   - `ndwi_full`: Min = -0.7705, Max = -0.2397, Mean = -0.5182 (0 infs)
5. **Class Balance:** Target of ~100 valid observations per crop is fulfilled across all 7 classes (99 to 103 per crop, 714 clean valid farms total).

---

## 4. Next Step
The pipeline has stopped before model training. Ready for spatial cross-validation benchmarking on the expanded 714-farm dataset.
