# Phase 7 — Test.csv Feature Extraction & Schema Audit Report

**Date:** 2026-10-23 15:13:08  
**Target Dataset:** `datasss/Test.csv` ($N = 2,718$ farm parcels)  
**Status:** EXTRACTION & FEATURE ENGINEERING COMPLETE  

---

## 1. Executive Summary

- **Total Test Parcels:** 2,718
- **Successfully Extracted:** 2265 (83.33%)
- **Extraction / Cloud Failures:** 453 (16.67%)
- **Feature Schema Parity:** 140 features ($S+T$ schema bit-for-bit matched with `phase6h_feature_schema_manifest.json`)
- **Leakage Invariant:** VERIFIED (Zero target, administrative, geometry, or identifier columns in predictor matrix)
- **Topological Repair:** 1 self-intersecting polygon (`FarmID 1321446`) successfully repaired via `shapely.validation.make_valid()`.

---

## 2. Extraction Pipeline Execution Details

- **Collection:** `COPERNICUS/S2_SR_HARMONIZED`
- **Cloud Threshold:** $\le 20\%$ scene cloud cover with SCL pixel-level cloud/cirrus/shadow masking.
- **Mosaicking:** Daily mosaic grouping to eliminate MGRS duplicate acquisition tile overlap.
- **Phenological Windows:**
  - Early Season: 0% to 25% of growth duration
  - Mid Season: 25% to 75% of growth duration
  - Late Season: 75% to 100% of growth duration
  - Full Season: 0% to 100% of growth duration
- **Spatial Resolution:** 10m bands reduced at 10m scale; 20m bands reduced at 20m scale. Reducer: `mean` + `stdDev`.
- **Batch Processing:** 11 atomic chunks (~250 farms/chunk) processed with multi-threaded GEE execution and exponential backoff retry.

---

## 3. Artifact Inventory

1. **Master Raw Bands:** `outputs\data\phase7\s2_test_raw_bands.csv` (2718 rows, 80 raw band statistics)
2. **Predictor Feature Matrix (140 features):** `outputs\data\phase7\s2_test_140_features.csv` (2718 rows, exactly 140 features)
3. **Full Dataset with Metadata:** `outputs\data\phase7\s2_test_full_features_with_metadata.csv`
4. **Individual Chunks:** `outputs\data\phase7/raw_bands_chunk_00.csv` to `chunk_10.csv`

---

## 4. Status Counts
```json
{
  "success": 2265,
  "failed": 453
}
```
