# System Architecture

## Architecture Overview
The system implements an AI-based remote sensing pipeline for multi-temporal Sentinel-2 crop classification and agricultural monitoring in Telangana, India.

```
datasss/Train.csv (7,888 farms)
       │
       ▼
[src/data/load_data.py] ──> GeoDataFrame / DataFrame
       │
       ▼
[src/data/validate_data.py] ──> Geometry & Label Verification
       │
       ▼
[src/satellite/s2_extractor.py] <── GEE (S2 Level-2A)
       │  (Daily Mosaic, Cloud Masking, Zonal Stats per Window)
       ▼
outputs/data/s2_phase6e_chunk_*.csv ──> Consolidated Raw Bands
       │
       ▼
[src/features/indices.py] ──> Spectral & Temporal Vegetation Indices
       │  (NDVI, EVI, SAVI, NDRE, NDMI, PSRI, Dynamic Deltas)
       ▼
[src/models/train_pilot_models.py]
       │
       ├── Track 1: Satellite-Only (Spectral indices across windows)
       ├── Track 2: Multimodal (Satellite + Geometry + Calendar)
       │
       ▼
Spatial Block CV (StratifiedGroupKFold on Sub-District, 5 Folds)
       │
       ▼
Outputs: OOF Predictions, Confusion Matrices, JSON & Markdown Benchmark Reports
```

## Core Modules
1. **`src.data.load_data`**: Loads tabular and spatial datasets from `datasss/`, parses WKT polygons into EPSG:4326 geometries, enforces schema datatypes.
2. **`src.data.validate_data`**: Validates non-null constraints, geometry validity, SDate/HDate temporal validity, duplicates, and returns structured quality reports.
3. **`src.satellite.s2_extractor`**: Handles Earth Engine communication, builds phenological date windows (early, mid, late, full), performs daily mosaicking to eliminate MGRS overlap duplicates, applies SCL cloud/shadow masking, computes polygon zonal statistics.
4. **`src.features.indices`**: Computes 12+ standard and red-edge vegetation, moisture, and chlorophyll indices with strict NaN preservation and zero division safeguards.
5. **`src.models.train_pilot_models`**: Encapsulates spatial CV cross-validation, median imputation strictly inside training folds, Random Forest and XGBoost classification, OOF generation, and metric calculation.
