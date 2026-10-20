# System Integrations

## External Services & Cloud APIs
- **Google Earth Engine (GEE)**
  - Project ID: `scientific-axle-446313-i6`
  - Collection: `COPERNICUS/S2_SR_HARMONIZED`
  - Authentication: User service credentials initialized via `ee.Initialize(project=...)`
  - Integration logic: `src/satellite/s2_extractor.py` (`EarthEngineManager`)
  - Bands queried: 10m bands (`B2`, `B3`, `B4`, `B8`), 20m bands (`B5`, `B6`, `B7`, `B8A`, `B11`, `B12`), and QA band `SCL` (Scene Classification Layer).

## Authoritative Data Sources
- **Raw Datasets (`datasss/`)**:
  - `Train.csv`: Authoritative ground truth survey data for 7,888 farms (or train split).
  - `Test.csv`: Held-out competition evaluation set (not to be conflated with spatial CV test folds).
  - `SampleSubmission.csv`: Benchmark submission template.
  - Coordinate Reference System: EPSG:4326 (WGS84) geographic coordinates.

## Intermediate Data Pipelines
- **Manifests (`outputs/data/`)**:
  - `s2_full_expansion_manifest_7164.csv`: Phase 6E extraction candidate manifest.
  - `s2_phase6d_manifest_724.csv`: Phase 6D manifest.
  - `s2_phase6c_manifest_350.csv`: Phase 6C manifest.
- **Extraction Chunks**:
  - `s2_phase6e_chunk_00_0_500_raw_bands.csv` through `s2_phase6e_chunk_14_7000_7164_raw_bands.csv`
  - Consolidated raw bands: `s2_phase6e_7164_raw_bands.csv`
- **Model Predictions & Checkpoints**:
  - `phase6c_oof_predictions.csv`, `phase6d_oof_predictions.csv`
  - Cached models in `outputs/models/`
