# Technology Stack

## Core Language & Runtime
- **Python**: 3.12 / 3.14 (Active runtime on Windows 10/11 x86_64)
- **Execution Shell**: PowerShell

## Geospatial & Earth Observation
- **Google Earth Engine (GEE)**: `earthengine-api` (project: `scientific-axle-446313-i6`)
  - Image Collection: `COPERNICUS/S2_SR_HARMONIZED` (Sentinel-2 Level-2A Surface Reflectance)
  - Processing: Server-side zonal statistics, cloud masking via SCL, daily MGRS mosaic compositing
- **Shapely**: Geometric parsing, polygon validation, WKT decoding
- **GeoPandas**: Spatial dataframe operations, area/perimeter calculations in EPSG:32644 (UTM 44N)
- **PyProj**: Coordinate reference system transformations (EPSG:4326 to EPSG:32644)

## Machine Learning & Data Processing
- **scikit-learn**:
  - Models: `RandomForestClassifier`, `DummyClassifier`
  - Validation: `StratifiedGroupKFold` (5 folds grouped by `Sub-District`)
  - Pipeline & Preprocessing: `Pipeline`, `ColumnTransformer`, `SimpleImputer`, `OrdinalEncoder`, `LabelEncoder`
  - Metrics: `accuracy_score`, `f1_score` (macro & weighted), `confusion_matrix`, `classification_report`
- **XGBoost**: `XGBClassifier` (gradient boosting with tree-based multiclass classification)
- **Pandas & NumPy**: Tabular data manipulation, vectorized arithmetic, NaN preservation
- **Joblib**: Model serialization and parallel computation

## Configuration & Quality Assurance
- **PyYAML**: Centralized configuration management (`config/project_config.yaml`)
- **unittest**: Regression testing, reproducibility tests, pre-extraction gate validation
- **Logging**: Python standard `logging` with timestamps and severity levels
