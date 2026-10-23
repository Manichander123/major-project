# Phase 7 — Leakage-Safe Test Set Evaluation Protocol

**Phase:** Phase 7, Plan 1 (Evaluation Protocol)  
**Date:** October 23, 2026  
**Status:** COMPLETE (Protocol Established; Awaiting User Approval to Execute Plan 2)  
**Evaluator:** Senior Machine Learning Engineer & Geospatial Remote Sensing Researcher  

---

## 1. Principles of Strict Data Isolation & Leakage Prevention

To guarantee complete scientific integrity and prevent information leakage:

1. **Strict Frozen Model Isolation:**
   - The selected Phase 6I champion model (`Weighted S+T`, 140 features) is completely frozen.
   - **Zero Hyperparameter Tuning:** No parameters (tree depth, learning rate, subsample ratio, regularization) may be adjusted or tuned on `Test.csv`.
   - **Zero Feature Selection:** No features may be added, pruned, or re-weighted based on test set variance, correlations, or performance.
   - **Zero Threshold Calibration:** Argmax decision rules remain fixed across the 7 classes.

2. **Strict Predictor Isolation:**
   - The predictor matrix $X_{\text{test}}$ must consist **exclusively** of the 140 Sentinel-2 Spectral and Temporal ($S+T$) features derived from satellite surface reflectance.
   - **Forbidden Predictors:**
     - Target labels: `Crop` (must only be accessed at the final evaluation stage as $y_{\text{test}}$).
     - Competition health column: `category`.
     - Identifiers: `FarmID`.
     - Administrative descriptors: `State`, `District`, `Sub-District`.
     - Survey ground attributes: `CropCoveredArea`, `CHeight`, `CNext`, `CLast`, `CTransp`, `IrriType`, `IrriSource`, `IrriCount`, `WaterCov`, `ExpYield`, `Season`.
     - Spatial coordinates and geometry metrics: centroid lat/lon, area, perimeter, compactness.
     - Spatial GLCM texture rasters (`P` group: 20 features), preserving the parsimonious $S+T$ feature schema.

3. **Preprocessing Parity & Rescaling Invariance:**
   - Feature engineering must strictly reuse the mathematical definitions from `src/features/indices.py`.
   - Missing data policy: If an optical cloud composite contains missing values due to persistent cloud cover, missing values must be preserved as `np.nan` (or imputed strictly using the training set median imputer from Phase 6F; never fit an imputer or scaler on `Test.csv`).
   - Feature column names and ordering must match `outputs/reports/phase6h/phase6h_feature_schema_manifest.json` bit-for-bit.

---

## 2. Dual Evaluation Paths & Contingencies

### Path A: Supervised Independent Evaluation (Authorized Path)
*Condition:* Verified in Preflight Audit (`phase7_test_preflight_audit.md`). `Test.csv` contains genuine ground-truth crop observations (`Crop`) recorded during field surveys across 35 unseen Mandals.

*Execution Protocol:*
1. **Extraction:** Extract 140 $S+T$ features across all 2,718 test parcels without consulting `Crop`.
2. **Inference:** Pass the 140-feature matrix $X_{\text{test}}$ to the frozen model `outputs/models/phase6h/xgb_weighted_s+t.joblib`.
3. **Class Mapping:** Convert numeric predictions (0..6) to categorical crop names using the alphabetical mapping:
   ```python
   EXPECTED_CLASSES = [
       "Bengal Gram", "Chillies", "Cotton", "Groundnut", 
       "Maize", "Paddy", "Red Gram"
   ]
   ```
4. **Independent Metrics Calculation:** Compare predictions $\hat{y}_{\text{test}}$ against true labels $y_{\text{test}}$ across all parcels with successful satellite extraction:
   - Primary metric: Macro-averaged F1 score ($\text{Macro-F1}$).
   - Overall Accuracy ($\text{OA}$) and Balanced Accuracy ($\text{BA}$).
   - Weighted-averaged F1 score ($\text{Weighted-F1}$).
   - Per-class metrics: Precision, Recall, F1-score, and Support for each of the 7 crops.
   - 7x7 Out-of-District Confusion Matrix.
   - Cloud failure accounting: Number and percentage of farms excluded due to cloudy observations.
5. **Comparison with Spatial Cross-Validation:**
   - Contextualize test performance against the 5-fold spatial cross-validation baseline ($\text{Macro-F1} = 0.7226$).
   - Explicitly document cohort differences: 116 training Mandals vs 35 entirely unseen test Mandals, highlighting true spatial generalization across agro-climatic boundaries.

### Path B: Unsupervised Inference & Diagnostic Auditing (Contingency Path)
*Condition:* Triggered ONLY if ground-truth labels are found to be corrupt, synthetic, or disputed.

*Execution Protocol:*
1. Generate predictions $\hat{y}_{\text{test}}$ on all valid test parcels.
2. Report prediction diagnostics: class distribution, prediction entropy/confidence distribution, missing feature rates, and spatial distribution across the 35 Mandals.
3. Strictly forbid reporting accuracy, precision, recall, or F1 scores.

---

## 3. Resumable & Bounded GEE Extraction Specifications (Plan 2 Blueprint)

When Plan 2 is authorized, feature extraction will execute under these parameters:

1. **Satellite Collection:** `COPERNICUS/S2_SR_HARMONIZED` (Level-2A Bottom-of-Atmosphere surface reflectance).
2. **Phenological Windows:**
   - Early Season: June 1, 2023 – July 31, 2023
   - Mid Season: August 1, 2023 – September 30, 2023
   - Late Season: October 1, 2023 – November 30, 2023
   - Full Season: June 1, 2023 – December 31, 2023
3. **Daily Mosaicking:** Daily mosaic grouping to collapse MGRS tile overlaps before temporal compositing.
4. **Cloud Masking:** Scene classification layer (`SCL`) bitmask filtering thick clouds, cirrus, and shadows; scene cloud threshold $\le 20\%$.
5. **Geometry Repair:** Single self-intersecting polygon (`FarmID 1321446`) repaired via `shapely.validation.make_valid()`.
6. **Resumable Chunking & Caching:**
   - Batch size: 250 parcels per chunk (11 total chunks for 2,718 parcels).
   - Checkpointing: Each chunk writes an atomic checkpoint to `outputs/data/phase7/raw_bands_chunk_*.csv`.
   - Idempotency: If interrupted, existing chunks are recognized by hash and skipped.
   - No data from `Train.csv` will be modified or overwritten.

---

## 4. Deliverables Matrix for Phase 7

| Plan ID | Description | Key Artifact Outputs | Execution Status |
| :---: | :--- | :--- | :---: |
| **Plan 1** | Read-Only Preflight & Evaluation Protocol | `outputs/reports/phase7/phase7_test_preflight_audit.md`<br>`outputs/reports/phase7/phase7_evaluation_protocol.md` | **COMPLETE** |
| **Plan 2** | Sentinel-2 GEE Feature Extraction | `outputs/data/phase7/s2_test_raw_bands.csv`<br>`outputs/data/phase7/s2_test_140_features.csv`<br>`outputs/reports/phase7/phase7_test_extraction_report.md` | **PENDING APPROVAL** |
| **Plan 3** | Model Inference & Supervised Evaluation | `outputs/data/phase7/test_predictions.csv`<br>`outputs/reports/phase7/phase7_test_evaluation_report.md`<br>`outputs/reports/phase7/phase7_test_metrics.json` | **PENDING APPROVAL** |
| **Plan 4** | Quality Control & Final Verification | `tests/test_phase7_inference.py`<br>`outputs/reports/phase7/PHASE7_FINAL_SUMMARY.md` | **PENDING APPROVAL** |

---

## 5. Verification Gate 1 Sign-Off

Plan 1 has fully validated all preflight requirements without modifying files outside the Phase 7 reports directory. No code has been staged or committed to Git. All experimental prerequisites are verified. The project halts here awaiting explicit user authorization to proceed to Plan 2.
