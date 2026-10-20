# Engineering & Scientific Conventions

## Scientific Integrity & Anti-Leakage Rules
1. **Target Distinction**:
   - `Crop` (7 classes: Paddy, Cotton, Maize, Bengal Gram, Groundnut, Chillies, Red Gram) is the primary target.
   - `category` (Healthy, Diseased, Pests, Stressed) is a secondary target; never swap or use as a feature for crop classification.
2. **Predictive Feature Boundaries**:
   - Strictly forbidden from feature matrices: `FarmID`, raw coordinates (`lat`, `lon`), administrative names (`District`, `Sub-District`), survey proxies.
3. **Spatial Cross-Validation**:
   - Evaluated via `StratifiedGroupKFold(n_splits=5)` grouped strictly by `Sub-District`.
   - Mandals in validation folds must be strictly unseen in corresponding training folds (0 group overlap).
4. **Strict Imputation Scope**:
   - Imputers (`SimpleImputer(strategy='median')`) and encoders are fit *strictly* on the training fold inside `sklearn.pipeline.Pipeline`.
   - Never compute global medians or fit scalers across the full dataset prior to fold splitting.
5. **Preservation of Missing Observations (NaN)**:
   - Spectral values missing due to cloud obscuration or lack of coverage must remain `NaN`. Never artificially fill missing observations with `0.0`.
6. **Feature Tracks**:
   - **Track A (Retrospective)**: Uses full-season and late-season observations. Disclosed as post-harvest / retrospective.
   - **Track B (Early-Season)**: Strictly uses observations available prior to a designated cutoff date, preventing future data leakage.

## Code Standards
- **Execution**: Scripts configure UTF-8 encoding for Windows terminals (`sys.stdout.reconfigure(encoding='utf-8', errors='replace')`).
- **File I/O**: Atomic writes using temporary files (`.tmp` -> `.csv`) to avoid corrupted checkpoints.
- **Logging**: Informative log output via standard library `logging`.
