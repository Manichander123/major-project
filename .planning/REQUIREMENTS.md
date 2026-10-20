# Phase 6F Requirements & Operational Rules

## Operational Guardrails & Gates
1. **Target Invariance**:
   - Primary target is `Crop` (7 classes). Do NOT substitute with `category`.
2. **Data Preservation**:
   - NEVER overwrite or mutate raw data (`datasss/Train.csv`, `datasss/Test.csv`), earlier phase outputs (Phase 6C/6D chunks, OOF predictions, benchmark reports).
3. **Missing Value Integrity**:
   - Strict NaN preservation for missing spectral observations. Never substitute missing values with `0.0`.
4. **Strict Spatial Cross-Validation**:
   - 5-Fold `StratifiedGroupKFold` grouped by `Sub-District`.
   - Zero spatial group overlap between train and validation folds.
5. **Leakage Prevention**:
   - Imputers, encoders, and scalers fit STRICTLY on training folds inside `Pipeline`.
   - No coordinates, FarmID, or administrative names in primary feature matrices.
6. **Feature Tracks**:
   - **Track A (Retrospective)**: Uses valid full-season and late-season features with transparent disclosure.
   - **Track B (Early-Season)**: Uses only features available by a defined prediction cutoff date.
7. **Verification Gates**:
   - **Gate A (Stage 1 Audit)**: Reconcile all 15 chunks, manifest coverage, duplicates, label alignment. Stop and present evidence before moving forward.
   - **Gate B (Stage 3 Feature Readiness)**: Validate final cohort compatibility, missingness analysis, and feature schema before model benchmarking.

---

# Phase 6H Requirements & Operational Rules

## Operational Guardrails & Gates
1. **Isolated Execution**:
   - Write all Phase 6H data, reports, and models exclusively into `outputs/data/phase6h/`, `outputs/reports/phase6h/`, and `outputs/models/phase6h/`.
   - Never mutate or overwrite historical Phase 6F or Phase 6G artifacts.
2. **Read-Only Scope for Stage 1**:
   - Stage 1 is strictly read-only: no new satellite extraction, no feature mutation, no model training.
   - Halt and require explicit user approval after Stage 1 verification.
3. **No Parcel Geometry Substitution**:
   - Do not conflate vector parcel geometry descriptors (`area_ha`, `perimeter_m`, `compactness`) with raster spatial GLCM texture.
4. **Spatial Grouped Isolation**:
   - Enforce 5-Fold `StratifiedGroupKFold` grouped strictly by `Sub-District` (Mandal) with 0 Mandal overlap for all ablation experiments.
5. **Matched Cohort Integrity**:
   - Ensure identical FarmID cohorts across all seven S/T/P ablation configurations.
6. **Class-Weighting Isolation**:
   - Initially benchmark all seven feature configurations without class weighting. Evaluate class weighting only after establishing the best feature combination.
