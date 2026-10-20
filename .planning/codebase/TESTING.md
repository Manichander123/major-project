# Testing Strategy & Coverage

## Test Runner
- Python standard `unittest` / `pytest`.
- Run full suite: `python -m unittest discover tests`.

## Existing Test Suites
1. **`tests/test_pre_extraction_gate.py`**:
   - Validates that duplicate Sentinel-2 scenes on the same calendar date (MGRS tile overlap) are collapsed into single observations.
   - Validates cloud and shadow masking before compositing.
   - Asserts strict NaN preservation (no fabricated zeros for missing observations).
   - Validates deterministic checkpointing and cache skipping.
   - Verifies label and FarmID alignment.
2. **`tests/test_s2_mosaic.py`**:
   - Asserts daily mosaic behavior on synthetic multi-temporal ImageCollections.
3. **`tests/test_phase6d_audit.py`**:
   - Verifies 4 failed farms resolution (348 clean evaluated out of 352 manifest).
   - Verifies exact OOF predictions integrity and row alignment.
   - Verifies exact numerical reproducibility of benchmark JSON metrics from OOF predictions.
   - Verifies zero spatial leakage across Sub-Districts in saved folds.
4. **`tests/test_phase6c.py` & `tests/test_baseline.py`**:
   - Tabular model baselines and Phase 6C regression tests.

## Current Test Status
- 27 unit tests executed and passing with zero failures.
