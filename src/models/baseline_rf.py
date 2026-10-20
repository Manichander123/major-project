"""
src/models/baseline_rf.py
=========================
DEPRECATION SHIM: This module has moved to `src.models.archive.baseline_rf`.

WARNING:
    This non-satellite baseline used post-harvest calendar predictors, explicit coordinates,
    and random 80/20 splitting (spatial autocorrelation leakage).
    It is SUPERSEDED by the Phase 6I satellite classification pipeline.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "src.models.baseline_rf is deprecated and archived in src.models.archive.baseline_rf. "
    "It reflects a non-satellite tabular baseline with post-harvest calendar leakage and spatial autocorrelation. "
    "Use the official Phase 6I satellite classification pipeline instead.",
    DeprecationWarning,
    stacklevel=2,
)

from src.models.archive.baseline_rf import (  # noqa: F401, E402
    CAT_FEATURE_COLS,
    CLASSES_7,
    EXCLUDED_COLS,
    NUM_FEATURE_COLS,
    REQUIRED_BASE_COLS,
    REQUIRED_GEOM_COLS,
    TARGET_COL,
    build_pipeline,
    prepare_features,
    run_baseline,
    validate_inputs,
)

__all__ = [
    "CAT_FEATURE_COLS",
    "CLASSES_7",
    "EXCLUDED_COLS",
    "NUM_FEATURE_COLS",
    "REQUIRED_BASE_COLS",
    "REQUIRED_GEOM_COLS",
    "TARGET_COL",
    "build_pipeline",
    "prepare_features",
    "run_baseline",
    "validate_inputs",
]

if __name__ == "__main__":
    from src.models.archive.baseline_rf import run_baseline
    run_baseline()
