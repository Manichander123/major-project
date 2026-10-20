"""
src/models/archive
==================
Archive of superseded, exploratory, and non-satellite baseline experiments.

Contains:
- `baseline_rf.py`: Phase 6A tabular/geospatial & phenological calendar baseline (NO satellite imagery).
  Uses post-harvest calendar predictors and coordinates with random 80/20 splitting.
  SUPERSEDED by the Phase 6F-6I satellite pipelines.
- `train_pilot_models.py`: Phase 6C initial pilot benchmark (350 farms).
  SUPERSEDED by Phase 6F-6I full-cohort (7,843 farms) spatial cross-validation.

Official Production Pipeline:
- Phase 6I: 140 Spectral + Temporal (S+T) features, inverse-frequency weighted regularized XGBoost,
  evaluated via 5-Fold StratifiedGroupKFold on Sub-District (Mandal) across 7,843 eligible farms.
  (Outputs documented in `outputs/reports/phase6i/phase6i_model_selection_report.md`).
"""

from src.models.archive.baseline_rf import (
    CLASSES_7,
    EXCLUDED_COLS,
    REQUIRED_BASE_COLS,
    REQUIRED_GEOM_COLS,
    TARGET_COL,
    build_pipeline,
    prepare_features,
    run_baseline,
    validate_inputs,
)

__all__ = [
    "CLASSES_7",
    "EXCLUDED_COLS",
    "REQUIRED_BASE_COLS",
    "REQUIRED_GEOM_COLS",
    "TARGET_COL",
    "build_pipeline",
    "prepare_features",
    "run_baseline",
    "validate_inputs",
]
