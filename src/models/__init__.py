"""
src/models
==========
Machine learning modeling package.

Official Production Pipeline:
- Phase 6I: 140 Sentinel-2 Spectral + Temporal (S+T) features, inverse-frequency weighted regularized XGBoost,
  evaluated via 5-Fold StratifiedGroupKFold on Sub-District (Mandal) across 7,843 eligible farms.
  (Outputs documented in `outputs/reports/phase6i/phase6i_model_selection_report.md`).

Archived / Historical Baselines:
- `src.models.archive.baseline_rf`: Phase 6A tabular baseline (NO satellite imagery, post-harvest leakage).
- `src.models.archive.train_pilot_models`: Phase 6C 350-farm pilot benchmark.
"""

from src.models.archive.baseline_rf import (
    CLASSES_7,
    TARGET_COL,
    prepare_features,
    run_baseline,
)

__all__ = ["run_baseline", "prepare_features", "TARGET_COL", "CLASSES_7"]
