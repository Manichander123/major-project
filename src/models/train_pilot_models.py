"""
src/models/train_pilot_models.py
================================
DEPRECATION SHIM: This module has moved to `src.models.archive.train_pilot_models`.

WARNING:
    This was an exploratory Phase 6C pilot benchmark (350 farms) with calendar leakage in Track 2.
    It is SUPERSEDED by the Phase 6F-6I full-cohort satellite classification pipeline.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "src.models.train_pilot_models is deprecated and archived in src.models.archive.train_pilot_models. "
    "It reflects an exploratory Phase 6C pilot on 350 farms. "
    "Use the official Phase 6I satellite classification pipeline instead.",
    DeprecationWarning,
    stacklevel=2,
)

from src.models.archive.train_pilot_models import (  # noqa: F401, E402
    CLASSES_7,
    GROUP_COL,
    TARGET_COL,
    build_pipeline_for_track,
    define_feature_subsets,
    evaluate_spatial_cv,
    run_benchmark,
)

__all__ = [
    "CLASSES_7",
    "GROUP_COL",
    "TARGET_COL",
    "build_pipeline_for_track",
    "define_feature_subsets",
    "evaluate_spatial_cv",
    "run_benchmark",
]

if __name__ == "__main__":
    from src.models.archive.train_pilot_models import run_benchmark
    import argparse
    parser = argparse.ArgumentParser(description="Archived pilot spatial CV benchmark.")
    parser.add_argument("--features-csv", type=str, default="outputs/data/s2_pilot_175_features.csv")
    parser.add_argument("--report-name", type=str, default="pilot_spatial_cv_benchmark.json")
    parser.add_argument("--model-prefix", type=str, default="pilot")
    args = parser.parse_args()
    run_benchmark(args.features_csv, args.report_name, args.model_prefix)
