"""
src/models/baseline_rf.py
=========================
Reproducible Random Forest Baseline for Seven-Class Crop Classification.

NOTE ON SCOPE & SCIENTIFIC NATURE:
    This model is a TABULAR / GEOSPATIAL & PHENOLOGICAL CALENDAR baseline.
    It relies on farm parcel geometries, administrative boundaries, and seasonal
    sowing/harvest schedules. It DOES NOT use multi-spectral Sentinel-2 satellite imagery.
    A multi-temporal satellite model will be trained once EO extraction is completed.

Primary Objective:
    Seven-class crop identification on the 'Crop' target variable:
    ['Bengal Gram', 'Chillies', 'Cotton', 'Groundnut', 'Maize', 'Paddy', 'Red Gram']

Feature Integrity & Leakage Prevention:
    - EXCLUDES 'category' (secondary target / crop health; must not be used here).
    - EXCLUDES 'FarmID' and raw 'geometry' WKT string.
    - EXCLUDES post-hoc farmer survey proxy fields:
        * 'CTransp': 100% Paddy is Transplanting, 0% all other crops (critical leakage).
        * 'ExpYield': post-hoc survey expected yield.
        * 'CLast', 'CNext': survey crop rotation responses.
        * 'CropCoveredArea', 'CHeight', 'WaterCov': in-situ field survey metrics.
        * 'IrriType', 'IrriSource', 'IrriCount': in-situ irrigation survey metrics.
    - INCLUDES strictly legitimate geospatial, geometric, and calendar/phenology variables:
        * Categorical: Season, District, Sub-District
        * Temporal/Calendar: sowing/harvest month, day-of-year, growing duration (days)
        * Geometric: area_ha, perimeter_m, compactness, vertex_count, centroid_lon, centroid_lat

Preprocessing & Modeling Pipeline:
    - Categorical variables are encoded strictly within an sklearn Pipeline using
      OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1) to prevent
      train/val leakage and gracefully handle unseen geographic levels at inference.
    - Numeric features pass through SimpleImputer(strategy='median') for inference safety.
    - The complete fitted preprocessing and Random Forest pipeline is persisted as a unit.

Validation Strategy:
    - Stratified train/test split (80/20, seed 42) for initial baseline.
    - Evaluated against DummyClassifier (majority class baseline).
    - PRELIMINARY NOTICE: These metrics are preliminary. Spatial autocorrelation
      may inflate scores on a random split; Spatial Block Cross-Validation
      is required to evaluate true field generalization.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

logger = logging.getLogger(__name__)

# Target definition
TARGET_COL: str = "Crop"
CLASSES_7: List[str] = [
    "Bengal Gram",
    "Chillies",
    "Cotton",
    "Groundnut",
    "Maize",
    "Paddy",
    "Red Gram",
]

# Explicitly excluded columns to prevent leakage or non-feature usage
EXCLUDED_COLS: List[str] = [
    "FarmID",
    "category",          # Secondary target
    "Crop",              # Target
    "geometry",          # Raw WKT
    "State",             # Constant (Telangana)
    "is_valid_geom",
    # Survey proxy fields / post-hoc survey questions
    "CTransp",           # Critical leakage: 100% Paddy is Transplanting
    "ExpYield",          # Post-hoc survey yield expectation
    "CLast",
    "CNext",
    "CropCoveredArea",
    "CHeight",
    "WaterCov",
    "IrriType",
    "IrriSource",
    "IrriCount",
]

# Required columns for feature preparation
REQUIRED_BASE_COLS: List[str] = [
    "Crop",
    "SDate",
    "HDate",
    "Season",
    "District",
    "Sub-District",
]

REQUIRED_GEOM_COLS: List[str] = [
    "area_ha",
    "perimeter_m",
    "compactness",
    "centroid_lon",
    "centroid_lat",
]

CAT_FEATURE_COLS: List[str] = ["Season", "District", "Sub-District"]

NUM_FEATURE_COLS: List[str] = [
    "duration_days",
    "sowing_month",
    "sowing_doy",
    "harvest_month",
    "harvest_doy",
    "area_ha",
    "perimeter_m",
    "compactness",
    "vertex_count",
    "centroid_lon",
    "centroid_lat",
    "bbox_min_lon",
    "bbox_min_lat",
    "bbox_max_lon",
    "bbox_max_lat",
]


def validate_inputs(df: pd.DataFrame, require_target: bool = True) -> None:
    """
    Validate dataset structure, required columns, date formats, and target labels.
    Raises ValueError with a descriptive message upon validation failure.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Input must be a pandas DataFrame, got {type(df)}.")

    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    # 1. Required column validation
    required_cols = list(REQUIRED_BASE_COLS) if require_target else [c for c in REQUIRED_BASE_COLS if c != "Crop"]
    missing_base = [c for c in required_cols if c not in df.columns]
    if missing_base:
        raise ValueError(f"Missing required base column(s): {missing_base}")

    missing_geom = [c for c in REQUIRED_GEOM_COLS if c not in df.columns]
    if missing_geom:
        raise ValueError(
            f"Missing required geometry column(s): {missing_geom}. "
            "Ensure geometry features have been added via geometry_utils.add_geometry_features()."
        )

    # 2. Target validation
    if require_target:
        if df[TARGET_COL].isna().any():
            n_nan = int(df[TARGET_COL].isna().sum())
            raise ValueError(f"Target column '{TARGET_COL}' contains {n_nan} missing/NaN values.")
        unique_targets = set(df[TARGET_COL].dropna().astype(str).str.strip())
        unknown_targets = unique_targets - set(CLASSES_7)
        if unknown_targets:
            raise ValueError(
                f"Unrecognized crop class(es) found in '{TARGET_COL}': {sorted(list(unknown_targets))}. "
                f"Expected subset of {CLASSES_7}."
            )

    # 3. Date validation
    try:
        sdate = pd.to_datetime(df["SDate"], format="mixed", dayfirst=True)
        hdate = pd.to_datetime(df["HDate"], format="mixed", dayfirst=True)
    except Exception as exc:
        raise ValueError(f"Failed to parse 'SDate' or 'HDate' into datetime: {exc}") from exc

    if sdate.isna().any() or hdate.isna().any():
        raise ValueError("Encountered unparseable / NaN values in 'SDate' or 'HDate'.")

    durations = (hdate - sdate).dt.days
    invalid_durations = durations <= 0
    if invalid_durations.any():
        n_inv = int(invalid_durations.sum())
        raise ValueError(
            f"Found {n_inv} rows where harvest date (HDate) is on or before sowing date (SDate) (duration <= 0)."
        )


def prepare_features(
    df: pd.DataFrame,
    require_target: bool = True,
) -> Tuple[pd.DataFrame, Optional[pd.Series], List[str], List[str]]:
    """
    Extract safe geospatial, geometric, and temporal features from dataset.
    Validates inputs and returns raw features (WITHOUT fitting encoders).

    Returns
    -------
    X : pd.DataFrame
        Unencoded feature matrix containing categorical (string) and numeric columns.
    y : pd.Series or None
        Target Series if require_target is True, else None.
    cat_cols : List[str]
        List of categorical column names present in X.
    num_cols : List[str]
        List of numeric column names present in X.
    """
    validate_inputs(df, require_target=require_target)

    df_feats = df.copy()

    # Parse dates safely
    sdate = pd.to_datetime(df_feats["SDate"], format="mixed", dayfirst=True)
    hdate = pd.to_datetime(df_feats["HDate"], format="mixed", dayfirst=True)

    df_feats["duration_days"] = (hdate - sdate).dt.days.astype(float)
    df_feats["sowing_month"] = sdate.dt.month.astype(float)
    df_feats["sowing_doy"] = sdate.dt.dayofyear.astype(float)
    df_feats["harvest_month"] = hdate.dt.month.astype(float)
    df_feats["harvest_doy"] = hdate.dt.dayofyear.astype(float)

    cat_cols = [c for c in CAT_FEATURE_COLS if c in df_feats.columns]
    num_cols = [c for c in NUM_FEATURE_COLS if c in df_feats.columns]

    # Explicitly enforce types without encoding
    for c in cat_cols:
        df_feats[c] = df_feats[c].astype(str)
    for c in num_cols:
        df_feats[c] = pd.to_numeric(df_feats[c], errors="coerce")

    feature_cols = cat_cols + num_cols
    X = df_feats[feature_cols].copy()
    y = df_feats[TARGET_COL].copy() if require_target else None

    return X, y, cat_cols, num_cols


def build_pipeline(
    cat_cols: List[str],
    num_cols: List[str],
    n_estimators: int = 300,
    max_depth: int = 15,
    min_samples_split: int = 5,
    random_state: int = 42,
) -> Pipeline:
    """
    Construct complete end-to-end preprocessing + Random Forest pipeline.
    Encoding and imputation are contained entirely within the Pipeline.
    """
    cat_transformer = Pipeline(
        steps=[
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    dtype=np.float64,
                ),
            ),
        ]
    )

    num_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", cat_transformer, cat_cols),
            ("num", num_transformer, num_cols),
        ],
        remainder="drop",
    )

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ]
    )


def run_baseline(
    data_path: str = "outputs/data/train_enriched_geometry.csv",
    test_size: float = 0.2,
    random_state: int = 42,
    save_artifacts: bool = True,
) -> Dict[str, Any]:
    """
    Train and evaluate the baseline Random Forest on tabular/geospatial features.
    """
    path = Path(data_path)
    if not path.exists():
        from src.data.load_data import load_train
        from src.geospatial.geometry_utils import add_geometry_features

        logger.info("Enriched data not found at %s. Loading Train.csv and adding geometry...", data_path)
        gdf = load_train(parse_geometry=True, as_geodataframe=True)
        df = add_geometry_features(gdf)
    else:
        df = pd.read_csv(path)

    X, y, cat_cols, num_cols = prepare_features(df, require_target=True)

    # Stratified split
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    # 1. Majority class baseline
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    dummy_preds = dummy.predict(X_val)

    dummy_acc = float(accuracy_score(y_val, dummy_preds))
    dummy_macro_f1 = float(f1_score(y_val, dummy_preds, average="macro", zero_division=0))
    dummy_weighted_f1 = float(f1_score(y_val, dummy_preds, average="weighted", zero_division=0))

    # 2. Pipeline-based Random Forest baseline
    pipeline = build_pipeline(
        cat_cols=cat_cols,
        num_cols=num_cols,
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        random_state=random_state,
    )

    # Fit pipeline strictly on training data
    pipeline.fit(X_train, y_train)
    rf_preds = pipeline.predict(X_val)

    rf_acc = float(accuracy_score(y_val, rf_preds))
    rf_macro_f1 = float(f1_score(y_val, rf_preds, average="macro", zero_division=0))
    rf_weighted_f1 = float(f1_score(y_val, rf_preds, average="weighted", zero_division=0))

    classes = sorted(list(y.unique()))
    report_dict = classification_report(
        y_val,
        rf_preds,
        labels=classes,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_val, rf_preds, labels=classes).tolist()

    # Extract feature importances through pipeline preprocessor
    fitted_rf = pipeline.named_steps["classifier"]
    all_features = cat_cols + num_cols
    importances = dict(
        sorted(
            zip(all_features, [round(float(v), 4) for v in fitted_rf.feature_importances_]),
            key=lambda item: item[1],
            reverse=True,
        )
    )

    results = {
        "n_samples": len(df),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "target": TARGET_COL,
        "classes": classes,
        "is_satellite_model": False,
        "model_description": "Tabular/geospatial & phenological calendar baseline (NO satellite imagery)",
        "validation_strategy": "preliminary_stratified_split",
        "spatial_generalization_claimed": False,
        "spatial_warning": (
            "Metrics are preliminary. Stratified random splitting allows spatial autocorrelation "
            "to artificially elevate accuracy. Spatial Block Cross-Validation is required before field deployment."
        ),
        "features_used": all_features,
        "features_excluded": EXCLUDED_COLS,
        "majority_baseline": {
            "accuracy": round(dummy_acc, 4),
            "macro_f1": round(dummy_macro_f1, 4),
            "weighted_f1": round(dummy_weighted_f1, 4),
        },
        "random_forest": {
            "accuracy": round(rf_acc, 4),
            "macro_f1": round(rf_macro_f1, 4),
            "weighted_f1": round(rf_weighted_f1, 4),
            "per_class": {
                c: {
                    "precision": round(report_dict[c]["precision"], 4),
                    "recall": round(report_dict[c]["recall"], 4),
                    "f1": round(report_dict[c]["f1-score"], 4),
                    "support": report_dict[c]["support"],
                }
                for c in classes
            },
            "confusion_matrix": cm,
            "top_features": list(importances.items())[:8],
        },
    }

    if save_artifacts:
        models_dir = Path("outputs/models")
        reports_dir = Path("outputs/reports")
        models_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)

        # Persist full pipeline (preprocessor + model)
        joblib.dump(pipeline, models_dir / "baseline_rf.joblib")
        with open(reports_dir / "baseline_rf_metrics.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    res = run_baseline()
    print("=" * 60)
    print("RANDOM FOREST BASELINE RESULTS (CROP 7-CLASS)")
    print("=" * 60)
    print(f"Scope: {res['model_description']}")
    print(f"Validation: {res['validation_strategy']} (Spatial generalization claimed: {res['spatial_generalization_claimed']})")
    print(f"Train / Val Split: {res['n_train']} / {res['n_val']}")
    print(f"Majority Baseline Acc: {res['majority_baseline']['accuracy']:.4f} | Macro F1: {res['majority_baseline']['macro_f1']:.4f}")
    print(f"Random Forest Acc:     {res['random_forest']['accuracy']:.4f} | Macro F1: {res['random_forest']['macro_f1']:.4f} | Weighted F1: {res['random_forest']['weighted_f1']:.4f}")
    print("\nPer-Class Breakdown:")
    for c, m in res["random_forest"]["per_class"].items():
        print(f"  {c:<12}: Precision={m['precision']:.3f}, Recall={m['recall']:.3f}, F1={m['f1']:.3f} (N={m['support']})")
    print("\nTop 5 Important Features:")
    for feat, imp in res["random_forest"]["top_features"][:5]:
        print(f"  - {feat}: {imp:.4f}")
    print("\nNotice:")
    print(res["spatial_warning"])
