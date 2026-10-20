"""
src/models/train_pilot_models.py
================================
Benchmarking Random Forest vs. XGBoost on Multi-Crop Sentinel-2 Datasets.

Evaluates 2 Distinct Modeling Tracks:
    1. Satellite-Only Model:
       - Uses purely Sentinel-2 multi-temporal spectral indices across 4 phenological
         stages (early, mid, late, full) and dynamic trajectory deltas.
       - No survey data, no geographic coordinates, no administrative identifiers.
    2. Multimodal Model:
       - Combines Sentinel-2 indices with safe geometric features (area, perimeter,
         compactness) and calendar metrics (sowing/harvest month, DOY, duration, Season).
       - No geographic coordinates or survey proxies.

Validation Strategy:
    - Spatial Cross-Validation: 5-Fold StratifiedGroupKFold grouped by 'Sub-District'.
    - Zero data leakage: Mandals in validation are strictly unseen in training fold.
    - Missingness handling: SimpleImputer(strategy='median') fitted strictly inside
      each training fold.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from xgboost import XGBClassifier

# Force UTF-8 stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

TARGET_COL: str = "Crop"
GROUP_COL: str = "Sub-District"
CLASSES_7: List[str] = [
    "Bengal Gram",
    "Chillies",
    "Cotton",
    "Groundnut",
    "Maize",
    "Paddy",
    "Red Gram",
]

OUTPUTS_MODELS_DIR = Path("outputs/models")
OUTPUTS_REPORTS_DIR = Path("outputs/reports")
OUTPUTS_MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def define_feature_subsets(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """
    Identify candidate columns and partition into satellite-only and multimodal subsets.
    """
    index_names = [
        "ndvi", "evi", "gndvi", "savi", "msavi", "ndmi", "ndwi",
        "msi", "ndre1", "ndre2", "psri", "bsi", "cire", "cig"
    ]
    windows = ["early", "mid", "late", "full"]

    # 1. Satellite index columns
    satellite_cols = []
    for idx_name in index_names:
        for w in windows:
            col = f"{idx_name}_{w}"
            if col in df.columns:
                satellite_cols.append(col)

    # 2. Multi-temporal dynamics
    temporal_dynamics = [
        "ndvi_growth_delta",
        "peak_to_early_ratio",
        "ndvi_senescence_delta",
        "moisture_loss_delta",
    ]
    for col in temporal_dynamics:
        if col in df.columns:
            satellite_cols.append(col)

    # 3. Multimodal additional safe columns (geometry & calendar)
    calendar_cols = [
        "duration_days",
        "sowing_month",
        "sowing_doy",
        "harvest_month",
        "harvest_doy",
        "Season",
    ]
    geometry_cols = [
        "area_ha",
        "perimeter_m",
        "compactness",
        "vertex_count",
    ]

    multimodal_added = [c for c in (calendar_cols + geometry_cols) if c in df.columns]
    multimodal_cols = satellite_cols + multimodal_added

    return satellite_cols, multimodal_cols, multimodal_added


def build_pipeline_for_track(
    feature_cols: List[str],
    has_season: bool,
    model_type: str = "rf",
    random_state: int = 42,
) -> Pipeline:
    """
    Build preprocessing and classifier pipeline.
    """
    num_cols = [c for c in feature_cols if c != "Season"]

    transformers = [
        ("num", Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]), num_cols),
    ]

    if has_season and "Season" in feature_cols:
        cat_transformer = Pipeline(
            steps=[
                ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.float64)),
                ("imputer", SimpleImputer(strategy="most_frequent")),
            ]
        )
        transformers.append(("cat", cat_transformer, ["Season"]))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    if model_type == "rf":
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_split=4,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
    elif model_type == "xgb":
        clf = XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            eval_metric="mlogloss",
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])


def evaluate_spatial_cv(
    df: pd.DataFrame,
    feature_cols: List[str],
    track_name: str,
    model_type: str,
    model_prefix: str = "pilot",
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Perform 5-fold StratifiedGroupKFold cross-validation grouped by Sub-District.
    """
    X = df[feature_cols].copy()
    y_raw = df[TARGET_COL].copy()
    groups = df[GROUP_COL].copy()

    # Encode target labels as integers 0..6
    label_encoder = LabelEncoder()
    label_encoder.fit(CLASSES_7)
    y = label_encoder.transform(y_raw)

    has_season = "Season" in feature_cols

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    oof_preds = np.zeros(len(df), dtype=int)
    oof_folds = np.zeros(len(df), dtype=int)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]

        # Verify zero group leakage between folds
        train_groups = set(groups.iloc[train_idx])
        val_groups = set(groups.iloc[val_idx])
        assert len(train_groups & val_groups) == 0, f"Spatial leakage detected in fold {fold}!"

        pipe = build_pipeline_for_track(
            feature_cols=feature_cols,
            has_season=has_season,
            model_type=model_type,
            random_state=random_state + fold,
        )

        pipe.fit(X_train, y_train)
        val_preds = pipe.predict(X_val)
        oof_preds[val_idx] = val_preds
        oof_folds[val_idx] = fold

        f_acc = accuracy_score(y_val, val_preds)
        f_macro_f1 = f1_score(y_val, val_preds, average="macro", zero_division=0)
        f_weighted_f1 = f1_score(y_val, val_preds, average="weighted", zero_division=0)

        fold_metrics.append({
            "fold": fold,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "val_groups_count": len(val_groups),
            "val_classes_represented": int(len(np.unique(y_val))),
            "accuracy": round(float(f_acc), 4),
            "macro_f1": round(float(f_macro_f1), 4),
            "weighted_f1": round(float(f_weighted_f1), 4),
        })

    # Overall OOF Metrics
    total_acc = accuracy_score(y, oof_preds)
    total_macro_f1 = f1_score(y, oof_preds, average="macro", zero_division=0)
    total_weighted_f1 = f1_score(y, oof_preds, average="weighted", zero_division=0)

    # Fold mean and standard deviation
    macro_scores = [f["macro_f1"] for f in fold_metrics]
    acc_scores = [f["accuracy"] for f in fold_metrics]

    report_dict = classification_report(
        y,
        oof_preds,
        target_names=CLASSES_7,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y, oof_preds, labels=range(len(CLASSES_7))).tolist()

    # Train final full-dataset pipeline for artifact saving
    final_pipeline = build_pipeline_for_track(
        feature_cols=feature_cols,
        has_season=has_season,
        model_type=model_type,
        random_state=random_state,
    )
    final_pipeline.fit(X, y)
    model_save_path = OUTPUTS_MODELS_DIR / f"{model_type}_{track_name}_{model_prefix}.joblib"
    joblib.dump(final_pipeline, model_save_path)

    results = {
        "track_name": track_name,
        "model_type": model_type,
        "n_samples": len(df),
        "n_features": len(feature_cols),
        "validation_strategy": "5-Fold StratifiedGroupKFold (grouped by Sub-District)",
        "spatial_leakage_between_folds": 0,
        "overall_oof_accuracy": round(float(total_acc), 4),
        "overall_oof_macro_f1": round(float(total_macro_f1), 4),
        "overall_oof_weighted_f1": round(float(total_weighted_f1), 4),
        "fold_macro_f1_mean": round(float(np.mean(macro_scores)), 4),
        "fold_macro_f1_std": round(float(np.std(macro_scores)), 4),
        "fold_accuracy_mean": round(float(np.mean(acc_scores)), 4),
        "fold_accuracy_std": round(float(np.std(acc_scores)), 4),
        "fold_results": fold_metrics,
        "per_class": {
            c: {
                "precision": round(report_dict[c]["precision"], 4),
                "recall": round(report_dict[c]["recall"], 4),
                "f1": round(report_dict[c]["f1-score"], 4),
                "support": report_dict[c]["support"],
            }
            for c in CLASSES_7
        },
        "confusion_matrix": cm,
        "persisted_model_path": str(model_save_path),
    }
    oof_labels = [CLASSES_7[i] for i in oof_preds]
    return results, oof_labels, oof_folds


def run_benchmark(
    features_csv: str = "outputs/data/s2_pilot_175_features.csv",
    report_name: str = "pilot_spatial_cv_benchmark.json",
    model_prefix: str = "pilot",
):
    logger.info("Loading features from %s...", features_csv)
    df = pd.read_csv(features_csv)

    # Filter strictly to successful extractions
    clean_df = df[df["status"] == "success"].copy().reset_index(drop=True)
    logger.info("Clean valid observations for spatial benchmarking: %d farms", len(clean_df))

    # Add phenological dates calendar metrics if not present
    if "duration_days" not in clean_df.columns:
        sdate = pd.to_datetime(clean_df["SDate"], format="mixed", dayfirst=True)
        hdate = pd.to_datetime(clean_df["HDate"], format="mixed", dayfirst=True)
        clean_df["duration_days"] = (hdate - sdate).dt.days.clip(lower=1)
        clean_df["sowing_month"] = sdate.dt.month
        clean_df["sowing_doy"] = sdate.dt.dayofyear
        clean_df["harvest_month"] = hdate.dt.month
        clean_df["harvest_doy"] = hdate.dt.dayofyear

    sat_cols, multi_cols, added_cols = define_feature_subsets(clean_df)
    logger.info("Satellite-only feature count: %d", len(sat_cols))
    logger.info("Multimodal feature count: %d (added: %s)", len(multi_cols), added_cols)

    # 1. Majority class baseline
    dummy = DummyClassifier(strategy="most_frequent")
    y_raw = clean_df[TARGET_COL]
    dummy.fit(clean_df[sat_cols], y_raw)
    dummy_preds = dummy.predict(clean_df[sat_cols])
    dummy_acc = accuracy_score(y_raw, dummy_preds)
    dummy_macro_f1 = f1_score(y_raw, dummy_preds, average="macro", zero_division=0)
    dummy_weighted_f1 = f1_score(y_raw, dummy_preds, average="weighted", zero_division=0)

    benchmark_results = {
        "dataset": features_csv,
        "n_samples": len(clean_df),
        "majority_baseline": {
            "accuracy": round(float(dummy_acc), 4),
            "macro_f1": round(float(dummy_macro_f1), 4),
            "weighted_f1": round(float(dummy_weighted_f1), 4),
        },
        "models": {},
    }

    configurations = [
        ("satellite_only", sat_cols, "rf"),
        ("satellite_only", sat_cols, "xgb"),
        ("multimodal", multi_cols, "rf"),
        ("multimodal", multi_cols, "xgb"),
    ]

    oof_df = pd.DataFrame({
        "FarmID": clean_df["FarmID"],
        "Crop": clean_df["Crop"],
        "District": clean_df.get("District", ""),
        "Sub-District": clean_df.get("Sub-District", ""),
        "pred_majority": dummy_preds,
    })

    assigned_folds = None
    for track, cols, model_type in configurations:
        key = f"{model_type.upper()}_{track}"
        logger.info("Running Spatial CV for %s...", key)
        res, oof_labels, oof_folds = evaluate_spatial_cv(
            df=clean_df,
            feature_cols=cols,
            track_name=track,
            model_type=model_type,
            model_prefix=model_prefix,
            n_splits=5,
            random_state=42,
        )
        benchmark_results["models"][key] = res
        oof_df[f"pred_{key}"] = oof_labels
        if assigned_folds is None:
            assigned_folds = oof_folds
            oof_df["fold"] = assigned_folds

    # Reorder columns so fold is next to Sub-District
    cols_order = ["FarmID", "Crop", "District", "Sub-District", "fold", "pred_majority"] + [f"pred_{m.upper()}_{t}" for t, _, m in configurations]
    oof_df = oof_df[[c for c in cols_order if c in oof_df.columns]]
    oof_path = Path("outputs/data") / f"{model_prefix}_oof_predictions.csv"
    oof_df.to_csv(oof_path, index=False)
    logger.info("Saved reproducible OOF predictions -> %s (%d rows)", oof_path, len(oof_df))

    # Save benchmark report
    report_path = OUTPUTS_REPORTS_DIR / report_name
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2)
    logger.info("Saved complete benchmark report -> %s", report_path)

    # Formatted terminal display
    print("\n" + "=" * 82)
    print(f"BENCHMARK RESULTS: {features_csv} ({len(clean_df)} valid farms)")
    print("Validation: 5-Fold StratifiedGroupKFold by Sub-District (Zero Group Leakage)")
    print("=" * 82)
    print(f"Majority Baseline Acc: {dummy_acc:.4f} | Macro F1: {dummy_macro_f1:.4f}\n")
    print(f"{'Model & Track':<24} | {'OOF Acc':<10} | {'Macro F1':<10} | {'Fold Macro F1':<18} | {'Weighted F1':<10}")
    print("-" * 82)
    for key, res in benchmark_results["models"].items():
        f_mean = res.get("fold_macro_f1_mean", res["overall_oof_macro_f1"])
        f_std = res.get("fold_macro_f1_std", 0.0)
        fold_str = f"{f_mean:.4f} +/- {f_std:.4f}"
        print(f"{key:<24} | {res['overall_oof_accuracy']:<10.4f} | {res['overall_oof_macro_f1']:<10.4f} | {fold_str:<18} | {res['overall_oof_weighted_f1']:<10.4f}")
    print("=" * 82)

    best_key = max(benchmark_results["models"].keys(), key=lambda k: benchmark_results["models"][k]["overall_oof_macro_f1"])
    best_res = benchmark_results["models"][best_key]
    print(f"\nTop Performing Model ({best_key}) - Per-Class Metrics:")
    for crop, m in best_res["per_class"].items():
        print(f"  - {crop:<12}: Precision={m['precision']:.3f}, Recall={m['recall']:.3f}, F1={m['f1']:.3f} (N={m['support']})")

    print("\nFold Breakdown for Best Model:")
    for f in best_res["fold_results"]:
        print(f"  Fold {f['fold']}: Acc={f['accuracy']:.4f}, Macro-F1={f['macro_f1']:.4f} (Val Farms={f['n_val']}, Val Groups={f['val_groups_count']})")
    print("\n" + "=" * 82)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate models with spatial CV.")
    parser.add_argument("--features-csv", type=str, default="outputs/data/s2_pilot_175_features.csv")
    parser.add_argument("--report-name", type=str, default="pilot_spatial_cv_benchmark.json")
    parser.add_argument("--model-prefix", type=str, default="pilot")
    args = parser.parse_args()

    run_benchmark(
        features_csv=args.features_csv,
        report_name=args.report_name,
        model_prefix=args.model_prefix,
    )
