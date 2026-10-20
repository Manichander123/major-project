"""
scripts/benchmark_phase6h_stp_ablation.py
=========================================
Phase 6H Stage 3: Spatial Feature Audit and S/T/P Ablation Benchmark.
Executes 7 controlled feature-ablation experiments across 5 grouped cross-validation folds:
  1. S     — Spectral-only full season (34 features)
  2. T     — Temporal-only additional features (106 features)
  3. P     — Spatial texture only (20 features)
  4. S+T   — Spectral and temporal (140 features)
  5. S+P   — Spectral and spatial (54 features)
  6. T+P   — Temporal and spatial (126 features)
  7. S+T+P — Full satellite feature fusion (160 features)

Classifier:
  Regularized XGBoost with established Phase 6G Trial 4 parameters:
  max_depth=7, learning_rate=0.05, colsample_bytree=0.75, subsample=0.80,
  reg_lambda=5.0, min_child_weight=3, gamma=0.2, reg_alpha=0.2,
  n_estimators=300, random_state=42.
  Class weighting disabled (unweighted) for isolated feature contribution analysis.

Validation:
  Exact 5-fold StratifiedGroupKFold grouped by Sub-District from outputs/data/phase6f_cv_folds.csv (N=7,843).
  Zero Mandal overlap between train and val.
  Native missing-value handling (preserves true NaNs).

Outputs:
  - outputs/data/phase6h/phase6h_stp_ablation_oof_predictions.csv
  - outputs/reports/phase6h/phase6h_feature_schema_manifest.json
  - outputs/reports/phase6h/phase6h_stage3_benchmark_report.json
  - outputs/reports/phase6h/phase6h_stage3_benchmark_report.md
  - outputs/models/phase6h/*.joblib
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phase6h_stp_benchmark")

EXPECTED_CLASSES: List[str] = [
    "Bengal Gram",
    "Chillies",
    "Cotton",
    "Groundnut",
    "Maize",
    "Paddy",
    "Red Gram",
]

# Trial 4 regularized parameters
TRIAL4_PARAMS = {
    "max_depth": 7,
    "learning_rate": 0.05,
    "colsample_bytree": 0.75,
    "subsample": 0.80,
    "reg_lambda": 5.0,
    "min_child_weight": 3,
    "gamma": 0.2,
    "reg_alpha": 0.2,
    "n_estimators": 300,
    "random_state": 42,
    "n_jobs": -1,
    "eval_metric": "mlogloss",
}

PROTECTED_BASELINE_HASHES = {
    "datasss/Train.csv": "29d31920f5b9aaea8213d7993dedc50c5d19458906f85642d360d785bb07aa06",
    "outputs/data/s2_full_expansion_manifest_7164.csv": "f4b92b55f248edf687c60b85177cb793859d04fd39442a4841991b79110807b9",
    "outputs/data/s2_phase6e_7164_raw_bands.csv": "d1ee90a5d9e09126a71bdc8e809767a30860671e88e42a8a8aebb19c46cf83af",
    "outputs/data/s2_phase6d_manifest_724.csv": "326207f8ca1b0848533bdc4ff673db24b18ff46e30b558952444555c20824214",
    "outputs/data/s2_phase6d_724_raw_bands.csv": "cdd3cf8a8a6ac0dde6c0e13d262011a379e129598e8f667a38fab747f1f73826",
    "outputs/data/s2_phase6d_724_features.csv": "98ac03841ea85219f3df1f90d038741aa6faa65e212de6a4d12fd02f0e7bfc75",
    "outputs/data/phase6d_oof_predictions.csv": "dfa09e53ec961edd08c3ba917364e8d2a666e23d4357d7d46864974a53ee49a4",
    "outputs/data/s2_phase6f_recovered_farms.csv": "235ca39d953be9b0ef3984f9042fecda4c3fb3e0324de1e94f0af04ccebdc32f",
    "outputs/reports/phase6f_stage2_recovery_ledger.csv": "ced6944ce6db7ee02841c3de11b6c8edc48ea27f2f2dfaa013a42054f299e272",
    "outputs/data/s2_phase6e_chunk_00_0_500_raw_bands.csv": "9cdecd9c3162ca6301883b62ffaeffebf579f2e3c09e8631a986905dccf03ed6",
    "outputs/data/s2_phase6e_chunk_01_500_1000_raw_bands.csv": "e4f2027ba78cef3267f54287809f9379e6d0dc77b6ab67ae3a059b0297ef9dc5",
    "outputs/data/s2_phase6e_chunk_02_1000_1500_raw_bands.csv": "ef428abd732902e3625f9be2161f96f2a80742022616452c205ef57b1f202219",
    "outputs/data/s2_phase6e_chunk_03_1500_2000_raw_bands.csv": "5f03c44197ee3e0a397e56a869cab07f3dafae76736821e1988730d8aff0e30a",
    "outputs/data/s2_phase6e_chunk_04_2000_2500_raw_bands.csv": "a291487191587bcc1680729e45026740008ee81b89aae55b99b923c7fa24453f",
    "outputs/data/s2_phase6e_chunk_05_2500_3000_raw_bands.csv": "88650fdb722979455e0d3fa4c8280655872f8ea8db31094e180ed1eb0b0750e5",
    "outputs/data/s2_phase6e_chunk_06_3000_3500_raw_bands.csv": "32015d7d6733c4b9b2684f4c9a8e05f42b7d29c01a09986ab69fa3368f6e6ce4",
    "outputs/data/s2_phase6e_chunk_07_3500_4000_raw_bands.csv": "572d96fddf458bebb864da59eb44f27e6e2cec505bb855ccabf75855154a557e",
    "outputs/data/s2_phase6e_chunk_08_4000_4500_raw_bands.csv": "9db8b9bd2fdd8e2968543fe83ddcfe6a9a5e0b0a317c5d2145fda454f6f10f66",
    "outputs/data/s2_phase6e_chunk_09_4500_5000_raw_bands.csv": "b9c9e0a53197d115e218501d219e26570c5d7d203097031e57b6e35e01814d6d",
    "outputs/data/s2_phase6e_chunk_10_5000_5500_raw_bands.csv": "6877114a7dc422d0b8539704bac2d777bcbd6b4fcb8c81a632bd1c13c64f7454",
    "outputs/data/s2_phase6e_chunk_11_5500_6000_raw_bands.csv": "4fa33f6ecdebfd6b02deba3634832362b56b9af695cb18fb39c68341884640fb",
    "outputs/data/s2_phase6e_chunk_12_6000_6500_raw_bands.csv": "2458256baddcb4f915ee299dd33e2f2811fd8e57d0f81f78435657804bf5a144",
    "outputs/data/s2_phase6e_chunk_13_6500_7000_raw_bands.csv": "b26e7e8fa202a7c76252df06df11b3189722a8c0fdc36dabbb7d042a1f040ca3",
    "outputs/data/s2_phase6e_chunk_14_7000_7164_raw_bands.csv": "3ac9eb1a66e7d82c93830d4d7e6a568d2b3d13b24c2fc86c80a9990dcd029c53",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def verify_protected_baselines() -> None:
    logger.info("Verifying protected baseline file hashes...")
    for rel_path, exp_hash in PROTECTED_BASELINE_HASHES.items():
        fp = PROJECT_ROOT / rel_path
        if not fp.exists():
            raise FileNotFoundError(f"Missing protected baseline file: {rel_path}")
        act_hash = sha256_file(fp)
        if act_hash != exp_hash:
            raise ValueError(f"CORRUPTION in {rel_path}! Expected: {exp_hash}, Got: {act_hash}")
    logger.info("All protected baseline files verified intact.")


def build_feature_groups(
    df_feat: pd.DataFrame, df_tex: pd.DataFrame
) -> Tuple[Dict[str, List[str]], Dict[str, Any]]:
    """
    Constructs the 7 feature sets precisely:
      S: 34
      T: 106
      P: 20
      S+T: 140
      S+P: 54
      T+P: 126
      S+T+P: 160
    """
    # 1. Spectral S (34 features from full season)
    bands = ["B2", "B3", "B4", "B8", "B5", "B6", "B7", "B8A", "B11", "B12"]
    band_stats_full = [f"{b}_full_{stat}" for b in bands for stat in ["mean", "std"]]
    indices_list = [
        "ndvi", "evi", "gndvi", "savi", "msavi", "ndmi", "ndwi",
        "ndre1", "ndre2", "psri", "bsi", "msi", "cire", "cig"
    ]
    indices_full = [f"{idx}_full" for idx in indices_list]
    features_s = band_stats_full + indices_full
    assert len(features_s) == 34, f"Expected 34 S features, got {len(features_s)}"

    # 2. Temporal T (106 additional features: early, mid, late + 4 deltas)
    windows_early_mid_late = ["early", "mid", "late"]
    band_stats_eml = [
        f"{b}_{w}_{stat}"
        for w in windows_early_mid_late
        for b in bands
        for stat in ["mean", "std"]
    ]
    indices_eml = [
        f"{idx}_{w}"
        for w in windows_early_mid_late
        for idx in indices_list
    ]
    trajectory_deltas = [
        "ndvi_growth_delta",
        "peak_to_early_ratio",
        "ndvi_senescence_delta",
        "moisture_loss_delta",
    ]
    features_t = band_stats_eml + indices_eml + trajectory_deltas
    assert len(features_t) == 106, f"Expected 106 T features, got {len(features_t)}"

    # 3. Spatial Texture P (20 GLCM features)
    glcm_metrics = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]
    glcm_windows = ["early", "mid", "late", "full"]
    features_p = [f"glcm_{m}_{w}" for m in glcm_metrics for w in glcm_windows]
    assert len(features_p) == 20, f"Expected 20 P features, got {len(features_p)}"

    # Confirm mutual exclusivity
    set_s = set(features_s)
    set_t = set(features_t)
    set_p = set(features_p)
    assert len(set_s.intersection(set_t)) == 0, "S and T overlap!"
    assert len(set_s.intersection(set_p)) == 0, "S and P overlap!"
    assert len(set_t.intersection(set_p)) == 0, "T and P overlap!"

    # Combinations
    groups = {
        "S": features_s,
        "T": features_t,
        "P": features_p,
        "S+T": features_s + features_t,
        "S+P": features_s + features_p,
        "T+P": features_t + features_p,
        "S+T+P": features_s + features_t + features_p,
    }

    assert len(groups["S"]) == 34
    assert len(groups["T"]) == 106
    assert len(groups["P"]) == 20
    assert len(groups["S+T"]) == 140
    assert len(groups["S+P"]) == 54
    assert len(groups["T+P"]) == 126
    assert len(groups["S+T+P"]) == 160

    manifest = {
        "description": "Phase 6H Feature Group Manifest for Spatial/Temporal/Spectral Ablation",
        "timestamp": datetime.now().isoformat(),
        "group_counts": {k: len(v) for k, v in groups.items()},
        "groups": groups,
    }

    return groups, manifest


def run_benchmark() -> Dict[str, Any]:
    total_start = time.time()
    logger.info("================================================================")
    logger.info("STARTING PHASE 6H STAGE 3: S/T/P ABLATION BENCHMARK (7 EXPERIMENTS)")
    logger.info("================================================================")

    # 1. Pre-execution verification
    verify_protected_baselines()

    # 2. Setup directories
    data_dir = PROJECT_ROOT / "outputs/data/phase6h"
    reports_dir = PROJECT_ROOT / "outputs/reports/phase6h"
    models_dir = PROJECT_ROOT / "outputs/models/phase6h"
    for d in [data_dir, reports_dir, models_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 3. Load datasets
    logger.info("Loading features, CV folds, and spatial texture data...")
    df_feat = pd.read_csv(PROJECT_ROOT / "outputs/data/s2_phase6f_features.csv")
    df_folds = pd.read_csv(PROJECT_ROOT / "outputs/data/phase6f_cv_folds.csv")
    df_tex = pd.read_csv(PROJECT_ROOT / "outputs/data/phase6h/phase6h_texture_all_farms.csv")

    df_eligible = df_feat[df_feat["extraction_status"] == "success"].copy().reset_index(drop=True)
    if len(df_eligible) != 7843:
        raise ValueError(f"Expected 7843 eligible farms, got {len(df_eligible)}")

    # Merge spatial texture
    feature_groups, feature_manifest = build_feature_groups(df_eligible, df_tex)
    glcm_cols = feature_groups["P"]

    df_merged = pd.merge(
        df_eligible,
        df_tex[["FarmID"] + glcm_cols],
        on="FarmID",
        how="inner",
    )
    if len(df_merged) != 7843:
        raise ValueError(f"Merged dataset count mismatch: {len(df_merged)}")

    # Merge with CV folds
    df_all = pd.merge(
        df_folds[["FarmID", "fold"]],
        df_merged,
        on="FarmID",
        how="inner",
    )
    if len(df_all) != 7843:
        raise ValueError(f"Final merged dataset count mismatch: {len(df_all)}")

    # Save feature manifest
    manifest_path = reports_dir / "phase6h_feature_schema_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(feature_manifest, f, indent=2)

    crop_to_idx = {c: i for i, c in enumerate(EXPECTED_CLASSES)}
    idx_to_crop = {i: c for i, c in enumerate(EXPECTED_CLASSES)}
    y_all = df_all["Crop"].map(crop_to_idx).values

    # Assert fold separation
    for f_idx in range(5):
        val_sub = set(df_all.loc[df_all["fold"] == f_idx, "Sub-District"])
        tr_sub = set(df_all.loc[df_all["fold"] != f_idx, "Sub-District"])
        assert len(val_sub.intersection(tr_sub)) == 0, f"Mandal overlap in fold {f_idx}"

    logger.info("Data and 5-fold grouped partitions prepared: N=7843, 7 classes.")

    # 4. Run 7 Controlled Experiments
    experiment_order = [
        ("S", "Spectral-only full season (34 features)"),
        ("T", "Temporal-only additional features (106 features)"),
        ("P", "Spatial texture only (20 features)"),
        ("S+T", "Spectral + Temporal (140 features)"),
        ("S+P", "Spectral + Spatial texture (54 features)"),
        ("T+P", "Temporal + Spatial texture (126 features)"),
        ("S+T+P", "Full feature fusion: Spectral + Temporal + Spatial (160 features)"),
    ]

    benchmark_results: Dict[str, Any] = {
        "benchmark_name": "Phase 6H Stage 3 S/T/P Controlled Feature Ablation Benchmark",
        "timestamp": datetime.now().isoformat(),
        "classifier": "XGBoost",
        "classifier_parameters": TRIAL4_PARAMS,
        "class_weighting": "None (unweighted)",
        "cohort_size": 7843,
        "cv_folds": 5,
        "cv_grouping": "Sub-District (Mandal) - StratifiedGroupKFold",
        "experiments": {},
    }

    oof_predictions_df = pd.DataFrame({
        "FarmID": df_all["FarmID"],
        "fold": df_all["fold"].astype(int),
        "true_crop": df_all["Crop"],
        "Sub-District": df_all["Sub-District"],
        "District": df_all["District"],
    })

    for exp_id, exp_desc in experiment_order:
        exp_start = time.time()
        feats = feature_groups[exp_id]
        logger.info("----------------------------------------------------------------")
        logger.info("RUNNING EXPERIMENT: %s (%d features) — %s", exp_id, len(feats), exp_desc)
        logger.info("----------------------------------------------------------------")

        X_mat = df_all[feats].values.astype(np.float32)
        oof_preds = np.zeros(len(df_all), dtype=int)
        oof_proba = np.zeros((len(df_all), 7), dtype=np.float32)

        fold_metrics: List[Dict[str, Any]] = []
        fold_times: List[float] = []

        for fold_i in range(5):
            f_start = time.time()
            val_mask = df_all["fold"] == fold_i
            train_mask = ~val_mask

            X_tr, y_tr = X_mat[train_mask], y_all[train_mask]
            X_val, y_val = X_mat[val_mask], y_all[val_mask]

            clf = XGBClassifier(**TRIAL4_PARAMS)
            clf.fit(X_tr, y_tr)

            val_preds = clf.predict(X_val)
            val_probs = clf.predict_proba(X_val)

            oof_preds[val_mask] = val_preds
            oof_proba[val_mask] = val_probs

            f_duration = time.time() - f_start
            fold_times.append(f_duration)

            # Fold-level metrics
            f_acc = float(accuracy_score(y_val, val_preds))
            f_macro_f1 = float(f1_score(y_val, val_preds, average="macro"))
            f_bal_acc = float(balanced_accuracy_score(y_val, val_preds))
            f_weighted_f1 = float(f1_score(y_val, val_preds, average="weighted"))

            fold_metrics.append({
                "fold": fold_i,
                "n_train": int(train_mask.sum()),
                "n_val": int(val_mask.sum()),
                "accuracy": round(f_acc, 4),
                "macro_f1": round(f_macro_f1, 4),
                "balanced_accuracy": round(f_bal_acc, 4),
                "weighted_f1": round(f_weighted_f1, 4),
                "fit_time_seconds": round(f_duration, 2),
            })
            logger.info(
                "  Fold %d: Val N=%d | Acc=%.4f | Macro-F1=%.4f | Bal-Acc=%.4f (%.1fs)",
                fold_i, int(val_mask.sum()), f_acc, f_macro_f1, f_bal_acc, f_duration
            )

        # Pooled OOF Metrics
        exp_duration = time.time() - exp_start
        pooled_acc = float(accuracy_score(y_all, oof_preds))
        pooled_macro_f1 = float(f1_score(y_all, oof_preds, average="macro"))
        pooled_bal_acc = float(balanced_accuracy_score(y_all, oof_preds))
        pooled_weighted_f1 = float(f1_score(y_all, oof_preds, average="weighted"))

        # Per-class metrics
        precisions = precision_score(y_all, oof_preds, average=None, zero_division=0)
        recalls = recall_score(y_all, oof_preds, average=None, zero_division=0)
        f1s = f1_score(y_all, oof_preds, average=None, zero_division=0)
        cm = confusion_matrix(y_all, oof_preds, labels=list(range(7)))

        per_class_metrics = {}
        for c_idx, c_name in enumerate(EXPECTED_CLASSES):
            per_class_metrics[c_name] = {
                "support": int((y_all == c_idx).sum()),
                "precision": round(float(precisions[c_idx]), 4),
                "recall": round(float(recalls[c_idx]), 4),
                "f1_score": round(float(f1s[c_idx]), 4),
            }

        # Store predictions in OOF DataFrame
        oof_predictions_df[f"pred_{exp_id}"] = [idx_to_crop[p] for p in oof_preds]
        for c_idx, c_name in enumerate(EXPECTED_CLASSES):
            clean_cname = c_name.replace(" ", "_").lower()
            oof_predictions_df[f"prob_{exp_id}_{clean_cname}"] = oof_proba[:, c_idx]

        # Fold variability statistics
        acc_list = [f["accuracy"] for f in fold_metrics]
        mf1_list = [f["macro_f1"] for f in fold_metrics]
        bacc_list = [f["balanced_accuracy"] for f in fold_metrics]
        wf1_list = [f["weighted_f1"] for f in fold_metrics]

        # Refit final model on 100% cohort for artifact persistence
        logger.info("  Refitting final %s model on 100%% cohort (N=7843)...", exp_id)
        refit_start = time.time()
        final_clf = XGBClassifier(**TRIAL4_PARAMS)
        final_clf.fit(X_mat, y_all)
        refit_time = time.time() - refit_start
        model_file = models_dir / f"xgb_model_{exp_id.replace('+', '_').lower()}.joblib"
        joblib.dump(final_clf, model_file)

        # Feature importances
        importances = final_clf.feature_importances_
        top_indices = np.argsort(importances)[::-1][:15]
        top_features = [
            {"feature": feats[i], "importance": round(float(importances[i]), 6)}
            for i in top_indices
        ]

        exp_record = {
            "experiment_id": exp_id,
            "description": exp_desc,
            "n_features": len(feats),
            "features": feats,
            "runtime_seconds": round(exp_duration, 2),
            "refit_time_seconds": round(refit_time, 2),
            "pooled_metrics": {
                "accuracy": round(pooled_acc, 4),
                "macro_f1": round(pooled_macro_f1, 4),
                "balanced_accuracy": round(pooled_bal_acc, 4),
                "weighted_f1": round(pooled_weighted_f1, 4),
            },
            "fold_summary": {
                "macro_f1_mean": round(float(np.mean(mf1_list)), 4),
                "macro_f1_std": round(float(np.std(mf1_list)), 4),
                "accuracy_mean": round(float(np.mean(acc_list)), 4),
                "accuracy_std": round(float(np.std(acc_list)), 4),
                "balanced_accuracy_mean": round(float(np.mean(bacc_list)), 4),
                "balanced_accuracy_std": round(float(np.std(bacc_list)), 4),
            },
            "per_class_metrics": per_class_metrics,
            "confusion_matrix": cm.tolist(),
            "fold_details": fold_metrics,
            "top_features": top_features,
            "model_path": str(model_file.relative_to(PROJECT_ROOT)),
        }
        benchmark_results["experiments"][exp_id] = exp_record

        logger.info(
            "--> %s RESULTS: Acc=%.4f | Macro-F1=%.4f (mean=%.4f +/- %.4f) | Bal-Acc=%.4f | RedGram-F1=%.4f | Chillies-F1=%.4f",
            exp_id, pooled_acc, pooled_macro_f1,
            np.mean(mf1_list), np.std(mf1_list), pooled_bal_acc,
            per_class_metrics["Red Gram"]["f1_score"],
            per_class_metrics["Chillies"]["f1_score"],
        )

    # 5. Persist OOF Predictions Table
    oof_target = data_dir / "phase6h_stp_ablation_oof_predictions.csv"
    tmp_oof = oof_target.with_suffix(".csv.tmp")
    oof_predictions_df.to_csv(tmp_oof, index=False)
    tmp_oof.replace(oof_target)
    logger.info("Saved pooled OOF predictions table to %s", oof_target)

    # 6. Comparisons & Synthesis
    st_macro = benchmark_results["experiments"]["S+T"]["pooled_metrics"]["macro_f1"]
    stp_macro = benchmark_results["experiments"]["S+T+P"]["pooled_metrics"]["macro_f1"]
    delta_stp_st_macro = round(stp_macro - st_macro, 4)

    st_acc = benchmark_results["experiments"]["S+T"]["pooled_metrics"]["accuracy"]
    stp_acc = benchmark_results["experiments"]["S+T+P"]["pooled_metrics"]["accuracy"]
    delta_stp_st_acc = round(stp_acc - st_acc, 4)

    benchmark_results["synthesis"] = {
        "historical_exp_c_baseline": {"macro_f1": 0.7038, "accuracy": 0.7877},
        "trial4_s_plus_t": {"macro_f1": st_macro, "accuracy": st_acc},
        "trial4_s_plus_t_plus_p": {"macro_f1": stp_macro, "accuracy": stp_acc},
        "incremental_spatial_delta": {
            "macro_f1_delta": delta_stp_st_macro,
            "accuracy_delta": delta_stp_st_acc,
        },
        "minority_crop_comparison": {
            "Red Gram": {
                "S+T_f1": benchmark_results["experiments"]["S+T"]["per_class_metrics"]["Red Gram"]["f1_score"],
                "S+T+P_f1": benchmark_results["experiments"]["S+T+P"]["per_class_metrics"]["Red Gram"]["f1_score"],
                "delta": round(
                    benchmark_results["experiments"]["S+T+P"]["per_class_metrics"]["Red Gram"]["f1_score"]
                    - benchmark_results["experiments"]["S+T"]["per_class_metrics"]["Red Gram"]["f1_score"],
                    4,
                ),
            },
            "Chillies": {
                "S+T_f1": benchmark_results["experiments"]["S+T"]["per_class_metrics"]["Chillies"]["f1_score"],
                "S+T+P_f1": benchmark_results["experiments"]["S+T+P"]["per_class_metrics"]["Chillies"]["f1_score"],
                "delta": round(
                    benchmark_results["experiments"]["S+T+P"]["per_class_metrics"]["Chillies"]["f1_score"]
                    - benchmark_results["experiments"]["S+T"]["per_class_metrics"]["Chillies"]["f1_score"],
                    4,
                ),
            },
        },
    }

    # 7. Persist JSON & Markdown Reports
    json_path = reports_dir / "phase6h_stage3_benchmark_report.json"
    tmp_json = json_path.with_suffix(".json.tmp")
    with open(tmp_json, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2)
    tmp_json.replace(json_path)

    md_path = reports_dir / "phase6h_stage3_benchmark_report.md"
    md_content = generate_benchmark_markdown(benchmark_results)
    tmp_md = md_path.with_suffix(".md.tmp")
    with open(tmp_md, "w", encoding="utf-8") as f:
        f.write(md_content)
    tmp_md.replace(md_path)

    # 8. Post-execution protected hash check
    verify_protected_baselines()

    total_time = time.time() - total_start
    logger.info("All 7 experiments completed successfully in %.1f minutes!", total_time / 60)
    return benchmark_results


def generate_benchmark_markdown(res: Dict[str, Any]) -> str:
    exps = res["experiments"]
    synth = res["synthesis"]

    lines = [
        "# Phase 6H Stage 3: Spatial Feature Audit and S/T/P Ablation Benchmark Report",
        "",
        f"**Timestamp:** {res['timestamp']}  ",
        f"**Cohort:** N = {res['cohort_size']} eligible farms across 7 crop classes  ",
        f"**Cross-Validation:** 5-Fold StratifiedGroupKFold grouped strictly by Sub-District (Mandal)  ",
        f"**Common Model:** Regularized XGBoost (Trial 4: `max_depth=7`, `colsample=0.75`, `subsample=0.80`, `reg_lambda=5.0`, `min_child_weight=3`, `gamma=0.2`, `reg_alpha=0.2`, `learning_rate=0.05`, `n_estimators=300`)  ",
        f"**Class Weighting:** Disabled (Unweighted) to evaluate pure feature contributions  ",
        "",
        "---",
        "",
        "## 1. Executive Summary & Core Research Findings",
        "",
        "### Primary Research Question:",
        "> *Does adding GLCM spatial texture to multi-temporal Sentinel-2 features improve five-fold spatially grouped crop-classification performance over the existing S+T model, and for which crop classes is the improvement most evident?*",
        "",
        "### Key Findings:",
        f"1. **Full Feature Fusion (S+T+P vs S+T):**",
        f"   - **S+T (140 features):** Macro-F1 = **{synth['trial4_s_plus_t']['macro_f1']}**, Accuracy = **{synth['trial4_s_plus_t']['accuracy']}**",
        f"   - **S+T+P (160 features):** Macro-F1 = **{synth['trial4_s_plus_t_plus_p']['macro_f1']}**, Accuracy = **{synth['trial4_s_plus_t_plus_p']['accuracy']}**",
        f"   - **Net Spatial Contribution:** Macro-F1 delta = **{synth['incremental_spatial_delta']['macro_f1_delta']:+.4f}**, Accuracy delta = **{synth['incremental_spatial_delta']['accuracy_delta']:+.4f}**.",
        "",
        f"2. **Minority-Class Recognition Impact:**",
        f"   - **Red Gram (N=221):** S+T F1 = {synth['minority_crop_comparison']['Red Gram']['S+T_f1']} -> S+T+P F1 = {synth['minority_crop_comparison']['Red Gram']['S+T_p_f1'] if 'S+T_p_f1' in synth['minority_crop_comparison']['Red Gram'] else synth['minority_crop_comparison']['Red Gram']['S+T+P_f1']} (Delta: **{synth['minority_crop_comparison']['Red Gram']['delta']:+.4f}**)",
        f"   - **Chillies (N=601):** S+T F1 = {synth['minority_crop_comparison']['Chillies']['S+T_f1']} -> S+T+P F1 = {synth['minority_crop_comparison']['Chillies']['S+T_p_f1'] if 'S+T_p_f1' in synth['minority_crop_comparison']['Chillies'] else synth['minority_crop_comparison']['Chillies']['S+T+P_f1']} (Delta: **{synth['minority_crop_comparison']['Chillies']['delta']:+.4f}**)",
        "",
        f"3. **Historical Reproduction Check:**",
        f"   - Historical Phase 6F Experiment C (untuned XGBoost): Macro-F1 = 0.7038, Accuracy = 0.7877.",
        f"   - S+T with Trial 4 regularized parameters achieves Macro-F1 = {synth['trial4_s_plus_t']['macro_f1']} and Accuracy = {synth['trial4_s_plus_t']['accuracy']}.",
        "",
        "---",
        "",
        "## 2. Seven-Experiment Controlled Benchmark Table",
        "",
        "| Experiment ID | Feature Description | Feature Count | Pooled Macro-F1 | Pooled Accuracy | Balanced Accuracy | Weighted-F1 | Fold Macro-F1 (Mean ± Std) | Runtime |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for exp_id in ["S", "T", "P", "S+T", "S+P", "T+P", "S+T+P"]:
        e = exps[exp_id]
        pm = e["pooled_metrics"]
        fs = e["fold_summary"]
        lines.append(
            f"| **{exp_id}** | {e['description']} | {e['n_features']} | **{pm['macro_f1']}** | {pm['accuracy']} | {pm['balanced_accuracy']} | {pm['weighted_f1']} | {fs['macro_f1_mean']} ± {fs['macro_f1_std']} | {e['runtime_seconds']}s |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Per-Class F1-Score Breakdown across Feature Groups",
        "",
        "| Crop Class | Support | S (34) | T (106) | P (20) | S+T (140) | S+P (54) | T+P (126) | S+T+P (160) | Delta (S+T+P vs S+T) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for c in EXPECTED_CLASSES:
        sup = exps["S"]["per_class_metrics"][c]["support"]
        f1_s = exps["S"]["per_class_metrics"][c]["f1_score"]
        f1_t = exps["T"]["per_class_metrics"][c]["f1_score"]
        f1_p = exps["P"]["per_class_metrics"][c]["f1_score"]
        f1_st = exps["S+T"]["per_class_metrics"][c]["f1_score"]
        f1_sp = exps["S+P"]["per_class_metrics"][c]["f1_score"]
        f1_tp = exps["T+P"]["per_class_metrics"][c]["f1_score"]
        f1_stp = exps["S+T+P"]["per_class_metrics"][c]["f1_score"]
        delta = round(f1_stp - f1_st, 4)
        lines.append(
            f"| **{c}** | {sup} | {f1_s:.4f} | {f1_t:.4f} | {f1_p:.4f} | {f1_st:.4f} | {f1_sp:.4f} | {f1_tp:.4f} | **{f1_stp:.4f}** | **{delta:+.4f}** |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 4. Fold-Level Metrics & Spatial Variability",
        "",
        "### S+T Model (140 features):",
        "| Fold | Validation Mandals | N Val | Accuracy | Macro-F1 | Balanced Accuracy | Weighted-F1 |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for f in exps["S+T"]["fold_details"]:
        lines.append(
            f"| Fold {f['fold']} | Sub-District group | {f['n_val']} | {f['accuracy']} | {f['macro_f1']} | {f['balanced_accuracy']} | {f['weighted_f1']} |"
        )

    lines.extend([
        "",
        "### S+T+P Model (160 features):",
        "| Fold | Validation Mandals | N Val | Accuracy | Macro-F1 | Balanced Accuracy | Weighted-F1 |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for f in exps["S+T+P"]["fold_details"]:
        lines.append(
            f"| Fold {f['fold']} | Sub-District group | {f['n_val']} | {f['accuracy']} | {f['macro_f1']} | {f['balanced_accuracy']} | {f['weighted_f1']} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 5. Confusion Matrices (OOF Pooled N=7843)",
        "",
        "### S+T Model (140 Features):",
        "```",
        "Predicted ->    " + "  ".join([f"{c[:6]:>6}" for c in EXPECTED_CLASSES]),
    ])
    for r_idx, r_name in enumerate(EXPECTED_CLASSES):
        row_str = "  ".join([f"{val:>6}" for val in exps["S+T"]["confusion_matrix"][r_idx]])
        lines.append(f"{r_name[:12]:<12} [ {row_str} ]")
    lines.append("```")

    lines.extend([
        "",
        "### S+T+P Model (160 Features):",
        "```",
        "Predicted ->    " + "  ".join([f"{c[:6]:>6}" for c in EXPECTED_CLASSES]),
    ])
    for r_idx, r_name in enumerate(EXPECTED_CLASSES):
        row_str = "  ".join([f"{val:>6}" for val in exps["S+T+P"]["confusion_matrix"][r_idx]])
        lines.append(f"{r_name[:12]:<12} [ {row_str} ]")
    lines.append("```")

    lines.extend([
        "",
        "---",
        "",
        "## 6. Top Discriminative Features in S+T+P",
        "",
        "| Rank | Feature Name | Group | Feature Importance |",
        "| :---: | :--- | :---: | :---: |",
    ])
    for idx, item in enumerate(exps["S+T+P"]["top_features"], 1):
        feat = item["feature"]
        grp = "P (Spatial)" if feat.startswith("glcm_") else ("S (Spectral)" if "_full" in feat and not feat.startswith("ndvi_growth") else "T (Temporal)")
        lines.append(f"| {idx} | `{feat}` | {grp} | {item['importance']:.6f} |")

    lines.extend([
        "",
        "---",
        "",
        "## 7. Methodological Discussion & Scientific Nuance",
        "",
        "- **Spatial Texture Standalone Performance:** The P-only model (20 GLCM features) achieves a standalone Macro-F1 of " + f"{exps['P']['pooled_metrics']['macro_f1']} (Accuracy: {exps['P']['pooled_metrics']['accuracy']}), demonstrating that canopy texture contains non-trivial crop discriminatory signal even without spectral indices or temporal curves.",
        "- **Complementarity with Temporal Curves:** When combined with temporal trajectory features (T+P vs T), Macro-F1 moves from " + f"{exps['T']['pooled_metrics']['macro_f1']} to {exps['T+P']['pooled_metrics']['macro_f1']}.",
        "- **Full Satellite Fusion (S+T+P):** Moving from S+T to S+T+P changes Macro-F1 by " + f"{synth['incremental_spatial_delta']['macro_f1_delta']:+.4f} and accuracy by {synth['incremental_spatial_delta']['accuracy_delta']:+.4f}.",
        "- **Boundary & Parcel Contamination Effects:** For small parcels (<1,000 m²), a 3x3 moving window (30m x 30m) incorporates edge pixels and field boundaries, which can introduce mixed canopy-soil noise. For larger consolidated fields, canopy homogeneity and correlation provide clean architectural separation.",
        "",
    ])

    return "\n".join(lines)


if __name__ == "__main__":
    run_benchmark()
