"""
scripts/benchmark_phase6g_pilot.py
==================================
Phase 6G Pilot Stage: Bounded Nested Grouped Cross-Validation on Outer Fold 0.
Strictly isolated within Phase 6G namespace.

Guarantees:
  1. Zero leakage: Outer Fold 0 validation partition (N=1,508) is completely untouched
     during model selection and hyperparameter evaluation.
  2. Inner Grouped Tuning: Inner 3-fold StratifiedGroupKFold on 93 training Mandals (N=6,335).
     Asserts 0 Mandal overlap in all inner splits and all 7 classes present.
  3. Bounded budget: Exactly 10 XGBoost candidate configurations evaluated via early stopping.
  4. Selection criterion: Highest inner validation macro-F1.
  5. Refit: Selected configuration refit on full outer training split and evaluated ONCE on Outer Fold 0.
  6. Audit & Preservation: All 24 historical baseline hashes re-verified before execution.
  7. Results explicitly labeled exploratory pilot findings, not final 5-fold estimates.
"""

from __future__ import annotations

import argparse
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
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phase6g_pilot")

EXPECTED_CLASSES = [
    "Bengal Gram",
    "Chillies",
    "Cotton",
    "Groundnut",
    "Maize",
    "Paddy",
    "Red Gram",
]

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

# 10 Bounded Candidate Configurations for Nested Grouped Search
CANDIDATE_CONFIGS = [
    {
        "trial_id": 0,
        "name": "Phase 6F Baseline (Reference)",
        "params": {
            "max_depth": 7,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 1.0,
            "min_child_weight": 1,
            "gamma": 0.0,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
        },
    },
    {
        "trial_id": 1,
        "name": "Depth 6 | Feature Subsample 0.8 | L2 Regularized",
        "params": {
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "gamma": 0.1,
            "reg_alpha": 0.1,
            "reg_lambda": 3.0,
        },
    },
    {
        "trial_id": 2,
        "name": "Depth 5 | Feature Subsample 0.75 | Conservative Trees",
        "params": {
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.85,
            "colsample_bytree": 0.75,
            "min_child_weight": 3,
            "gamma": 0.1,
            "reg_alpha": 0.1,
            "reg_lambda": 3.0,
        },
    },
    {
        "trial_id": 3,
        "name": "Depth 6 | Feature Subsample 0.70 | Slower Learning Rate (0.03)",
        "params": {
            "max_depth": 6,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "min_child_weight": 3,
            "gamma": 0.2,
            "reg_alpha": 0.5,
            "reg_lambda": 5.0,
        },
    },
    {
        "trial_id": 4,
        "name": "Depth 7 | Feature Subsample 0.75 | L2 Regularized (5.0)",
        "params": {
            "max_depth": 7,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.75,
            "min_child_weight": 3,
            "gamma": 0.2,
            "reg_alpha": 0.2,
            "reg_lambda": 5.0,
        },
    },
    {
        "trial_id": 5,
        "name": "Depth 8 | Feature Subsample 0.70 | High Leaf Weight (5)",
        "params": {
            "max_depth": 8,
            "learning_rate": 0.04,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "min_child_weight": 5,
            "gamma": 0.3,
            "reg_alpha": 0.5,
            "reg_lambda": 5.0,
        },
    },
    {
        "trial_id": 6,
        "name": "Depth 6 | Strong Decorrelation (colsample 0.65) | Faster LR (0.06)",
        "params": {
            "max_depth": 6,
            "learning_rate": 0.06,
            "subsample": 0.75,
            "colsample_bytree": 0.65,
            "min_child_weight": 1,
            "gamma": 0.1,
            "reg_alpha": 0.0,
            "reg_lambda": 2.0,
        },
    },
    {
        "trial_id": 7,
        "name": "Depth 5 | colsample 0.80 | Fast Rate (0.08)",
        "params": {
            "max_depth": 5,
            "learning_rate": 0.08,
            "subsample": 0.85,
            "colsample_bytree": 0.8,
            "min_child_weight": 1,
            "gamma": 0.0,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
        },
    },
    {
        "trial_id": 8,
        "name": "Depth 7 | colsample 0.85 | Heavy Regularization (L1=1, L2=10)",
        "params": {
            "max_depth": 7,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.85,
            "min_child_weight": 5,
            "gamma": 0.2,
            "reg_alpha": 1.0,
            "reg_lambda": 10.0,
        },
    },
    {
        "trial_id": 9,
        "name": "Depth 6 | colsample 0.85 | Balanced Mild Regularization",
        "params": {
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.85,
            "min_child_weight": 1,
            "gamma": 0.0,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
        },
    },
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def verify_protected_baselines() -> None:
    logger.info("Verifying all 24 protected baseline files before pilot...")
    for rel_path, exp_hash in PROTECTED_BASELINE_HASHES.items():
        fp = PROJECT_ROOT / rel_path
        if not fp.exists():
            raise FileNotFoundError(f"Missing protected baseline file: {rel_path}")
        act_hash = sha256_file(fp)
        if act_hash != exp_hash:
            raise ValueError(f"CORRUPTION in {rel_path}! Expected: {exp_hash}, Got: {act_hash}")
    logger.info("All 24 protected baseline files verified bit-for-bit intact.")


def run_pilot(allow_overwrite: bool = True) -> Dict[str, Any]:
    start_time_all = time.time()
    logger.info("==================================================")
    logger.info("STARTING PHASE 6G PILOT: Bounded Nested Grouped CV (Outer Fold 0)")
    logger.info("==================================================")

    # 1. Verification of historical baselines
    verify_protected_baselines()

    # 2. Namespace directories
    phase6g_data_dir = PROJECT_ROOT / "outputs/data/phase6g"
    phase6g_reports_dir = PROJECT_ROOT / "outputs/reports/phase6g"
    phase6g_models_dir = PROJECT_ROOT / "outputs/models/phase6g"
    for d in [phase6g_data_dir, phase6g_reports_dir, phase6g_models_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 3. Load feature matrix and fold mapping
    logger.info("Loading feature matrix and CV folds...")
    feat_path = PROJECT_ROOT / "outputs/data/s2_phase6f_features.csv"
    folds_path = PROJECT_ROOT / "outputs/data/phase6f_cv_folds.csv"
    manifest_path = PROJECT_ROOT / "outputs/reports/phase6f_feature_schema_manifest.json"

    df_features = pd.read_csv(feat_path)
    df_folds = pd.read_csv(folds_path)

    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    exp_c_features = manifest["experiments"]["experiment_c_multitemporal_trajectory"]["features"]
    assert len(exp_c_features) == 140, f"Expected 140 features, got {len(exp_c_features)}"

    # Filter to eligible farms
    df_eligible = df_features[df_features["cloud_obscured_failure"] == False].copy()
    assert len(df_eligible) == 7843, f"Expected 7,843 eligible farms, got {len(df_eligible)}"
    assert len(df_folds) == 7843, f"Expected 7,843 farms in CV folds, got {len(df_folds)}"

    # Merge features with fold info
    df_merged = pd.merge(
        df_folds[["FarmID", "fold", "Sub-District", "Crop"]],
        df_eligible[["FarmID"] + exp_c_features],
        on="FarmID",
        how="inner",
    )
    assert len(df_merged) == 7843

    # Label encoding
    crop_to_idx = {c: i for i, c in enumerate(EXPECTED_CLASSES)}
    idx_to_crop = {i: c for i, c in enumerate(EXPECTED_CLASSES)}
    df_merged["target"] = df_merged["Crop"].map(crop_to_idx)

    # 4. Outer Fold 0 Partitioning
    outer_val_mask = df_merged["fold"] == 0
    outer_train_mask = df_merged["fold"] != 0

    df_outer_train = df_merged[outer_train_mask].copy().reset_index(drop=True)
    df_outer_val = df_merged[outer_val_mask].copy().reset_index(drop=True)

    assert len(df_outer_val) == 1508, f"Expected 1,508 farms in Outer Fold 0 val, got {len(df_outer_val)}"
    assert len(df_outer_train) == 6335, f"Expected 6,335 farms in Outer Fold 0 train, got {len(df_outer_train)}"

    # Assert Outer Group Separation
    outer_val_mandals = set(df_outer_val["Sub-District"].unique())
    outer_train_mandals = set(df_outer_train["Sub-District"].unique())
    outer_mandal_overlap = outer_val_mandals.intersection(outer_train_mandals)
    assert len(outer_mandal_overlap) == 0, f"CRITICAL: Outer Mandal overlap detected: {outer_mandal_overlap}"
    logger.info(f"Outer Fold 0: Train N={len(df_outer_train)} (93 Mandals), Val N={len(df_outer_val)} (23 Mandals). Overlap = 0.")

    # 5. Setup Inner 3-Fold StratifiedGroupKFold on Outer Train
    inner_kfold = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    inner_splits = list(
        inner_kfold.split(
            df_outer_train,
            df_outer_train["target"].values,
            groups=df_outer_train["Sub-District"].values,
        )
    )

    # Assert Inner Group Separation & Class Completeness
    logger.info("Validating Inner StratifiedGroupKFold splits on Outer Train...")
    for split_i, (in_tr_idx, in_val_idx) in enumerate(inner_splits):
        in_tr = df_outer_train.iloc[in_tr_idx]
        in_vl = df_outer_train.iloc[in_val_idx]
        overlap = set(in_tr["Sub-District"]).intersection(set(in_vl["Sub-District"]))
        assert len(overlap) == 0, f"Inner split {split_i} Mandal leakage: {overlap}"
        assert in_vl["target"].nunique() == 7, f"Inner split {split_i} missing crop classes!"
        rg_cnt = int((in_vl["Crop"] == "Red Gram").sum())
        logger.info(f"  Inner Split {split_i}: Train N={len(in_tr)}, Val N={len(in_vl)} ({in_vl['Sub-District'].nunique()} Mandals), Red Gram N={rg_cnt}. Overlap=0. PASS.")

    # 6. Execute Inner Grouped Search across 10 Candidate Configurations
    X_outer_tr = df_outer_train[exp_c_features].values
    y_outer_tr = df_outer_train["target"].values

    trial_results: List[Dict[str, Any]] = []

    logger.info("--------------------------------------------------")
    logger.info(f"Evaluating {len(CANDIDATE_CONFIGS)} Candidate XGBoost Configurations across 3 Inner Folds...")
    logger.info("--------------------------------------------------")

    for cfg in CANDIDATE_CONFIGS:
        t_id = cfg["trial_id"]
        c_name = cfg["name"]
        params = cfg["params"]
        logger.info(f"Trial {t_id}/9: {c_name} ...")

        t_start = time.time()
        inner_val_preds = np.zeros(len(df_outer_train), dtype=int) - 1
        best_iters: List[int] = []

        for split_i, (in_tr_idx, in_val_idx) in enumerate(inner_splits):
            X_in_tr, y_in_tr = X_outer_tr[in_tr_idx], y_outer_tr[in_tr_idx]
            X_in_vl, y_in_vl = X_outer_tr[in_val_idx], y_outer_tr[in_val_idx]

            clf = XGBClassifier(
                n_estimators=400,
                early_stopping_rounds=30,
                eval_metric="mlogloss",
                random_state=42 + split_i,
                n_jobs=-1,
                **params,
            )
            clf.fit(X_in_tr, y_in_tr, eval_set=[(X_in_vl, y_in_vl)], verbose=False)
            best_iters.append(clf.best_iteration)

            preds = clf.predict(X_in_vl)
            inner_val_preds[in_val_idx] = preds

        assert (inner_val_preds >= 0).all(), "Unassigned inner validation predictions!"

        # Compute Inner Validation Metrics
        in_macro_f1 = round(float(f1_score(y_outer_tr, inner_val_preds, average="macro")), 4)
        in_acc = round(float(accuracy_score(y_outer_tr, inner_val_preds)), 4)
        in_bacc = round(float(balanced_accuracy_score(y_outer_tr, inner_val_preds)), 4)

        # Red Gram and Chillies Inner Recall
        y_rg = y_outer_tr == crop_to_idx["Red Gram"]
        p_rg = inner_val_preds == crop_to_idx["Red Gram"]
        rg_f1 = round(float(f1_score(y_rg, p_rg, zero_division=0)), 4)
        rg_rec = round(float(recall_score(y_rg, p_rg, zero_division=0)), 4)

        y_ch = y_outer_tr == crop_to_idx["Chillies"]
        p_ch = inner_val_preds == crop_to_idx["Chillies"]
        ch_f1 = round(float(f1_score(y_ch, p_ch, zero_division=0)), 4)
        ch_rec = round(float(recall_score(y_ch, p_ch, zero_division=0)), 4)

        t_elapsed = round(time.time() - t_start, 2)
        mean_best_iter = int(np.round(np.mean(best_iters)))

        logger.info(
            f"  -> Trial {t_id} Completed in {t_elapsed}s | "
            f"Inner Macro-F1: {in_macro_f1:.4f} | Acc: {in_acc:.4f} | BAcc: {in_bacc:.4f} | "
            f"Red Gram F1: {rg_f1:.4f} (Rec: {rg_rec:.4f}) | Mean Iter: {mean_best_iter}"
        )

        trial_results.append({
            "trial_id": t_id,
            "name": c_name,
            "params": params,
            "inner_macro_f1": in_macro_f1,
            "inner_accuracy": in_acc,
            "inner_balanced_accuracy": in_bacc,
            "inner_red_gram_f1": rg_f1,
            "inner_red_gram_recall": rg_rec,
            "inner_chillies_f1": ch_f1,
            "inner_chillies_recall": ch_rec,
            "best_iterations": best_iters,
            "mean_best_iteration": mean_best_iter,
            "fit_time_seconds": t_elapsed,
        })

    # Save Trial Ledger
    df_trials = pd.DataFrame(trial_results)
    ledger_path = phase6g_reports_dir / "pilot_fold0_trial_ledger.csv"
    df_trials.to_csv(ledger_path, index=False)
    logger.info(f"Trial ledger saved to {ledger_path}")

    # 7. Select Best Model by Inner Validation Macro-F1
    best_trial = max(trial_results, key=lambda x: x["inner_macro_f1"])
    logger.info("==================================================")
    logger.info(
        f"SELECTION: Trial {best_trial['trial_id']} ({best_trial['name']}) "
        f"with Inner Macro-F1 = {best_trial['inner_macro_f1']:.4f} (Acc = {best_trial['inner_accuracy']:.4f})"
    )
    logger.info(f"Selected Hyperparameters: {best_trial['params']}")
    logger.info(f"Optimal Iteration Budget: {best_trial['mean_best_iteration']} trees")
    logger.info("==================================================")

    # 8. Refit Selected Configuration on Entire Outer Training Partition
    logger.info("Refitting selected model on entire Outer Fold 0 training partition (N=6,335)...")
    t_refit_start = time.time()
    final_n_estimators = max(100, best_trial["mean_best_iteration"])

    final_model = XGBClassifier(
        n_estimators=final_n_estimators,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        **best_trial["params"],
    )
    final_model.fit(X_outer_tr, y_outer_tr)
    refit_duration = round(time.time() - t_refit_start, 2)

    # Persist Refit Model Pipeline
    model_save_path = phase6g_models_dir / "xgb_pilot_fold0_selected.joblib"
    joblib.dump(final_model, model_save_path)
    logger.info(f"Final model persisted to {model_save_path}")

    # 9. Evaluate ONCE on Untouched Outer Fold 0 Validation Split (N=1,508)
    logger.info("Evaluating selected model on UNTOUCHED Outer Fold 0 Validation Partition (N=1,508)...")
    X_outer_val = df_outer_val[exp_c_features].values
    y_outer_val = df_outer_val["target"].values

    outer_preds = final_model.predict(X_outer_val)
    outer_pred_labels = [idx_to_crop[i] for i in outer_preds]
    outer_true_labels = [idx_to_crop[i] for i in y_outer_val]

    # Compute Outer Validation Metrics
    outer_macro_f1 = round(float(f1_score(y_outer_val, outer_preds, average="macro")), 4)
    outer_acc = round(float(accuracy_score(y_outer_val, outer_preds)), 4)
    outer_bacc = round(float(balanced_accuracy_score(y_outer_val, outer_preds)), 4)
    outer_weighted_f1 = round(float(f1_score(y_outer_val, outer_preds, average="weighted")), 4)

    # Per-Class Metrics
    per_class_metrics: Dict[str, Any] = {}
    for c_name in EXPECTED_CLASSES:
        c_idx = crop_to_idx[c_name]
        mask_true = y_outer_val == c_idx
        mask_pred = outer_preds == c_idx
        p = round(float(precision_score(mask_true, mask_pred, zero_division=0)), 4)
        r = round(float(recall_score(mask_true, mask_pred, zero_division=0)), 4)
        f = round(float(f1_score(mask_true, mask_pred, zero_division=0)), 4)
        supp = int(mask_true.sum())
        per_class_metrics[c_name] = {"precision": p, "recall": r, "f1": f, "support": supp}

    # Confusion Matrix
    cm = confusion_matrix(outer_true_labels, outer_pred_labels, labels=EXPECTED_CLASSES)

    # Phase 6F Outer Fold 0 Baseline Reference
    phase6f_fold0_baseline = {
        "macro_f1": 0.7095,
        "accuracy": 0.7818,
        "balanced_accuracy": 0.6973,
        "weighted_f1": 0.7723,
        "red_gram_support": 56,
        "red_gram_precision": 0.7000,
        "red_gram_recall": 0.2500,
        "red_gram_f1": 0.3684,
    }

    # Delta against Phase 6F Baseline
    delta_macro_f1 = round(outer_macro_f1 - phase6f_fold0_baseline["macro_f1"], 4)
    delta_acc = round(outer_acc - phase6f_fold0_baseline["accuracy"], 4)
    delta_bacc = round(outer_bacc - phase6f_fold0_baseline["balanced_accuracy"], 4)

    logger.info("==================================================")
    logger.info(f"PILOT OUTER FOLD 0 RESULT: Macro-F1 = {outer_macro_f1:.4f} (Baseline: {phase6f_fold0_baseline['macro_f1']:.4f}, Delta: {delta_macro_f1:+.4f})")
    logger.info(f"PILOT OUTER FOLD 0 ACCURACY: {outer_acc:.4f} (Baseline: {phase6f_fold0_baseline['accuracy']:.4f}, Delta: {delta_acc:+.4f})")
    logger.info(f"Red Gram Outer F1: {per_class_metrics['Red Gram']['f1']:.4f} (Recall: {per_class_metrics['Red Gram']['recall']:.4f}, Prec: {per_class_metrics['Red Gram']['precision']:.4f})")
    logger.info("==================================================")

    # 10. Persist Out-of-Fold Predictions for Outer Fold 0
    df_outer_oof = pd.DataFrame({
        "FarmID": df_outer_val["FarmID"],
        "fold": 0,
        "Sub-District": df_outer_val["Sub-District"],
        "true_crop": outer_true_labels,
        "predicted_crop": outer_pred_labels,
        "selected_trial_id": best_trial["trial_id"],
        "pilot_run": "phase6g_pilot_fold0",
    })
    oof_path = phase6g_data_dir / "pilot_fold0_oof_predictions.csv"
    df_outer_oof.to_csv(oof_path, index=False)
    logger.info(f"Outer Fold 0 OOF predictions saved to {oof_path}")

    # 11. Compile Comprehensive Audit & Results Dict
    pilot_summary = {
        "pilot_stage": "Phase 6G Pilot (Outer Fold 0 Nested Grouped CV)",
        "timestamp": pd.Timestamp.now().isoformat(),
        "status": "COMPLETED SUCCESSFULLY",
        "scope_boundary_note": "EXPLORATORY PILOT RESULT ON OUTER FOLD 0. NOT A FINAL 5-FOLD GENERALIZATION ESTIMATE.",
        "outer_fold": 0,
        "outer_train_samples": len(df_outer_train),
        "outer_val_samples": len(df_outer_val),
        "feature_count": len(exp_c_features),
        "total_trials": len(trial_results),
        "inner_cv_folds": 3,
        "selected_trial": best_trial,
        "outer_validation_metrics": {
            "macro_f1": outer_macro_f1,
            "accuracy": outer_acc,
            "balanced_accuracy": outer_bacc,
            "weighted_f1": outer_weighted_f1,
        },
        "phase6f_outer_fold0_baseline": phase6f_fold0_baseline,
        "delta_vs_baseline": {
            "macro_f1_delta": delta_macro_f1,
            "accuracy_delta": delta_acc,
            "balanced_accuracy_delta": delta_bacc,
        },
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": cm.tolist(),
        "trial_ledger": trial_results,
        "total_runtime_seconds": round(time.time() - start_time_all, 2),
    }

    # Save JSON Report
    json_path = phase6g_reports_dir / "phase6g_pilot_fold0_report.json"
    with open(json_path, "w") as f:
        json.dump(pilot_summary, f, indent=2)
    logger.info(f"Pilot JSON report written to {json_path}")

    # Save Markdown Report
    md_path = phase6g_reports_dir / "phase6g_pilot_fold0_report.md"
    write_pilot_markdown(pilot_summary, md_path)
    logger.info(f"Pilot Markdown report written to {md_path}")

    return pilot_summary


def write_pilot_markdown(summary: Dict[str, Any], path: Path) -> None:
    best = summary["selected_trial"]
    val_m = summary["outer_validation_metrics"]
    base_m = summary["phase6f_outer_fold0_baseline"]
    delta_m = summary["delta_vs_baseline"]
    per_class = summary["per_class_metrics"]

    lines = [
        "# Phase 6G Pilot Report: Bounded Nested Grouped Cross-Validation (Outer Fold 0)",
        "",
        "**Project:** AI-Based Remote Sensing Framework for Crop Monitoring and Irrigation Decision Support Using Multi-Temporal Sentinel-2 Satellite Imagery  ",
        "**Phase:** Phase 6G — Scientific Model Improvement  ",
        "**Stage:** Pilot Stage (Outer Fold 0 Bounded Hyperparameter Optimization)  ",
        "**Design:** 3-Fold Inner `StratifiedGroupKFold` on 93 Mandals ($N=6,335$) $\\rightarrow$ Final Refit $\\rightarrow$ Untouched Outer Fold 0 Validation ($N=1,508$, 23 Mandals)  ",
        "**Evaluation Scope:** **EXPLORATORY PILOT RESULT ONLY. NOT A FINAL 5-FOLD GENERALIZATION ESTIMATE.**  ",
        "",
        "---",
        "",
        "## 1. Executive Summary & Outer Fold 0 Comparison",
        "",
        "The nested grouped cross-validation pilot was executed successfully with 10 bounded XGBoost candidate configurations on Outer Fold 0 without spatial data leakage. Zero Sub-District groups crossed training/validation boundaries in any inner or outer split.",
        "",
        "| Metric | Phase 6F Untuned Baseline (Fold 0) | Phase 6G Tuned Pilot (Fold 0) | Delta |",
        "| :--- | :---: | :---: | :---: |",
        f"| **Outer Validation Macro-F1** | **{base_m['macro_f1']:.4f}** | **{val_m['macro_f1']:.4f}** | **{delta_m['macro_f1_delta']:+.4f}** |",
        f"| **Outer Validation Accuracy** | {base_m['accuracy']:.4f} | {val_m['accuracy']:.4f} | {delta_m['accuracy_delta']:+.4f} |",
        f"| **Balanced Accuracy** | {base_m['balanced_accuracy']:.4f} | {val_m['balanced_accuracy']:.4f} | {delta_m['balanced_accuracy_delta']:+.4f} |",
        f"| **Weighted-F1** | {base_m['weighted_f1']:.4f} | {val_m['weighted_f1']:.4f} | {val_m['weighted_f1'] - base_m['weighted_f1']:+.4f} |",
        f"| **Red Gram F1 Score** | {base_m['red_gram_f1']:.4f} | {per_class['Red Gram']['f1']:.4f} | {per_class['Red Gram']['f1'] - base_m['red_gram_f1']:+.4f} |",
        f"| **Red Gram Recall** | {base_m['red_gram_recall']:.4f} | {per_class['Red Gram']['recall']:.4f} | {per_class['Red Gram']['recall'] - base_m['red_gram_recall']:+.4f} |",
        "",
        "---",
        "",
        "## 2. Inner Grouped Search: 10-Trial Optimization Ledger",
        "",
        "Selected strictly by **Mean Inner Validation Macro-F1** across 3 inner grouped folds on the 93 outer training Mandals ($N=6,335$):",
        "",
        "| Trial ID | Configuration Name | Depth | LR | Colsample | Subsample | L2 Reg | Mean Trees | Inner Macro-F1 | Inner Acc | Red Gram F1 |",
        "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for t in summary["trial_ledger"]:
        p = t["params"]
        is_sel = " **(SELECTED)**" if t["trial_id"] == best["trial_id"] else ""
        lines.append(
            f"| {t['trial_id']} | {t['name']}{is_sel} | {p['max_depth']} | {p['learning_rate']} | "
            f"{p['colsample_bytree']} | {p['subsample']} | {p['reg_lambda']} | {t['mean_best_iteration']} | "
            f"**{t['inner_macro_f1']:.4f}** | {t['inner_accuracy']:.4f} | {t['inner_red_gram_f1']:.4f} |"
        )

    lines.extend([
        "",
        f"> **Selection Rationale:** Trial {best['trial_id']} (`{best['name']}`) achieved the highest inner validation macro-F1 ({best['inner_macro_f1']:.4f}) across all 3 inner grouped splits. Refit was performed with {best['mean_best_iteration']} trees on all 6,335 training farms.",
        "",
        "---",
        "",
        "## 3. Outer Fold 0 Per-Class Performance Breakdown",
        "",
        "| Crop Class | Support | Precision | Recall | F1 Score | Phase 6F Baseline F1 | F1 Delta |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    # We also have phase 6f baseline per class from Stage 4B fold 0
    for c_name in EXPECTED_CLASSES:
        m = per_class[c_name]
        lines.append(f"| **{c_name}** | {m['support']} | {m['precision']:.4f} | {m['recall']:.4f} | **{m['f1']:.4f}** | *Ref fold 0* | — |")

    lines.extend([
        "",
        "---",
        "",
        "## 4. Integrity, Leakage & Safety Audit",
        "",
        "1. **Outer Validation Isolation:** Outer Fold 0 validation farms ($N=1,508$) were never evaluated during inner hyperparameter search or early stopping. Predictions were generated exactly once after final refit.",
        "2. **Zero Spatial Group Leakage:** 0 Mandals overlap between inner training and inner validation across all 3 splits; 0 Mandals overlap between outer training (93 Mandals) and outer validation (23 Mandals).",
        "3. **Protected Baseline Immutability:** All 24 historical baseline files verified 100% bit-for-bit identical to recorded sha256 hashes.",
        "4. **Runtime & Computational Budget:** Total pilot runtime was " + f"{summary['total_runtime_seconds']} seconds (~{round(summary['total_runtime_seconds']/60, 1)} minutes) for 30 inner fits + 1 outer refit.",
        "",
        "---",
        "",
        "## 5. Artifact Registry",
        "",
        "- Pilot Trial Ledger: [`outputs/reports/phase6g/pilot_fold0_trial_ledger.csv`](file:///c:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/reports/phase6g/pilot_fold0_trial_ledger.csv)",
        "- Pilot JSON Report: [`outputs/reports/phase6g/phase6g_pilot_fold0_report.json`](file:///c:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/reports/phase6g/phase6g_pilot_fold0_report.json)",
        "- Pilot Markdown Report: [`outputs/reports/phase6g/phase6g_pilot_fold0_report.md`](file:///c:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/reports/phase6g/phase6g_pilot_fold0_report.md)",
        "- Outer Fold 0 Predictions: [`outputs/data/phase6g/pilot_fold0_oof_predictions.csv`](file:///c:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/data/phase6g/pilot_fold0_oof_predictions.csv)",
        "- Trained Refit Checkpoint: [`outputs/models/phase6g/xgb_pilot_fold0_selected.joblib`](file:///c:/Users/Manichander/OneDrive/Desktop/cbit/major%20project/major_proj_final/outputs/models/phase6g/xgb_pilot_fold0_selected.joblib)",
    ])

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    run_pilot()
