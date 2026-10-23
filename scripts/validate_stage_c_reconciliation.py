"""
scripts/validate_stage_c_reconciliation.py
=========================================
Stage C Automated Validation Script:
Verifies Phase 1 (Hyperparameter Tuning & Partition Counts) and
Phase 2 (Test Confusion Matrix & Metrics Reconciliation).
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CLASSES = [
    "Bengal Gram",
    "Chillies",
    "Cotton",
    "Groundnut",
    "Maize",
    "Paddy",
    "Red Gram",
]

def validate_phase1_partitions() -> dict:
    cv_folds_path = PROJECT_ROOT / "outputs" / "data" / "phase6f_cv_folds.csv"
    assert cv_folds_path.exists(), f"Missing {cv_folds_path}"
    df = pd.read_csv(cv_folds_path)
    
    # 1. Total cohort count
    assert len(df) == 7843, f"Expected 7843 rows, got {len(df)}"
    
    # 2. Fold 0 partition counts
    fold0_val = df[df["fold"] == 0]
    fold0_train = df[df["fold"] != 0]
    assert len(fold0_train) == 6335, f"Expected 6335 train, got {len(fold0_train)}"
    assert len(fold0_val) == 1508, f"Expected 1508 val, got {len(fold0_val)}"
    
    # 3. Mandals and zero overlap
    train_mandals = set(fold0_train["Sub-District"])
    val_mandals = set(fold0_val["Sub-District"])
    assert len(train_mandals) == 93, f"Expected 93 train mandals, got {len(train_mandals)}"
    assert len(val_mandals) == 23, f"Expected 23 val mandals, got {len(val_mandals)}"
    overlap = train_mandals.intersection(val_mandals)
    assert len(overlap) == 0, f"Expected 0 overlap, got {overlap}"
    
    # 4. Check all 5 folds
    fold_counts = {
        0: (6335, 1508, 23),
        1: (6243, 1600, 21),
        2: (6248, 1595, 22),
        3: (6313, 1530, 26),
        4: (6233, 1610, 24),
    }
    for fold_idx, (exp_tr, exp_val, exp_mandals) in fold_counts.items():
        v = df[df["fold"] == fold_idx]
        tr = df[df["fold"] != fold_idx]
        assert len(tr) == exp_tr
        assert len(v) == exp_val
        assert v["Sub-District"].nunique() == exp_mandals
        assert len(set(tr["Sub-District"]).intersection(set(v["Sub-District"]))) == 0

    # 5. Check Phase 6G pilot report JSON
    pilot_json_path = PROJECT_ROOT / "outputs" / "reports" / "phase6g" / "phase6g_pilot_fold0_report.json"
    assert pilot_json_path.exists()
    with open(pilot_json_path) as f:
        pdata = json.load(f)
    assert pdata["outer_train_samples"] == 6335
    assert pdata["outer_val_samples"] == 1508
    assert pdata["selected_trial"]["trial_id"] == 4
    assert np.isclose(pdata["selected_trial"]["inner_macro_f1"], 0.6797, atol=1e-4)
    assert np.isclose(pdata["outer_validation_metrics"]["macro_f1"], 0.7024, atol=1e-4)
    assert np.isclose(pdata["phase6f_outer_fold0_baseline"]["macro_f1"], 0.7095, atol=1e-4)
    
    return {
        "status": "PASS",
        "total_eligible": 7843,
        "fold0_train": 6335,
        "fold0_val": 1508,
        "fold0_train_mandals": 93,
        "fold0_val_mandals": 23,
        "trial4_inner_macro_f1": 0.6797,
        "trial4_outer_macro_f1": 0.7024,
        "phase6f_baseline_outer_macro_f1": 0.7095,
    }


def validate_phase2_test_metrics() -> dict:
    test_pred_path = PROJECT_ROOT / "outputs" / "data" / "phase7" / "test_predictions.csv"
    assert test_pred_path.exists(), f"Missing {test_pred_path}"
    df = pd.read_csv(test_pred_path)
    
    # 1. Total and status split
    assert len(df) == 2718, f"Expected 2718 rows, got {len(df)}"
    eval_df = df[df["extraction_status"] == "success"].copy()
    excluded_df = df[df["extraction_status"] != "success"].copy()
    assert len(eval_df) == 2265, f"Expected 2265 evaluated, got {len(eval_df)}"
    assert len(excluded_df) == 453, f"Expected 453 excluded, got {len(excluded_df)}"
    
    # 2. Excluded rows have null predictions
    assert excluded_df["predicted_crop"].isnull().sum() == 453
    assert eval_df["predicted_crop"].isnull().sum() == 0
    assert eval_df["true_crop"].isnull().sum() == 0
    
    # 3. Train vs Test Mandal overlap check
    train_df = pd.read_csv(PROJECT_ROOT / "outputs" / "data" / "phase6f_cv_folds.csv")
    train_mandals = set(train_df["Sub-District"].unique())
    test_mandals = set(df["Sub-District"].unique())
    assert len(test_mandals) == 35, f"Expected 35 test mandals, got {len(test_mandals)}"
    overlap = train_mandals.intersection(test_mandals)
    assert len(overlap) == 0, f"Expected 0 mandal overlap, got {overlap}"
    
    # 4. Confusion matrix recomputation
    y_true = eval_df["true_crop"]
    y_pred = eval_df["predicted_crop"]
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    
    expected_cm = np.array([
        [36,   1,   4,   0,   2,   1,   2],
        [ 0, 243,  40,   5,  12,  11,   0],
        [ 7,  14, 385,  20,  15,  28,   3],
        [ 0,   9,  11, 211,   6,   4,   0],
        [ 6,  41,  36,  12, 495,  40,   1],
        [ 1,  18,  16,   5,  18, 470,   1],
        [ 3,   2,  21,   0,   1,   1,   7]
    ])
    assert np.array_equal(cm, expected_cm), f"CM mismatch:\n{cm}\nvs\n{expected_cm}"
    
    # 5. Row and Column sums
    row_sums = cm.sum(axis=1)
    col_sums = cm.sum(axis=0)
    expected_row_sums = np.array([46, 311, 472, 241, 631, 529, 35])
    expected_col_sums = np.array([53, 328, 513, 253, 549, 555, 14])
    assert np.array_equal(row_sums, expected_row_sums)
    assert np.array_equal(col_sums, expected_col_sums)
    
    # 6. Diagonal (correct) and off-diagonal (incorrect)
    correct = int(np.trace(cm))
    incorrect = int(len(eval_df) - correct)
    assert correct == 1847
    assert incorrect == 418
    
    # 7. Headline metrics
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", labels=CLASSES))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", labels=CLASSES))
    
    assert np.isclose(acc, 0.8154525, atol=1e-5)
    assert np.isclose(bal_acc, 0.732585, atol=1e-5)
    assert np.isclose(macro_f1, 0.730810, atol=1e-5)
    assert np.isclose(weighted_f1, 0.813671, atol=1e-5)
    
    # 8. Check phase7_test_metrics.json
    metrics_json_path = PROJECT_ROOT / "outputs" / "reports" / "phase7" / "phase7_test_metrics.json"
    assert metrics_json_path.exists()
    with open(metrics_json_path) as f:
        mdata = json.load(f)
    assert np.isclose(mdata["metrics"]["accuracy"], 0.8155, atol=1e-4)
    assert np.isclose(mdata["metrics"]["balanced_accuracy"], 0.7326, atol=1e-4)
    assert np.isclose(mdata["metrics"]["macro_f1"], 0.7308, atol=1e-4)
    assert np.isclose(mdata["metrics"]["weighted_f1"], 0.8137, atol=1e-4)
    assert mdata["metrics"]["correct_predictions"] == 1847
    assert mdata["metrics"]["incorrect_predictions"] == 418
    
    # 9. Class-specific checks (Red Gram & Paddy)
    rg = mdata["per_class"]["Red Gram"]
    assert rg["evaluated_support"] == 35
    assert rg["precision"] == 0.5
    assert rg["recall"] == 0.2
    assert np.isclose(rg["f1_score"], 0.2857, atol=1e-4)
    
    pad = mdata["per_class"]["Paddy"]
    assert pad["evaluated_support"] == 529
    assert pad["cloud_excluded"] == 185
    assert np.isclose(pad["precision"], 0.8468, atol=1e-4)
    assert np.isclose(pad["recall"], 0.8885, atol=1e-4)
    assert np.isclose(pad["f1_score"], 0.8672, atol=1e-4)
    
    return {
        "status": "PASS",
        "total_test": 2718,
        "evaluated": 2265,
        "excluded": 453,
        "correct": 1847,
        "incorrect": 418,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "confusion_matrix": cm.tolist(),
        "row_sums": row_sums.tolist(),
        "col_sums": col_sums.tolist(),
    }


def main():
    print("Running Stage C Automated Validation...")
    p1 = validate_phase1_partitions()
    print("Phase 1 Validation: PASS")
    p2 = validate_phase2_test_metrics()
    print("Phase 2 Validation: PASS")
    
    out_dir = PROJECT_ROOT / "outputs" / "remediation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "regenerated_test_metrics.json"
    with open(out_file, "w") as f:
        json.dump({"phase1": p1, "phase2": p2}, f, indent=2)
    print(f"Saved machine-readable audit outputs to: {out_file}")

if __name__ == "__main__":
    main()
