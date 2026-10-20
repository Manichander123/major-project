"""
scripts/generate_phase6f_cv_folds.py
====================================
Generates the authoritative 5-fold StratifiedGroupKFold spatial CV partition
grouped strictly by 'Sub-District' (Mandal) across the 7,843 eligible satellite farms.
Ensures:
  - 0 group overlap between training and validation partitions.
  - All 7 target crop classes represented in every fold.
  - Fixed random seed = 42 for absolute determinism.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_csv(df: pd.DataFrame, target_path: Path) -> str:
    temp_path = target_path.with_suffix(".csv.tmp")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(temp_path, index=False)
    temp_path.replace(target_path)
    return compute_sha256(target_path)


def atomic_write_json(data: dict, target_path: Path) -> None:
    temp_path = target_path.with_suffix(".json.tmp")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temp_path.replace(target_path)


def generate_folds(allow_overwrite: bool = False) -> tuple[pd.DataFrame, dict]:
    target_csv = PROJECT_ROOT / "outputs/data/phase6f_cv_folds.csv"
    target_json = PROJECT_ROOT / "outputs/reports/phase6f_cv_folds_audit.json"

    if not allow_overwrite:
        if target_csv.exists() or target_json.exists():
            raise FileExistsError("Target CV fold files already exist. Use --overwrite to replace.")

    features_path = PROJECT_ROOT / "outputs/data/s2_phase6f_features.csv"
    df = pd.read_csv(features_path)

    # Filter strictly to eligible satellite cohort
    eligible_df = df[df["extraction_status"] == "success"].copy().reset_index(drop=True)
    if len(eligible_df) != 7843:
        raise ValueError(f"Expected 7843 eligible farms, got {len(eligible_df)}")

    X = eligible_df[["FarmID"]]
    y = eligible_df["Crop"]
    groups = eligible_df["Sub-District"]

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    fold_assignments = pd.Series(index=eligible_df.index, dtype=int)
    fold_audit_data = []

    for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        fold_assignments.iloc[val_idx] = fold_idx

        val_df = eligible_df.iloc[val_idx]
        train_df = eligible_df.iloc[train_idx]

        val_groups = set(val_df["Sub-District"])
        train_groups = set(train_df["Sub-District"])
        overlap = val_groups.intersection(train_groups)

        if len(overlap) > 0:
            raise ValueError(f"Fold {fold_idx} has group overlap: {overlap}")

        val_classes = {str(k): int(v) for k, v in val_df["Crop"].value_counts().items()}
        train_classes = {str(k): int(v) for k, v in train_df["Crop"].value_counts().items()}

        if len(val_classes) != 7 or len(train_classes) != 7:
            raise ValueError(f"Fold {fold_idx} does not represent all 7 classes!")

        fold_audit_data.append({
            "fold": fold_idx,
            "train_samples": len(train_idx),
            "val_samples": len(val_idx),
            "val_groups_count": len(val_groups),
            "group_overlap": len(overlap),
            "val_class_distribution": val_classes,
            "val_red_gram_support": val_classes.get("Red Gram", 0),
        })

    eligible_df["fold"] = fold_assignments.values

    # Output CSV containing FarmID, fold, Sub-District, Crop, Season
    folds_df = eligible_df[["FarmID", "fold", "Sub-District", "Crop", "Season"]].copy()

    # Validation assertions
    assert len(folds_df) == 7843
    assert folds_df["FarmID"].nunique() == 7843
    assert folds_df["fold"].isin([0, 1, 2, 3, 4]).all()
    assert (folds_df.groupby("Sub-District")["fold"].nunique() == 1).all(), "Some Sub-Districts split across folds!"

    print(f"Writing CV fold assignments to {target_csv}...")
    csv_hash = atomic_write_csv(folds_df, target_csv)

    audit_summary = {
        "dataset_name": "phase6f_cv_folds.csv",
        "sha256": csv_hash,
        "n_splits": 5,
        "group_column": "Sub-District",
        "total_eligible_farms": 7843,
        "total_unique_groups": int(eligible_df["Sub-District"].nunique()),
        "folds": fold_audit_data,
        "rare_class_summary": {
            "crop": "Red Gram",
            "total_eligible": int((eligible_df["Crop"] == "Red Gram").sum()),
            "support_by_fold": {f["fold"]: f["val_red_gram_support"] for f in fold_audit_data},
        },
    }

    print(f"Writing CV fold audit to {target_json}...")
    atomic_write_json(audit_summary, target_json)
    print("Fold generation completed and verified.")
    return folds_df, audit_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Phase 6F CV Folds")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting target files")
    args = parser.parse_args()

    try:
        generate_folds(allow_overwrite=args.overwrite)
    except Exception as e:
        print(f"Error during fold generation: {e}", file=sys.stderr)
        sys.exit(1)
