"""
scripts/audit_spatial_validation.py
====================================
Spatial Validation Feasibility and Sensitivity Audit for Phase 6D:
- Projects farm centroids from WGS84 (EPSG:4326) to UTM Zone 44N (EPSG:32644)
- Measures nearest-neighbor distances across Sub-District boundaries
- Tests spatial block grid sizes (25km, 50km, 75km)
- Evaluates buffer exclusions (5km, 10km, 15km) and quantifies class dropout
- Evaluates spatial generalization metrics under spatial blocks
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial.distance import cdist
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold, StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.train_pilot_models import CLASSES_7, define_feature_subsets

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)


def main():
    feat_path = Path("outputs/data/s2_phase6c_350_features.csv")
    df = pd.read_csv(feat_path)
    clean = df[df["status"] == "success"].copy().reset_index(drop=True)
    logger.info("Loaded %d clean valid farms", len(clean))

    # CRS Transformation: EPSG:4326 -> EPSG:32644 (UTM Zone 44N, Telangana)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32644", always_xy=True)
    coords_utm = np.array([transformer.transform(lon, lat) for lon, lat in zip(clean["centroid_lon"], clean["centroid_lat"])])
    clean["utm_x"] = coords_utm[:, 0]
    clean["utm_y"] = coords_utm[:, 1]

    # 1. Inter-Mandal Boundary Distance Distribution
    logger.info("Computing pairwise inter-mandal distances...")
    dist_matrix = cdist(coords_utm, coords_utm, metric="euclidean")
    np.fill_diagonal(dist_matrix, np.inf)

    # For each farm, find nearest neighbor in a DIFFERENT Sub-District
    mandal_nn_distances = []
    for i, row in clean.iterrows():
        mandal_i = row["Sub-District"]
        mask = clean["Sub-District"] != mandal_i
        diff_mandal_dists = dist_matrix[i, mask]
        if len(diff_mandal_dists) > 0:
            mandal_nn_distances.append(np.min(diff_mandal_dists))

    mandal_nn_dists = np.array(mandal_nn_distances)
    logger.info("Nearest Neighbor across different Sub-Districts (meters):")
    logger.info("Min: %.1f m | 5th: %.1f m | 10th: %.1f m | Median: %.1f m | Mean: %.1f m",
                np.min(mandal_nn_dists), np.percentile(mandal_nn_dists, 5),
                np.percentile(mandal_nn_dists, 10), np.median(mandal_nn_dists), np.mean(mandal_nn_dists))

    buffer_thresholds = [1000, 5000, 10000, 15000, 20000]
    boundary_proximity = {}
    for b in buffer_thresholds:
        count = int(np.sum(mandal_nn_dists < b))
        pct = (count / len(clean)) * 100
        boundary_proximity[f"within_{b//1000}km"] = {"count": count, "pct": round(pct, 2)}
        logger.info("Farms within %d km of adjacent mandal: %d (%.2f%%)", b // 1000, count, pct)

    # 2. Spatial Block Partitioning Analysis
    # Let's test grid block sizes: 25 km, 50 km, 75 km
    min_x, max_x = clean["utm_x"].min(), clean["utm_x"].max()
    min_y, max_y = clean["utm_y"].min(), clean["utm_y"].max()
    logger.info("Spatial extent: X: %.1f km, Y: %.1f km", (max_x - min_x)/1000, (max_y - min_y)/1000)

    grid_results = {}
    for block_size_km in [25, 50, 75]:
        b_size = block_size_km * 1000
        clean[f"grid_{block_size_km}km"] = (
            ((clean["utm_x"] - min_x) // b_size).astype(int).astype(str)
            + "_"
            + ((clean["utm_y"] - min_y) // b_size).astype(int).astype(str)
        )
        n_blocks = clean[f"grid_{block_size_km}km"].nunique()
        counts_per_block = clean[f"grid_{block_size_km}km"].value_counts()
        
        # Check class coverage across blocks
        crosstab = pd.crosstab(clean[f"grid_{block_size_km}km"], clean["Crop"])
        blocks_with_all_7 = int(np.sum((crosstab > 0).sum(axis=1) == 7))
        
        grid_results[f"{block_size_km}km"] = {
            "n_blocks": n_blocks,
            "min_farms_per_block": int(counts_per_block.min()),
            "max_farms_per_block": int(counts_per_block.max()),
            "median_farms_per_block": float(counts_per_block.median()),
            "blocks_with_all_7_classes": blocks_with_all_7,
        }
        logger.info("%d km Blocks: Total blocks=%d, Blocks with all 7 classes=%d", block_size_km, n_blocks, blocks_with_all_7)

    # 3. Spatial Cross-Validation Comparison: Sub-District vs. 50km Spatial Block
    # Prepare features
    if "duration_days" not in clean.columns:
        sdate = pd.to_datetime(clean["SDate"], format="mixed", dayfirst=True)
        hdate = pd.to_datetime(clean["HDate"], format="mixed", dayfirst=True)
        clean["duration_days"] = (hdate - sdate).dt.days.clip(lower=1)
        clean["sowing_month"] = sdate.dt.month
        clean["sowing_doy"] = sdate.dt.dayofyear
        clean["harvest_month"] = hdate.dt.month
        clean["harvest_doy"] = hdate.dt.dayofyear

    sat_cols, multi_cols, _ = define_feature_subsets(clean)

    # Compare 5-fold StratifiedGroupKFold on Sub-District vs 5-fold GroupKFold on 50km block
    y_raw = clean["Crop"]
    y_idx = np.array([CLASSES_7.index(c) for c in y_raw])
    X = clean[multi_cols].copy()
    num_cols = [c for c in multi_cols if c != "Season"]
    # Simple median imputation for benchmark evaluation
    X[num_cols] = X[num_cols].fillna(X[num_cols].median())
    X["Season"] = (X["Season"] == "Rabi").astype(int)

    # Model: Random Forest
    def evaluate_grouping(groups, grouping_name):
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        oof_preds = np.zeros(len(clean), dtype=int)
        fold_f1s = []
        class_missing_in_folds = 0

        for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y_idx, groups)):
            y_train, y_val = y_idx[train_idx], y_idx[val_idx]
            if len(np.unique(y_val)) < 7:
                class_missing_in_folds += 1

            clf = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_split=4, class_weight="balanced", random_state=42 + fold, n_jobs=-1)
            clf.fit(X.iloc[train_idx], y_train)
            preds = clf.predict(X.iloc[val_idx])
            oof_preds[val_idx] = preds
            fold_f1s.append(f1_score(y_val, preds, average="macro", zero_division=0))

        macro_f1 = f1_score(y_idx, oof_preds, average="macro", zero_division=0)
        acc = accuracy_score(y_idx, oof_preds)
        return {
            "grouping": grouping_name,
            "overall_macro_f1": round(float(macro_f1), 4),
            "overall_accuracy": round(float(acc), 4),
            "fold_macro_f1_mean": round(float(np.mean(fold_f1s)), 4),
            "fold_macro_f1_std": round(float(np.std(fold_f1s)), 4),
            "folds_with_missing_classes": class_missing_in_folds,
        }

    res_subdistrict = evaluate_grouping(clean["Sub-District"], "Sub-District (Administrative Mandal)")
    res_block50 = evaluate_grouping(clean["grid_50km"], "50km Projected Spatial Block")

    logger.info("Evaluation comparison:")
    logger.info("Sub-District: Macro-F1=%.4f (+/- %.4f) | Acc=%.4f", res_subdistrict["overall_macro_f1"], res_subdistrict["fold_macro_f1_std"], res_subdistrict["overall_accuracy"])
    logger.info("50km Block:   Macro-F1=%.4f (+/- %.4f) | Acc=%.4f", res_block50["overall_macro_f1"], res_block50["fold_macro_f1_std"], res_block50["overall_accuracy"])

    report = {
        "crs": "EPSG:32644 (UTM Zone 44N, Telangana - units in meters)",
        "inter_mandal_distance_stats_meters": {
            "min": round(float(np.min(mandal_nn_dists)), 1),
            "percentile_5": round(float(np.percentile(mandal_nn_dists, 5)), 1),
            "percentile_10": round(float(np.percentile(mandal_nn_dists, 10)), 1),
            "median": round(float(np.median(mandal_nn_dists)), 1),
            "mean": round(float(np.mean(mandal_nn_dists)), 1),
        },
        "boundary_proximity": boundary_proximity,
        "grid_block_analysis": grid_results,
        "model_comparison": {
            "sub_district": res_subdistrict,
            "block_50km": res_block50,
        },
    }

    out_file = Path("outputs/reports/spatial_validation_audit.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved spatial validation audit report -> %s", out_file)


if __name__ == "__main__":
    main()
