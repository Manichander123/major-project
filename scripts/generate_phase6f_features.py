"""
scripts/generate_phase6f_features.py
====================================
Generates the authoritative Phase 6F feature matrix from the harmonized cohort.
Computes:
  - 14 spectral indices for each of the 4 phenological windows (56 index columns)
  - 4 temporal trajectory deltas
  - 3 pre-sowing calendar features (Season, sowing_month, sowing_doy)
  - 3 field geometry descriptors (area_ha, perimeter_m, compactness)
Strictly excludes all survey proxies, post-harvest variables, and spatial coordinates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.indices import compute_all_spectral_indices, compute_temporal_dynamics

PROTECTED_BASELINE_HASHES = {
    "datasss/Train.csv": "29d31920f5b9aaea8213d7993dedc50c5d19458906f85642d360d785bb07aa06",
    "outputs/data/s2_phase6f_harmonized_cohort.csv": "caed402949eb577425b9d2e2cac0ee2c8462f635ca22b466a7675962ebb5a7af",
    "outputs/data/train_enriched_geometry.csv": "80645c793132e01df391d1e44f80e55627255bb4d44aa6c4644a42b15747cb9b",
}


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


def generate_features(allow_overwrite: bool = False) -> tuple[pd.DataFrame, dict]:
    target_csv = PROJECT_ROOT / "outputs/data/s2_phase6f_features.csv"
    target_manifest = PROJECT_ROOT / "outputs/reports/phase6f_feature_schema_manifest.json"

    if not allow_overwrite:
        if target_csv.exists() or target_manifest.exists():
            raise FileExistsError("Target feature files already exist. Use --overwrite to replace.")

    # 1. Pre-execution verification of harmonized cohort
    harm_path = PROJECT_ROOT / "outputs/data/s2_phase6f_harmonized_cohort.csv"
    if compute_sha256(harm_path) != PROTECTED_BASELINE_HASHES["outputs/data/s2_phase6f_harmonized_cohort.csv"]:
        raise ValueError("Protected baseline s2_phase6f_harmonized_cohort.csv has been mutated!")

    df_harm = pd.read_csv(harm_path)
    if len(df_harm) != 7888 or df_harm["FarmID"].nunique() != 7888:
        raise ValueError(f"Cohort size mismatch: {len(df_harm)} rows, {df_harm['FarmID'].nunique()} unique IDs")

    # 2. Compute spectral indices across all 4 windows
    windows = ["early", "mid", "late", "full"]
    indices_dfs = []

    for w in windows:
        band_cols = {
            "blue_col": f"B2_{w}_mean",
            "green_col": f"B3_{w}_mean",
            "red_col": f"B4_{w}_mean",
            "re1_col": f"B5_{w}_mean",
            "re2_col": f"B6_{w}_mean",
            "re3_col": f"B7_{w}_mean",
            "nir_col": f"B8_{w}_mean",
            "swir1_col": f"B11_{w}_mean",
            "swir2_col": f"B12_{w}_mean",
        }
        w_indices = compute_all_spectral_indices(df_harm, prefix="", scale_factor=10000.0, **band_cols)
        # Rename columns to {index}_{window}
        w_indices.columns = [f"{col}_{w}" for col in w_indices.columns]
        indices_dfs.append(w_indices)

    all_indices = pd.concat(indices_dfs, axis=1)

    # 3. Compute 4 temporal dynamics deltas
    temporal_dynamics = compute_temporal_dynamics(all_indices)

    # 4. Extract pre-sowing calendar features
    sdate = pd.to_datetime(df_harm["SDate"], format="mixed")
    calendar_df = pd.DataFrame({
        "sowing_month": sdate.dt.month.astype(int),
        "sowing_doy": sdate.dt.dayofyear.astype(int),
    }, index=df_harm.index)

    # 5. Load geometry shape features
    geom_path = PROJECT_ROOT / "outputs/data/train_enriched_geometry.csv"
    df_geom = pd.read_csv(geom_path).set_index("FarmID")
    geom_features = df_geom.loc[df_harm["FarmID"], ["area_ha", "perimeter_m", "compactness"]].reset_index(drop=True)

    # 6. Assemble complete features DataFrame
    # Retain metadata & canonical raw bands (100 columns), then append 56 indices + 4 deltas + 2 calendar + 3 geometry
    full_df = pd.concat([df_harm, all_indices, temporal_dynamics, calendar_df, geom_features], axis=1)

    # 7. Define Experiment Feature Column Sets
    exp_a_features = (
        [f"B{b}_full_{stat}" for b in ["2", "3", "4", "8", "5", "6", "7", "8A", "11", "12"] for stat in ["mean", "std"]]
        + [f"{idx}_full" for idx in ["ndvi", "evi", "gndvi", "savi", "msavi", "ndmi", "ndwi", "ndre1", "ndre2", "psri", "bsi", "msi", "cire", "cig"]]
    )

    exp_b_features = (
        [f"B{b}_early_{stat}" for b in ["2", "3", "4", "8", "5", "6", "7", "8A", "11", "12"] for stat in ["mean", "std"]]
        + [f"{idx}_early" for idx in ["ndvi", "evi", "gndvi", "savi", "msavi", "ndmi", "ndwi", "ndre1", "ndre2", "psri", "bsi", "msi", "cire", "cig"]]
    )

    exp_c_features = (
        [f"B{b}_{w}_{stat}" for w in windows for b in ["2", "3", "4", "8", "5", "6", "7", "8A", "11", "12"] for stat in ["mean", "std"]]
        + [f"{idx}_{w}" for w in windows for idx in ["ndvi", "evi", "gndvi", "savi", "msavi", "ndmi", "ndwi", "ndre1", "ndre2", "psri", "bsi", "msi", "cire", "cig"]]
        + ["ndvi_growth_delta", "peak_to_early_ratio", "ndvi_senescence_delta", "moisture_loss_delta"]
    )

    exp_d_features = exp_c_features + ["Season", "sowing_month", "sowing_doy", "area_ha", "perimeter_m", "compactness"]

    # Assertions on exact dimensions
    assert len(exp_a_features) == 34, f"Exp A expected 34 cols, got {len(exp_a_features)}"
    assert len(exp_b_features) == 34, f"Exp B expected 34 cols, got {len(exp_b_features)}"
    assert len(exp_c_features) == 140, f"Exp C expected 140 cols, got {len(exp_c_features)}"
    assert len(exp_d_features) == 146, f"Exp D expected 146 cols, got {len(exp_d_features)}"

    # Check that all features exist in full_df
    for feat_set, name in [(exp_a_features, "Exp A"), (exp_b_features, "Exp B"), (exp_c_features, "Exp C"), (exp_d_features, "Exp D")]:
        missing = [c for c in feat_set if c not in full_df.columns]
        assert not missing, f"Missing columns in {name}: {missing}"

    # Atomic write to CSV
    print(f"Writing feature matrix to {target_csv}...")
    csv_hash = atomic_write_csv(full_df, target_csv)

    manifest_data = {
      "dataset_name": "s2_phase6f_features.csv",
      "sha256": csv_hash,
      "total_rows": len(full_df),
      "total_columns": len(full_df.columns),
      "experiments": {
          "experiment_a_full_season_retrospective": {
              "description": "Satellite-only retrospective full-season benchmark",
              "feature_count": len(exp_a_features),
              "features": exp_a_features,
          },
          "experiment_b_early_season_in_season": {
              "description": "Satellite-only in-season early-stage benchmark",
              "feature_count": len(exp_b_features),
              "features": exp_b_features,
          },
          "experiment_c_multitemporal_trajectory": {
              "description": "Multi-temporal phenological trajectory satellite benchmark",
              "feature_count": len(exp_c_features),
              "features": exp_c_features,
          },
          "experiment_d_multimodal_benchmark": {
              "description": "Multimodal benchmark combining multi-temporal satellite, pre-sowing calendar, and shape geometry",
              "feature_count": len(exp_d_features),
              "features": exp_d_features,
          },
      },
      "excluded_columns_rationale": {
          "prohibited_survey_proxies": [
              "ExpYield", "CropCoveredArea", "CHeight", "CLast", "CNext", "CTransp",
              "IrriType", "IrriSource", "IrriCount", "WaterCov"
          ],
          "prohibited_spatial_coordinates": ["lat", "lon", "centroid_lat", "centroid_lon", "bbox_min_lat", "bbox_max_lat", "bbox_min_lon", "bbox_max_lon"],
          "prohibited_administrative_names": ["State", "District", "Sub-District"],
          "prohibited_targets_and_ids": ["Crop", "category", "FarmID"],
          "prohibited_post_harvest": ["duration_days", "harvest_month", "harvest_doy", "HDate"]
      }
    }

    print(f"Writing feature schema manifest to {target_manifest}...")
    atomic_write_json(manifest_data, target_manifest)
    print("Feature generation completed successfully.")
    return full_df, manifest_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Phase 6F Feature Matrix")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting target files")
    args = parser.parse_args()

    try:
        generate_features(allow_overwrite=args.overwrite)
    except Exception as e:
        print(f"Error during feature generation: {e}", file=sys.stderr)
        sys.exit(1)
