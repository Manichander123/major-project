"""
scripts/extract_s2_multicrop_pilot.py
=====================================
Multi-Crop Sentinel-2 Extraction & Feature Engineering Pilot.

Objective:
    Extract multi-temporal Sentinel-2 L2A surface reflectance data for a balanced,
    geographically diverse pilot sample of 25 farms per crop class across 7 classes
    (175 farms total):
    ['Bengal Gram', 'Chillies', 'Cotton', 'Groundnut', 'Maize', 'Paddy', 'Red Gram']

Guarantees & Integrity:
    - Primary target is 'Crop' (label only, strictly excluded from feature inputs).
    - 'category' (secondary target) is excluded.
    - All post-hoc survey proxies ('CTransp', 'ExpYield', etc.) are excluded.
    - True farm polygon geometry is reduced natively in Earth Engine (no centroid box).
    - Cloud masking using Scene Classification Layer (SCL) surface reflectance mask.
    - Native band resolutions respected: 10m bands at 10m, 20m bands at 20m.
    - Surface reflectance scale factor applied exactly once (dividing DN by 10,000).
    - Missing observations and cloud-obscured windows preserved as NaN.
    - Multi-threaded GEE extraction (max_workers=4, from project_config.yaml) with
      disk checkpointing every 5 farms and automatic resume support.

Usage:
    python scripts/extract_s2_multicrop_pilot.py --smoke-test   # Run 7 farms (1 per crop)
    python scripts/extract_s2_multicrop_pilot.py                # Run full 175-farm pilot
"""

from __future__ import annotations

import argparse
from concurrent.futures import as_completed, ThreadPoolExecutor
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from shapely import wkt

# UTF-8 stdout configuration for Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.indices import compute_all_spectral_indices, compute_temporal_dynamics
from src.satellite.s2_extractor import (
    DEFAULT_CLOUD_COVER_THRESHOLD,
    EarthEngineManager,
    S2_ALL_BANDS,
    S2_SR_COLLECTION,
    extract_zonal_stats_for_farm,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Constants
CLASSES_7: List[str] = [
    "Bengal Gram",
    "Chillies",
    "Cotton",
    "Groundnut",
    "Maize",
    "Paddy",
    "Red Gram",
]

WINDOWS: List[str] = ["early", "mid", "late", "full"]

OUTPUTS_DATA_DIR = PROJECT_ROOT / "outputs" / "data"
OUTPUTS_REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
OUTPUTS_DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def select_pilot_manifest(
    enriched_data_path: Path,
    n_per_crop: int = 25,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Select n_per_crop distinct, geographically diverse farms for each of the 7 crops.
    """
    logger.info("Loading enriched data from %s...", enriched_data_path)
    df = pd.read_csv(enriched_data_path)

    selected_dfs = []
    for crop in CLASSES_7:
        crop_subset = df[df["Crop"] == crop].copy()
        if len(crop_subset) < n_per_crop:
            raise ValueError(
                f"Crop '{crop}' has only {len(crop_subset)} farms, fewer than requested {n_per_crop}."
            )

        # Maximize geographic diversity by grouping across districts
        districts = crop_subset["District"].unique()
        shuffled = crop_subset.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        samples_by_dist = []
        grouped = {d: shuffled[shuffled["District"] == d].copy() for d in districts}

        count = 0
        while count < n_per_crop:
            for d in districts:
                if count >= n_per_crop:
                    break
                d_df = grouped[d]
                if not d_df.empty:
                    samples_by_dist.append(d_df.iloc[[0]])
                    grouped[d] = d_df.iloc[1:]
                    count += 1

        crop_sample = pd.concat(samples_by_dist, ignore_index=True)
        selected_dfs.append(crop_sample)

    manifest = pd.concat(selected_dfs, ignore_index=True)
    manifest["FarmID"] = manifest["FarmID"].astype(int)
    return manifest


def _extract_single_farm(
    row_dict: Dict[str, Any],
    cloud_cover_threshold: float,
) -> Dict[str, Any]:
    """Helper worker to extract one farm."""
    farm_id = int(row_dict["FarmID"])
    crop = row_dict["Crop"]
    district = row_dict["District"]
    sdate = str(row_dict["SDate"])[:10]
    hdate = str(row_dict["HDate"])[:10]
    geom = wkt.loads(str(row_dict["geometry"]))

    try:
        res = extract_zonal_stats_for_farm(
            farm_id=farm_id,
            geometry=geom,
            sdate_str=sdate,
            hdate_str=hdate,
            collection_name=S2_SR_COLLECTION,
            cloud_cover_threshold=cloud_cover_threshold,
        )
    except Exception as exc:
        err_msg = str(exc)
        logger.error("Extraction error for FarmID %d: %s", farm_id, err_msg)
        res = {
            "FarmID": farm_id,
            "status": "failed",
            "error_message": err_msg,
        }

    res["Crop"] = crop
    res["District"] = district
    res["Sub-District"] = row_dict.get("Sub-District", "")
    res["Season"] = row_dict.get("Season", "")
    res["SDate"] = sdate
    res["HDate"] = hdate
    return res


def extract_pilot_raw_bands(
    manifest: pd.DataFrame,
    raw_bands_cache_file: Path,
    cloud_cover_threshold: float = DEFAULT_CLOUD_COVER_THRESHOLD,
    max_workers: int = 4,
    checkpoint_interval: int = 5,
) -> pd.DataFrame:
    """
    Execute multi-temporal zonal extraction with parallel threads and disk checkpointing.
    """
    cached_records: Dict[int, Dict[str, Any]] = {}
    successful_ids = set()

    if raw_bands_cache_file.exists():
        try:
            cached_df = pd.read_csv(raw_bands_cache_file)
            for rec in cached_df.to_dict("records"):
                fid = int(rec["FarmID"])
                if rec.get("status") == "success":
                    successful_ids.add(fid)
                    cached_records[fid] = rec
            logger.info(
                "Cache check: %d successfully extracted farms already on disk.",
                len(successful_ids),
            )
        except Exception as e:
            logger.warning("Could not read existing cache file %s: %s", raw_bands_cache_file, e)

    records_to_process = manifest[~manifest["FarmID"].isin(successful_ids)]
    logger.info("Farms to extract: %d (Skipping %d cached)", len(records_to_process), len(successful_ids))

    all_records = list(cached_records.values())

    if records_to_process.empty:
        logger.info("All manifest farms already successfully extracted.")
        return pd.DataFrame(all_records)

    row_dicts = records_to_process.to_dict("records")
    completed_in_run = 0

    logger.info("Launching parallel extraction with %d worker threads...", max_workers)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(_extract_single_farm, r, cloud_cover_threshold): r["FarmID"]
            for r in row_dicts
        }

        for future in as_completed(future_map):
            completed_in_run += 1
            fid = future_map[future]
            try:
                res = future.result()
                all_records.append(res)
                logger.info(
                    "Progress [%d/%d] Completed FarmID %d | Status: %s",
                    completed_in_run,
                    len(row_dicts),
                    fid,
                    res.get("status"),
                )
            except Exception as e:
                logger.error("Future exception for FarmID %d: %s", fid, e)
                all_records.append({
                    "FarmID": fid,
                    "status": "failed",
                    "error_message": str(e),
                })

            # Checkpoint to disk every checkpoint_interval
            if completed_in_run % checkpoint_interval == 0 or completed_in_run == len(row_dicts):
                checkpoint_df = pd.DataFrame(all_records)
                checkpoint_df.to_csv(raw_bands_cache_file, index=False)
                logger.info("Saved checkpoint (%d total records) -> %s", len(checkpoint_df), raw_bands_cache_file)

    final_raw_df = pd.DataFrame(all_records)
    final_raw_df.to_csv(raw_bands_cache_file, index=False)
    return final_raw_df


def compute_pilot_features(
    raw_df: pd.DataFrame,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute 14 spectral indices for each window, plus multi-temporal dynamics.
    Merges safe geometry features without leaking survey variables.
    """
    df = raw_df.copy()

    # Calculate 14 spectral indices for each of the 4 windows
    indices_dfs = []
    for w in WINDOWS:
        b4_col = f"B4_{w}_mean"
        b8_col = f"B8_{w}_mean"
        if b4_col in df.columns and b8_col in df.columns:
            w_indices = compute_all_spectral_indices(
                df,
                prefix="",
                scale_factor=10000.0,
                blue_col=f"B2_{w}_mean",
                green_col=f"B3_{w}_mean",
                red_col=f"B4_{w}_mean",
                re1_col=f"B5_{w}_mean",
                re2_col=f"B6_{w}_mean",
                re3_col=f"B7_{w}_mean",
                nir_col=f"B8_{w}_mean",
                swir1_col=f"B11_{w}_mean",
                swir2_col=f"B12_{w}_mean",
            )
            # Rename columns to {index}_{window} (e.g. ndvi_early, ndwi_mid)
            w_indices.columns = [f"{c}_{w}" for c in w_indices.columns]
            indices_dfs.append(w_indices)

    if indices_dfs:
        all_indices_df = pd.concat(indices_dfs, axis=1)
        temporal_dyn = compute_temporal_dynamics(all_indices_df)
        feature_block = pd.concat([df, all_indices_df, temporal_dyn], axis=1)
    else:
        feature_block = df

    # Merge safe geometry features from manifest
    safe_geom_cols = [
        "FarmID",
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
    manifest_safe = manifest[[c for c in safe_geom_cols if c in manifest.columns]].copy()
    manifest_safe["FarmID"] = manifest_safe["FarmID"].astype(int)
    feature_block["FarmID"] = feature_block["FarmID"].astype(int)

    final_features_df = feature_block.merge(manifest_safe, on="FarmID", how="left")
    return final_features_df


def generate_pilot_report(
    features_df: pd.DataFrame,
    manifest: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Generate an audit summary of the extracted pilot dataset.
    """
    total_manifest = len(manifest)
    total_extracted = len(features_df)
    status_counts = features_df["status"].value_counts().to_dict()
    crop_counts = features_df["Crop"].value_counts().to_dict()

    # Image acquisition statistics
    img_cols = [c for c in features_df.columns if "n_images_valid" in c]
    img_stats = {}
    for c in img_cols:
        img_stats[c] = {
            "mean": round(float(features_df[c].mean()), 1),
            "min": int(features_df[c].min()),
            "max": int(features_df[c].max()),
            "zero_image_farms": int((features_df[c] == 0).sum()),
        }

    # Missingness analysis on core vegetation indices
    core_indices = [
        "ndvi_early", "ndvi_mid", "ndvi_late", "ndvi_full",
        "ndmi_early", "ndmi_mid", "ndmi_late", "ndmi_full",
        "evi_full", "ndre1_full", "msi_full",
    ]
    missing_pct = {}
    for idx_name in core_indices:
        if idx_name in features_df.columns:
            missing_pct[idx_name] = round(float(features_df[idx_name].isna().mean() * 100), 2)

    # Missingness per crop class
    crop_missing = {}
    for crop in CLASSES_7:
        c_df = features_df[features_df["Crop"] == crop]
        if not c_df.empty and "ndvi_full" in c_df.columns:
            crop_missing[crop] = {
                "n_farms": len(c_df),
                "ndvi_full_missing_pct": round(float(c_df["ndvi_full"].isna().mean() * 100), 2),
                "mid_season_missing_pct": round(float(c_df["ndvi_mid"].isna().mean() * 100), 2) if "ndvi_mid" in c_df.columns else 100.0,
            }

    # Value ranges for validation
    ranges = {}
    for c in ["ndvi_full", "evi_full", "ndwi_full", "ndmi_full"]:
        if c in features_df.columns:
            valid_vals = features_df[c].dropna()
            if not valid_vals.empty:
                ranges[c] = {
                    "min": round(float(valid_vals.min()), 4),
                    "max": round(float(valid_vals.max()), 4),
                    "mean": round(float(valid_vals.mean()), 4),
                }

    report = {
        "dataset_name": "Sentinel-2 Multi-Crop Pilot (Phase 6A)",
        "total_manifest_farms": total_manifest,
        "total_extracted_records": total_extracted,
        "status_distribution": status_counts,
        "crop_distribution": crop_counts,
        "image_acquisitions": img_stats,
        "index_missing_pct": missing_pct,
        "crop_missingness": crop_missing,
        "index_ranges": ranges,
        "suitable_for_modeling": status_counts.get("success", 0) >= 0.85 * total_manifest,
    }
    return report


def main():
    parser = argparse.ArgumentParser(description="Extract Sentinel-2 Multi-Crop Pilot Dataset.")
    parser.add_argument("--smoke-test", action="store_true", help="Extract only 1 farm per crop (7 farms total).")
    parser.add_argument("--n-per-crop", type=int, default=25, help="Number of farms to sample per crop class.")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent worker threads for GEE extraction.")
    args = parser.parse_args()

    n_per_crop = 1 if args.smoke_test else args.n_per_crop
    workers = 2 if args.smoke_test else args.workers

    logger.info("Initializing Earth Engine...")
    if not EarthEngineManager.initialize():
        logger.error("Earth Engine initialization failed. Aborting.")
        sys.exit(1)

    enriched_csv = OUTPUTS_DATA_DIR / "train_enriched_geometry.csv"
    if not enriched_csv.exists():
        logger.error("Enriched geometry file %s not found. Run geometry enrichment first.", enriched_csv)
        sys.exit(1)

    # 1. Select Manifest
    logger.info("Selecting balanced manifest: %d farms per crop across %d classes...", n_per_crop, len(CLASSES_7))
    manifest = select_pilot_manifest(enriched_csv, n_per_crop=n_per_crop, random_state=42)

    suffix = "_smoke" if args.smoke_test else "_175"
    manifest_file = OUTPUTS_DATA_DIR / f"pilot_manifest{suffix}.csv"
    raw_bands_file = OUTPUTS_DATA_DIR / f"s2_pilot{suffix}_raw_bands.csv"
    features_file = OUTPUTS_DATA_DIR / f"s2_pilot{suffix}_features.csv"
    report_file = OUTPUTS_REPORTS_DIR / f"s2_pilot{suffix}_report.json"

    manifest.to_csv(manifest_file, index=False)
    logger.info("Saved pilot manifest (%d farms) -> %s", len(manifest), manifest_file)

    # 2. Extract Raw Band Stats from Earth Engine
    logger.info("Starting batch extraction of multi-temporal Sentinel-2 imagery with %d workers...", workers)
    raw_df = extract_pilot_raw_bands(
        manifest=manifest,
        raw_bands_cache_file=raw_bands_file,
        max_workers=workers,
        checkpoint_interval=2 if args.smoke_test else 5,
    )

    # 3. Compute Spectral Indices and Temporal Dynamics
    logger.info("Computing 14 spectral vegetation/water indices across all phenological windows...")
    features_df = compute_pilot_features(raw_df=raw_df, manifest=manifest)
    features_df.to_csv(features_file, index=False)
    logger.info("Saved complete pilot feature table (%d rows, %d cols) -> %s", len(features_df), len(features_df.columns), features_file)

    # 4. Generate & Save Quality Report
    report = generate_pilot_report(features_df=features_df, manifest=manifest)
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved quality audit report -> %s", report_file)

    print("\n" + "=" * 65)
    print(f"PILOT EXTRACTION COMPLETE ({'SMOKE TEST' if args.smoke_test else 'FULL 175-FARM PILOT'})")
    print("=" * 65)
    print(f"Manifest Farms:  {report['total_manifest_farms']}")
    print(f"Extraction Status: {report['status_distribution']}")
    print("\nCrop Representation in Extracted Features:")
    for crop, count in report["crop_distribution"].items():
        print(f"  - {crop:<12}: {count} farms")
    print("\nSample Spectral Index Ranges:")
    for idx_name, r in report["index_ranges"].items():
        print(f"  - {idx_name:<12}: min={r['min']:.3f}, max={r['max']:.3f}, mean={r['mean']:.3f}")
    print("\nCore Index Missingness (%):")
    for idx_name, pct in report["index_missing_pct"].items():
        print(f"  - {idx_name:<12}: {pct:.1f}%")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
