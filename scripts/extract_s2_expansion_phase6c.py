"""
scripts/extract_s2_expansion_phase6c.py
=======================================
Phase 6C: Controlled Sentinel-2 Dataset Expansion (Target: 50 farms/crop x 7 = 350 total).

Responsibilities:
    - Reuses the 173 successfully extracted farms from Phase 6A.
    - Samples 177 additional distinct, geographically diverse farms from Train.csv:
      * Bengal Gram: +25
      * Chillies: +26 (offsetting 1 failed in 6A)
      * Cotton: +25
      * Groundnut: +25
      * Maize: +26 (offsetting 1 failed in 6A)
      * Paddy: +25
      * Red Gram: +25
    - Saves manifests before extraction:
      * outputs/data/s2_phase6c_batch2_manifest.csv (177 new farms)
      * outputs/data/s2_phase6c_manifest_350.csv (combined 352 manifest farms)
    - Multi-threaded GEE extraction (max_workers=4) with disk checkpointing every 5 farms.
    - Merges with Phase 6A raw bands -> outputs/data/s2_phase6c_350_raw_bands.csv.
    - Computes 14 spectral indices for 4 windows + temporal dynamics + geometry.
    - Exports clean dataset -> outputs/data/s2_phase6c_350_features.csv.
    - Exports quality report -> outputs/reports/phase6c_data_quality.json.
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
from typing import Any, Dict, List

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
    S2_SR_COLLECTION,
    extract_zonal_stats_for_farm,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

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


def select_expansion_manifest(
    enriched_data_path: Path,
    phase6a_manifest_path: Path,
    phase6a_raw_path: Path,
    target_per_crop: int = 50,
    random_state: int = 142,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Select additional farms needed to reach target_per_crop per class,
    maximizing district diversity and guaranteeing zero overlap with Phase 6A.
    """
    df = pd.read_csv(enriched_data_path)
    p1_manifest = pd.read_csv(phase6a_manifest_path)
    p1_raw = pd.read_csv(phase6a_raw_path)

    p1_success = p1_raw[p1_raw["status"] == "success"]
    existing_counts = p1_success["Crop"].value_counts().to_dict()
    p1_ids = set(p1_manifest["FarmID"])

    remaining_df = df[~df["FarmID"].isin(p1_ids)].copy()

    batch2_list = []
    for c in CLASSES_7:
        n_have = existing_counts.get(c, 0)
        n_needed = max(0, target_per_crop - n_have)
        subset = remaining_df[remaining_df["Crop"] == c].copy()

        districts = subset["District"].unique()
        shuffled = subset.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        grouped = {d: shuffled[shuffled["District"] == d].copy() for d in districts}

        picks = []
        cnt = 0
        while cnt < n_needed:
            for d in districts:
                if cnt >= n_needed:
                    break
                d_df = grouped[d]
                if not d_df.empty:
                    picks.append(d_df.iloc[[0]])
                    grouped[d] = d_df.iloc[1:]
                    cnt += 1

        sampled_c = pd.concat(picks, ignore_index=True)
        batch2_list.append(sampled_c)
        logger.info("Crop %-12s: currently have %2d, sampled %2d new farms", c, n_have, len(sampled_c))

    batch2_manifest = pd.concat(batch2_list, ignore_index=True)
    batch2_manifest["FarmID"] = batch2_manifest["FarmID"].astype(int)

    # Combined manifest
    combined_manifest = pd.concat([p1_manifest, batch2_manifest], ignore_index=True)
    combined_manifest["FarmID"] = combined_manifest["FarmID"].astype(int)

    return batch2_manifest, combined_manifest


def _extract_single_farm(
    row_dict: Dict[str, Any],
    cloud_cover_threshold: float,
) -> Dict[str, Any]:
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


def extract_batch_raw_bands(
    manifest: pd.DataFrame,
    cache_file: Path,
    cloud_cover_threshold: float = DEFAULT_CLOUD_COVER_THRESHOLD,
    max_workers: int = 4,
    checkpoint_interval: int = 5,
) -> pd.DataFrame:
    cached_records: Dict[int, Dict[str, Any]] = {}
    successful_ids = set()

    if cache_file.exists():
        try:
            cached_df = pd.read_csv(cache_file)
            for rec in cached_df.to_dict("records"):
                fid = int(rec["FarmID"])
                if rec.get("status") == "success":
                    successful_ids.add(fid)
                    cached_records[fid] = rec
            logger.info("Found %d already successfully extracted farms in cache %s.", len(successful_ids), cache_file.name)
        except Exception as e:
            logger.warning("Could not read cache %s: %s", cache_file, e)

    records_to_process = manifest[~manifest["FarmID"].isin(successful_ids)]
    logger.info("Farms to extract in this batch: %d (Skipping %d cached)", len(records_to_process), len(successful_ids))

    all_records = list(cached_records.values())
    if records_to_process.empty:
        logger.info("All batch farms already successfully extracted.")
        return pd.DataFrame(all_records)

    row_dicts = records_to_process.to_dict("records")
    completed = 0

    logger.info("Launching parallel extraction with %d workers...", max_workers)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(_extract_single_farm, r, cloud_cover_threshold): r["FarmID"]
            for r in row_dicts
        }

        for future in as_completed(future_map):
            completed += 1
            fid = future_map[future]
            try:
                res = future.result()
                all_records.append(res)
                logger.info(
                    "Progress [%d/%d] FarmID %d | Crop: %-12s | Status: %s",
                    completed,
                    len(row_dicts),
                    fid,
                    res.get("Crop", ""),
                    res.get("status"),
                )
            except Exception as e:
                logger.error("Future exception for FarmID %d: %s", fid, e)
                all_records.append({"FarmID": fid, "status": "failed", "error_message": str(e)})

            if completed % checkpoint_interval == 0 or completed == len(row_dicts):
                checkpoint_df = pd.DataFrame(all_records)
                checkpoint_df.to_csv(cache_file, index=False)
                logger.info("Saved checkpoint (%d total records) -> %s", len(checkpoint_df), cache_file)

    final_df = pd.DataFrame(all_records)
    final_df.to_csv(cache_file, index=False)
    return final_df


def compute_combined_features(
    raw_df: pd.DataFrame,
    combined_manifest: pd.DataFrame,
) -> pd.DataFrame:
    df = raw_df.copy()

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
            w_indices.columns = [f"{c}_{w}" for c in w_indices.columns]
            indices_dfs.append(w_indices)

    if indices_dfs:
        all_indices_df = pd.concat(indices_dfs, axis=1)
        temporal_dyn = compute_temporal_dynamics(all_indices_df)
        feature_block = pd.concat([df, all_indices_df, temporal_dyn], axis=1)
    else:
        feature_block = df

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
    manifest_safe = combined_manifest[[c for c in safe_geom_cols if c in combined_manifest.columns]].copy()
    manifest_safe["FarmID"] = manifest_safe["FarmID"].astype(int)
    feature_block["FarmID"] = feature_block["FarmID"].astype(int)

    final_features_df = feature_block.merge(manifest_safe, on="FarmID", how="left")
    return final_features_df


def generate_expansion_report(features_df: pd.DataFrame, manifest: pd.DataFrame) -> Dict[str, Any]:
    total_manifest = len(manifest)
    total_extracted = len(features_df)
    status_counts = features_df["status"].value_counts().to_dict()
    crop_counts = features_df["Crop"].value_counts().to_dict()

    core_indices = [
        "ndvi_early", "ndvi_mid", "ndvi_late", "ndvi_full",
        "ndmi_early", "ndmi_mid", "ndmi_late", "ndmi_full",
        "evi_full", "ndre1_full", "msi_full",
    ]
    missing_pct = {
        idx: round(float(features_df[idx].isna().mean() * 100), 2)
        for idx in core_indices if idx in features_df.columns
    }

    ranges = {}
    for c in ["ndvi_full", "evi_full", "ndwi_full", "ndmi_full"]:
        if c in features_df.columns:
            v = features_df[c].dropna()
            if not v.empty:
                ranges[c] = {"min": round(float(v.min()), 4), "max": round(float(v.max()), 4), "mean": round(float(v.mean()), 4)}

    report = {
        "dataset_name": "Sentinel-2 Multi-Crop Expanded Dataset (Phase 6C)",
        "total_manifest_farms": total_manifest,
        "total_extracted_records": total_extracted,
        "status_distribution": status_counts,
        "crop_distribution": crop_counts,
        "index_missing_pct": missing_pct,
        "index_ranges": ranges,
        "suitable_for_modeling": status_counts.get("success", 0) >= 0.85 * total_manifest,
    }
    return report


def main():
    parser = argparse.ArgumentParser(description="Phase 6C Dataset Expansion.")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent GEE workers.")
    args = parser.parse_args()

    logger.info("Initializing Earth Engine...")
    if not EarthEngineManager.initialize():
        logger.error("Earth Engine initialization failed. Aborting.")
        sys.exit(1)

    enriched_csv = OUTPUTS_DATA_DIR / "train_enriched_geometry.csv"
    p1_manifest_csv = OUTPUTS_DATA_DIR / "pilot_manifest_175.csv"
    p1_raw_csv = OUTPUTS_DATA_DIR / "s2_pilot_175_raw_bands.csv"

    # 1. Manifest formulation
    b2_manifest_file = OUTPUTS_DATA_DIR / "s2_phase6c_batch2_manifest.csv"
    c_manifest_file = OUTPUTS_DATA_DIR / "s2_phase6c_manifest_350.csv"
    b2_raw_file = OUTPUTS_DATA_DIR / "s2_phase6c_batch2_raw_bands.csv"
    combined_raw_file = OUTPUTS_DATA_DIR / "s2_phase6c_350_raw_bands.csv"
    features_file = OUTPUTS_DATA_DIR / "s2_phase6c_350_features.csv"
    report_file = OUTPUTS_REPORTS_DIR / "phase6c_data_quality.json"

    logger.info("Formulating expansion manifests...")
    b2_manifest, c_manifest = select_expansion_manifest(
        enriched_data_path=enriched_csv,
        phase6a_manifest_path=p1_manifest_csv,
        phase6a_raw_path=p1_raw_csv,
        target_per_crop=50,
        random_state=142,
    )
    b2_manifest.to_csv(b2_manifest_file, index=False)
    c_manifest.to_csv(c_manifest_file, index=False)
    logger.info("Saved batch 2 manifest (%d farms) -> %s", len(b2_manifest), b2_manifest_file)
    logger.info("Saved combined manifest (%d farms) -> %s", len(c_manifest), c_manifest_file)

    # 2. Extract batch 2 from Earth Engine
    logger.info("Starting extraction for %d new farms...", len(b2_manifest))
    b2_raw_df = extract_batch_raw_bands(
        manifest=b2_manifest,
        cache_file=b2_raw_file,
        max_workers=args.workers,
        checkpoint_interval=5,
    )

    # 3. Combine with Phase 6A raw bands
    p1_raw_df = pd.read_csv(p1_raw_csv)
    # Deduplicate by FarmID (in case of any overlap)
    combined_raw = pd.concat([p1_raw_df, b2_raw_df], ignore_index=True).drop_duplicates(subset=["FarmID"], keep="last")
    combined_raw.to_csv(combined_raw_file, index=False)
    logger.info("Combined raw bands dataset: %d total farms -> %s", len(combined_raw), combined_raw_file)

    # 4. Compute spectral features
    logger.info("Computing spectral indices and temporal dynamics for combined dataset...")
    features_df = compute_combined_features(raw_df=combined_raw, combined_manifest=c_manifest)
    features_df.to_csv(features_file, index=False)
    logger.info("Saved complete expanded feature table (%d rows, %d cols) -> %s", len(features_df), len(features_df.columns), features_file)

    # 5. Quality report
    report = generate_expansion_report(features_df=features_df, manifest=c_manifest)
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved quality report -> %s", report_file)

    print("\n" + "=" * 65)
    print("PHASE 6C DATASET EXPANSION COMPLETE")
    print("=" * 65)
    print(f"Total Farms: {report['total_extracted_records']} (Manifest: {report['total_manifest_farms']})")
    print(f"Status Counts: {report['status_distribution']}")
    print("\nCrop Distribution:")
    for crop, count in report["crop_distribution"].items():
        print(f"  - {crop:<12}: {count} farms")
    print("\nCore Missingness (%):")
    for idx_name, pct in report["index_missing_pct"].items():
        print(f"  - {idx_name:<12}: {pct:.1f}%")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
