"""
scripts/extract_s2_expansion_phase6d.py
=======================================
Phase 6D: Sentinel-2 Dataset Expansion with Daily Mosaicking Guard.
Extracts 372 additional farms from Train.csv to reach ~103-105 valid farms per class across 7 crops (~724 total).

Safety Guarantees:
- Applies daily mosaicking before temporal median, eliminating MGRS tile duplicate weighting.
- Preserves all Phase 6C artifacts intact (outputs/data/s2_phase6c_*).
- Checkpoints to outputs/data/s2_phase6d_batch3_raw_bands.csv every 5 farms.
- Retains retryability for failed farms (status='failed' not cached as complete).
- Versioned outputs:
    * outputs/data/s2_phase6d_batch3_raw_bands.csv (372 new batch)
    * outputs/data/s2_phase6d_724_raw_bands.csv (merged 352 Phase 6C + 372 Phase 6D)
    * outputs/data/s2_phase6d_724_features.csv (clean 14 spectral indices + temporal dynamics)
    * outputs/reports/phase6d_data_quality.json (data quality summary)
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

# UTF-8 stdout configuration for Windows terminals
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


def _extract_single_farm(
    row_dict: Dict[str, Any],
    cloud_cover_threshold: float,
    daily_mosaic: bool = True,
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
            daily_mosaic=daily_mosaic,
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


def extract_batch_raw_bands_phase6d(
    manifest: pd.DataFrame,
    cache_file: Path,
    cloud_cover_threshold: float = DEFAULT_CLOUD_COVER_THRESHOLD,
    max_workers: int = 4,
    checkpoint_interval: int = 5,
    daily_mosaic: bool = True,
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
    logger.info("Farms to extract in Phase 6D batch: %d (Skipping %d cached)", len(records_to_process), len(successful_ids))

    all_records = list(cached_records.values())
    if records_to_process.empty:
        logger.info("All batch farms already successfully extracted.")
        return pd.DataFrame(all_records)

    row_dicts = records_to_process.to_dict("records")
    completed = 0

    logger.info("Launching parallel extraction with %d workers (daily_mosaic=%s)...", max_workers, daily_mosaic)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(_extract_single_farm, r, cloud_cover_threshold, daily_mosaic): r["FarmID"]
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


def compute_combined_features_phase6d(
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
                blue_col=f"B2_{w}_mean",
                green_col=f"B3_{w}_mean",
                red_col=f"B4_{w}_mean",
                nir_col=f"B8_{w}_mean",
                re1_col=f"B5_{w}_mean",
                re2_col=f"B6_{w}_mean",
                re3_col=f"B7_{w}_mean",
                swir1_col=f"B11_{w}_mean",
                swir2_col=f"B12_{w}_mean",
                scale_factor=10000.0,
            )
            w_indices.columns = [f"{c}_{w}" for c in w_indices.columns]
            indices_dfs.append(w_indices)

    if indices_dfs:
        all_indices_df = pd.concat(indices_dfs, axis=1)
        temporal_dyn = compute_temporal_dynamics(all_indices_df)
        feature_block = pd.concat([df, all_indices_df, temporal_dyn], axis=1)
    else:
        feature_block = df

    # Merge geometry attributes from manifest
    safe_geom_cols = [
        "FarmID", "area_ha", "perimeter_m", "compactness", "vertex_count",
        "centroid_lon", "centroid_lat", "bbox_min_lon", "bbox_min_lat",
        "bbox_max_lon", "bbox_max_lat", "geometry",
    ]
    available_geom_cols = [c for c in safe_geom_cols if c in combined_manifest.columns]
    if available_geom_cols:
        geom_subset = combined_manifest[available_geom_cols].drop_duplicates("FarmID").copy()
        geom_subset["FarmID"] = geom_subset["FarmID"].astype(int)
        feature_block["FarmID"] = feature_block["FarmID"].astype(int)
        feature_block = feature_block.merge(geom_subset, on="FarmID", how="left")

    return feature_block


def generate_phase6d_report(features_df: pd.DataFrame, manifest: pd.DataFrame) -> Dict[str, Any]:
    total_manifest = len(manifest)
    total_extracted = len(features_df)
    status_counts = features_df["status"].value_counts().to_dict()
    crop_counts = features_df["Crop"].value_counts().to_dict()
    clean_df = features_df[features_df["status"] == "success"]
    clean_crop_counts = clean_df["Crop"].value_counts().to_dict()

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
        "dataset_name": "Sentinel-2 Multi-Crop Expanded Dataset (Phase 6D)",
        "total_manifest_farms": total_manifest,
        "total_extracted_records": total_extracted,
        "status_distribution": status_counts,
        "total_valid_farms": len(clean_df),
        "clean_crop_distribution": clean_crop_counts,
        "raw_crop_distribution": crop_counts,
        "index_missing_pct": missing_pct,
        "index_ranges": ranges,
        "suitable_for_modeling": status_counts.get("success", 0) >= 0.85 * total_manifest,
    }
    return report


def main():
    parser = argparse.ArgumentParser(description="Phase 6D Sentinel-2 Expansion Extractor")
    parser.add_argument("--dry-run", action="store_true", help="Print manifest info without launching extraction")
    parser.add_argument("--max-farms", type=int, default=None, help="Limit extraction to first N farms")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel worker threads")
    args = parser.parse_args()

    manifest_path = OUTPUTS_DATA_DIR / "s2_phase6d_recommended_manifest_372.csv"
    if not manifest_path.exists():
        logger.error("Recommended manifest %s not found. Run scripts/generate_phase6d_reconciliation_and_manifest.py first.", manifest_path)
        sys.exit(1)

    manifest_372 = pd.read_csv(manifest_path)
    logger.info("Loaded Phase 6D manifest (%d farms). Class distribution:\n%s", len(manifest_372), manifest_372["Crop"].value_counts().to_string())

    p6c_manifest_path = OUTPUTS_DATA_DIR / "s2_phase6c_manifest_350.csv"
    p6c_raw_path = OUTPUTS_DATA_DIR / "s2_phase6c_350_raw_bands.csv"
    p6c_manifest = pd.read_csv(p6c_manifest_path)

    # 1. Verification before execution
    overlap = set(manifest_372["FarmID"]).intersection(set(p6c_manifest["FarmID"]))
    assert len(overlap) == 0, f"Error: Unexpected FarmID overlap between Phase 6C and 6D: {overlap}"
    assert manifest_372["FarmID"].nunique() == len(manifest_372), "Duplicate FarmIDs in manifest"
    logger.info("Pre-execution check passed: 372 unique FarmIDs, zero overlap with Phase 6C.")

    if args.dry_run:
        logger.info("DRY RUN requested. Extraction not started.")
        return

    if not EarthEngineManager.initialize():
        logger.error("Could not authenticate or initialize Google Earth Engine. Aborting.")
        sys.exit(1)

    batch_cache_path = OUTPUTS_DATA_DIR / "s2_phase6d_batch3_raw_bands.csv"
    target_manifest = manifest_372.head(args.max_farms) if args.max_farms else manifest_372

    # 2. Extract Phase 6D batch with daily mosaicking
    b3_raw_df = extract_batch_raw_bands_phase6d(
        manifest=target_manifest,
        cache_file=batch_cache_path,
        max_workers=args.workers,
        daily_mosaic=True,
    )

    # 3. Combine with Phase 6C raw bands (preserves Phase 6C untouched!)
    p6c_raw_df = pd.read_csv(p6c_raw_path)
    combined_raw = pd.concat([p6c_raw_df, b3_raw_df], ignore_index=True).drop_duplicates(subset=["FarmID"], keep="last")
    combined_raw_path = OUTPUTS_DATA_DIR / "s2_phase6d_724_raw_bands.csv"
    combined_raw.to_csv(combined_raw_path, index=False)
    logger.info("Saved combined raw bands (%d farms) -> %s", len(combined_raw), combined_raw_path)

    # 4. Combine manifests
    combined_manifest = pd.concat([p6c_manifest, manifest_372], ignore_index=True).drop_duplicates(subset=["FarmID"], keep="last")
    combined_manifest_path = OUTPUTS_DATA_DIR / "s2_phase6d_manifest_724.csv"
    combined_manifest.to_csv(combined_manifest_path, index=False)
    logger.info("Saved combined manifest (%d farms) -> %s", len(combined_manifest), combined_manifest_path)

    # 5. Compute combined features
    logger.info("Computing 14 spectral indices + temporal dynamics for combined dataset...")
    features_df = compute_combined_features_phase6d(raw_df=combined_raw, combined_manifest=combined_manifest)
    features_path = OUTPUTS_DATA_DIR / "s2_phase6d_724_features.csv"
    features_df.to_csv(features_path, index=False)
    logger.info("Saved combined feature table (%d rows, %d cols) -> %s", len(features_df), len(features_df.columns), features_path)

    # 6. Generate quality report
    report = generate_phase6d_report(features_df=features_df, manifest=combined_manifest)
    report_path = OUTPUTS_REPORTS_DIR / "phase6d_data_quality.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved quality report -> %s", report_path)

    print("\n" + "=" * 70)
    print("PHASE 6D DATASET EXPANSION COMPLETE")
    print("=" * 70)
    print(f"Total Extracted Farms : {report['total_extracted_records']} (Manifest: {report['total_manifest_farms']})")
    print(f"Total Valid Clean     : {report['total_valid_farms']}")
    print(f"Status Counts         : {report['status_distribution']}")
    print("\nClean Valid Support per Crop:")
    for crop, count in report["clean_crop_distribution"].items():
        print(f"  - {crop:<12}: {count} farms")
    print("\nCore Missingness (%):")
    for idx_name, pct in report["index_missing_pct"].items():
        print(f"  - {idx_name:<12}: {pct:.1f}%")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
