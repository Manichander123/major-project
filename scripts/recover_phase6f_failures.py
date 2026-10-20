"""
scripts/recover_phase6f_failures.py
===================================
Phase 6F Stage 2: Bounded Failure Recovery & Schema Validation.
Executes sequential, targeted window reductions for the 5 eligible Phase 6E failures:
- 1330053 (missing: full)
- 1330055 (missing: mid, late, full)
- 1330056 (missing: mid, late, full)
- 1330059 (missing: early, mid, late, full)
- 1330230 (missing: early, mid, late, full)

Total reductions: 15 window operations.
Strictly sequential (concurrency = 1).
Max 3 attempts per farm-window with bounded exponential backoff.
Preserves existing valid partial observations.
Never modifies original Phase 6E artifacts.

Outputs:
- outputs/data/s2_phase6f_recovered_farms.csv
- outputs/reports/phase6f_stage2_recovery_ledger.csv
- outputs/reports/phase6f_stage2_recovery_report.json
- outputs/reports/phase6f_stage2_recovery_report.md
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

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

from src.satellite.s2_extractor import (
    DEFAULT_CLOUD_COVER_THRESHOLD,
    EarthEngineManager,
    S2_ALL_BANDS,
    S2_BANDS_10M,
    S2_BANDS_20M,
    S2_SR_COLLECTION,
    compute_phenological_windows,
    get_ee_polygon,
    mask_s2_clouds_sr,
    mosaic_same_date_scenes,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phase6f_recovery")

# HARD SCOPE ALLOWLIST
ALLOWED_FARM_IDS = [1330053, 1330055, 1330056, 1330059, 1330230]

# RECOVERY TARGET MATRIX
RECOVERY_MATRIX = {
    1330053: ["full"],
    1330055: ["mid", "late", "full"],
    1330056: ["mid", "late", "full"],
    1330059: ["early", "mid", "late", "full"],
    1330230: ["early", "mid", "late", "full"],
}

CANONICAL_COLUMNS = [
    "FarmID", "status", "error_message",
    "n_images_found_early", "n_images_valid_early",
    "B2_early_mean", "B2_early_std", "B3_early_mean", "B3_early_std", "B4_early_mean", "B4_early_std",
    "B8_early_mean", "B8_early_std", "B5_early_mean", "B5_early_std", "B6_early_mean", "B6_early_std",
    "B7_early_mean", "B7_early_std", "B8A_early_mean", "B8A_early_std", "B11_early_mean", "B11_early_std",
    "B12_early_mean", "B12_early_std",
    "n_images_found_mid", "n_images_valid_mid",
    "B2_mid_mean", "B2_mid_std", "B3_mid_mean", "B3_mid_std", "B4_mid_mean", "B4_mid_std",
    "B8_mid_mean", "B8_mid_std", "B5_mid_mean", "B5_mid_std", "B6_mid_mean", "B6_mid_std",
    "B7_mid_mean", "B7_mid_std", "B8A_mid_mean", "B8A_mid_std", "B11_mid_mean", "B11_mid_std",
    "B12_mid_mean", "B12_mid_std",
    "n_images_found_late", "n_images_valid_late",
    "B2_late_mean", "B2_late_std", "B3_late_mean", "B3_late_std", "B4_late_mean", "B4_late_std",
    "B8_late_mean", "B8_late_std", "B5_late_mean", "B5_late_std", "B6_late_mean", "B6_late_std",
    "B7_late_mean", "B7_late_std", "B8A_late_mean", "B8A_late_std", "B11_late_mean", "B11_late_std",
    "B12_late_mean", "B12_late_std",
    "n_images_found_full", "n_images_valid_full",
    "B2_full_mean", "B2_full_std", "B3_full_mean", "B3_full_std", "B4_full_mean", "B4_full_std",
    "B8_full_mean", "B8_full_std", "B5_full_mean", "B5_full_std", "B6_full_mean", "B6_full_std",
    "B7_full_mean", "B7_full_std", "B8A_full_mean", "B8A_full_std", "B11_full_mean", "B11_full_std",
    "B12_full_mean", "B12_full_std",
    "Crop", "District", "Sub-District", "Season", "SDate", "HDate",
]


def extract_single_window_reduction(
    farm_id: int,
    geometry: Any,
    window_name: str,
    w_start: str,
    w_end: str,
    cloud_cover_threshold: float = DEFAULT_CLOUD_COVER_THRESHOLD,
    scale_10m: int = 10,
    scale_20m: int = 20,
    daily_mosaic: bool = True,
) -> Dict[str, Any]:
    """
    Extract zonal stats for exactly one phenological window over a farm polygon.
    Strictly preserves the Phase 6E extraction pipeline.
    """
    import ee

    ee_region = get_ee_polygon(geometry)

    zonal_reducer = ee.Reducer.mean().combine(
        reducer2=ee.Reducer.stdDev(),
        sharedInputs=True,
    )

    # 1. Filter collection
    col = (
        ee.ImageCollection(S2_SR_COLLECTION)
        .filterDate(w_start, w_end)
        .filterBounds(ee_region)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", cloud_cover_threshold))
    )

    n_images_found = int(col.size().getInfo())

    results: Dict[str, Any] = {
        f"n_images_found_{window_name}": n_images_found,
        f"n_images_valid_{window_name}": 0,
        "has_valid_data": False,
        "outcome": "empty_collection" if n_images_found == 0 else "unknown",
    }

    # Initialize all band stats to NaN
    for b in S2_ALL_BANDS:
        results[f"{b}_{window_name}_mean"] = np.nan
        results[f"{b}_{window_name}_std"] = np.nan

    if n_images_found == 0:
        return results

    # 2. SCL cloud and shadow masking
    col_masked = col.map(mask_s2_clouds_sr)

    # 3. Count valid scenes containing unmasked pixels over farm
    def count_unmasked(img):
        unmasked = img.select("B2").mask().reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=ee_region,
            scale=scale_10m,
            maxPixels=1e6,
        ).get("B2")
        return img.set("unmasked_pixels", unmasked)

    col_with_unmasked = col_masked.map(count_unmasked)
    n_images_valid = int(col_with_unmasked.filter(ee.Filter.gt("unmasked_pixels", 0)).size().getInfo())
    results[f"n_images_valid_{window_name}"] = n_images_valid

    if n_images_valid == 0:
        results["outcome"] = "cloud_obscured"
        return results

    # 4. Daily mosaicking before median composite
    if daily_mosaic:
        daily_col = mosaic_same_date_scenes(col_masked)
        composite = daily_col.median()
    else:
        composite = col_masked.median()

    # 5. Extract zonal statistics at respective native resolutions
    stats_10m = composite.select(S2_BANDS_10M).reduceRegion(
        reducer=zonal_reducer,
        geometry=ee_region,
        scale=scale_10m,
        maxPixels=1e8,
    ).getInfo()

    stats_20m = composite.select(S2_BANDS_20M).reduceRegion(
        reducer=zonal_reducer,
        geometry=ee_region,
        scale=scale_20m,
        maxPixels=1e8,
    ).getInfo()

    stats_combined = {}
    if stats_10m:
        stats_combined.update(stats_10m)
    if stats_20m:
        stats_combined.update(stats_20m)

    has_valid_band = False
    for b in S2_ALL_BANDS:
        m_val = stats_combined.get(f"{b}_mean")
        s_val = stats_combined.get(f"{b}_stdDev")
        if m_val is not None:
            results[f"{b}_{window_name}_mean"] = float(m_val)
            has_valid_band = True
        if s_val is not None:
            results[f"{b}_{window_name}_std"] = float(s_val)

    results["has_valid_data"] = has_valid_band
    results["outcome"] = "success" if has_valid_band else "no_data_in_bounds"
    return results


def run_recovery():
    logger.info("Initializing Phase 6F Stage 2 Recovery...")

    # 1. Earth Engine Authentication Precondition
    if not EarthEngineManager.initialize():
        logger.error("FATAL: Earth Engine initialization failed. Aborting recovery.")
        sys.exit(1)

    # 2. Load Manifest & Original Extraction Data
    manifest_path = PROJECT_ROOT / "outputs" / "data" / "s2_full_expansion_manifest_7164.csv"
    orig_path = PROJECT_ROOT / "outputs" / "data" / "s2_phase6e_7164_raw_bands.csv"

    df_manifest = pd.read_csv(manifest_path)
    df_orig = pd.read_csv(orig_path)

    # 3. Target Allowlist Validation
    manifest_sub = df_manifest[df_manifest["FarmID"].isin(ALLOWED_FARM_IDS)].copy()
    orig_sub = df_orig[df_orig["FarmID"].isin(ALLOWED_FARM_IDS)].copy()

    assert len(manifest_sub) == len(ALLOWED_FARM_IDS), "Allowlist FarmIDs missing from manifest!"
    assert len(orig_sub) == len(ALLOWED_FARM_IDS), "Allowlist FarmIDs missing from original extraction!"

    ledger_records: List[Dict[str, Any]] = []
    recovered_farm_rows: List[Dict[str, Any]] = []

    total_planned_reductions = sum(len(wins) for wins in RECOVERY_MATRIX.values())
    logger.info(f"Target farms: {len(ALLOWED_FARM_IDS)} | Total planned reductions: {total_planned_reductions}")
    assert total_planned_reductions == 15, f"Expected 15 reductions, got {total_planned_reductions}"

    # 4. Sequential Recovery Loop (concurrency = 1)
    for farm_id in ALLOWED_FARM_IDS:
        m_row = manifest_sub[manifest_sub["FarmID"] == farm_id].iloc[0]
        o_row = orig_sub[orig_sub["FarmID"] == farm_id].iloc[0]

        crop = m_row["Crop"]
        district = m_row["District"]
        sub_district = m_row["Sub-District"]
        season = m_row["Season"]
        sdate = m_row["SDate"]
        hdate = m_row["HDate"]
        geom_wkt = m_row["geometry"]
        geometry = wkt.loads(geom_wkt)

        windows_dict = compute_phenological_windows(sdate, hdate)
        needed_windows = RECOVERY_MATRIX[farm_id]

        logger.info(f"--- Processing FarmID {farm_id} ({crop}, {sub_district}) ---")
        logger.info(f"Windows to recover: {needed_windows}")

        # Start with a copy of the original row data to preserve valid partial features
        merged_row = o_row.to_dict()

        for win_name in ["early", "mid", "late", "full"]:
            if win_name not in needed_windows:
                # Window was already valid in original data; record in ledger as preserved
                ledger_records.append({
                    "FarmID": farm_id,
                    "Crop": crop,
                    "District": district,
                    "Sub-District": sub_district,
                    "Season": season,
                    "window": win_name,
                    "was_previously_valid": True,
                    "requested": False,
                    "attempts": 0,
                    "outcome": "preserved_existing",
                    "n_images_valid": merged_row.get(f"n_images_valid_{win_name}"),
                    "error_message": "",
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                })
                continue

            # This window needs to be recovered via GEE
            w_start, w_end = windows_dict[win_name]
            attempts = 0
            max_attempts = 3
            initial_delay = 2.0
            last_err = ""
            win_result = None

            while attempts < max_attempts:
                attempts += 1
                logger.info(f"FarmID {farm_id} | Window: {win_name} | Attempt {attempts}/{max_attempts}...")
                try:
                    win_result = extract_single_window_reduction(
                        farm_id=farm_id,
                        geometry=geometry,
                        window_name=win_name,
                        w_start=w_start,
                        w_end=w_end,
                        cloud_cover_threshold=DEFAULT_CLOUD_COVER_THRESHOLD,
                        scale_10m=10,
                        scale_20m=20,
                        daily_mosaic=True,
                    )
                    logger.info(f"FarmID {farm_id} | Window: {win_name} | Outcome: {win_result['outcome']} (valid images: {win_result.get(f'n_images_valid_{win_name}', 0)})")
                    break
                except Exception as exc:
                    last_err = str(exc)
                    logger.warning(f"FarmID {farm_id} | Window: {win_name} | Attempt {attempts} failed: {last_err}")
                    # Check for fatal auth/permission errors
                    if any(s in last_err.lower() for s in ["auth", "permission", "credential"]):
                        logger.error(f"FATAL authentication error encountered: {last_err}. Aborting.")
                        sys.exit(1)
                    if attempts < max_attempts:
                        delay = min(10.0, initial_delay * (2 ** (attempts - 1)) + random.uniform(0.1, 1.0))
                        logger.info(f"Backing off for {delay:.2f}s before retry...")
                        time.sleep(delay)

            if win_result is not None:
                outcome = win_result.get("outcome", "unknown")
                # Merge recovered fields into the merged_row
                for k, v in win_result.items():
                    if k not in ["has_valid_data", "outcome"]:
                        merged_row[k] = v

                ledger_records.append({
                    "FarmID": farm_id,
                    "Crop": crop,
                    "District": district,
                    "Sub-District": sub_district,
                    "Season": season,
                    "window": win_name,
                    "was_previously_valid": False,
                    "requested": True,
                    "attempts": attempts,
                    "outcome": outcome,
                    "n_images_valid": win_result.get(f"n_images_valid_{win_name}", 0),
                    "error_message": "",
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                })
            else:
                outcome = "failed_timeout" if "timeout" in last_err.lower() else "failed_error"
                ledger_records.append({
                    "FarmID": farm_id,
                    "Crop": crop,
                    "District": district,
                    "Sub-District": sub_district,
                    "Season": season,
                    "window": win_name,
                    "was_previously_valid": False,
                    "requested": True,
                    "attempts": attempts,
                    "outcome": outcome,
                    "n_images_valid": 0,
                    "error_message": last_err,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                })

        # Assess farm-level final status
        # Count how many windows have valid data
        windows_with_data = 0
        for w in ["early", "mid", "late", "full"]:
            if pd.notna(merged_row.get(f"B8_{w}_mean")):
                windows_with_data += 1

        if windows_with_data == 4:
            merged_row["status"] = "success"
            merged_row["error_message"] = ""
        elif windows_with_data > 0:
            merged_row["status"] = "partial_success"
            merged_row["error_message"] = f"Partial recovery: {windows_with_data}/4 windows valid"
        else:
            merged_row["status"] = "failed"
            merged_row["error_message"] = "Recovery failed across all requested windows"

        # Ensure canonical metadata
        merged_row["FarmID"] = farm_id
        merged_row["Crop"] = crop
        merged_row["District"] = district
        merged_row["Sub-District"] = sub_district
        merged_row["Season"] = season
        merged_row["SDate"] = sdate
        merged_row["HDate"] = hdate

        # Keep strictly canonical columns
        clean_row = {col: merged_row.get(col, np.nan) for col in CANONICAL_COLUMNS}
        recovered_farm_rows.append(clean_row)

    # 5. Output Artifacts Generation
    df_recovered = pd.DataFrame(recovered_farm_rows, columns=CANONICAL_COLUMNS)
    df_ledger = pd.DataFrame(ledger_records)

    out_data_dir = PROJECT_ROOT / "outputs" / "data"
    out_rep_dir = PROJECT_ROOT / "outputs" / "reports"
    out_data_dir.mkdir(parents=True, exist_ok=True)
    out_rep_dir.mkdir(parents=True, exist_ok=True)

    rec_csv_path = out_data_dir / "s2_phase6f_recovered_farms.csv"
    ledger_csv_path = out_rep_dir / "phase6f_stage2_recovery_ledger.csv"

    # Atomic write for recovered CSV
    tmp_rec = rec_csv_path.with_suffix(".csv.tmp")
    df_recovered.to_csv(tmp_rec, index=False)
    os.replace(tmp_rec, rec_csv_path)
    logger.info(f"Saved recovered farms to {rec_csv_path}")

    # Atomic write for ledger CSV
    tmp_led = ledger_csv_path.with_suffix(".csv.tmp")
    df_ledger.to_csv(tmp_led, index=False)
    os.replace(tmp_led, ledger_csv_path)
    logger.info(f"Saved recovery ledger to {ledger_csv_path}")

    # 6. JSON & Markdown Reports
    fully_recovered = int((df_recovered["status"] == "success").sum())
    partially_recovered = int((df_recovered["status"] == "partial_success").sum())
    still_failed = int((df_recovered["status"] == "failed").sum())
    requested_ops = df_ledger[df_ledger["requested"] == True]
    success_ops = int((requested_ops["outcome"] == "success").sum())

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "total_farms_attempted": len(ALLOWED_FARM_IDS),
        "fully_recovered_farms": fully_recovered,
        "partially_recovered_farms": partially_recovered,
        "still_failed_farms": still_failed,
        "total_window_reductions_requested": len(requested_ops),
        "successful_window_reductions": success_ops,
        "farm_status_summary": df_recovered[["FarmID", "Crop", "Sub-District", "status", "error_message"]].to_dict(orient="records"),
        "artifacts_created": [
            str(rec_csv_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            str(ledger_csv_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        ],
    }

    rep_json_path = out_rep_dir / "phase6f_stage2_recovery_report.json"
    with open(rep_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    rep_md_path = out_rep_dir / "phase6f_stage2_recovery_report.md"
    with open(rep_md_path, "w", encoding="utf-8") as f:
        f.write("# Phase 6F Stage 2: Failure Recovery & Schema Validation Report\n\n")
        f.write(f"**Execution Date:** {summary['timestamp_utc']}  \n")
        f.write(f"**Target Farms Attempted:** {summary['total_farms_attempted']}  \n")
        f.write(f"**Fully Recovered:** {summary['fully_recovered_farms']}  \n")
        f.write(f"**Partially Recovered:** {summary['partially_recovered_farms']}  \n")
        f.write(f"**Still Failed:** {summary['still_failed_farms']}  \n\n")
        f.write("---\n\n")
        f.write("## 1. Recovered Farm Summary\n\n")
        f.write("| FarmID | Crop | Sub-District | Final Status | Note |\n")
        f.write("| :---: | :--- | :--- | :---: | :--- |\n")
        for rec in summary["farm_status_summary"]:
            f.write(f"| {rec['FarmID']} | {rec['Crop']} | {rec['Sub-District']} | **`{rec['status']}`** | {rec['error_message']} |\n")
        f.write("\n")
        f.write("## 2. Operation Ledger\n\n")
        f.write("| FarmID | Window | Requested | Attempts | Outcome | Valid Scenes |\n")
        f.write("| :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for _, r in df_ledger.iterrows():
            f.write(f"| {r['FarmID']} | {r['window']} | {r['requested']} | {r['attempts']} | `{r['outcome']}` | {r['n_images_valid']} |\n")
        f.write("\n")

    logger.info("Stage 2 recovery script complete.")
    return summary


if __name__ == "__main__":
    run_recovery()
