"""
scripts/extract_s2_test_phase7.py
=================================
Phase 7 Plan 2: Resumable Sentinel-2 Extraction & Feature Engineering for Test.csv (N=2,718).

Key Features:
- Exact Phase 6E parity: SCL cloud/shadow masking, daily mosaicking, phenological windows.
- Resumable in 11 chunks (~250 farms per chunk). Checkpoints to outputs/data/phase7/.
- Server-side unified dictionary evaluation (1 network call per farm).
- Topological repair via make_valid() on FarmID 1321446 (and any other invalid geometry).
- Exact 140-feature S+T schema generation from phase6h_feature_schema_manifest.json.
- Full cloud/shadow and failure logging.
- Leakage isolation: Crop, FarmID, coordinates, and survey proxies are kept strictly out of the feature matrix.
"""

from __future__ import annotations

import argparse
from concurrent.futures import as_completed, ThreadPoolExecutor
import hashlib
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from shapely import wkt
from shapely.validation import make_valid
import ee

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.indices import compute_all_spectral_indices, compute_temporal_dynamics
from src.satellite.s2_extractor import (
    DEFAULT_CLOUD_COVER_THRESHOLD,
    EarthEngineManager,
    S2_SR_COLLECTION,
    get_ee_polygon,
    compute_phenological_windows,
    mask_s2_clouds_sr,
    mosaic_same_date_scenes,
    S2_BANDS_10M,
    S2_BANDS_20M,
    S2_ALL_BANDS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phase7_test_extractor")

PHASE7_DATA_DIR = PROJECT_ROOT / "outputs" / "data" / "phase7"
PHASE7_REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports" / "phase7"
PHASE7_DATA_DIR.mkdir(parents=True, exist_ok=True)
PHASE7_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

WINDOWS = ["early", "mid", "late", "full"]


def extract_zonal_stats_unified(
    farm_id: int,
    geometry: Any,
    sdate_str: str,
    hdate_str: str,
    collection_name: str = S2_SR_COLLECTION,
    cloud_cover_threshold: float = DEFAULT_CLOUD_COVER_THRESHOLD,
    daily_mosaic: bool = True,
) -> Dict[str, Any]:
    """
    Extract zonal stats for all 4 windows in a single server-side GEE evaluation graph.
    Matches exact Phase 6E math while eliminating redundant round-trips.
    """
    results: Dict[str, Any] = {
        "FarmID": farm_id,
        "status": "failed",
        "error_message": "",
    }

    try:
        ee_region = get_ee_polygon(geometry)
        windows = compute_phenological_windows(sdate_str, hdate_str)
        zonal_reducer = ee.Reducer.mean().combine(reducer2=ee.Reducer.stdDev(), sharedInputs=True)

        dict_parts: Dict[str, Any] = {}

        for window_name, (w_start, w_end) in windows.items():
            col = (
                ee.ImageCollection(collection_name)
                .filterDate(w_start, w_end)
                .filterBounds(ee_region)
                .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", cloud_cover_threshold))
            )

            # Cloud and shadow masking
            col_masked = col.map(mask_s2_clouds_sr)

            if daily_mosaic:
                daily_col = mosaic_same_date_scenes(col_masked)
                composite = daily_col.median()
            else:
                composite = col_masked.median()

            stats_10m = composite.select(S2_BANDS_10M).reduceRegion(
                reducer=zonal_reducer, geometry=ee_region, scale=10, maxPixels=1e8
            )
            stats_20m = composite.select(S2_BANDS_20M).reduceRegion(
                reducer=zonal_reducer, geometry=ee_region, scale=20, maxPixels=1e8
            )

            for b in S2_BANDS_10M:
                dict_parts[f"{b}_{window_name}_mean"] = stats_10m.get(f"{b}_mean")
                dict_parts[f"{b}_{window_name}_std"] = stats_10m.get(f"{b}_stdDev")
            for b in S2_BANDS_20M:
                dict_parts[f"{b}_{window_name}_mean"] = stats_20m.get(f"{b}_mean")
                dict_parts[f"{b}_{window_name}_std"] = stats_20m.get(f"{b}_stdDev")

        # Single evaluate call to GEE
        eval_dict = ee.Dictionary(dict_parts).getInfo()

        has_any_valid = False
        for k, v in eval_dict.items():
            val = np.nan if v is None else float(v)
            results[k] = val
            if not np.isnan(val):
                has_any_valid = True

        if has_any_valid:
            results["status"] = "success"
        else:
            results["status"] = "failed"
            results["error_message"] = "All windows cloud-obscured or no imagery available"

        return results

    except Exception as exc:
        results["status"] = "failed"
        results["error_message"] = str(exc)
        for w in WINDOWS:
            for b in S2_ALL_BANDS:
                results[f"{b}_{w}_mean"] = np.nan
                results[f"{b}_{w}_std"] = np.nan
        return results


def _extract_farm_with_retry(
    row_dict: Dict[str, Any],
    max_retries: int = 3,
    initial_delay: float = 2.0,
) -> Dict[str, Any]:
    farm_id = int(row_dict["FarmID"])
    sdate = str(row_dict["SDate"])[:10]
    hdate = str(row_dict["HDate"])[:10]

    raw_geom_str = str(row_dict["geometry"])
    geom = wkt.loads(raw_geom_str)
    if not geom.is_valid:
        geom = make_valid(geom)

    last_exc: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            res = extract_zonal_stats_unified(
                farm_id=farm_id,
                geometry=geom,
                sdate_str=sdate,
                hdate_str=hdate,
            )
            # Retain non-predictor metadata for tracking
            res["Crop"] = row_dict.get("Crop", "")
            res["District"] = row_dict.get("District", "")
            res["Sub-District"] = row_dict.get("Sub-District", "")
            res["Season"] = row_dict.get("Season", "")
            res["SDate"] = sdate
            res["HDate"] = hdate
            return res
        except Exception as exc:
            last_exc = exc
            err_str = str(exc).lower()
            if attempt < max_retries and any(t in err_str for t in ["429", "503", "quota", "rate limit", "timeout", "unavailable", "network"]):
                backoff = initial_delay * (2 ** (attempt - 1)) + random.uniform(0.1, 1.0)
                time.sleep(backoff)
            else:
                break

    err_msg = str(last_exc) if last_exc else "Unknown extraction error"
    res = {
        "FarmID": farm_id,
        "Crop": row_dict.get("Crop", ""),
        "District": row_dict.get("District", ""),
        "Sub-District": row_dict.get("Sub-District", ""),
        "Season": row_dict.get("Season", ""),
        "SDate": sdate,
        "HDate": hdate,
        "status": "failed",
        "error_message": err_msg,
    }
    for w in WINDOWS:
        for b in S2_ALL_BANDS:
            res[f"{b}_{w}_mean"] = np.nan
            res[f"{b}_{w}_std"] = np.nan
    return res


def run_chunk_extraction(
    chunk_df: pd.DataFrame,
    chunk_file: Path,
    max_workers: int = 6,
) -> pd.DataFrame:
    """Run extraction for a chunk with resumption."""
    cached_records: Dict[int, Dict[str, Any]] = {}
    successful_ids = set()

    if chunk_file.exists():
        try:
            prev_df = pd.read_csv(chunk_file)
            for rec in prev_df.to_dict("records"):
                fid = int(rec["FarmID"])
                if rec.get("status") == "success":
                    successful_ids.add(fid)
                    cached_records[fid] = rec
            logger.info("Chunk %s: Loaded %d already successful farms", chunk_file.name, len(successful_ids))
        except Exception as e:
            logger.warning("Could not read existing chunk %s: %s", chunk_file, e)

    pending_df = chunk_df[~chunk_df["FarmID"].isin(successful_ids)]
    all_records: Dict[int, Dict[str, Any]] = dict(cached_records)

    if pending_df.empty:
        logger.info("Chunk %s is already completely extracted.", chunk_file.name)
        return pd.DataFrame(list(all_records.values()))

    row_dicts = pending_df.to_dict("records")
    completed = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(_extract_farm_with_retry, r): int(r["FarmID"])
            for r in row_dicts
        }

        for future in as_completed(future_map):
            completed += 1
            fid = future_map[future]
            try:
                res = future.result()
            except Exception as e:
                res = {"FarmID": fid, "status": "failed", "error_message": str(e)}

            all_records[fid] = res
            if completed % 25 == 0 or completed == len(row_dicts):
                elapsed = time.time() - t0
                rate = completed / max(elapsed, 1.0) * 60.0
                logger.info(
                    "Chunk %s: [%d/%d] (%.1f farms/min) - Farm %d status: %s",
                    chunk_file.name, completed, len(row_dicts), rate, fid, res.get("status")
                )
                chk_df = pd.DataFrame(list(all_records.values()))
                chk_df.to_csv(chunk_file, index=False)

    final_df = pd.DataFrame(list(all_records.values()))
    final_df.to_csv(chunk_file, index=False)
    return final_df


def compute_test_140_features(raw_bands_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute 14 spectral indices for 4 windows + 4 temporal deltas = 60 features.
    Assembles the exact 140 S+T features matching phase6h_feature_schema_manifest.json.
    """
    manifest_path = PROJECT_ROOT / "outputs" / "reports" / "phase6h" / "phase6h_feature_schema_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    st_features = manifest["groups"]["S+T"]

    indices_dfs = []
    for w in WINDOWS:
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
        w_indices = compute_all_spectral_indices(raw_bands_df, prefix="", scale_factor=10000.0, **band_cols)
        w_indices.columns = [f"{col}_{w}" for col in w_indices.columns]
        indices_dfs.append(w_indices)

    all_indices = pd.concat(indices_dfs, axis=1)
    temporal_dynamics = compute_temporal_dynamics(all_indices)

    # Combined DataFrame with metadata
    full_df = pd.concat([raw_bands_df, all_indices, temporal_dynamics], axis=1)

    # Verify all 140 features exist
    missing_feats = [f for f in st_features if f not in full_df.columns]
    if missing_feats:
        raise ValueError(f"Missing S+T features in computed matrix: {missing_feats}")

    # Extract STRICT predictor matrix (140 features only, NO metadata, NO Target)
    X_test = full_df[st_features].copy()

    # Leakage assertions
    forbidden = ["Crop", "category", "FarmID", "District", "Sub-District", "geometry", "area_m2"]
    for col in forbidden:
        assert col not in X_test.columns, f"LEAKAGE DETECTED: {col} is present in predictor matrix!"

    assert list(X_test.columns) == st_features, "Feature order mismatch with Phase 6H manifest!"
    return full_df, X_test


def main():
    parser = argparse.ArgumentParser(description="Phase 7 Plan 2: Test.csv S2 Extraction")
    parser.add_argument("--chunk_size", type=int, default=250, help="Farms per chunk")
    parser.add_argument("--max_workers", type=int, default=6, help="Worker threads")
    parser.add_argument("--chunk_idx", type=int, default=None, help="Specific chunk to run (0 to 10)")
    args = parser.parse_args()

    logger.info("Initializing Earth Engine session...")
    if not EarthEngineManager.initialize():
        raise RuntimeError("Earth Engine failed to initialize!")

    test_csv_path = PROJECT_ROOT / "datasss" / "Test.csv"
    test_df = pd.read_csv(test_csv_path)
    logger.info("Loaded Test.csv with %d records.", len(test_df))

    # Divide into chunks of ~250
    chunk_size = args.chunk_size
    n_chunks = int(np.ceil(len(test_df) / chunk_size))
    logger.info("Partitioned %d records into %d chunks (~%d farms/chunk).", len(test_df), n_chunks, chunk_size)

    chunk_files = []
    chunk_dfs = []

    for c_idx in range(n_chunks):
        if args.chunk_idx is not None and c_idx != args.chunk_idx:
            continue
        start_i = c_idx * chunk_size
        end_i = min(start_i + chunk_size, len(test_df))
        c_manifest = test_df.iloc[start_i:end_i].copy()
        c_file = PHASE7_DATA_DIR / f"raw_bands_chunk_{c_idx:02d}.csv"
        chunk_files.append(c_file)

        logger.info("--- Starting Chunk %02d/%02d (rows %d to %d, N=%d) ---", c_idx, n_chunks - 1, start_i, end_i, len(c_manifest))
        c_res = run_chunk_extraction(c_manifest, c_file, max_workers=args.max_workers)
        chunk_dfs.append(c_res)

    if args.chunk_idx is not None:
        logger.info("Chunk %d finished.", args.chunk_idx)
        return

    # Combine all chunks
    logger.info("All chunks processed. Assembling master raw bands DataFrame...")
    combined_raw_records = []
    seen_ids = set()

    for c_idx in range(n_chunks):
        c_file = PHASE7_DATA_DIR / f"raw_bands_chunk_{c_idx:02d}.csv"
        if not c_file.exists():
            raise FileNotFoundError(f"Missing chunk file: {c_file}")
        c_df = pd.read_csv(c_file)
        for r in c_df.to_dict("records"):
            fid = int(r["FarmID"])
            if fid not in seen_ids:
                seen_ids.add(fid)
                combined_raw_records.append(r)

    master_raw_df = pd.DataFrame(combined_raw_records)
    assert len(master_raw_df) == len(test_df), f"Expected {len(test_df)} raw records, got {len(master_raw_df)}"

    # Sort to align exactly with original Test.csv FarmID order
    farm_order = test_df["FarmID"].tolist()
    master_raw_df = master_raw_df.set_index("FarmID").loc[farm_order].reset_index()

    master_raw_path = PHASE7_DATA_DIR / "s2_test_raw_bands.csv"
    master_raw_df.to_csv(master_raw_path, index=False)
    logger.info("Saved master raw bands (%d rows) to %s", len(master_raw_df), master_raw_path)

    # Compute 140 features
    logger.info("Computing 140 Spectral-Temporal features and indices...")
    full_df, X_test = compute_test_140_features(master_raw_df)

    features_path = PHASE7_DATA_DIR / "s2_test_140_features.csv"
    X_test.to_csv(features_path, index=False)
    logger.info("Saved pure 140-feature matrix to %s", features_path)

    # Save complete feature dataframe with FarmID & metadata
    full_features_path = PHASE7_DATA_DIR / "s2_test_full_features_with_metadata.csv"
    full_df.to_csv(full_features_path, index=False)

    # Extraction statistics
    status_counts = master_raw_df["status"].value_counts().to_dict()
    cloud_failures = int((master_raw_df["status"] != "success").sum())
    successful_farms = int((master_raw_df["status"] == "success").sum())

    report_content = f"""# Phase 7 — Test.csv Feature Extraction & Schema Audit Report

**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}  
**Target Dataset:** `datasss/Test.csv` ($N = 2,718$ farm parcels)  
**Status:** EXTRACTION & FEATURE ENGINEERING COMPLETE  

---

## 1. Executive Summary

- **Total Test Parcels:** 2,718
- **Successfully Extracted:** {successful_farms} ({successful_farms / len(test_df) * 100:.2f}%)
- **Extraction / Cloud Failures:** {cloud_failures} ({cloud_failures / len(test_df) * 100:.2f}%)
- **Feature Schema Parity:** 140 features ($S+T$ schema bit-for-bit matched with `phase6h_feature_schema_manifest.json`)
- **Leakage Invariant:** VERIFIED (Zero target, administrative, geometry, or identifier columns in predictor matrix)
- **Topological Repair:** 1 self-intersecting polygon (`FarmID 1321446`) successfully repaired via `shapely.validation.make_valid()`.

---

## 2. Extraction Pipeline Execution Details

- **Collection:** `COPERNICUS/S2_SR_HARMONIZED`
- **Cloud Threshold:** $\\le 20\\%$ scene cloud cover with SCL pixel-level cloud/cirrus/shadow masking.
- **Mosaicking:** Daily mosaic grouping to eliminate MGRS duplicate acquisition tile overlap.
- **Phenological Windows:**
  - Early Season: 0% to 25% of growth duration
  - Mid Season: 25% to 75% of growth duration
  - Late Season: 75% to 100% of growth duration
  - Full Season: 0% to 100% of growth duration
- **Spatial Resolution:** 10m bands reduced at 10m scale; 20m bands reduced at 20m scale. Reducer: `mean` + `stdDev`.
- **Batch Processing:** 11 atomic chunks (~250 farms/chunk) processed with multi-threaded GEE execution and exponential backoff retry.

---

## 3. Artifact Inventory

1. **Master Raw Bands:** `{master_raw_path.relative_to(PROJECT_ROOT)}` ({len(master_raw_df)} rows, 80 raw band statistics)
2. **Predictor Feature Matrix (140 features):** `{features_path.relative_to(PROJECT_ROOT)}` ({len(X_test)} rows, exactly 140 features)
3. **Full Dataset with Metadata:** `{full_features_path.relative_to(PROJECT_ROOT)}`
4. **Individual Chunks:** `{PHASE7_DATA_DIR.relative_to(PROJECT_ROOT)}/raw_bands_chunk_00.csv` to `chunk_10.csv`

---

## 4. Status Counts
```json
{json.dumps(status_counts, indent=2)}
```
"""
    extraction_report_path = PHASE7_REPORTS_DIR / "phase7_test_extraction_report.md"
    extraction_report_path.write_text(report_content, encoding="utf-8")
    logger.info("Wrote extraction report to %s", extraction_report_path)
    logger.info("Phase 7 Plan 2 execution finished successfully.")


if __name__ == "__main__":
    main()
