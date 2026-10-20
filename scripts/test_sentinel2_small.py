"""
scripts/test_sentinel2_small.py
# -*- coding: utf-8 -*-
===============================
Small Sentinel-2 Extraction & Validation Test
Telangana Crop Health Challenge

Performs a controlled validation test on 15 Train farms:
1. Loads 15 actual farms with genuine geometries, SDate, and HDate.
2. Queries Sentinel-2 SR Harmonized collection with cloud cover threshold (<= 20%).
3. Evaluates cloud and shadow masking.
4. Reports image counts found before and after quality filtering for each farm.
5. Generates temporal composites across phenological windows (Early, Mid, Late, Full).
6. Extracts band reflectances (mean + stdDev) at native spatial resolutions (10m and 20m).
7. Calculates the full suite of 14 spectral vegetation and moisture indices.
8. Ensures failed farms are recorded with status='failed' and error_message, remaining retryable.
9. Exports the feature table to outputs/s2_small_test_features.csv for review.

Usage:
    python scripts/test_sentinel2_small.py
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Force UTF-8 stdout on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import numpy as np
import pandas as pd
import requests
from shapely import wkt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.indices import compute_all_spectral_indices, compute_temporal_dynamics
from src.geospatial.geometry_utils import geometry_to_ee_bbox
from src.satellite.s2_extractor import (
    compute_phenological_windows,
    extract_zonal_stats_for_farm,
    EarthEngineManager,
    S2_SR_COLLECTION,
    DEFAULT_CLOUD_COVER_THRESHOLD,
    S2_ALL_BANDS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = OUTPUTS_DIR / "satellite_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def query_stac_image_counts(
    bbox: List[float],
    sdate_str: str,
    hdate_str: str,
    max_cloud_cover: float = DEFAULT_CLOUD_COVER_THRESHOLD,
) -> Dict[str, Dict[str, int]]:
    """
    Query Microsoft Planetary Computer STAC for the exact Sentinel-2 L2A
    Harmonized collection to count images before and after quality filtering
    per phenological window.
    """
    windows = compute_phenological_windows(sdate_str, hdate_str)
    counts: Dict[str, Dict[str, int]] = {}

    stac_url = "https://planetarycomputer.microsoft.com/api/stac/v1/search"

    for w_name, (w_start, w_end) in windows.items():
        try:
            # 1. Total images in spatiotemporal bounds (before cloud filter)
            payload_all = {
                "collections": ["sentinel-2-l2a"],
                "bbox": bbox,
                "datetime": f"{w_start}/{w_end}",
                "limit": 250,
            }
            r_all = requests.post(stac_url, json=payload_all, timeout=10)
            items_all = r_all.json().get("features", [])
            n_raw = len(items_all)

            # 2. Images with cloud cover <= threshold (quality filtered)
            items_valid = [
                item for item in items_all
                if item.get("properties", {}).get("eo:cloud_cover", 100.0) <= max_cloud_cover
            ]
            n_valid = len(items_valid)

            counts[w_name] = {
                "n_found": n_raw,
                "n_valid": n_valid,
            }
        except Exception as e:
            logger.debug("STAC count query error: %s", e)
            counts[w_name] = {"n_found": 0, "n_valid": 0}

    return counts


def run_small_test(num_farms: int = 15):
    print("=" * 75)
    print("  PHASE 4 & 5 VALIDATION: SMALL SENTINEL-2 EXTRACTION TEST (15 FARMS)")
    print("=" * 75)

    # 1. Load actual train farms
    train_enriched_path = OUTPUTS_DIR / "data" / "train_enriched_geometry.csv"
    if train_enriched_path.exists():
        train_df = pd.read_csv(train_enriched_path)
    else:
        from src.data.load_data import load_train
        from src.geospatial.geometry_utils import add_geometry_features
        train_df = add_geometry_features(load_train(), repair=True)

    # Convert geometry column to shapely objects if needed
    if isinstance(train_df["geometry"].iloc[0], str):
        train_df["geometry"] = train_df["geometry"].apply(wkt.loads)

    # Pick 15 diverse farms across different crops and districts
    sample_farms = train_df.head(num_farms).copy()
    logger.info("Selected %d test farms from Train dataset.", len(sample_farms))

    # 2. Check GEE authentication and project configuration
    ee_auth = EarthEngineManager.is_authenticated()
    if ee_auth:
        logger.info("Earth Engine Status: READY (authenticated and project configured)")
    else:
        # Distinguish: do OAuth credentials exist on disk?
        from pathlib import Path as _P
        creds_file = _P.home() / ".config" / "earthengine" / "credentials"
        if creds_file.exists():
            logger.warning(
                "Earth Engine Status: INITIALIZATION FAILED — OAuth credentials exist "
                "but ee.Initialize() failed. Check that 'earth_engine.project' in "
                "config/project_config.yaml is set to a valid Google Cloud project ID."
            )
        else:
            logger.warning(
                "Earth Engine Status: NOT AUTHENTICATED — No credentials found. "
                "Run `python -c \"import ee; ee.Authenticate()\"` first."
            )

    results_list: List[Dict[str, Any]] = []
    image_counts_per_farm: Dict[int, Dict[str, Any]] = {}

    for idx, row in sample_farms.iterrows():
        farm_id = row["FarmID"]
        crop = row["Crop"]
        district = row["District"]
        geom = row["geometry"]
        sdate = str(row["SDate"])[:10]
        hdate = str(row["HDate"])[:10]
        bbox = geometry_to_ee_bbox(geom)

        print(f"\n───────────────────────────────────────────────────────────────────────────")
        print(f"  Farm #{farm_id} | Crop: {crop} | District: {district} | SDate: {sdate} -> HDate: {hdate}")
        print(f"───────────────────────────────────────────────────────────────────────────")

        # Query image availability via STAC
        stac_counts = query_stac_image_counts(bbox, sdate, hdate, max_cloud_cover=DEFAULT_CLOUD_COVER_THRESHOLD)
        image_counts_per_farm[farm_id] = {
            "crop": crop,
            "district": district,
            "sdate": sdate,
            "hdate": hdate,
            "windows": stac_counts,
        }

        for w_name, cnts in stac_counts.items():
            print(f"    Window [{w_name:5s}]: S2 scenes found: {cnts['n_found']:2d} | Valid (cloud < 20%): {cnts['n_valid']:2d}")

        # Attempt GEE extraction using s2_extractor
        if ee_auth:
            res = extract_zonal_stats_for_farm(
                farm_id=farm_id,
                geometry=geom,
                sdate_str=sdate,
                hdate_str=hdate,
                collection_name=S2_SR_COLLECTION,
                cloud_cover_threshold=DEFAULT_CLOUD_COVER_THRESHOLD,
            )
        else:
            # GEE not authenticated: strictly record status='failed' with descriptive error
            err_msg = (
                "Earth Engine is not authenticated on this system. "
                "Please run `earthengine authenticate` in your command line, or ee.Authenticate() in Python."
            )
            res = {
                "FarmID": farm_id,
                "status": "failed",
                "error_message": err_msg,
            }
            # Populate image count metadata from STAC audit
            for w_name, cnts in stac_counts.items():
                res[f"n_images_found_{w_name}"] = cnts["n_found"]
                res[f"n_images_valid_{w_name}"] = cnts["n_valid"]
                for b in S2_ALL_BANDS:
                    res[f"{b}_{w_name}_mean"] = np.nan
                    res[f"{b}_{w_name}_std"] = np.nan

        res["Crop"] = crop
        res["District"] = district
        res["SDate"] = sdate
        res["HDate"] = hdate
        results_list.append(res)

    results_df = pd.DataFrame(results_list)

    # 3. Calculate indices if bands are present (even with NaNs, ensures pipeline compatibility)
    # Check if full window bands are present
    has_full_bands = "B4_full_mean" in results_df.columns and "B8_full_mean" in results_df.columns
    if has_full_bands:
        indices_full = compute_all_spectral_indices(
            results_df,
            prefix="",
            blue_col="B2_full_mean",
            green_col="B3_full_mean",
            red_col="B4_full_mean",
            re1_col="B5_full_mean",
            re2_col="B6_full_mean",
            re3_col="B7_full_mean",
            nir_col="B8_full_mean",
            swir1_col="B11_full_mean",
            swir2_col="B12_full_mean",
        )
        # Prefix index names with 'full_'
        indices_full.columns = [f"full_{c}" for c in indices_full.columns]
        results_df = pd.concat([results_df, indices_full], axis=1)

    # 4. Save feature table to outputs/
    output_csv = OUTPUTS_DIR / "s2_small_test_features.csv"
    results_df.to_csv(output_csv, index=False)
    logger.info("Saved small test feature table -> %s (%d rows × %d cols)", output_csv, len(results_df), len(results_df.columns))

    # 5. Print summary
    n_success = (results_df["status"] == "success").sum()
    n_failed = (results_df["status"] == "failed").sum()

    print("\n" + "=" * 75)
    print("  TEST SUMMARY RESULTS")
    print("=" * 75)
    print(f"  Total test farms evaluated : {len(results_df)}")
    print(f"  Successful extractions     : {n_success}")
    print(f"  Failed extractions         : {n_failed}")
    print(f"  Output feature table saved : {output_csv}")
    print("=" * 75)

    return results_df, image_counts_per_farm, ee_auth


if __name__ == "__main__":
    run_small_test(num_farms=15)
