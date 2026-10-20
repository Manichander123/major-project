"""
scripts/run_phase6d_pilot.py
============================
Phase 6D Pilot Verification:
Compares Sentinel-2 raw temporal median (Phase 6C) vs daily-mosaicked compositing (Phase 6D)
across 7 representative farms spanning all 7 target crops, Kharif & Rabi seasons,
and varying tile overlap conditions (single tile vs 2-tile vs 3-4 tile overlaps).
Saves results to outputs/data/phase6d_pilot_comparison.csv.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from shapely import wkt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.satellite.s2_extractor import (
    EarthEngineManager,
    extract_zonal_stats_for_farm,
    compute_phenological_windows,
    get_ee_polygon,
    mask_s2_clouds_sr,
    S2_BANDS_10M,
    S2_BANDS_20M,
)
from src.features.indices import calc_ndvi, calc_ndmi

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

PILOT_FARMS = [
    {"farm_id": 1327181, "crop": "Bengal Gram", "season": "Rabi", "tile_condition": "Heavy 3-4 Tile Overlap"},
    {"farm_id": 1331695, "crop": "Red Gram", "season": "Kharif", "tile_condition": "Moderate 2-Tile Overlap"},
    {"farm_id": 1327617, "crop": "Maize", "season": "Rabi", "tile_condition": "Heavy 3-4 Tile Overlap"},
    {"farm_id": 1324368, "crop": "Paddy", "season": "Rabi", "tile_condition": "Moderate 2-Tile Overlap"},
    {"farm_id": 1307790, "crop": "Cotton", "season": "Kharif", "tile_condition": "Kharif Overlap"},
    {"farm_id": 1322087, "crop": "Groundnut", "season": "Rabi", "tile_condition": "Standard Overlap"},
    {"farm_id": 1330504, "crop": "Chillies", "season": "Rabi", "tile_condition": "Standard Overlap"},
]


def run_pilot():
    if not EarthEngineManager.initialize():
        logger.error("Earth Engine initialization failed.")
        sys.exit(1)

    import ee

    train_enriched = pd.read_csv("outputs/data/train_enriched_geometry.csv")
    results = []

    logger.info("Starting Phase 6D Pilot across %d representative farms...", len(PILOT_FARMS))

    for item in PILOT_FARMS:
        fid = item["farm_id"]
        row = train_enriched[train_enriched["FarmID"] == fid].iloc[0]

        geom = wkt.loads(str(row["geometry"]))
        sdate = str(row["SDate"])[:10]
        hdate = str(row["HDate"])[:10]
        crop = row["Crop"]
        season = row["Season"]

        logger.info("Extracting Farm %d (%s, %s)...", fid, crop, season)

        # 1. Extract with raw median (daily_mosaic=False)
        raw_res = extract_zonal_stats_for_farm(
            farm_id=fid,
            geometry=geom,
            sdate_str=sdate,
            hdate_str=hdate,
            daily_mosaic=False,
        )

        # 2. Extract with daily mosaic (daily_mosaic=True)
        daily_res = extract_zonal_stats_for_farm(
            farm_id=fid,
            geometry=geom,
            sdate_str=sdate,
            hdate_str=hdate,
            daily_mosaic=True,
        )

        # Calculate tile overlap ratio in full window
        windows = compute_phenological_windows(sdate, hdate)
        w_start, w_end = windows["full"]
        ee_geom = get_ee_polygon(geom)
        col = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterDate(w_start, w_end)
            .filterBounds(ee_geom)
            .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", 20.0))
            .map(mask_s2_clouds_sr)
        )
        n_scenes = int(col.size().getInfo())
        dates = ee.List(
            col.map(lambda img: ee.Feature(None, {"date": img.date().format("yyyy-MM-dd")})).aggregate_array("date")
        ).distinct()
        n_dates = int(dates.size().getInfo())
        overlap_ratio = round(n_scenes / n_dates, 2) if n_dates > 0 else 1.0

        # Compare full season indices
        b4_raw = raw_res.get("B4_full_mean", np.nan)
        b8_raw = raw_res.get("B8_full_mean", np.nan)
        b11_raw = raw_res.get("B11_full_mean", np.nan)

        b4_daily = daily_res.get("B4_full_mean", np.nan)
        b8_daily = daily_res.get("B8_full_mean", np.nan)
        b11_daily = daily_res.get("B11_full_mean", np.nan)

        ndvi_raw = float(calc_ndvi(np.array([b8_raw]), np.array([b4_raw]))[0])
        ndvi_daily = float(calc_ndvi(np.array([b8_daily]), np.array([b4_daily]))[0])
        ndvi_delta = round(ndvi_daily - ndvi_raw, 5)

        ndmi_raw = float(calc_ndmi(np.array([b8_raw]), np.array([b11_raw]))[0])
        ndmi_daily = float(calc_ndmi(np.array([b8_daily]), np.array([b11_daily]))[0])
        ndmi_delta = round(ndmi_daily - ndmi_raw, 5)

        rec = {
            "FarmID": fid,
            "Crop": crop,
            "Season": season,
            "Tile_Condition": item["tile_condition"],
            "Total_Scenes_Full": n_scenes,
            "Unique_Dates_Full": n_dates,
            "Overlap_Ratio": overlap_ratio,
            "B4_raw": round(b4_raw, 2),
            "B4_daily": round(b4_daily, 2),
            "B4_delta": round(b4_daily - b4_raw, 2),
            "B8_raw": round(b8_raw, 2),
            "B8_daily": round(b8_daily, 2),
            "B8_delta": round(b8_daily - b8_raw, 2),
            "B11_raw": round(b11_raw, 2),
            "B11_daily": round(b11_daily, 2),
            "B11_delta": round(b11_daily - b11_raw, 2),
            "NDVI_raw": round(ndvi_raw, 5),
            "NDVI_daily": round(ndvi_daily, 5),
            "NDVI_delta": ndvi_delta,
            "NDMI_raw": round(ndmi_raw, 5),
            "NDMI_daily": round(ndmi_daily, 5),
            "NDMI_delta": ndmi_delta,
            "Status": "PASSED" if not np.isnan(ndvi_daily) else "FAILED",
        }
        results.append(rec)
        logger.info(
            "Farm %d | Overlap: %.2fx | NDVI Raw: %.4f -> Daily: %.4f (Delta: %+.4f) | NDMI Delta: %+.4f",
            fid, overlap_ratio, ndvi_raw, ndvi_daily, ndvi_delta, ndmi_delta
        )

    res_df = pd.DataFrame(results)
    out_csv = PROJECT_ROOT / "outputs" / "data" / "phase6d_pilot_comparison.csv"
    res_df.to_csv(out_csv, index=False)
    logger.info("Saved pilot comparison table to %s", out_csv)
    print("\n" + "=" * 80)
    print("PHASE 6D PILOT VERIFICATION RESULTS")
    print("=" * 80)
    print(res_df[["FarmID", "Crop", "Season", "Overlap_Ratio", "NDVI_raw", "NDVI_daily", "NDVI_delta", "NDMI_delta", "Status"]].to_string(index=False))


if __name__ == "__main__":
    run_pilot()
