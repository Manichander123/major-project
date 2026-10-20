"""
scripts/audit_mgrs_multi_farms.py
==================================
Multi-Farm Sensitivity Audit for Sentinel-2 MGRS Tile Overlaps:
Evaluates whether raw temporal median compositing distorts reflectance
and spectral indices compared to daily-mosaicked compositing across
multiple representative farms spanning different crops and cultivation windows.
"""

import json
import logging
import sys
from pathlib import Path

import ee
import pandas as pd
from shapely import wkt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.satellite.s2_extractor import (
    EarthEngineManager,
    S2_BANDS_10M,
    S2_BANDS_20M,
    compute_phenological_windows,
    get_ee_polygon,
    mask_s2_clouds_sr,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

# Representative multi-tile farms spanning different crops, seasons, and durations
AUDIT_FARMS = [
    {"farm_id": 1327181, "crop": "Bengal Gram", "season": "Rabi", "desc": "Extreme 3-tile overlap"},
    {"farm_id": 1331695, "crop": "Red Gram", "season": "Kharif", "desc": "Long duration Kharif monsoon"},
    {"farm_id": 1327617, "crop": "Maize", "season": "Rabi", "desc": "Moderate dual-tile overlap"},
    {"farm_id": 1324368, "crop": "Paddy", "season": "Rabi", "desc": "Dual-tile wet crop overlap"},
]


def audit_farm_mgrs(farm_meta: dict, df_feat: pd.DataFrame) -> dict:
    farm_id = farm_meta["farm_id"]
    row = df_feat[df_feat["FarmID"] == farm_id].iloc[0]

    geom_wkt = row["geometry"]
    geom = wkt.loads(geom_wkt)
    ee_region = get_ee_polygon(geom)

    sdate = row["SDate"]
    hdate = row["HDate"]
    windows = compute_phenological_windows(sdate, hdate)
    w_start, w_end = windows["full"]

    col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(w_start, w_end)
        .filterBounds(ee_region)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", 20.0))
        .map(mask_s2_clouds_sr)
    )

    n_total = int(col.size().getInfo())
    if n_total == 0:
        return {"farm_id": farm_id, "status": "no_images"}

    # Method A: Current Raw Median
    comp_raw = col.median()
    stats_raw_10m = comp_raw.select(S2_BANDS_10M).reduceRegion(ee.Reducer.mean(), ee_region, 10).getInfo()
    stats_raw_20m = comp_raw.select(S2_BANDS_20M).reduceRegion(ee.Reducer.mean(), ee_region, 20).getInfo()
    raw_bands = {**stats_raw_10m, **stats_raw_20m}

    # Method B: Daily Mosaicking
    dates = ee.List(
        col.map(lambda img: ee.Feature(None, {"date": img.date().format("YYYY-MM-dd")})).aggregate_array("date")
    ).distinct()
    n_unique_dates = int(dates.size().getInfo())

    def make_daily(d):
        t0 = ee.Date(d)
        t1 = t0.advance(1, "day")
        return col.filterDate(t0, t1).mosaic().set("system:time_start", t0.millis())

    daily_col = ee.ImageCollection(dates.map(make_daily))
    comp_daily = daily_col.median()
    stats_daily_10m = comp_daily.select(S2_BANDS_10M).reduceRegion(ee.Reducer.mean(), ee_region, 10).getInfo()
    stats_daily_20m = comp_daily.select(S2_BANDS_20M).reduceRegion(ee.Reducer.mean(), ee_region, 20).getInfo()
    daily_bands = {**stats_daily_10m, **stats_daily_20m}

    # Band comparisons
    band_diffs = {}
    for b in ["B2", "B3", "B4", "B8", "B11", "B12", "B5", "B6", "B7", "B8A"]:
        r = raw_bands.get(b)
        d = daily_bands.get(b)
        if r is not None and d is not None:
            diff = d - r
            pct = (diff / r) * 100 if r != 0 else 0.0
            band_diffs[b] = {
                "raw": round(r, 2),
                "daily": round(d, 2),
                "diff": round(diff, 2),
                "pct_diff": round(pct, 3),
            }

    # Key Index Comparisons
    def calc_indices(b):
        b8, b4, b11 = b["B8"], b["B4"], b["B11"]
        ndvi = (b8 - b4) / (b8 + b4) if (b8 + b4) != 0 else 0
        ndmi = (b8 - b11) / (b8 + b11) if (b8 + b11) != 0 else 0
        return {"ndvi": ndvi, "ndmi": ndmi}

    idx_raw = calc_indices(raw_bands)
    idx_daily = calc_indices(daily_bands)

    return {
        "farm_id": farm_id,
        "crop": farm_meta["crop"],
        "season": farm_meta["season"],
        "desc": farm_meta["desc"],
        "window": [w_start, w_end],
        "total_tile_scenes": n_total,
        "unique_acquisition_dates": n_unique_dates,
        "tile_overlap_ratio": round(n_total / n_unique_dates, 2) if n_unique_dates > 0 else 1.0,
        "band_differences": band_diffs,
        "indices": {
            "ndvi_raw": round(idx_raw["ndvi"], 5),
            "ndvi_daily": round(idx_daily["ndvi"], 5),
            "ndvi_delta": round(idx_daily["ndvi"] - idx_raw["ndvi"], 5),
            "ndmi_raw": round(idx_raw["ndmi"], 5),
            "ndmi_daily": round(idx_daily["ndmi"], 5),
            "ndmi_delta": round(idx_daily["ndmi"] - idx_raw["ndmi"], 5),
        },
    }


def main():
    EarthEngineManager.initialize()
    df_feat = pd.read_csv("outputs/data/s2_phase6c_350_features.csv")
    geom_df = pd.read_csv("outputs/data/train_enriched_geometry.csv")
    df_merged = df_feat.merge(geom_df[["FarmID", "geometry"]], on="FarmID", how="left")

    results = []
    for f in AUDIT_FARMS:
        logger.info("Auditing Farm %d (%s, %s)...", f["farm_id"], f["crop"], f["season"])
        res = audit_farm_mgrs(f, df_merged)
        results.append(res)
        logger.info(
            "Farm %d: Tile Scenes=%d, Dates=%d, Overlap Ratio=%.2fx | NDVI Delta=%+.5f, NDMI Delta=%+.5f",
            res["farm_id"],
            res["total_tile_scenes"],
            res["unique_acquisition_dates"],
            res["tile_overlap_ratio"],
            res["indices"]["ndvi_delta"],
            res["indices"]["ndmi_delta"],
        )

    out_path = Path("outputs/reports/mgrs_multi_farm_audit.json")
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)
    logger.info("Saved multi-farm MGRS audit report -> %s", out_path)


if __name__ == "__main__":
    main()
