"""
scripts/audit_mgrs_mosaic.py
============================
Audit MGRS overlapping tiles and quantify impact of daily mosaicking
vs raw temporal median on FarmID 1326576.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import ee
from shapely import wkt
from src.satellite.s2_extractor import (
    EarthEngineManager,
    get_ee_polygon,
    mask_s2_clouds_sr,
    compute_phenological_windows,
    S2_BANDS_10M,
    S2_BANDS_20M,
)

EarthEngineManager.initialize()

# Farm 1326576 known to have 3 overlapping MGRS tiles (43QHA, 43QHV, 44QKE)
farm_id = 1326576
wkt_str = (
    "POLYGON ((78.18143246076087 17.978884603571068, "
    "78.18149324901445 17.97898640558611, "
    "78.18175397935514 17.978865561467536, "
    "78.18166316316905 17.978728604799823, "
    "78.18143246076087 17.978884603571068))"
)
geom = wkt.loads(wkt_str)
ee_region = get_ee_polygon(geom)

sdate = "2023-11-25"
hdate = "2024-04-14"
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

# Method A: Current raw median across all tile scenes
comp_raw = col.median()
stats_raw_10m = comp_raw.select(S2_BANDS_10M).reduceRegion(ee.Reducer.mean(), ee_region, 10).getInfo()
stats_raw_20m = comp_raw.select(S2_BANDS_20M).reduceRegion(ee.Reducer.mean(), ee_region, 20).getInfo()

# Method B: Daily mosaicking before temporal median
# Note: In EE Joda-Time, lowercase 'dd' is day-of-month (uppercase 'DD' is day-of-year)
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

print(f"Farm {farm_id} in Full Window ({w_start} to {w_end}):")
print(f"Total S2 tile records: {n_total} | Unique acquisition dates: {n_unique_dates}")

raw_all = {**stats_raw_10m, **stats_raw_20m}
daily_all = {**stats_daily_10m, **stats_daily_20m}

print("\nComparison of Band Reflectance (DN):")
print(f"{'Band':<6} | {'Raw Median':<12} | {'Daily Mosaic':<12} | {'Diff (Daily - Raw)':<18} | {'% Diff':<8}")
print("-" * 65)
for b in ["B2", "B3", "B4", "B8", "B5", "B6", "B7", "B8A", "B11", "B12"]:
    r = raw_all.get(b, float("nan"))
    d = daily_all.get(b, float("nan"))
    diff = d - r
    pct = (diff / r) * 100 if r != 0 else 0
    print(f"{b:<6} | {r:<12.3f} | {d:<12.3f} | {diff:<18.4f} | {pct:<8.3f}%")

# Compare NDVI & NDMI
ndvi_raw = (raw_all["B8"] - raw_all["B4"]) / (raw_all["B8"] + raw_all["B4"])
ndvi_daily = (daily_all["B8"] - daily_all["B4"]) / (daily_all["B8"] + daily_all["B4"])
ndmi_raw = (raw_all["B8"] - raw_all["B11"]) / (raw_all["B8"] + raw_all["B11"])
ndmi_daily = (daily_all["B8"] - daily_all["B11"]) / (daily_all["B8"] + daily_all["B11"])

print("\nIndex Comparison:")
print(f"NDVI Raw: {ndvi_raw:.6f} | NDVI Daily: {ndvi_daily:.6f} | Delta: {ndvi_daily - ndvi_raw:+.6f}")
print(f"NDMI Raw: {ndmi_raw:.6f} | NDMI Daily: {ndmi_daily:.6f} | Delta: {ndmi_daily - ndmi_raw:+.6f}")
