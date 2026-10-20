"""
scripts/diagnose_s2_inventory.py
================================
Sentinel-2 Image Inventory Diagnostic for FarmID 1326576

Purpose:
    Audit exactly what the GEE ImageCollection returns for a single farm
    across all 4 phenological windows. Determines whether high image counts
    are caused by:
      - Both Sentinel-2A + Sentinel-2B satellites
      - Multiple overlapping MGRS tiles
      - Duplicate image records
      - Genuinely different acquisition dates

    Does NOT modify any extraction, cloud-masking, or feature logic.

Usage:
    python scripts/diagnose_s2_inventory.py
"""

from __future__ import annotations

import logging
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.satellite.s2_extractor import (
    EarthEngineManager,
    S2_SR_COLLECTION,
    DEFAULT_CLOUD_COVER_THRESHOLD,
    compute_phenological_windows,
    get_ee_polygon,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Target farm ──────────────────────────────────────────────────────────────
FARM_ID = 1326576
FARM_WKT = (
    "POLYGON ((78.18143246076087 17.978884603571068, "
    "78.18149324901445 17.97898640558611, "
    "78.18175397935514 17.978865561467536, "
    "78.18166316316905 17.978728604799823, "
    "78.18143246076087 17.978884603571068))"
)
FARM_SDATE = "2023-11-25"
FARM_HDATE = "2024-04-14"


def collect_image_metadata(ee_region, start_date: str, end_date: str, window_name: str):
    """
    Query COPERNICUS/S2_SR_HARMONIZED for the given region and date range
    WITHOUT applying any cloud-cover filter at the collection level
    (so we can see ALL records and then report which pass the filter).

    Returns a list of dicts with per-image metadata.
    """
    import ee

    # Query WITHOUT cloud filter first — to see everything GEE returns
    col_all = (
        ee.ImageCollection(S2_SR_COLLECTION)
        .filterDate(start_date, end_date)
        .filterBounds(ee_region)
    )

    n_total = int(col_all.size().getInfo())
    logger.info(
        "Window [%s] %s → %s: total GEE records (no cloud filter): %d",
        window_name, start_date, end_date, n_total,
    )

    if n_total == 0:
        return []

    # Extract metadata properties for every image
    # We use aggregate_array to batch-fetch rather than getInfo per image
    metadata_props = [
        "system:index",
        "system:time_start",
        "PRODUCT_ID",
        "SPACECRAFT_NAME",
        "MGRS_TILE",
        "CLOUDY_PIXEL_PERCENTAGE",
    ]

    prop_arrays = {}
    for prop in metadata_props:
        prop_arrays[prop] = col_all.aggregate_array(prop).getInfo()

    records = []
    for i in range(n_total):
        time_ms = prop_arrays["system:time_start"][i]
        acq_dt = datetime.utcfromtimestamp(time_ms / 1000.0)
        acq_date = acq_dt.strftime("%Y-%m-%d")
        acq_datetime = acq_dt.strftime("%Y-%m-%d %H:%M:%S")
        cloud_pct = prop_arrays["CLOUDY_PIXEL_PERCENTAGE"][i]

        records.append({
            "FarmID": FARM_ID,
            "Window": window_name,
            "Window_Start": start_date,
            "Window_End": end_date,
            "Image_ID": prop_arrays["system:index"][i],
            "Acquisition_Date": acq_date,
            "Acquisition_DateTime": acq_datetime,
            "Product_ID": prop_arrays["PRODUCT_ID"][i],
            "Spacecraft": prop_arrays["SPACECRAFT_NAME"][i],
            "MGRS_Tile": prop_arrays["MGRS_TILE"][i],
            "Cloudy_Pixel_Percentage": round(cloud_pct, 2) if cloud_pct is not None else None,
            "Passes_Cloud_Filter": (cloud_pct is not None and cloud_pct <= DEFAULT_CLOUD_COVER_THRESHOLD),
        })

    return records


def print_window_report(records: list, window_name: str, start_date: str, end_date: str):
    """Print a detailed diagnostic summary for one temporal window."""
    if not records:
        print(f"\n  WINDOW: {window_name.upper()}")
        print(f"  Date range: {start_date} → {end_date}")
        print(f"  Total GEE records: 0")
        return

    df = pd.DataFrame(records)

    n_total = len(df)
    unique_ids = df["Image_ID"].nunique()
    unique_dates = df["Acquisition_Date"].nunique()
    unique_products = df["Product_ID"].nunique()
    unique_spacecraft = df["Spacecraft"].nunique()
    unique_tiles = df["MGRS_Tile"].nunique()

    passes_cloud = df[df["Passes_Cloud_Filter"]]
    n_pass = len(passes_cloud)
    unique_dates_after_cloud = passes_cloud["Acquisition_Date"].nunique()

    # Spacecraft breakdown
    s2a = df[df["Spacecraft"] == "Sentinel-2A"]
    s2b = df[df["Spacecraft"] == "Sentinel-2B"]
    s2a_dates = s2a["Acquisition_Date"].nunique()
    s2b_dates = s2b["Acquisition_Date"].nunique()

    # Tile breakdown
    tile_counts = df["MGRS_Tile"].value_counts()

    # Duplicate diagnostics
    dup_image_ids = n_total - unique_ids
    dup_product_ids = n_total - unique_products
    dup_dates = n_total - unique_dates

    # Same-date analysis: dates with multiple records
    date_record_counts = df.groupby("Acquisition_Date").size()
    multi_record_dates = date_record_counts[date_record_counts > 1]

    print(f"\n  {'='*72}")
    print(f"  WINDOW: {window_name.upper()}")
    print(f"  Date range: {start_date} → {end_date}")
    print(f"  {'='*72}")
    print(f"  A. Total GEE ImageCollection records   : {n_total}")
    print(f"  B. Unique system:index (Image IDs)      : {unique_ids}")
    print(f"  C. Unique acquisition dates              : {unique_dates}")
    print(f"  D. Unique PRODUCT_ID values              : {unique_products}")
    print(f"  E. Unique SPACECRAFT_NAME values         : {unique_spacecraft}  → {sorted(df['Spacecraft'].unique())}")
    print(f"  F. Unique MGRS_TILE values               : {unique_tiles}  → {sorted(df['MGRS_Tile'].unique())}")
    print(f"  G. Records passing cloud ≤{DEFAULT_CLOUD_COVER_THRESHOLD}%           : {n_pass}")
    print(f"  H. Unique dates after cloud filter       : {unique_dates_after_cloud}")
    print()
    print(f"  Sentinel-2A records: {len(s2a):3d}  (unique dates: {s2a_dates})")
    print(f"  Sentinel-2B records: {len(s2b):3d}  (unique dates: {s2b_dates})")
    print()
    print(f"  MGRS Tile breakdown:")
    for tile, cnt in tile_counts.items():
        tile_dates = df[df["MGRS_Tile"] == tile]["Acquisition_Date"].nunique()
        print(f"    {tile}: {cnt} records, {tile_dates} unique dates")
    print()
    print(f"  Duplicate diagnostics:")
    print(f"    Duplicate Image_ID count      : {dup_image_ids}")
    print(f"    Duplicate Product_ID count     : {dup_product_ids}")
    print(f"    Duplicate Acquisition_Date cnt : {dup_dates}  (dates with >1 record)")

    if len(multi_record_dates) > 0:
        print(f"\n  Dates with multiple records ({len(multi_record_dates)} dates):")
        for date_str, cnt in multi_record_dates.items():
            subset = df[df["Acquisition_Date"] == date_str]
            tiles = sorted(subset["MGRS_Tile"].unique())
            sats = sorted(subset["Spacecraft"].unique())
            print(f"    {date_str}: {cnt} records | Tiles: {tiles} | Satellites: {sats}")
    else:
        print(f"\n  All dates have exactly 1 record (no multi-tile or multi-satellite overlap)")

    print()


def main():
    print("=" * 75)
    print("  SENTINEL-2 IMAGE INVENTORY DIAGNOSTIC")
    print(f"  FarmID: {FARM_ID}")
    print(f"  Collection: {S2_SR_COLLECTION}")
    print(f"  Cloud threshold: ≤{DEFAULT_CLOUD_COVER_THRESHOLD}%")
    print("=" * 75)

    # Initialize Earth Engine
    if not EarthEngineManager.initialize():
        logger.error("Cannot initialize Earth Engine. Aborting diagnostic.")
        sys.exit(1)

    import ee
    from shapely import wkt

    geom = wkt.loads(FARM_WKT)
    ee_region = get_ee_polygon(geom)

    # Compute phenological windows (same logic as the extractor)
    windows = compute_phenological_windows(FARM_SDATE, FARM_HDATE)
    print(f"\n  Phenological windows computed from SDate={FARM_SDATE}, HDate={FARM_HDATE}:")
    for wn, (ws, we) in windows.items():
        print(f"    {wn:5s}: {ws} → {we}")

    # Collect metadata for all windows
    all_records = []
    for window_name, (w_start, w_end) in windows.items():
        records = collect_image_metadata(ee_region, w_start, w_end, window_name)
        all_records.extend(records)
        print_window_report(records, window_name, w_start, w_end)

    # Save full inventory CSV
    if all_records:
        inv_df = pd.DataFrame(all_records)
        csv_path = OUTPUTS_DIR / f"s2_image_inventory_{FARM_ID}.csv"
        inv_df.to_csv(csv_path, index=False)
        logger.info("Saved image inventory → %s (%d records)", csv_path, len(inv_df))

        # Global summary
        print("=" * 75)
        print("  GLOBAL SUMMARY (across all windows)")
        print("=" * 75)

        # Use FULL window for global dedup since it spans the entire period
        full_df = inv_df[inv_df["Window"] == "full"]
        if len(full_df) > 0:
            print(f"  Full-season total GEE records     : {len(full_df)}")
            print(f"  Full-season unique Image IDs       : {full_df['Image_ID'].nunique()}")
            print(f"  Full-season unique dates            : {full_df['Acquisition_Date'].nunique()}")
            print(f"  Full-season unique MGRS tiles       : {full_df['MGRS_Tile'].nunique()} → {sorted(full_df['MGRS_Tile'].unique())}")
            print(f"  Full-season unique spacecraft       : {full_df['Spacecraft'].nunique()} → {sorted(full_df['Spacecraft'].unique())}")
            s2a_f = full_df[full_df["Spacecraft"] == "Sentinel-2A"]
            s2b_f = full_df[full_df["Spacecraft"] == "Sentinel-2B"]
            print(f"  Full-season Sentinel-2A records     : {len(s2a_f)} ({s2a_f['Acquisition_Date'].nunique()} unique dates)")
            print(f"  Full-season Sentinel-2B records     : {len(s2b_f)} ({s2b_f['Acquisition_Date'].nunique()} unique dates)")
            cloud_pass = full_df[full_df["Passes_Cloud_Filter"]]
            print(f"  Full-season passing cloud ≤{DEFAULT_CLOUD_COVER_THRESHOLD}%    : {len(cloud_pass)}")
            print(f"  Full-season unique dates after cloud: {cloud_pass['Acquisition_Date'].nunique()}")

        print()
        print("=" * 75)
        print("  KEY DIAGNOSTIC ANSWERS")
        print("=" * 75)

        early_df = inv_df[inv_df["Window"] == "early"]
        if len(early_df) > 0:
            e_total = len(early_df)
            e_dates = early_df["Acquisition_Date"].nunique()
            e_tiles = sorted(early_df["MGRS_Tile"].unique())
            e_sats = sorted(early_df["Spacecraft"].unique())
            print(f"\n  Q: 'Are the {e_total} Early-window records {e_total} unique acquisition dates?'")
            if e_dates == e_total:
                print(f"  A: YES — {e_total} records correspond to {e_dates} unique dates.")
            else:
                print(f"  A: NO — {e_total} records correspond to only {e_dates} unique dates.")
                print(f"     The extra records are due to: tiles={e_tiles}, satellites={e_sats}")

        print(f"\n  Q: 'Does our query use Sentinel-2A, Sentinel-2B, or both?'")
        all_sats = inv_df["Spacecraft"].unique()
        print(f"  A: {', '.join(sorted(all_sats))}")

        all_tiles = inv_df["MGRS_Tile"].unique()
        if len(all_tiles) > 1:
            print(f"\n  Q: 'Does the farm intersect multiple MGRS tiles?'")
            print(f"  A: YES — {len(all_tiles)} tiles: {sorted(all_tiles)}")
            print(f"     This is the primary reason for inflated image counts.")
            print(f"     Each acquisition date produces one record PER overlapping tile.")
        else:
            print(f"\n  Q: 'Does the farm intersect multiple MGRS tiles?'")
            print(f"  A: NO — single tile: {all_tiles[0]}")

        # Check for actual duplicates (same Image_ID appearing twice)
        dup_ids = inv_df[inv_df.duplicated(subset=["Image_ID"], keep=False)]
        # But within different windows this is expected; check within same window
        for wn in ["early", "mid", "late", "full"]:
            w_df = inv_df[inv_df["Window"] == wn]
            dup_within = w_df[w_df.duplicated(subset=["Image_ID"], keep=False)]
            if len(dup_within) > 0:
                print(f"\n  WARNING: {len(dup_within)} duplicate Image_IDs within window '{wn}'")

        print()
        print("=" * 75)
    else:
        print("  No records retrieved from GEE.")


if __name__ == "__main__":
    main()
