"""
src/satellite/s2_extractor.py
# -*- coding: utf-8 -*-
=============================
Sentinel-2 Multi-Temporal Remote Sensing Extractor
Telangana Crop Health Challenge

Responsibilities:
    - Authenticate and initialize Google Earth Engine (GEE).
    - Query Sentinel-2 L2A (Surface Reflectance: COPERNICUS/S2_SR_HARMONIZED)
      imagery filtered by farm geometry, date range, and CLOUDY_PIXEL_PERCENTAGE threshold.
    - Mask clouds, cirrus, and cloud shadows using SCL (Scene Classification Layer) or QA60 bitmask.
    - Partition the crop growing season into 3 phenological windows:
        1. Early Season (Emergence/Early Vegetative: 0% - 25% of duration)
        2. Mid Season (Peak Growth/Flowering: 25% - 75% of duration)
        3. Late Season (Maturation/Pre-harvest: 75% - 100% of duration)
        4. Full Season (Median baseline: 0% - 100% of duration)
    - Generate median composites across valid, cloud-masked observations for each window.
    - Extract zonal statistics using Mean + Standard Deviation reducers:
        ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True).
    - Spatial Resolution Strategy:
        Sentinel-2 MSI has different spatial resolutions across spectral bands:
          - 10 m resolution bands: B2 (Blue), B3 (Green), B4 (Red), B8 (NIR Broad).
          - 20 m resolution bands: B5 (RedEdge 1), B6 (RedEdge 2), B7 (RedEdge 3),
                                   B8A (NIR Narrow), B11 (SWIR 1), B12 (SWIR 2).
        To preserve genuine radiometric properties and avoid artificial 10 m interpolation
        artifacts on 20 m sensors, 10 m bands are reduced at scale=10 m, while 20 m bands
        are reduced at scale=20 m.
    - Explicitly handle empty collections and cloudy windows without failure.
    - Checkpoint results to disk cache, ensuring failed farms are marked status='failed'
      with error messages and remain retryable (never considered successfully cached).
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from shapely.geometry import mapping, MultiPolygon, Polygon

logger = logging.getLogger(__name__)

# Band definitions by native spatial resolution
S2_BANDS_10M: List[str] = ["B2", "B3", "B4", "B8"]
S2_BANDS_20M: List[str] = ["B5", "B6", "B7", "B8A", "B11", "B12"]
S2_ALL_BANDS: List[str] = S2_BANDS_10M + S2_BANDS_20M

# Default Earth Engine Collections & Thresholds
S2_SR_COLLECTION: str = "COPERNICUS/S2_SR_HARMONIZED"
S2_TOA_COLLECTION: str = "COPERNICUS/S2"
DEFAULT_CLOUD_COVER_THRESHOLD: float = 20.0  # Max cloudy pixel % for scene-level filter


class EarthEngineManager:
    """Manages Google Earth Engine authentication, initialization, and status."""

    _initialized: bool = False

    @classmethod
    def initialize(cls, project_id: Optional[str] = None) -> bool:
        """
        Initialize Earth Engine session.

        Project resolution order:
        1. Explicit ``project_id`` argument
        2. ``EE_PROJECT`` environment variable
        3. ``earth_engine.project`` in ``config/project_config.yaml``
        4. Fall through to ``ee.Initialize()`` with no project (may fail)
        """
        if cls._initialized:
            return True

        import ee

        # --- resolve project ---
        project = project_id or os.environ.get("EE_PROJECT", None)

        if not project:
            # Try loading from project config YAML
            try:
                import yaml
                config_path = Path(__file__).resolve().parent.parent.parent / "config" / "project_config.yaml"
                if config_path.exists():
                    with open(config_path, "r", encoding="utf-8") as f:
                        cfg = yaml.safe_load(f)
                    project = (cfg or {}).get("earth_engine", {}).get("project", "") or None
            except Exception:
                pass  # Config read failed; proceed without

        try:
            if project:
                ee.Initialize(project=project)
            else:
                ee.Initialize()
            cls._initialized = True
            logger.info("Google Earth Engine initialized successfully (project=%s).", project or "default")
            return True
        except Exception as exc:
            exc_str = str(exc)
            # Distinguish missing project from missing credentials
            if "no project" in exc_str.lower() or "project is not found" in exc_str.lower():
                logger.warning(
                    "Earth Engine initialization failed — OAuth credentials exist but no "
                    "Google Cloud project was resolved. Set 'earth_engine.project' in "
                    "config/project_config.yaml or the EE_PROJECT environment variable. "
                    "Error: %s", exc,
                )
            else:
                logger.warning("Google Earth Engine initialization failure: %s", exc)
            return False

    @classmethod
    def is_authenticated(cls) -> bool:
        """Test if EE API calls can be executed."""
        if not cls._initialized:
            if not cls.initialize():
                return False
        import ee
        try:
            _ = ee.Number(1).getInfo()
            return True
        except Exception:
            return False


def mask_s2_clouds_sr(image):
    """
    Mask clouds in Sentinel-2 Surface Reflectance using SCL (Scene Classification Layer).
    Values to mask:
        3 = Cloud shadow
        8 = Cloud medium probability
        9 = Cloud high probability
        10 = Thin cirrus
        11 = Snow/ice
    """
    import ee
    scl = image.select("SCL")
    mask = (
        scl.neq(3)
        .And(scl.neq(8))
        .And(scl.neq(9))
        .And(scl.neq(10))
        .And(scl.neq(11))
    )
    return image.updateMask(mask)


def mask_s2_clouds_qa60(image):
    """
    Mask clouds in Sentinel-2 TOA or L2A using QA60 bitmask band.
    Bit 10: Opaque clouds
    Bit 11: Cirrus clouds
    """
    import ee
    qa = image.select("QA60")
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    return image.updateMask(mask)


def mosaic_same_date_scenes(col):
    """
    Mosaic Sentinel-2 scenes acquired on the same calendar date (e.g. overlapping
    MGRS tiles) prior to temporal compositing.

    This ensures that calendar dates covered by multiple swaths/tiles contribute exactly
    one observation to temporal reductions, eliminating spatial over-weighting bias.
    """
    import ee

    dates = ee.List(
        col.map(lambda img: ee.Feature(None, {"date": img.date().format("YYYY-MM-dd")})).aggregate_array("date")
    ).distinct()

    def make_daily(d):
        t0 = ee.Date(d)
        t1 = t0.advance(1, "day")
        return col.filterDate(t0, t1).mosaic().set({
            "system:time_start": t0.millis(),
            "date": d,
        })

    return ee.ImageCollection(dates.map(make_daily))


def compute_phenological_windows(
    sdate_str: Union[str, pd.Timestamp],
    hdate_str: Union[str, pd.Timestamp],
) -> Dict[str, Tuple[str, str]]:
    """
    Split the growing season into 3 phenological stages plus full season:
        - early: 0% to 25% of duration (Emergence & early vegetative)
        - mid: 25% to 75% of duration (Peak vegetative, flowering, grain-fill)
        - late: 75% to 100% of duration (Ripening, maturation, pre-harvest)
        - full: 0% to 100% of duration (Full seasonal baseline)

    Guarantees:
        - Every window has strictly valid boundaries (start < end).
        - All window intervals lie within [sdate, hdate].
        - Robust against arbitrary durations (1 day, 2 days, 11 days, 30 days, normal).
    """
    sdate = pd.to_datetime(sdate_str, format="mixed", dayfirst=True)
    hdate = pd.to_datetime(hdate_str, format="mixed", dayfirst=True)

    if pd.isna(sdate) or pd.isna(hdate) or hdate <= sdate:
        if pd.notna(sdate):
            hdate = sdate + timedelta(days=120)
        else:
            sdate = pd.to_datetime("2023-11-01")
            hdate = pd.to_datetime("2024-03-31")

    duration = (hdate - sdate).days
    fmt = "%Y-%m-%d"

    if duration <= 1:
        # 1-day duration: all windows span the single available day
        return {
            "early": (sdate.strftime(fmt), hdate.strftime(fmt)),
            "mid": (sdate.strftime(fmt), hdate.strftime(fmt)),
            "late": (sdate.strftime(fmt), hdate.strftime(fmt)),
            "full": (sdate.strftime(fmt), hdate.strftime(fmt)),
        }
    elif duration == 2:
        # 2-day duration: early is day 1, late is day 2, mid and full span both days
        mid_date = sdate + timedelta(days=1)
        return {
            "early": (sdate.strftime(fmt), mid_date.strftime(fmt)),
            "mid": (sdate.strftime(fmt), hdate.strftime(fmt)),
            "late": (mid_date.strftime(fmt), hdate.strftime(fmt)),
            "full": (sdate.strftime(fmt), hdate.strftime(fmt)),
        }
    else:
        # duration >= 3: proportional partition guaranteed strictly within [sdate, hdate]
        d1 = max(1, int(round(duration * 0.25)))
        d2 = max(d1 + 1, int(round(duration * 0.75)))
        d2 = min(d2, duration - 1)
        d1 = min(d1, d2 - 1)

        early_end = sdate + timedelta(days=d1)
        mid_end = sdate + timedelta(days=d2)

        return {
            "early": (sdate.strftime(fmt), early_end.strftime(fmt)),
            "mid": (early_end.strftime(fmt), mid_end.strftime(fmt)),
            "late": (mid_end.strftime(fmt), hdate.strftime(fmt)),
            "full": (sdate.strftime(fmt), hdate.strftime(fmt)),
        }


def get_ee_polygon(geometry: Union[Polygon, MultiPolygon]):
    """Convert a Shapely Polygon or MultiPolygon to an ee.Geometry."""
    import ee
    geo_json = mapping(geometry)
    return ee.Geometry(geo_json)


def extract_zonal_stats_for_farm(
    farm_id: Union[int, str],
    geometry: Union[Polygon, MultiPolygon],
    sdate_str: str,
    hdate_str: str,
    collection_name: str = S2_SR_COLLECTION,
    cloud_cover_threshold: float = DEFAULT_CLOUD_COVER_THRESHOLD,
    scale_10m: int = 10,
    scale_20m: int = 20,
    daily_mosaic: bool = True,
) -> Dict[str, Any]:
    """
    Query Earth Engine for Sentinel-2 multi-temporal composites and extract
    zonal band reflectances (mean + standard deviation) for a single farm polygon.

    Spatial Resolution Strategy:
        - 10 m bands (B2, B3, B4, B8) reduced at scale_10m (10 m).
        - 20 m bands (B5, B6, B7, B8A, B11, B12) reduced at scale_20m (20 m).

    Parameters
    ----------
    daily_mosaic : bool, default True
        If True, mosaics scenes acquired on the same date before temporal median,
        preventing duplicate weighting from overlapping MGRS swaths.

    Returns
    -------
    dict containing:
        - FarmID
        - status: 'success' or 'failed'
        - error_message: '' if successful, error description if failed
        - {band}_{window}_mean, {band}_{window}_std for each band and window
        - n_images_found_{window}, n_images_valid_{window}
    """
    import ee

    if not EarthEngineManager.is_authenticated():
        msg = "Earth Engine is not authenticated. Please run `earthengine authenticate`."
        logger.error("FarmID %s: %s", farm_id, msg)
        return {
            "FarmID": farm_id,
            "status": "failed",
            "error_message": msg,
        }

    results: Dict[str, Any] = {
        "FarmID": farm_id,
        "status": "failed",
        "error_message": "",
    }

    try:
        ee_region = get_ee_polygon(geometry)
        windows = compute_phenological_windows(sdate_str, hdate_str)
        windows_with_data = 0

        # Define reducer: Mean + Standard Deviation
        zonal_reducer = ee.Reducer.mean().combine(
            reducer2=ee.Reducer.stdDev(),
            sharedInputs=True,
        )

        for window_name, (w_start, w_end) in windows.items():
            # 1. Filter collection by Date, Bounds, and Cloud Cover Threshold
            col = (
                ee.ImageCollection(collection_name)
                .filterDate(w_start, w_end)
                .filterBounds(ee_region)
                .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", cloud_cover_threshold))
            )

            # Count total images found matching criteria
            n_images_found = int(col.size().getInfo())

            if n_images_found == 0:
                logger.info(
                    "FarmID %s | Window: %-5s (%s to %s) | S2 images found: 0 | Valid: 0 | Status: empty_collection",
                    farm_id, window_name, w_start, w_end
                )
                results[f"n_images_found_{window_name}"] = 0
                results[f"n_images_valid_{window_name}"] = 0
                for b in S2_ALL_BANDS:
                    results[f"{b}_{window_name}_mean"] = np.nan
                    results[f"{b}_{window_name}_std"] = np.nan
                continue

            # 2. Apply cloud and shadow masking
            if "SR" in collection_name:
                col_masked = col.map(mask_s2_clouds_sr)
            else:
                col_masked = col.map(mask_s2_clouds_qa60)

            # 3. Count valid images that contain unmasked pixels over the farm polygon
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

            results[f"n_images_found_{window_name}"] = n_images_found
            results[f"n_images_valid_{window_name}"] = n_images_valid

            if n_images_valid == 0:
                logger.info(
                    "FarmID %s | Window: %-5s (%s to %s) | S2 images found: %d | Valid: 0 | Status: cloud_obscured",
                    farm_id, window_name, w_start, w_end, n_images_found
                )
                for b in S2_ALL_BANDS:
                    results[f"{b}_{window_name}_mean"] = np.nan
                    results[f"{b}_{window_name}_std"] = np.nan
                continue

            # 4. Daily mosaicking before temporal median composite to eliminate
            #    duplicate weighting from overlapping MGRS tiles on the same date.
            if daily_mosaic:
                daily_col = mosaic_same_date_scenes(col_masked)
                composite = daily_col.median()
            else:
                composite = col_masked.median()

            # 5. Extract zonal statistics at respective native resolutions
            # 10 m bands reduced at scale=10
            stats_10m = composite.select(S2_BANDS_10M).reduceRegion(
                reducer=zonal_reducer,
                geometry=ee_region,
                scale=scale_10m,
                maxPixels=1e8,
            ).getInfo()

            # 20 m bands reduced at scale=20
            stats_20m = composite.select(S2_BANDS_20M).reduceRegion(
                reducer=zonal_reducer,
                geometry=ee_region,
                scale=scale_20m,
                maxPixels=1e8,
            ).getInfo()

            # Combine reduction outputs
            stats_combined = {}
            if stats_10m:
                stats_combined.update(stats_10m)
            if stats_20m:
                stats_combined.update(stats_20m)

            has_valid_band = False
            for b in S2_ALL_BANDS:
                m_val = stats_combined.get(f"{b}_mean")
                s_val = stats_combined.get(f"{b}_stdDev")
                results[f"{b}_{window_name}_mean"] = np.nan if m_val is None else float(m_val)
                results[f"{b}_{window_name}_std"] = np.nan if s_val is None else float(s_val)
                if m_val is not None:
                    has_valid_band = True

            if has_valid_band:
                windows_with_data += 1

            logger.info(
                "FarmID %s | Window: %-5s (%s to %s) | S2 images found: %d | Valid: %d | Status: success",
                farm_id, window_name, w_start, w_end, n_images_found, n_images_valid
            )

        if windows_with_data > 0:
            results["status"] = "success"
            results["error_message"] = ""
        else:
            results["status"] = "failed"
            results["error_message"] = "No valid cloud-free imagery found across any phenological window"

        return results

    except Exception as exc:
        err_msg = str(exc)
        logger.error("FarmID %s extraction failed: %s", farm_id, err_msg)
        results["status"] = "failed"
        results["error_message"] = err_msg
        return results


class BatchSentinelExtractor:
    """
    Handles robust batch extraction of Sentinel-2 multi-temporal data
    with disk checkpoint caching, error isolation, rate-limiting,
    and automatic retrying of previously failed records.
    """

    def __init__(
        self,
        output_dir: Union[str, Path] = "outputs/satellite_cache",
        checkpoint_interval: int = 25,
        rate_limit_delay: float = 0.1,
        cloud_cover_threshold: float = DEFAULT_CLOUD_COVER_THRESHOLD,
        daily_mosaic: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_interval = checkpoint_interval
        self.rate_limit_delay = rate_limit_delay
        self.cloud_cover_threshold = cloud_cover_threshold
        self.daily_mosaic = daily_mosaic

    def extract_dataset(
        self,
        gdf: pd.DataFrame,
        cache_filename: str = "s2_extracted_features.csv",
        max_farms: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Process farms in gdf. Only farms with status == 'success' in cache
        are skipped; failed farms remain retryable and will be processed.
        """
        cache_file = self.output_dir / cache_filename
        cached_records: Dict[Any, Dict[str, Any]] = {}
        successful_ids = set()

        if cache_file.exists():
            existing_df = pd.read_csv(cache_file)
            for rec in existing_df.to_dict("records"):
                fid = rec["FarmID"]
                if rec.get("status") == "success":
                    successful_ids.add(fid)
                    cached_records[fid] = rec
                else:
                    logger.debug("FarmID %s was marked failed in cache; keeping in retry queue.", fid)

            logger.info(
                "Cache check: %d successfully extracted farms loaded. %d failed farms remain in retry queue.",
                len(successful_ids), len(existing_df) - len(successful_ids)
            )

        # Target queue: only farms NOT successfully processed yet
        target_gdf = gdf[~gdf["FarmID"].isin(successful_ids)]
        if max_farms:
            target_gdf = target_gdf.head(max_farms)

        logger.info("Starting Sentinel-2 extraction for %d farms...", len(target_gdf))

        all_records = list(cached_records.values())

        for count, (_, row) in enumerate(target_gdf.iterrows(), start=1):
            farm_id = row["FarmID"]
            geom = row["geometry"]
            sdate = row["SDate"]
            hdate = row["HDate"]

            try:
                rec = extract_zonal_stats_for_farm(
                    farm_id=farm_id,
                    geometry=geom,
                    sdate_str=sdate,
                    hdate_str=hdate,
                    cloud_cover_threshold=self.cloud_cover_threshold,
                    daily_mosaic=self.daily_mosaic,
                )
                all_records.append(rec)
            except Exception as e:
                err_msg = str(e)
                logger.error("Unexpected error for FarmID %s: %s", farm_id, err_msg)
                all_records.append({
                    "FarmID": farm_id,
                    "status": "failed",
                    "error_message": err_msg,
                })

            time.sleep(self.rate_limit_delay)

            # Checkpoint to disk
            if count % self.checkpoint_interval == 0:
                pd.DataFrame(all_records).to_csv(cache_file, index=False)
                logger.info("Checkpoint saved: %d records -> %s", len(all_records), cache_file)

        final_df = pd.DataFrame(all_records)
        final_df.to_csv(cache_file, index=False)
        logger.info("Batch extraction complete: %d records -> %s", len(final_df), cache_file)
        return final_df
