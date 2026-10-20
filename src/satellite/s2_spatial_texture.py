"""
src/satellite/s2_spatial_texture.py
===================================
Phase 6H: Dedicated Sentinel-2 Spatial-Texture Extraction Module.
Computes Gray-Level Co-occurrence Matrix (GLCM) canopy texture metrics
from Sentinel-2 Level-2A surface reflectance across phenological windows.

Methodological Specifications:
- Band: B8 (NIR Broad, 842 nm) at native 10 m resolution.
- Quantization: 32 discrete gray levels [0, 31] via linear scaling of
  surface reflectance [0, 10,000].
- Moving Window: 3x3 kernel (30 m x 30 m), directionally averaged
  (omnidirectional isotropic) across 4 directions (0, 45, 90, 135 deg).
- Metrics (5 Haralick descriptors):
    1. Contrast: local gray-level variations.
    2. Dissimilarity: absolute gray-level differences.
    3. Homogeneity: Inverse Difference Moment (IDM).
    4. Energy: square root of Angular Second Moment (sqrt(ASM)).
    5. Correlation: linear dependency of gray levels.
- Phenological Windows: early, mid, late, full (4 windows x 5 metrics = 20 features).
- Polygon Aggregation: Zonal mean (ee.Reducer.mean()) over farm parcel geometry.
- Missingness & Integrity: Unobserved/cloudy windows preserve NaN. Zero artificial zeros.
  Strictly label-independent (operates only on reflectance and geometry).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon

from src.satellite.s2_extractor import (
    DEFAULT_CLOUD_COVER_THRESHOLD,
    EarthEngineManager,
    S2_SR_COLLECTION,
    compute_phenological_windows,
    get_ee_polygon,
    mask_s2_clouds_sr,
    mosaic_same_date_scenes,
)

logger = logging.getLogger(__name__)

GLCM_METRICS: List[str] = [
    "contrast",
    "dissimilarity",
    "homogeneity",
    "energy",
    "correlation",
]

GLCM_WINDOWS: List[str] = ["early", "mid", "late", "full"]

# Exactly 20 spatial texture feature names
GLCM_FEATURE_NAMES: List[str] = [
    f"glcm_{metric}_{window}"
    for metric in GLCM_METRICS
    for window in GLCM_WINDOWS
]


def quantize_band(
    image,
    band_name: str = "B8",
    max_reflectance: float = 10000.0,
    quantization_levels: int = 32,
):
    """
    Linearly quantize surface reflectance into discrete integer levels [0, quantization_levels - 1].
    """
    import ee

    band = image.select(band_name)
    max_idx = quantization_levels - 1
    quantized = (
        band.divide(max_reflectance)
        .multiply(max_idx)
        .clamp(0, max_idx)
        .round()
        .toInt32()
    )
    return quantized


def compute_glcm_texture_image(
    quantized_image,
    band_name: str = "B8",
    window_name: str = "full",
    kernel_size: int = 3,
):
    """
    Compute omnidirectional GLCM texture metrics on quantized band and extract the 5 core Haralick metrics.
    """
    import ee

    # glcmTexture on integer image with default directional averaging
    glcm = quantized_image.glcmTexture(size=kernel_size)

    # In GEE glcmTexture, band names are prefixed by the input band: e.g. B8_contrast, B8_diss, B8_idm, B8_asm, B8_corr
    contrast = glcm.select(f"{band_name}_contrast").rename(f"glcm_contrast_{window_name}")
    dissimilarity = glcm.select(f"{band_name}_diss").rename(f"glcm_dissimilarity_{window_name}")
    homogeneity = glcm.select(f"{band_name}_idm").rename(f"glcm_homogeneity_{window_name}")
    # Energy is sqrt(ASM)
    energy = glcm.select(f"{band_name}_asm").sqrt().rename(f"glcm_energy_{window_name}")
    correlation = glcm.select(f"{band_name}_corr").rename(f"glcm_correlation_{window_name}")

    return ee.Image.cat([contrast, dissimilarity, homogeneity, energy, correlation])


def extract_spatial_texture_for_farm(
    farm_id: Union[int, str],
    geometry: Union[Polygon, MultiPolygon],
    sdate_str: str,
    hdate_str: str,
    collection_name: str = S2_SR_COLLECTION,
    cloud_cover_threshold: float = DEFAULT_CLOUD_COVER_THRESHOLD,
    scale: int = 10,
    quantization_levels: int = 32,
    kernel_size: int = 3,
    daily_mosaic: bool = True,
) -> Dict[str, Any]:
    """
    Extract 20 GLCM spatial texture features for a single farm parcel across 4 phenological windows.

    All 4 windows are combined into a single multi-band Earth Engine image to minimize
    network round-trips and maximize server-side parallel execution.
    """
    import ee

    if not EarthEngineManager.is_authenticated():
        msg = "Earth Engine is not authenticated."
        logger.error("FarmID %s: %s", farm_id, msg)
        return {"FarmID": farm_id, "status": "failed", "error_message": msg}

    results: Dict[str, Any] = {
        "FarmID": farm_id,
        "status": "failed",
        "error_message": "",
    }
    # Initialize all 20 features to NaN
    for feat_name in GLCM_FEATURE_NAMES:
        results[feat_name] = np.nan

    try:
        ee_region = get_ee_polygon(geometry)
        windows = compute_phenological_windows(sdate_str, hdate_str)
        window_images = []

        for window_name, (w_start, w_end) in windows.items():
            col = (
                ee.ImageCollection(collection_name)
                .filterDate(w_start, w_end)
                .filterBounds(ee_region)
                .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", cloud_cover_threshold))
            )

            if "SR" in collection_name:
                col_masked = col.map(mask_s2_clouds_sr)
            else:
                from src.satellite.s2_extractor import mask_s2_clouds_qa60
                col_masked = col.map(mask_s2_clouds_qa60)

            # Check if collection is empty
            n_images = int(col_masked.size().getInfo())
            results[f"glcm_n_images_{window_name}"] = n_images

            expected_band_names = [f"glcm_{m}_{window_name}" for m in GLCM_METRICS]

            if n_images == 0:
                # Fully masked placeholder band to maintain 20-band graph without failing
                dummy = (
                    ee.Image.constant([0.0] * len(expected_band_names))
                    .rename(expected_band_names)
                    .selfMask()
                )
                window_images.append(dummy)
                continue

            if daily_mosaic:
                daily_col = mosaic_same_date_scenes(col_masked)
                composite = daily_col.median()
            else:
                composite = col_masked.median()

            # Quantize B8 and compute GLCM
            b8_quant = quantize_band(
                composite,
                band_name="B8",
                max_reflectance=10000.0,
                quantization_levels=quantization_levels,
            )
            w_texture = compute_glcm_texture_image(
                b8_quant,
                band_name="B8",
                window_name=window_name,
                kernel_size=kernel_size,
            )
            window_images.append(w_texture)

        # Concatenate all window textures into a single 20-band image
        full_texture_img = ee.Image.cat(window_images)

        # Single zonal mean reduction over the farm polygon
        zonal_stats = full_texture_img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=ee_region,
            scale=scale,
            maxPixels=1e8,
        ).getInfo()

        valid_count = 0
        if zonal_stats:
            for feat_name in GLCM_FEATURE_NAMES:
                val = zonal_stats.get(feat_name)
                if val is not None:
                    results[feat_name] = float(val)
                    valid_count += 1
                else:
                    results[feat_name] = np.nan

        if valid_count > 0:
            results["status"] = "success"
            results["error_message"] = ""
        else:
            results["status"] = "failed"
            results["error_message"] = "No valid cloud-free texture extracted across any window"

        return results

    except Exception as exc:
        err_msg = str(exc)
        logger.error("FarmID %s spatial texture extraction failed: %s", farm_id, err_msg)
        results["status"] = "failed"
        results["error_message"] = err_msg
        return results
