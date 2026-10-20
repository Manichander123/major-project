"""
src/features/indices.py
# -*- coding: utf-8 -*-
=======================
Spectral and Vegetation Indices Computation Module
Telangana Crop Health Challenge

Computes standard and red-edge vegetation, moisture, chlorophyll, and soil indices
from Sentinel-2 multi-spectral bands (B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12).

Authoritative Remote Sensing Definitions:
    - NDVI  : Normalized Difference Vegetation Index (Rouse et al., 1974)
              Proxy indicator for photosynthetic capacity and canopy greenness.
    - EVI   : Enhanced Vegetation Index (Huete et al., 2002)
              Proxy indicator for canopy structural complexity with reduced atmospheric/soil sensitivity.
    - GNDVI : Green Normalized Difference Vegetation Index (Gitelson et al., 1996)
              Indicator associated with leaf chlorophyll concentration.
    - SAVI  : Soil Adjusted Vegetation Index (Huete, 1988)
              Vegetation index incorporating a soil brightness correction factor (L = 0.5).
    - MSAVI : Modified Soil Adjusted Vegetation Index (Qi et al., 1994)
              Adaptive self-adjusting soil factor for sparse early-season canopy cover.
    - NDMI  : Normalized Difference Moisture Index (Gao, 1996; Wilson & Sader, 2002)
              Uses NIR (B8) and SWIR1 (B11). Proxy indicator associated with canopy liquid water content.
    - NDWI  : Normalized Difference Water Index (McFeeters, 1996)
              Uses Green (B3) and NIR (B8). Delineates open surface water; yields negative values for vegetation.
              Distinct from NIR-SWIR moisture indices like NDMI.
    - NDRE1 : Normalized Difference Red Edge 1 (Barnes et al., 2000; Gitelson & Merzlyak, 1996)
              Uses NIR (B8) and Red Edge 1 (B5). Sensitive indicator of early canopy stress.
    - NDRE2 : Normalized Difference Red Edge 2
              Uses NIR (B8) and Red Edge 2 (B6). Indicator associated with canopy chlorophyll status.
    - PSRI  : Plant Senescence Reflectance Index (Merzlyak et al., 1999)
              Formula: (R680 - R500) / R750 -> (B4 - B2) / B6.
              Indicator associated with carotenoid-to-chlorophyll ratio and canopy senescence.
    - BSI   : Bare Soil Index (Rikimaru et al., 2002)
              Proxy separating bare soil / exposed background from vegetative canopy.
    - MSI   : Moisture Stress Index (Rock et al., 1986; Hunt & Rock, 1989)
              Formula: SWIR1 / NIR -> B11 / B8. Indicator associated with leaf water deficit.
    - CIRE  : Chlorophyll Index Red Edge (Gitelson et al., 2003)
              Formula: (NIR / Red Edge 1) - 1 -> (B8 / B5) - 1. Proxy indicator for leaf chlorophyll.
    - CIG   : Chlorophyll Index Green (Gitelson et al., 2005)
              Formula: (NIR / Green) - 1 -> (B8 / B3) - 1. Proxy indicator for total canopy chlorophyll.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def safe_normalized_diff(band_a: np.ndarray, band_b: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """
    Computes (band_a - band_b) / (band_a + band_b) safely.
    - Preserves NaN if either input band is NaN / missing.
    - Returns 0.0 for genuine near-zero denominators (|denom| <= eps) with finite inputs.
    - Output clipped to [-1.0, 1.0].
    """
    nan_mask = np.isnan(band_a) | np.isnan(band_b)
    num = band_a - band_b
    denom = band_a + band_b
    with np.errstate(divide="ignore", invalid="ignore"):
        res = np.where(np.abs(denom) > eps, num / denom, 0.0)
    res = np.where(nan_mask, np.nan, res)
    return np.clip(res, -1.0, 1.0)


# ── Core Vegetation Indices ───────────────────────────────────────────────────

def calc_ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """
    NDVI = (NIR - Red) / (NIR + Red)
    Rouse et al. (1974). Proxy indicator for photosynthetic vigor and green biomass.
    """
    return safe_normalized_diff(nir, red)


def calc_evi(
    nir: np.ndarray,
    red: np.ndarray,
    blue: np.ndarray,
    G: float = 2.5,
    C1: float = 6.0,
    C2: float = 7.5,
    L: float = 1.0,
) -> np.ndarray:
    """
    EVI = G * (NIR - Red) / (NIR + C1 * Red - C2 * Blue + L)
    Huete et al. (2002). Proxy for canopy structural complexity with reduced atmospheric/soil distortion.
    Requires surface reflectance in standard [0, 1] range.
    """
    nan_mask = np.isnan(nir) | np.isnan(red) | np.isnan(blue)
    num = G * (nir - red)
    denom = nir + C1 * red - C2 * blue + L
    with np.errstate(divide="ignore", invalid="ignore"):
        res = np.where(np.abs(denom) > 1e-6, num / denom, 0.0)
    res = np.where(nan_mask, np.nan, res)
    return np.clip(res, -1.5, 1.5)


def calc_gndvi(nir: np.ndarray, green: np.ndarray) -> np.ndarray:
    """
    GNDVI = (NIR - Green) / (NIR + Green)
    Gitelson et al. (1996). Indicator associated with green-band chlorophyll absorption.
    """
    return safe_normalized_diff(nir, green)


def calc_savi(nir: np.ndarray, red: np.ndarray, L: float = 0.5) -> np.ndarray:
    """
    SAVI = ((NIR - Red) / (NIR + Red + L)) * (1 + L)
    Huete (1988). Vegetation vigor indicator incorporating a soil brightness correction factor (L=0.5).
    Requires surface reflectance in standard [0, 1] range.
    """
    nan_mask = np.isnan(nir) | np.isnan(red)
    num = (nir - red) * (1.0 + L)
    denom = nir + red + L
    with np.errstate(divide="ignore", invalid="ignore"):
        res = np.where(np.abs(denom) > 1e-6, num / denom, 0.0)
    res = np.where(nan_mask, np.nan, res)
    return np.clip(res, -1.0, 1.5)


def calc_msavi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """
    MSAVI = (2 * NIR + 1 - sqrt((2 * NIR + 1)^2 - 8 * (NIR - Red))) / 2
    Qi et al. (1994). Modified SAVI with an adaptive self-adjusting soil factor.
    Requires surface reflectance in standard [0, 1] range.
    """
    nan_mask = np.isnan(nir) | np.isnan(red)
    term = (2.0 * nir + 1.0) ** 2 - 8.0 * (nir - red)
    with np.errstate(invalid="ignore"):
        term_clean = np.where(nan_mask, np.nan, np.clip(term, a_min=0.0, a_max=None))
        msavi = (2.0 * nir + 1.0 - np.sqrt(term_clean)) / 2.0
    msavi = np.where(nan_mask, np.nan, msavi)
    return np.clip(msavi, -1.0, 1.5)


# ── Moisture & Water Indices ──────────────────────────────────────────────────

def calc_ndmi(nir: np.ndarray, swir1: np.ndarray) -> np.ndarray:
    """
    Normalized Difference Moisture Index: NDMI = (NIR - SWIR1) / (NIR + SWIR1)
    Gao (1996); Wilson & Sader (2002).
    Uses Sentinel-2 B8 (NIR, 842 nm) and B11 (SWIR1, 1610 nm).
    Proxy indicator associated with leaf canopy liquid water content.
    Note: Often referred to as NDWI (Gao), but designated NDMI here to distinguish
    from the McFeeters surface water index.
    """
    return safe_normalized_diff(nir, swir1)


def calc_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """
    Normalized Difference Water Index: NDWI = (Green - NIR) / (Green + NIR)
    McFeeters (1996).
    Uses Sentinel-2 B3 (Green, 560 nm) and B8 (NIR, 842 nm).
    Delineates open surface water bodies from land; generally yields negative values over vegetation.
    Clearly distinct from NIR-SWIR canopy moisture indices like NDMI.
    """
    return safe_normalized_diff(green, nir)


def calc_msi(swir1: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """
    Moisture Stress Index: MSI = SWIR1 / (NIR + eps)
    Rock et al. (1986); Hunt & Rock (1989).
    Uses Sentinel-2 B11 / B8.
    Proxy indicator associated with relative leaf water deficit (higher values indicate greater stress).
    """
    nan_mask = np.isnan(swir1) | np.isnan(nir)
    with np.errstate(divide="ignore", invalid="ignore"):
        msi = np.where(nir > 1e-5, swir1 / (nir + 1e-6), 0.0)
    msi = np.where(nan_mask, np.nan, msi)
    return np.clip(msi, 0.0, 5.0)


# ── Red Edge, Chlorophyll & Senescence Indices ────────────────────────────────

def calc_ndre1(nir: np.ndarray, red_edge1: np.ndarray) -> np.ndarray:
    """
    NDRE1 = (NIR - Red Edge 1) / (NIR + Red Edge 1)
    Barnes et al. (2000). Uses Sentinel-2 B8 and B5 (705 nm).
    Sensitive proxy indicator for early physiological stress before visible chlorosis appears.
    """
    return safe_normalized_diff(nir, red_edge1)


def calc_ndre2(nir: np.ndarray, red_edge2: np.ndarray) -> np.ndarray:
    """
    NDRE2 = (NIR - Red Edge 2) / (NIR + Red Edge 2)
    Uses Sentinel-2 B8 and B6 (740 nm).
    Indicator associated with intermediate red-edge canopy chlorophyll status.
    """
    return safe_normalized_diff(nir, red_edge2)


def calc_psri(red: np.ndarray, blue: np.ndarray, red_edge2: np.ndarray) -> np.ndarray:
    """
    Plant Senescence Reflectance Index: PSRI = (Red - Blue) / Red Edge
    Authoritative reference: Merzlyak et al. (1999) [Physiologia Plantarum 106: 135-141].
    Original definition: PSRI = (R680 - R500) / R750.
    In Sentinel-2 MSI:
        - R680 is captured by Band 4 (Red, center 665 nm)
        - R500 is captured by Band 2 (Blue, center 490 nm)
        - R750 is captured by Band 6 (Red Edge 2, center 740 nm)
    Therefore: PSRI = (B4 - B2) / B6.
    Proxy indicator associated with carotenoid-to-chlorophyll ratio, sensitive to senescence and canopy aging.
    """
    nan_mask = np.isnan(red) | np.isnan(blue) | np.isnan(red_edge2)
    with np.errstate(divide="ignore", invalid="ignore"):
        psri = np.where(np.abs(red_edge2) > 1e-6, (red - blue) / red_edge2, 0.0)
    psri = np.where(nan_mask, np.nan, psri)
    return np.clip(psri, -1.0, 1.0)


def calc_bsi(swir1: np.ndarray, red: np.ndarray, nir: np.ndarray, blue: np.ndarray) -> np.ndarray:
    """
    Bare Soil Index: BSI = ((SWIR1 + Red) - (NIR + Blue)) / ((SWIR1 + Red) + (NIR + Blue))
    Rikimaru et al. (2002).
    Proxy index separating exposed soil background from photosynthetic vegetation.
    """
    nan_mask = np.isnan(swir1) | np.isnan(red) | np.isnan(nir) | np.isnan(blue)
    num = (swir1 + red) - (nir + blue)
    denom = (swir1 + red) + (nir + blue)
    with np.errstate(divide="ignore", invalid="ignore"):
        bsi = np.where(np.abs(denom) > 1e-6, num / denom, 0.0)
    bsi = np.where(nan_mask, np.nan, bsi)
    return np.clip(bsi, -1.0, 1.0)


def calc_cire(nir: np.ndarray, red_edge1: np.ndarray) -> np.ndarray:
    """
    Chlorophyll Index Red Edge: CIRE = (NIR / Red Edge 1) - 1
    Gitelson et al. (2003). Uses Sentinel-2 (B8 / B5) - 1.
    Proxy indicator associated with leaf-level chlorophyll content.
    """
    nan_mask = np.isnan(nir) | np.isnan(red_edge1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cire = np.where(red_edge1 > 1e-5, (nir / red_edge1) - 1.0, 0.0)
    cire = np.where(nan_mask, np.nan, cire)
    return np.clip(cire, -1.0, 10.0)


def calc_cig(nir: np.ndarray, green: np.ndarray) -> np.ndarray:
    """
    Chlorophyll Index Green: CIG = (NIR / Green) - 1
    Gitelson et al. (2005). Uses Sentinel-2 (B8 / B3) - 1.
    Proxy indicator associated with total canopy chlorophyll mass.
    """
    nan_mask = np.isnan(nir) | np.isnan(green)
    with np.errstate(divide="ignore", invalid="ignore"):
        cig = np.where(green > 1e-5, (nir / green) - 1.0, 0.0)
    cig = np.where(nan_mask, np.nan, cig)
    return np.clip(cig, -1.0, 10.0)


# ── Bulk Feature Calculator ───────────────────────────────────────────────────

def compute_all_spectral_indices(
    df: pd.DataFrame,
    prefix: str = "",
    scale_factor: float = 10000.0,
    blue_col: str = "B2",
    green_col: str = "B3",
    red_col: str = "B4",
    re1_col: str = "B5",
    re2_col: str = "B6",
    re3_col: str = "B7",
    nir_col: str = "B8",
    swir1_col: str = "B11",
    swir2_col: str = "B12",
) -> pd.DataFrame:
    """
    Compute full suite of 14 spectral vegetation, moisture, and soil indices
    from a DataFrame containing Sentinel-2 band columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing band reflectance columns.
    prefix : str
        Prefix for input band columns and output index names.
    scale_factor : float, default 10000.0
        Divisor to convert raw Sentinel-2 L2A Digital Numbers (DN) to
        physical surface reflectance [0, 1]. For COPERNICUS/S2_SR_HARMONIZED
        from GEE, values are stored as integers 0-10000, so scale_factor=10000.0
        ensures additive indices (EVI, SAVI, MSAVI) receive calibrated [0, 1] reflectance.
        Set to 1.0 if band inputs are already in [0, 1] reflectance scale.
    """
    p = f"{prefix}_" if prefix else ""
    out_df = pd.DataFrame(index=df.index)

    blue = df[f"{prefix}{blue_col}" if prefix else blue_col].values.astype(float)
    green = df[f"{prefix}{green_col}" if prefix else green_col].values.astype(float)
    red = df[f"{prefix}{red_col}" if prefix else red_col].values.astype(float)
    re1 = df[f"{prefix}{re1_col}" if prefix else re1_col].values.astype(float)
    re2 = df[f"{prefix}{re2_col}" if prefix else re2_col].values.astype(float)
    nir = df[f"{prefix}{nir_col}" if prefix else nir_col].values.astype(float)
    swir1 = df[f"{prefix}{swir1_col}" if prefix else swir1_col].values.astype(float)

    if scale_factor and scale_factor > 1.0:
        blue_s = blue / scale_factor
        green_s = green / scale_factor
        red_s = red / scale_factor
        re1_s = re1 / scale_factor
        re2_s = re2 / scale_factor
        nir_s = nir / scale_factor
        swir1_s = swir1 / scale_factor
    else:
        blue_s, green_s, red_s = blue, green, red
        re1_s, re2_s, nir_s, swir1_s = re1, re2, nir, swir1

    out_df[f"{p}ndvi"] = calc_ndvi(nir_s, red_s)
    out_df[f"{p}evi"] = calc_evi(nir_s, red_s, blue_s)
    out_df[f"{p}gndvi"] = calc_gndvi(nir_s, green_s)
    out_df[f"{p}savi"] = calc_savi(nir_s, red_s)
    out_df[f"{p}msavi"] = calc_msavi(nir_s, red_s)
    out_df[f"{p}ndmi"] = calc_ndmi(nir_s, swir1_s)
    out_df[f"{p}ndwi"] = calc_ndwi(green_s, nir_s)
    out_df[f"{p}ndre1"] = calc_ndre1(nir_s, re1_s)
    out_df[f"{p}ndre2"] = calc_ndre2(nir_s, re2_s)
    out_df[f"{p}psri"] = calc_psri(red_s, blue_s, re2_s)
    out_df[f"{p}bsi"] = calc_bsi(swir1_s, red_s, nir_s, blue_s)
    out_df[f"{p}msi"] = calc_msi(swir1_s, nir_s)
    out_df[f"{p}cire"] = calc_cire(nir_s, re1_s)
    out_df[f"{p}cig"] = calc_cig(nir_s, green_s)

    return out_df


def compute_temporal_dynamics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute multi-temporal trajectory features (deltas/changes) across growth stages:
        - ndvi_growth_delta: (ndvi_mid - ndvi_early) [canopy vigor change between early and peak stages]
        - ndvi_senescence_delta: (ndvi_mid - ndvi_late) [canopy change between peak and late stages]
        - moisture_loss_delta: (ndmi_mid - ndmi_late) [canopy water index change between peak and late stages]
        - peak_to_early_ratio: ndvi_mid / max(ndvi_early, 0.05) [growth dynamic ratio]
    """
    dyn = pd.DataFrame(index=df.index)

    if "ndvi_early" in df.columns and "ndvi_mid" in df.columns:
        dyn["ndvi_growth_delta"] = df["ndvi_mid"] - df["ndvi_early"]
        dyn["peak_to_early_ratio"] = df["ndvi_mid"] / (df["ndvi_early"].clip(lower=0.05))

    if "ndvi_mid" in df.columns and "ndvi_late" in df.columns:
        dyn["ndvi_senescence_delta"] = df["ndvi_mid"] - df["ndvi_late"]

    if "ndmi_mid" in df.columns and "ndmi_late" in df.columns:
        dyn["moisture_loss_delta"] = df["ndmi_mid"] - df["ndmi_late"]

    return dyn
