"""
src/geospatial/geometry_utils.py
=================================
Geometry utility functions for the Telangana Crop Health Challenge.

Responsibilities:
    - Parse WKT POLYGON strings safely.
    - Check and repair invalid geometries (using shapely.make_valid).
    - Compute polygon areas in both geographic and projected units.
    - Compute polygon perimeters and compactness indices.
    - Compute polygon centroids (lon/lat) and bounding box extents.
    - Project geometries to EPSG:32644 (UTM Zone 44N covers Telangana) —
      a metric projected CRS suitable for local area and distance calculations.
    - Support geometry scaling (around centroid) as in the official starter notebook.
    - Generate bounding-box coordinates for Earth Engine queries.
    - Summarise farm geometry statistics.
    - Perform a regional sanity check ensuring all polygons fall within the Telangana regional bounding box.

CRS Notes:
    - Input data: EPSG:4326 (WGS84 geographic) — coordinates in decimal degrees.
    - Metric projection: EPSG:32644 (WGS84 / UTM Zone 44N) — covers Telangana.
      Metric projected CRS suitable for local area and distance calculations (area in square metres).

DO NOT:
    - Modify raw geometry strings in datasss/.
    - Fabricate satellite data.
    - Guess CRS — EPSG:4326 is confirmed from the official challenge data.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import affinity, make_valid
from shapely import wkt as shp_wkt
from shapely.errors import ShapelyError
from shapely.geometry import box, mapping, MultiPolygon, Polygon

logger = logging.getLogger(__name__)

# ── CRS constants ────────────────────────────────────────────────────────────
CRS_GEO: str = "EPSG:4326"       # Geographic WGS84 — matches raw data
CRS_METRIC: str = "EPSG:32644"   # WGS84 / UTM Zone 44N — for area & perimeter in metres

# ── Telangana approximate bounding box (WGS84) ───────────────────────────────
TELANGANA_BBOX: Tuple[float, float, float, float] = (
    76.75,  # min_lon (west)
    15.70,  # min_lat (south)
    81.50,  # max_lon (east)
    20.00,  # max_lat (north)
)


# ── Parsing helpers ──────────────────────────────────────────────────────────

def parse_wkt(wkt_string: str) -> Optional[Polygon | MultiPolygon]:
    """
    Safely parse a WKT geometry string.

    Parameters
    ----------
    wkt_string : str
        A WKT string (expected: POLYGON ((lon lat, ...))).

    Returns
    -------
    shapely geometry or None if parsing fails.
    """
    try:
        return shp_wkt.loads(str(wkt_string).strip())
    except (ShapelyError, ValueError, TypeError) as exc:
        logger.debug("WKT parse failure: %s... | %s", str(wkt_string)[:60], exc)
        return None


def parse_wkt_series(series: pd.Series) -> pd.Series:
    """
    Parse an entire WKT Series to Shapely geometry objects.

    Parameters
    ----------
    series : pd.Series
        Series of WKT strings.

    Returns
    -------
    pd.Series of Shapely geometries; invalid rows -> None.
    """
    return series.apply(parse_wkt)


# ── Validity and Repair ───────────────────────────────────────────────────────

def check_geometry_validity(gdf: gpd.GeoDataFrame) -> pd.Series:
    """
    Return a boolean Series: True if geometry is valid (Shapely is_valid).
    """
    return gdf.geometry.is_valid


def _clean_repaired_geom(geom):
    """Ensure geometry repaired by make_valid retains only polygonal parts."""
    if geom is None or geom.is_empty:
        return geom
    if not geom.is_valid:
        geom = make_valid(geom)
    if geom.geom_type in ("Polygon", "MultiPolygon"):
        return geom
    elif geom.geom_type == "GeometryCollection":
        polys = [g for g in geom.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
        if len(polys) == 1:
            return polys[0]
        elif len(polys) > 1:
            from shapely.ops import unary_union
            return unary_union(polys)
    return geom


def repair_invalid_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Fix any self-intersecting or non-simple geometries using shapely.make_valid.
    Filters any non-polygonal components (points, lines) from GeometryCollections
    to guarantee strictly Polygon or MultiPolygon output.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame

    Returns
    -------
    gpd.GeoDataFrame with repaired geometries.
    """
    gdf = gdf.copy()
    invalid_mask = ~gdf.geometry.is_valid
    n_invalid = invalid_mask.sum()
    if n_invalid > 0:
        logger.info("Repairing %d invalid geometries using make_valid...", n_invalid)
        repaired = gdf.loc[invalid_mask, "geometry"].apply(_clean_repaired_geom)
        gdf.loc[invalid_mask, "geometry"] = repaired
    return gdf


# ── Geometry Transformation & Scaling ─────────────────────────────────────────

def scale_geometry(geom: Polygon | MultiPolygon, scale_factor: float = 5.0) -> Polygon | MultiPolygon:
    """
    Scale a geometry around its centroid.
    Compatible with the official starter notebook for GEE imagery query expansion.

    Parameters
    ----------
    geom : shapely geometry (Polygon or MultiPolygon)
    scale_factor : float, default 5.0

    Returns
    -------
    Scaled shapely geometry
    """
    if geom is None or geom.is_empty:
        return geom
    centroid = geom.centroid
    translated = affinity.translate(geom, -centroid.x, -centroid.y)
    scaled = affinity.scale(translated, xfact=scale_factor, yfact=scale_factor, origin=(0, 0))
    return affinity.translate(scaled, centroid.x, centroid.y)


# ── Projection helpers ───────────────────────────────────────────────────────

def to_metric_gdf(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Re-project a GeoDataFrame to the metric CRS (EPSG:32644 UTM 44N).

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Must have CRS set.

    Returns
    -------
    gpd.GeoDataFrame
        New GeoDataFrame in EPSG:32644.
    """
    if gdf.crs is None:
        raise ValueError("GeoDataFrame has no CRS set. Set crs='EPSG:4326' first.")
    return gdf.to_crs(CRS_METRIC)


def compute_area_m2(gdf: gpd.GeoDataFrame) -> pd.Series:
    """
    Compute polygon area in square metres using EPSG:32644 (UTM Zone 44N).
    """
    metric_gdf = to_metric_gdf(gdf)
    return metric_gdf.geometry.area


def compute_area_ha(gdf: gpd.GeoDataFrame) -> pd.Series:
    """
    Compute polygon area in hectares (1 ha = 10,000 m²).
    """
    return compute_area_m2(gdf) / 10_000.0


def compute_perimeter_m(gdf: gpd.GeoDataFrame) -> pd.Series:
    """
    Compute polygon perimeter in metres using EPSG:32644 (UTM Zone 44N).
    """
    metric_gdf = to_metric_gdf(gdf)
    return metric_gdf.geometry.length


def compute_compactness(gdf: gpd.GeoDataFrame) -> pd.Series:
    """
    Compute the isoperimetric quotient (Polsby-Popper compactness):
        compactness = 4 * pi * Area / (Perimeter ^ 2)
    Values range from 0 (highly irregular/elongated) to 1 (perfect circle).
    """
    metric_gdf = to_metric_gdf(gdf)
    area = metric_gdf.geometry.area
    perimeter = metric_gdf.geometry.length
    compactness = (4.0 * math.pi * area) / (perimeter ** 2 + 1e-8)
    return compactness.clip(lower=0.0, upper=1.0)


def compute_centroids(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Compute polygon centroid coordinates (longitude, latitude) accurately
    by projecting to metric CRS (EPSG:32644) and re-projecting centroid points to WGS84.
    """
    metric = to_metric_gdf(gdf)
    metric_centroids = metric.geometry.centroid
    geo_centroids = metric_centroids.to_crs(CRS_GEO)
    return pd.DataFrame({
        "centroid_lon": geo_centroids.x,
        "centroid_lat": geo_centroids.y,
    }, index=gdf.index)


def compute_bounding_boxes(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Compute bounding box coordinates (min_lon, min_lat, max_lon, max_lat) in WGS84.
    """
    geo = gdf if gdf.crs and gdf.crs.to_epsg() == 4326 else gdf.to_crs(CRS_GEO)
    bounds = geo.geometry.bounds
    bounds.columns = ["bbox_min_lon", "bbox_min_lat", "bbox_max_lon", "bbox_max_lat"]
    return bounds


# ── Bounding box helpers for GEE ─────────────────────────────────────────────

def geometry_to_ee_bbox(geometry) -> List[float]:
    """
    Return the bounding box of a Shapely geometry as [min_lon, min_lat, max_lon, max_lat].
    """
    b = geometry.bounds
    return [b[0], b[1], b[2], b[3]]


def geometry_to_geojson(geometry) -> dict:
    """
    Convert a Shapely geometry to a GeoJSON-compatible dict.
    """
    return mapping(geometry)


# ── Spatial validation ───────────────────────────────────────────────────────

def validate_within_telangana(gdf: gpd.GeoDataFrame) -> pd.Series:
    """
    Perform a regional sanity check verifying that each farm polygon's centroid
    falls within the approximate Telangana bounding box (76.75°E - 81.50°E, 15.70°N - 20.00°N).

    Note: This is a regional geographic sanity check to detect coordinate-swap
    or gross geo-referencing errors, not an exact administrative boundary test.
    """
    min_lon, min_lat, max_lon, max_lat = TELANGANA_BBOX
    centroids = compute_centroids(gdf)
    cx = centroids["centroid_lon"]
    cy = centroids["centroid_lat"]
    return (cx >= min_lon) & (cx <= max_lon) & (cy >= min_lat) & (cy <= max_lat)


# ── Summary statistics ───────────────────────────────────────────────────────

def geometry_stats(gdf: gpd.GeoDataFrame) -> Dict[str, Any]:
    """
    Compute a comprehensive summary of farm polygon geometry statistics.
    """
    n_total = len(gdf)
    valid_mask = check_geometry_validity(gdf)
    n_valid = int(valid_mask.sum())

    areas_ha = compute_area_ha(gdf)
    perimeters_m = compute_perimeter_m(gdf)
    compactness = compute_compactness(gdf)
    within = validate_within_telangana(gdf)

    vertex_counts = gdf.geometry.apply(
        lambda g: len(g.exterior.coords) if g and hasattr(g, "exterior") and g.exterior else np.nan
    )

    return {
        "n_total": n_total,
        "n_valid_geom": n_valid,
        "n_invalid_geom": n_total - n_valid,
        "n_within_telangana_bbox": int(within.sum()),
        "n_outside_telangana_bbox": int((~within).sum()),
        "area_ha_min": round(float(areas_ha.min()), 4),
        "area_ha_max": round(float(areas_ha.max()), 4),
        "area_ha_mean": round(float(areas_ha.mean()), 4),
        "area_ha_median": round(float(areas_ha.median()), 4),
        "area_m2_min": round(float((areas_ha * 10_000).min()), 1),
        "area_m2_max": round(float((areas_ha * 10_000).max()), 1),
        "perimeter_m_min": round(float(perimeters_m.min()), 1),
        "perimeter_m_max": round(float(perimeters_m.max()), 1),
        "perimeter_m_mean": round(float(perimeters_m.mean()), 1),
        "compactness_mean": round(float(compactness.mean()), 4),
        "compactness_median": round(float(compactness.median()), 4),
        "vertex_count_min": float(vertex_counts.min()),
        "vertex_count_max": float(vertex_counts.max()),
        "vertex_count_mean": round(float(vertex_counts.mean()), 2),
        "crs_input": str(gdf.crs),
        "crs_metric": CRS_METRIC,
    }


def add_geometry_features(gdf: gpd.GeoDataFrame, repair: bool = True) -> gpd.GeoDataFrame:
    """
    Add computed geometry columns to a GeoDataFrame.

    Adds:
        - area_ha : farm area in hectares
        - area_m2 : farm area in square metres
        - perimeter_m : perimeter in metres
        - compactness : Polsby-Popper compactness (0 to 1)
        - centroid_lon, centroid_lat : centroid coordinates in WGS84
        - bbox_min_lon, bbox_min_lat, bbox_max_lon, bbox_max_lat
        - vertex_count : number of vertices
        - is_valid_geom : boolean validity check

    Parameters
    ----------
    gdf : gpd.GeoDataFrame (in EPSG:4326)
    repair : bool, default True
        If True, fixes any invalid geometries with make_valid.

    Returns
    -------
    gpd.GeoDataFrame with new columns.
    """
    gdf = gdf.copy()
    if repair:
        gdf = repair_invalid_geometries(gdf)

    areas_ha = compute_area_ha(gdf)
    perimeters_m = compute_perimeter_m(gdf)
    compactness = compute_compactness(gdf)
    centroids = compute_centroids(gdf)
    bboxes = compute_bounding_boxes(gdf)

    gdf["area_ha"] = areas_ha
    gdf["area_m2"] = areas_ha * 10_000.0
    gdf["perimeter_m"] = perimeters_m
    gdf["compactness"] = compactness
    gdf["centroid_lon"] = centroids["centroid_lon"]
    gdf["centroid_lat"] = centroids["centroid_lat"]
    gdf["bbox_min_lon"] = bboxes["bbox_min_lon"]
    gdf["bbox_min_lat"] = bboxes["bbox_min_lat"]
    gdf["bbox_max_lon"] = bboxes["bbox_max_lon"]
    gdf["bbox_max_lat"] = bboxes["bbox_max_lat"]
    gdf["vertex_count"] = gdf.geometry.apply(
        lambda g: len(g.exterior.coords) if g and hasattr(g, "exterior") and g.exterior else 0
    )
    gdf["is_valid_geom"] = check_geometry_validity(gdf)

    return gdf
