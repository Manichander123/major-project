"""
scripts/visualize_spatial.py
# -*- coding: utf-8 -*-
============================
Phase 3 — Farm Spatial Visualisation Runner
Telangana Crop Health Challenge

Generates:
    1. outputs/figures/telangana_farms_interactive_map.html
       (Multi-layer Folium map with satellite/street basemaps, farm vector polygons,
        crop health popups, density heatmaps, and district cluster counts).
    2. outputs/figures/telangana_spatial_overview.png
       (4-panel static high-res overview map of crops, health categories, and irrigation).

Usage:
    python scripts/visualize_spatial.py
"""

from __future__ import annotations

import io
import logging
import os
import sys

# Force UTF-8 stdout on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from pathlib import Path
import geopandas as gpd
import pandas as pd
from shapely import wkt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_data import load_train, load_test
from src.geospatial.geometry_utils import add_geometry_features
from src.geospatial.spatial_viz import (
    create_interactive_farm_map,
    generate_static_spatial_figure,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

FIGS_DIR = PROJECT_ROOT / "outputs" / "figures"
DATA_DIR = PROJECT_ROOT / "outputs" / "data"
FIGS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 65)
    print("  PHASE 3: FARM SPATIAL VISUALISATION")
    print("=" * 65)

    # Check if enriched files exist, otherwise compute
    train_enriched_csv = DATA_DIR / "train_enriched_geometry.csv"
    test_enriched_csv = DATA_DIR / "test_enriched_geometry.csv"

    if train_enriched_csv.exists() and test_enriched_csv.exists():
        logger.info("Loading precomputed enriched datasets from %s...", DATA_DIR)
        train_df = pd.read_csv(train_enriched_csv)
        train_df["geometry"] = train_df["geometry"].apply(wkt.loads)
        train_gdf = gpd.GeoDataFrame(train_df, crs="EPSG:4326")

        test_df = pd.read_csv(test_enriched_csv)
        test_df["geometry"] = test_df["geometry"].apply(wkt.loads)
        test_gdf = gpd.GeoDataFrame(test_df, crs="EPSG:4326")
    else:
        logger.info("Enriched data not found, loading raw datasets...")
        train_gdf = add_geometry_features(load_train(), repair=True)
        test_gdf = add_geometry_features(load_test(), repair=True)

    # 1. Interactive Folium map
    html_out = FIGS_DIR / "telangana_farms_interactive_map.html"
    logger.info("Generating interactive Folium map -> %s", html_out)
    create_interactive_farm_map(
        gdf=train_gdf,
        output_html_path=html_out,
        sample_size=1200,
        random_state=42,
    )

    # 2. Static 4-panel figure
    png_out = FIGS_DIR / "telangana_spatial_overview.png"
    logger.info("Generating static spatial overview figure -> %s", png_out)
    generate_static_spatial_figure(
        train_gdf=train_gdf,
        test_gdf=test_gdf,
        output_png_path=png_out,
    )

    print("\n" + "=" * 65)
    print("  ✓  Phase 3 spatial visualisation complete!")
    print(f"  Interactive Map : {html_out}")
    print(f"  Static Figure   : {png_out}")
    print("=" * 65)


if __name__ == "__main__":
    main()
