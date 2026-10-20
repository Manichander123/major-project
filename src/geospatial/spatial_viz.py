"""
src/geospatial/spatial_viz.py
# -*- coding: utf-8 -*-
=============================
Spatial visualization utilities for the Telangana Crop Health Challenge.

Responsibilities:
    - Build rich, interactive Folium maps with satellite/street basemaps.
    - Render farm polygons color-coded by crop health category or crop type.
    - Provide informative HTML popups (FarmID, crop, irrigation, area, dates).
    - Add farm density heatmaps (HeatMap plugin) and district clusters.
    - Generate publication-quality static overview maps with matplotlib/cartopy style.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import folium
from folium import LayerControl, TileLayer
from folium.plugins import Fullscreen, HeatMap, MarkerCluster, MeasureControl
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# Color palettes
CATEGORY_COLORS = {
    "Healthy": "#2ecc71",   # Green
    "Pests": "#e67e22",     # Orange
    "Diseased": "#e74c3c",  # Red
    "Stressed": "#f1c40f",  # Amber/Yellow
    "Unknown": "#95a5a6",   # Grey for test set
}

CROP_COLORS = {
    "Paddy": "#1abc9c",
    "Cotton": "#3498db",
    "Maize": "#9b59b6",
    "Bengal Gram": "#e67e22",
    "Groundnut": "#d35400",
    "Chillies": "#c0392b",
    "Red Gram": "#16a085",
}

TELANGANA_CENTER = [17.8749, 78.1008]  # Lat, Lon center of Telangana


def create_interactive_farm_map(
    gdf: gpd.GeoDataFrame,
    output_html_path: Union[str, Path],
    sample_size: int = 1200,
    random_state: int = 42,
) -> folium.Map:
    """
    Create a standalone interactive Folium map of farm polygons.

    Features:
        - Multiple base maps: Esri World Imagery (satellite), CartoDB Positron, OSM.
        - Stratified sample of farms across all districts and categories.
        - Vector polygon overlays colored by crop health category.
        - Rich HTML popups with agronomic and geometric attributes.
        - HeatMap density overlay.
        - Layer control and measurement tools.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame (enriched with centroid and geometric features)
    output_html_path : path to write the HTML output
    sample_size : number of farms to plot as interactive vector polygons
    random_state : int for reproducibility

    Returns
    -------
    folium.Map instance
    """
    logger.info("Building interactive Folium map with %d sampled farms...", sample_size)

    # 1. Stratified sampling
    if len(gdf) > sample_size:
        strat_col = "category" if "category" in gdf.columns else "Crop"
        sample_gdf = gdf.groupby(strat_col, group_keys=False).apply(
            lambda x: x.sample(
                max(1, int(round(len(x) / len(gdf) * sample_size))),
                random_state=random_state,
            )
        ).reset_index(drop=True)
    else:
        sample_gdf = gdf.copy()

    # 2. Base map setup
    m = folium.Map(
        location=TELANGANA_CENTER,
        zoom_start=8,
        tiles=None,
        control_scale=True,
    )

    # Add tile layers
    TileLayer(
        tiles="CartoDB positron",
        name="Clean Light (CartoDB)",
        control=True,
    ).add_to(m)

    TileLayer(
        tiles="OpenStreetMap",
        name="OpenStreetMap",
        control=True,
    ).add_to(m)

    TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Satellite (Esri Imagery)",
        control=True,
    ).add_to(m)

    # 3. Farm polygons feature group
    fg_polygons = folium.FeatureGroup(name="Farm Boundaries (by Health Category)", show=True)

    for _, row in sample_gdf.iterrows():
        cat = row.get("category", "Unknown")
        color = CATEGORY_COLORS.get(cat, "#34495e")
        crop = row.get("Crop", "N/A")
        district = row.get("District", "N/A")
        sub_dist = row.get("Sub-District", "N/A")
        farm_id = row.get("FarmID", "N/A")
        area_ha = row.get("area_ha", 0.0)
        area_m2 = row.get("area_m2", 0.0)
        irri_type = row.get("IrriType", "N/A")
        irri_source = row.get("IrriSource", "N/A")
        irri_count = row.get("IrriCount", "N/A")
        water_cov = row.get("WaterCov", "N/A")
        sdate = str(row.get("SDate", "N/A"))[:10]
        hdate = str(row.get("HDate", "N/A"))[:10]
        season = row.get("Season", "N/A")
        exp_yield = row.get("ExpYield", "N/A")

        popup_html = f"""
        <div style="font-family: Arial, sans-serif; min-width: 200px; font-size: 12px; line-height: 1.4;">
            <h4 style="margin: 0 0 6px 0; color: #2c3e50; border-bottom: 2px solid {color}; padding-bottom: 3px;">
                Farm #{farm_id} — <span style="color: {color};">{cat}</span>
            </h4>
            <table style="width: 100%; border-collapse: collapse;">
                <tr><td><b>Crop:</b></td><td>{crop}</td></tr>
                <tr><td><b>Season:</b></td><td>{season}</td></tr>
                <tr><td><b>District:</b></td><td>{district} ({sub_dist})</td></tr>
                <tr><td><b>Area:</b></td><td>{area_ha:.3f} ha ({area_m2:,.0f} m²)</td></tr>
                <tr><td><b>Sowing:</b></td><td>{sdate}</td></tr>
                <tr><td><b>Harvest:</b></td><td>{hdate}</td></tr>
                <tr><td><b>Irrigation:</b></td><td>{irri_type} / {irri_source}</td></tr>
                <tr><td><b>Irri Count:</b></td><td>{irri_count} rounds</td></tr>
                <tr><td><b>Water Cov:</b></td><td>{water_cov}%</td></tr>
                <tr><td><b>Exp. Yield:</b></td><td>{exp_yield} q/acre</td></tr>
            </table>
        </div>
        """

        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        if geom.geom_type == "Polygon":
            coords = [[pt[1], pt[0]] for pt in geom.exterior.coords]
            folium.Polygon(
                locations=coords,
                color=color,
                weight=2,
                fill=True,
                fill_color=color,
                fill_opacity=0.45,
                tooltip=f"Farm #{farm_id} | {crop} ({cat})",
                popup=folium.Popup(popup_html, max_width=320),
            ).add_to(fg_polygons)
        elif geom.geom_type == "MultiPolygon":
            for poly in geom.geoms:
                coords = [[pt[1], pt[0]] for pt in poly.exterior.coords]
                folium.Polygon(
                    locations=coords,
                    color=color,
                    weight=2,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.45,
                    tooltip=f"Farm #{farm_id} | {crop} ({cat})",
                    popup=folium.Popup(popup_html, max_width=320),
                ).add_to(fg_polygons)

    fg_polygons.add_to(m)

    # 4. HeatMap Layer (Farm Density)
    valid_centroids = gdf.dropna(subset=["centroid_lat", "centroid_lon"])
    heat_data = [[row["centroid_lat"], row["centroid_lon"], 1.0] for _, row in valid_centroids.iterrows()]
    fg_heatmap = folium.FeatureGroup(name="Farm Density Heatmap", show=False)
    HeatMap(
        heat_data,
        radius=14,
        blur=18,
        max_zoom=10,
        gradient={0.2: "blue", 0.4: "lime", 0.7: "orange", 1.0: "red"},
    ).add_to(fg_heatmap)
    fg_heatmap.add_to(m)

    # 5. District Cluster Markers
    fg_districts = folium.FeatureGroup(name="District Clusters", show=False)
    district_summary = (
        gdf.groupby("District")
        .agg(
            lat=("centroid_lat", "mean"),
            lon=("centroid_lon", "mean"),
            count=("FarmID", "count"),
        )
        .reset_index()
    )

    for _, row in district_summary.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=min(30, max(8, int(np.sqrt(row["count"]) * 0.8))),
            color="#2c3e50",
            fill=True,
            fill_color="#3498db",
            fill_opacity=0.7,
            tooltip=f"{row['District']}: {row['count']} farms",
            popup=f"<b>{row['District']} District</b><br>Total Farms: {row['count']:,}",
        ).add_to(fg_districts)
    fg_districts.add_to(m)

    # 6. Controls
    Fullscreen(position="topleft").add_to(m)
    MeasureControl(position="bottomleft", primary_length_unit="meters", primary_area_unit="hectares").add_to(m)
    LayerControl(position="topright", collapsed=False).add_to(m)

    # Save to disk
    output_path = Path(output_html_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_path))
    logger.info("Interactive map saved successfully -> %s", output_path)

    return m


def generate_static_spatial_figure(
    train_gdf: gpd.GeoDataFrame,
    test_gdf: gpd.GeoDataFrame,
    output_png_path: Union[str, Path],
) -> None:
    """
    Generate a high-resolution 4-panel spatial diagnostic figure.

    Panels:
        1. Telangana Regional Farm Distribution (Train vs Test).
        2. Farm Health Status Spatial Breakdown (Train).
        3. Dominant Crop Geography across Districts.
        4. Irrigation Source Distribution Map.
    """
    logger.info("Generating static 4-panel spatial figure...")
    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    fig.suptitle(
        "Telangana Crop Health Challenge — Spatial Farm Geography & Distributions",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    # Panel 1: Train vs Test Centroid Distribution
    ax = axes[0, 0]
    ax.scatter(
        train_gdf["centroid_lon"],
        train_gdf["centroid_lat"],
        c="#2980b9",
        s=12,
        alpha=0.5,
        label=f"Train ({len(train_gdf):,} farms)",
    )
    ax.scatter(
        test_gdf["centroid_lon"],
        test_gdf["centroid_lat"],
        c="#e67e22",
        s=12,
        alpha=0.5,
        label=f"Test ({len(test_gdf):,} farms)",
    )
    ax.set_title("1. Farm Locations across Telangana (Train vs Test)", fontweight="bold")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.legend(loc="upper right", frameon=True)

    # Panel 2: Health Status Spatial Distribution
    ax = axes[0, 1]
    for cat, color in CATEGORY_COLORS.items():
        if cat == "Unknown":
            continue
        subset = train_gdf[train_gdf["category"] == cat]
        ax.scatter(
            subset["centroid_lon"],
            subset["centroid_lat"],
            c=color,
            s=16,
            alpha=0.6,
            label=f"{cat} ({len(subset):,})",
        )
    ax.set_title("2. Crop Health Category Spatial Distribution", fontweight="bold")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.legend(loc="upper right", frameon=True)

    # Panel 3: Crop Distribution by District
    ax = axes[1, 0]
    for crop, color in CROP_COLORS.items():
        subset = train_gdf[train_gdf["Crop"] == crop]
        ax.scatter(
            subset["centroid_lon"],
            subset["centroid_lat"],
            c=color,
            s=14,
            alpha=0.5,
            label=f"{crop} ({len(subset):,})",
        )
    ax.set_title("3. Geographic Distribution by Crop Type", fontweight="bold")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.legend(loc="upper right", frameon=True, fontsize=8)

    # Panel 4: Irrigation Source Spatial Map
    ax = axes[1, 1]
    irri_colors = {
        "Groundwater": "#16a085",
        "Canal": "#2980b9",
        "Rainfed": "#d35400",
        "Well": "#8e44ad",
        "Tubewell": "#27ae60",
    }
    for source, color in irri_colors.items():
        subset = train_gdf[train_gdf["IrriSource"] == source]
        ax.scatter(
            subset["centroid_lon"],
            subset["centroid_lat"],
            c=color,
            s=14,
            alpha=0.5,
            label=f"{source} ({len(subset):,})",
        )
    ax.set_title("4. Irrigation Water Source Distribution", fontweight="bold")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    output_path = Path(output_png_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Saved static spatial overview figure -> %s", output_path)
