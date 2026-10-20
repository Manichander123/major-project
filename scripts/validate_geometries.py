"""
scripts/validate_geometries.py
# -*- coding: utf-8 -*-
===============================
Phase 2 — Farm Geometry Parsing, Repair & Validation Script
Telangana Crop Health Challenge

Responsibilities:
    - Load Train and Test geospatial datasets.
    - Check geometry validity (detect self-intersections or bowtie anomalies).
    - Automatically repair invalid geometries using shapely.make_valid.
    - Compute geometric features (area ha/m², perimeter m, compactness, centroids, bounds, vertex count).
    - Validate coordinates fall within Telangana geographical bounds.
    - Generate summary statistics reports (JSON and TXT).
    - Generate diagnostic figures (area, perimeter, compactness, vertices, spatial distribution).
    - Save enriched datasets to outputs/data/ for downstream modeling and satellite extraction.

Usage:
    python scripts/validate_geometries.py
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys

# Force UTF-8 stdout on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_data import load_train, load_test
from src.geospatial.geometry_utils import (
    add_geometry_features,
    geometry_stats,
    repair_invalid_geometries,
    check_geometry_validity,
    validate_within_telangana,
    CRS_METRIC,
    CRS_GEO,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

FIGS_DIR = PROJECT_ROOT / "outputs" / "figures"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
DATA_OUT_DIR = PROJECT_ROOT / "outputs" / "data"
FIGS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_OUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="darkgrid", palette="muted")


def run_geometry_pipeline():
    print("=" * 65)
    print("  PHASE 2: FARM GEOMETRY PARSING, REPAIR & VALIDATION")
    print("=" * 65)

    # 1. Load raw data as GeoDataFrames
    logger.info("Loading Train and Test datasets...")
    train_gdf = load_train()
    test_gdf = load_test()

    logger.info("Train rows: %d | Test rows: %d", len(train_gdf), len(test_gdf))

    # 2. Check initial validity
    train_initial_valid = check_geometry_validity(train_gdf).sum()
    test_initial_valid = check_geometry_validity(test_gdf).sum()
    train_invalid_count = len(train_gdf) - train_initial_valid
    test_invalid_count = len(test_gdf) - test_initial_valid

    logger.info("Train initial valid: %d / %d (invalid: %d)", train_initial_valid, len(train_gdf), train_invalid_count)
    logger.info("Test initial valid:  %d / %d (invalid: %d)", test_initial_valid, len(test_gdf), test_invalid_count)

    # 3. Add geometry features with automated repair
    logger.info("Enriching Train geometry features...")
    train_enriched = add_geometry_features(train_gdf, repair=True)
    logger.info("Enriching Test geometry features...")
    test_enriched = add_geometry_features(test_gdf, repair=True)

    # 4. Check post-repair validity
    train_post_valid = train_enriched["is_valid_geom"].sum()
    test_post_valid = test_enriched["is_valid_geom"].sum()

    logger.info("Train post-repair valid: %d / %d", train_post_valid, len(train_enriched))
    logger.info("Test post-repair valid:  %d / %d", test_post_valid, len(test_enriched))

    # 5. Compute stats
    train_stats = geometry_stats(train_enriched)
    test_stats = geometry_stats(test_enriched)

    print("\n------------------------------------------------------------")
    print("  TRAIN GEOMETRY SUMMARY")
    print("------------------------------------------------------------")
    for k, v in train_stats.items():
        print(f"  {k:30s}: {v}")

    print("\n------------------------------------------------------------")
    print("  TEST GEOMETRY SUMMARY")
    print("------------------------------------------------------------")
    for k, v in test_stats.items():
        print(f"  {k:30s}: {v}")

    # 6. Save reports
    report_dict = {
        "train": {
            "initial_invalid": int(train_invalid_count),
            "post_repair_valid": int(train_post_valid),
            **train_stats,
        },
        "test": {
            "initial_invalid": int(test_invalid_count),
            "post_repair_valid": int(test_post_valid),
            **test_stats,
        },
    }

    json_path = REPORTS_DIR / "geometry_validation_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)
    logger.info("Saved JSON report -> %s", json_path)

    txt_path = REPORTS_DIR / "geometry_validation_report.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 65 + "\n")
        f.write("  GEOMETRY VALIDATION & SUMMARY REPORT\n")
        f.write("=" * 65 + "\n\n")
        f.write("[TRAIN DATASET]\n")
        f.write(f"  Total farms             : {len(train_gdf):,}\n")
        f.write(f"  Initial invalid         : {train_invalid_count}\n")
        f.write(f"  Post-repair valid       : {train_post_valid:,}\n")
        f.write(f"  Area (ha) [min/mean/max]: {train_stats['area_ha_min']:.4f} / {train_stats['area_ha_mean']:.4f} / {train_stats['area_ha_max']:.4f}\n")
        f.write(f"  Area (m²) [mean/median] : {train_stats['area_ha_mean']*10000:.1f} / {train_stats['area_ha_median']*10000:.1f}\n")
        f.write(f"  Perimeter (m) [mean]    : {train_stats['perimeter_m_mean']:.1f}\n")
        f.write(f"  Compactness [mean]      : {train_stats['compactness_mean']:.4f}\n")
        f.write(f"  Vertex count [mean]     : {train_stats['vertex_count_mean']:.2f}\n")
        f.write(f"  Within Telangana bbox   : {train_stats['n_within_telangana_bbox']:,}\n\n")
        f.write("[TEST DATASET]\n")
        f.write(f"  Total farms             : {len(test_gdf):,}\n")
        f.write(f"  Initial invalid         : {test_invalid_count}\n")
        f.write(f"  Post-repair valid       : {test_post_valid:,}\n")
        f.write(f"  Area (ha) [min/mean/max]: {test_stats['area_ha_min']:.4f} / {test_stats['area_ha_mean']:.4f} / {test_stats['area_ha_max']:.4f}\n")
        f.write(f"  Area (m²) [mean/median] : {test_stats['area_ha_mean']*10000:.1f} / {test_stats['area_ha_median']*10000:.1f}\n")
        f.write(f"  Perimeter (m) [mean]    : {test_stats['perimeter_m_mean']:.1f}\n")
        f.write(f"  Compactness [mean]      : {test_stats['compactness_mean']:.4f}\n")
        f.write(f"  Vertex count [mean]     : {test_stats['vertex_count_mean']:.2f}\n")
        f.write(f"  Within Telangana bbox   : {test_stats['n_within_telangana_bbox']:,}\n")
    logger.info("Saved TXT report -> %s", txt_path)

    # 7. Generate diagnostic figures
    logger.info("Generating geometry diagnostic figures...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Telangana Crop Health Challenge — Farm Geometry Diagnostics", fontsize=15, fontweight="bold", y=0.98)

    # (a) Area Distribution (ha) - Log scale comparison
    ax = axes[0, 0]
    sns.histplot(train_enriched["area_ha"], bins=40, kde=True, color="#2980b9", ax=ax, label="Train", alpha=0.6)
    sns.histplot(test_enriched["area_ha"], bins=40, kde=True, color="#e67e22", ax=ax, label="Test", alpha=0.6)
    ax.set_title("Farm Area Distribution (Hectares)", fontweight="bold")
    ax.set_xlabel("Area (ha)")
    ax.set_ylabel("Count")
    ax.legend()

    # (b) Area by Crop type
    ax = axes[0, 1]
    crop_order = train_enriched.groupby("Crop")["area_ha"].median().sort_values(ascending=False).index
    sns.boxplot(data=train_enriched, x="Crop", y="area_ha", order=crop_order, palette="viridis", ax=ax, showfliers=False)
    ax.set_title("Farm Area (ha) by Crop Type", fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Area (ha)")
    ax.tick_params(axis="x", rotation=30)

    # (c) Perimeter vs Area Scatter
    ax = axes[0, 2]
    sample = train_enriched.sample(min(1500, len(train_enriched)), random_state=42)
    sns.scatterplot(
        data=sample,
        x="area_m2",
        y="perimeter_m",
        hue="Crop",
        palette="tab10",
        alpha=0.6,
        s=25,
        ax=ax,
    )
    ax.set_title("Perimeter vs Area (1,500 Farm Sample)", fontweight="bold")
    ax.set_xlabel("Area (m²)")
    ax.set_ylabel("Perimeter (m)")
    ax.legend(title="Crop", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)

    # (d) Compactness Distribution
    ax = axes[1, 0]
    sns.kdeplot(train_enriched["compactness"], color="#27ae60", fill=True, alpha=0.4, label="Train", ax=ax)
    sns.kdeplot(test_enriched["compactness"], color="#8e44ad", fill=True, alpha=0.4, label="Test", ax=ax)
    ax.axvline(train_enriched["compactness"].mean(), color="#27ae60", linestyle="--", label=f"Train mean ({train_enriched['compactness'].mean():.2f})")
    ax.set_title("Compactness (Isoperimetric Quotient)", fontweight="bold")
    ax.set_xlabel("Compactness (0=elongated, 1=circular)")
    ax.set_ylabel("Density")
    ax.legend(loc="upper left")

    # (e) Vertex Count Distribution
    ax = axes[1, 1]
    vc = train_enriched["vertex_count"].value_counts().sort_index().head(12)
    ax.bar(vc.index.astype(str), vc.values, color="#34495e", edgecolor="white")
    ax.set_title("Top Polygon Vertex Counts (Train)", fontweight="bold")
    ax.set_xlabel("Number of Boundary Vertices")
    ax.set_ylabel("Farm Count")
    for i, (idx, val) in enumerate(vc.items()):
        ax.text(i, val + 50, f"{val}", ha="center", fontsize=8)

    # (f) Geographic Spatial Distribution (Centroids)
    ax = axes[1, 2]
    sns.scatterplot(
        data=sample,
        x="centroid_lon",
        y="centroid_lat",
        hue="District",
        palette="tab20",
        alpha=0.7,
        s=30,
        ax=ax,
    )
    ax.set_title("Geographic Centroids by District", fontweight="bold")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.legend(title="District", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)

    plt.tight_layout()
    fig_path = FIGS_DIR / "geometry_diagnostics.png"
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Saved geometry diagnostics figure -> %s", fig_path)

    # 8. Save enriched data for downstream usage
    train_out_csv = DATA_OUT_DIR / "train_enriched_geometry.csv"
    test_out_csv = DATA_OUT_DIR / "test_enriched_geometry.csv"

    # Save CSV with geometry as WKT string
    train_enriched_df = pd.DataFrame(train_enriched.copy())
    train_enriched_df["geometry"] = train_enriched["geometry"].apply(lambda g: g.wkt if g else "")
    train_enriched_df.to_csv(train_out_csv, index=False)

    test_enriched_df = pd.DataFrame(test_enriched.copy())
    test_enriched_df["geometry"] = test_enriched["geometry"].apply(lambda g: g.wkt if g else "")
    test_enriched_df.to_csv(test_out_csv, index=False)

    logger.info("Saved enriched train data -> %s (%d rows)", train_out_csv, len(train_enriched_df))
    logger.info("Saved enriched test data  -> %s (%d rows)", test_out_csv, len(test_enriched_df))

    print("\n" + "=" * 65)
    print("  ✓  Phase 2 geometry pipeline complete!")
    print(f"  Diagnostics saved to : {fig_path}")
    print(f"  Reports saved to     : {REPORTS_DIR}")
    print(f"  Enriched datasets    : {DATA_OUT_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    run_geometry_pipeline()
