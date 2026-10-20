"""
scripts/analyze_dataset.py
# -*- coding: utf-8 -*-
===========================
Phase 1 — Dataset Analysis Script
Telangana Crop Health Challenge

Run this script to produce a complete data-quality report and visualisations
without touching the raw data files.

Usage:
    python scripts/analyze_dataset.py

Outputs (written to outputs/figures/ and outputs/reports/):
    - data_quality_report.txt / .json
    - crop_distribution.png
    - category_distribution.png
    - season_district_heatmap.png
    - date_range_plot.png
    - farm_area_distribution.png
    - farm_boundaries_map.png   (static map of a sample of farms)

Requirements: geopandas, matplotlib, seaborn are the only required extras.
              matplotlib's backend is set to 'Agg' for headless execution.
"""

from __future__ import annotations

import io
import logging
import os
import sys
# Force UTF-8 stdout to avoid cp1252 UnicodeEncodeError on Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless -- no GUI window needed

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

# ── Resolve project root and configure imports ───────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_data import load_train, load_test
from src.data.validate_data import validate
from src.geospatial.geometry_utils import (
    add_geometry_features,
    geometry_stats,
    validate_within_telangana,
)

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Output directories ────────────────────────────────────────────────────────
FIGS_DIR = PROJECT_ROOT / "outputs" / "figures"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
FIGS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Plotting style ────────────────────────────────────────────────────────────
sns.set_theme(style="darkgrid", palette="muted")
PALETTE = {
    "Healthy": "#2ecc71",
    "Diseased": "#e74c3c",
    "Pests": "#e67e22",
    "Stressed": "#3498db",
}
CROP_COLORS = sns.color_palette("tab10", 7)


# ════════════════════════════════════════════════════════════════════════════════
# Section helpers
# ════════════════════════════════════════════════════════════════════════════════

def print_section(title: str) -> None:
    bar = "-" * 60
    print(f"\n{bar}\n  {title}\n{bar}")


def print_kv(key: str, value) -> None:
    print(f"  {key:<35} {value}")


# ════════════════════════════════════════════════════════════════════════════════
# 1. Load data
# ════════════════════════════════════════════════════════════════════════════════

def load_and_report(data_dir: Path | None = None):
    logger.info("Loading train and test datasets…")
    train = load_train(data_dir=data_dir, parse_geometry=False, as_geodataframe=False)
    test = load_test(data_dir=data_dir, parse_geometry=False, as_geodataframe=False)
    return train, test


# ════════════════════════════════════════════════════════════════════════════════
# 2. Basic dataset info
# ════════════════════════════════════════════════════════════════════════════════

def report_basic_info(train: pd.DataFrame, test: pd.DataFrame) -> None:
    print_section("DATASET SHAPE & COLUMNS")

    print_kv("Train rows × cols:", f"{train.shape[0]:,} × {train.shape[1]}")
    print_kv("Test  rows × cols:", f"{test.shape[0]:,} × {test.shape[1]}")

    print("\n  Train columns:")
    for col in train.columns:
        print(f"    {col:<25} dtype={train[col].dtype}  "
              f"unique={train[col].nunique()}")

    print("\n  Test columns (difference from train):")
    diff_cols = set(train.columns) - set(test.columns)
    print(f"    Columns only in Train: {sorted(diff_cols)}")
    diff_cols2 = set(test.columns) - set(train.columns)
    print(f"    Columns only in Test : {sorted(diff_cols2)}")

    print_section("MISSING VALUES")
    for col in train.columns:
        n = train[col].isna().sum()
        flag = "  ← MISSING" if n > 0 else ""
        print(f"  Train.{col:<25} {n}{flag}")
    for col in test.columns:
        n = test[col].isna().sum()
        flag = "  ← MISSING" if n > 0 else ""
        print(f"  Test. {col:<25} {n}{flag}")


# ════════════════════════════════════════════════════════════════════════════════
# 3. Crop & category distribution
# ════════════════════════════════════════════════════════════════════════════════

def report_and_plot_crops(train: pd.DataFrame) -> None:
    print_section("CROP DISTRIBUTION (PRIMARY TARGET - TRAIN)")
    crop_counts = train["Crop"].value_counts()
    total = len(train)
    for crop, cnt in crop_counts.items():
        print(f"  {crop:<20} {cnt:>5,}  ({cnt/total*100:.1f}%)")

    # Bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Crop counts
    ax = axes[0]
    bars = ax.barh(crop_counts.index[::-1], crop_counts.values[::-1], color=CROP_COLORS)
    ax.set_xlabel("Number of Farms", fontsize=11)
    ax.set_title("Crop Distribution (Train)", fontsize=13, fontweight="bold")
    for bar, val in zip(bars, crop_counts.values[::-1]):
        ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=9)
    ax.set_xlim(0, crop_counts.max() * 1.15)

    # Category distribution
    cat_counts = train["category"].value_counts()
    colors = [PALETTE.get(c, "#999") for c in cat_counts.index]
    wedges, texts, autotexts = axes[1].pie(
        cat_counts.values,
        labels=cat_counts.index,
        autopct="%1.1f%%",
        colors=colors,
        startangle=140,
        wedgeprops=dict(edgecolor="white", linewidth=1.5),
    )
    for at in autotexts:
        at.set_fontsize(10)
    axes[1].set_title("Crop Health Category Distribution (Train)", fontsize=13, fontweight="bold")

    plt.tight_layout()
    out = FIGS_DIR / "crop_and_category_distribution.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", out)


def report_category(train: pd.DataFrame) -> None:
    print_section("CATEGORY (SECONDARY HEALTH TARGET) DISTRIBUTION (TRAIN)")
    cat_counts = train["category"].value_counts()
    total = len(train)
    for cat, cnt in cat_counts.items():
        print(f"  {cat:<15} {cnt:>5,}  ({cnt/total*100:.1f}%)")


# ════════════════════════════════════════════════════════════════════════════════
# 4. Date range analysis
# ════════════════════════════════════════════════════════════════════════════════

def report_and_plot_dates(train: pd.DataFrame, test: pd.DataFrame) -> None:
    print_section("DATE RANGES")

    for name, df in [("Train", train), ("Test", test)]:
        sdate = pd.to_datetime(df["SDate"], errors="coerce", infer_datetime_format=True)
        hdate = pd.to_datetime(df["HDate"], errors="coerce", infer_datetime_format=True)
        duration = (hdate - sdate).dt.days
        print(f"\n  [{name}]")
        print(f"    SDate range  : {sdate.min().date()} → {sdate.max().date()}")
        print(f"    HDate range  : {hdate.min().date()} → {hdate.max().date()}")
        print(f"    Bad SDate    : {sdate.isna().sum()}")
        print(f"    Bad HDate    : {hdate.isna().sum()}")
        print(f"    Duration (days) min/mean/max: "
              f"{duration.min():.0f} / {duration.mean():.0f} / {duration.max():.0f}")

    # Duration histogram by crop
    sdate_tr = pd.to_datetime(train["SDate"], errors="coerce", infer_datetime_format=True)
    hdate_tr = pd.to_datetime(train["HDate"], errors="coerce", infer_datetime_format=True)
    train_dur = (hdate_tr - sdate_tr).dt.days
    train_plot = train.copy()
    train_plot["duration_days"] = train_dur

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Overall duration histogram
    axes[0].hist(train_dur.dropna(), bins=40, color="#3498db", edgecolor="white", alpha=0.85)
    axes[0].set_xlabel("Crop Duration (Harvest − Sowing, days)", fontsize=11)
    axes[0].set_ylabel("Number of Farms", fontsize=11)
    axes[0].set_title("Farm Duration Distribution (Train)", fontsize=13, fontweight="bold")
    axes[0].axvline(train_dur.median(), color="#e74c3c", linestyle="--", label=f"Median: {train_dur.median():.0f} days")
    axes[0].legend()

    # SDate by year-month
    sdate_counts = sdate_tr.dt.to_period("M").value_counts().sort_index()
    axes[1].bar(sdate_counts.index.astype(str), sdate_counts.values, color="#2ecc71", edgecolor="white")
    axes[1].set_xlabel("Sowing Month", fontsize=11)
    axes[1].set_ylabel("Number of Farms", fontsize=11)
    axes[1].set_title("Sowing Date Distribution by Month (Train)", fontsize=13, fontweight="bold")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    out = FIGS_DIR / "date_range_analysis.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", out)


# ════════════════════════════════════════════════════════════════════════════════
# 5. District / Sub-district / Season distributions
# ════════════════════════════════════════════════════════════════════════════════

def report_and_plot_districts(train: pd.DataFrame) -> None:
    print_section("GEOGRAPHIC DISTRIBUTION")
    dist_counts = train["District"].value_counts()
    print(f"  Unique Districts    : {train['District'].nunique()}")
    print(f"  Unique Sub-Districts: {train['Sub-District'].nunique()}")
    print("\n  District breakdown:")
    for d, cnt in dist_counts.items():
        print(f"    {d:<25} {cnt:>5,}")

    # Season distribution
    print_section("SEASON DISTRIBUTION")
    season_counts = train["Season"].value_counts()
    for s, cnt in season_counts.items():
        print(f"  {s:<12} {cnt:>5,}  ({cnt/len(train)*100:.1f}%)")

    # Heatmap: District × Crop count
    pivot = train.groupby(["District", "Crop"]).size().unstack(fill_value=0)
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    sns.heatmap(
        pivot,
        ax=axes[0],
        annot=True,
        fmt="d",
        cmap="YlOrRd",
        linewidths=0.5,
        cbar_kws={"label": "Farm count"},
    )
    axes[0].set_title("Farms per District × Crop (Train)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Crop", fontsize=11)
    axes[0].set_ylabel("District", fontsize=11)
    axes[0].tick_params(axis="x", rotation=30)

    # District × Category heatmap
    pivot2 = train.groupby(["District", "category"]).size().unstack(fill_value=0)
    sns.heatmap(
        pivot2,
        ax=axes[1],
        annot=True,
        fmt="d",
        cmap="Blues",
        linewidths=0.5,
        cbar_kws={"label": "Farm count"},
    )
    axes[1].set_title("Health Category per District (Train)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Category", fontsize=11)
    axes[1].set_ylabel("District", fontsize=11)

    plt.tight_layout()
    out = FIGS_DIR / "district_crop_heatmaps.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", out)


# ════════════════════════════════════════════════════════════════════════════════
# 6. Irrigation analysis
# ════════════════════════════════════════════════════════════════════════════════

def report_and_plot_irrigation(train: pd.DataFrame) -> None:
    print_section("IRRIGATION ANALYSIS")

    for col in ["IrriType", "IrriSource"]:
        counts = train[col].value_counts()
        print(f"\n  {col}:")
        for val, cnt in counts.items():
            print(f"    {val:<20} {cnt:>5,}  ({cnt/len(train)*100:.1f}%)")

    irri_count = train["IrriCount"].describe()
    water_cov = train["WaterCov"].describe()
    print(f"\n  IrriCount stats:\n{irri_count.to_string()}")
    print(f"\n  WaterCov stats:\n{water_cov.to_string()}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # IrriType
    irri_type = train["IrriType"].value_counts()
    axes[0, 0].bar(irri_type.index, irri_type.values, color=sns.color_palette("Set2", len(irri_type)))
    axes[0, 0].set_title("Irrigation Type", fontweight="bold")
    axes[0, 0].set_ylabel("Farms")
    axes[0, 0].tick_params(axis="x", rotation=15)

    # IrriSource
    irri_src = train["IrriSource"].value_counts()
    axes[0, 1].bar(irri_src.index, irri_src.values, color=sns.color_palette("Set3", len(irri_src)))
    axes[0, 1].set_title("Irrigation Source", fontweight="bold")
    axes[0, 1].set_ylabel("Farms")
    axes[0, 1].tick_params(axis="x", rotation=15)

    # IrriCount distribution
    axes[1, 0].hist(train["IrriCount"].dropna(), bins=range(0, 8), color="#9b59b6",
                    edgecolor="white", align="left")
    axes[1, 0].set_title("Irrigation Count (times/season)", fontweight="bold")
    axes[1, 0].set_xlabel("IrriCount")
    axes[1, 0].set_ylabel("Farms")

    # WaterCov distribution
    axes[1, 1].hist(train["WaterCov"].dropna(), bins=25, color="#1abc9c", edgecolor="white")
    axes[1, 1].set_title("Water Coverage (%)", fontweight="bold")
    axes[1, 1].set_xlabel("WaterCov (%)")
    axes[1, 1].set_ylabel("Farms")

    plt.suptitle("Irrigation Analysis — Train Dataset", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = FIGS_DIR / "irrigation_analysis.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", out)


# ════════════════════════════════════════════════════════════════════════════════
# 7. Geometry analysis
# ════════════════════════════════════════════════════════════════════════════════

def report_and_plot_geometry(data_dir: Path | None = None) -> None:
    """Load with geometry, compute area stats, plot farm area distribution and boundary map."""
    print_section("GEOMETRY & FARM AREA ANALYSIS")

    try:
        import geopandas as gpd
    except ImportError:
        logger.warning("geopandas not available — skipping geometry analysis.")
        return

    logger.info("Loading GeoDataFrame for geometry analysis…")
    train_gdf = load_train(data_dir=data_dir, parse_geometry=True, as_geodataframe=True)
    test_gdf = load_test(data_dir=data_dir, parse_geometry=True, as_geodataframe=True)

    # Drop invalid geom rows
    train_gdf = train_gdf[train_gdf.geometry.notna()]
    test_gdf = test_gdf[test_gdf.geometry.notna()]

    # Add geometry features (area in ha, centroids, validity)
    train_gdf = add_geometry_features(train_gdf)

    # Print stats
    stats = geometry_stats(train_gdf)
    print_kv("Total train farms:", stats["n_total"])
    print_kv("Valid geometries:", stats["n_valid_geom"])
    print_kv("Invalid geometries:", stats["n_invalid_geom"])
    print_kv("Within Telangana bbox:", stats["n_within_telangana_bbox"])
    print_kv("Outside Telangana bbox:", stats["n_outside_telangana_bbox"])
    print_kv("Area (ha) min:", f"{stats['area_ha_min']:.4f}")
    print_kv("Area (ha) max:", f"{stats['area_ha_max']:.4f}")
    print_kv("Area (ha) mean:", f"{stats['area_ha_mean']:.4f}")
    print_kv("Area (ha) median:", f"{stats['area_ha_median']:.4f}")
    print_kv("Area (m²) min:", f"{stats['area_m2_min']:.1f}")
    print_kv("Area (m²) max:", f"{stats['area_m2_max']:.1f}")
    print_kv("Vertex count min:", stats["vertex_count_min"])
    print_kv("Vertex count max:", stats["vertex_count_max"])
    print_kv("Vertex count mean:", f"{stats['vertex_count_mean']:.1f}")
    print_kv("Input CRS:", stats["crs_input"])
    print_kv("Metric CRS used:", stats["crs_metric"])

    # ── Farm area distribution plot ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Area histogram
    axes[0].hist(train_gdf["area_ha"].dropna(), bins=50, color="#3498db", edgecolor="white", alpha=0.85)
    axes[0].set_xlabel("Farm Area (hectares)", fontsize=11)
    axes[0].set_ylabel("Number of Farms", fontsize=11)
    axes[0].set_title("Farm Area Distribution (Train)", fontsize=13, fontweight="bold")
    axes[0].axvline(train_gdf["area_ha"].median(), color="#e74c3c", linestyle="--",
                    label=f"Median: {train_gdf['area_ha'].median():.4f} ha")
    axes[0].legend()

    # Scatter: centroid lon/lat coloured by crop
    unique_crops = train_gdf["Crop"].unique()
    crop_palette = dict(zip(unique_crops, CROP_COLORS))
    for crop, grp in train_gdf.groupby("Crop"):
        axes[1].scatter(
            grp["centroid_lon"], grp["centroid_lat"],
            c=[crop_palette[crop]],
            s=3, alpha=0.6, label=crop,
        )
    axes[1].set_xlabel("Longitude", fontsize=11)
    axes[1].set_ylabel("Latitude", fontsize=11)
    axes[1].set_title("Farm Centroids by Crop (Train) — EPSG:4326", fontsize=13, fontweight="bold")
    axes[1].legend(markerscale=4, fontsize=9)

    plt.tight_layout()
    out = FIGS_DIR / "farm_area_and_centroids.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", out)

    # ── Static boundary map (sample) ─────────────────────────────────────────
    # Take a random sample of 400 farms to keep the plot readable
    sample_size = min(400, len(train_gdf))
    sample_gdf = train_gdf.sample(sample_size, random_state=42)

    fig, ax = plt.subplots(figsize=(12, 10))
    cat_palette = {
        "Healthy": "#2ecc71",
        "Diseased": "#e74c3c",
        "Pests": "#e67e22",
        "Stressed": "#3498db",
    }
    for cat, grp in sample_gdf.groupby("category"):
        grp.plot(ax=ax, color=cat_palette.get(cat, "#aaa"), alpha=0.6,
                 edgecolor="white", linewidth=0.3, label=cat)

    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude", fontsize=11)
    ax.set_title(f"Farm Boundaries — Random Sample of {sample_size} Farms (Train)\n"
                 "CRS: EPSG:4326  |  Colour: Health Category", fontsize=13, fontweight="bold")
    legend_patches = [mpatches.Patch(color=v, label=k) for k, v in cat_palette.items()]
    ax.legend(handles=legend_patches, fontsize=11)

    plt.tight_layout()
    out = FIGS_DIR / "farm_boundaries_map.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", out)


# ════════════════════════════════════════════════════════════════════════════════
# 8. Numeric feature distributions
# ════════════════════════════════════════════════════════════════════════════════

def report_and_plot_numeric(train: pd.DataFrame) -> None:
    print_section("NUMERIC FEATURE DISTRIBUTIONS")

    num_cols = ["CropCoveredArea", "CHeight", "IrriCount", "WaterCov", "ExpYield"]
    for col in num_cols:
        s = pd.to_numeric(train[col], errors="coerce")
        print(f"  {col:<22} min={s.min():.0f}  max={s.max():.0f}  "
              f"mean={s.mean():.1f}  std={s.std():.1f}")

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for i, col in enumerate(num_cols):
        s = pd.to_numeric(train[col], errors="coerce")
        axes[i].hist(s.dropna(), bins=30, color=CROP_COLORS[i], edgecolor="white", alpha=0.85)
        axes[i].set_title(col, fontsize=12, fontweight="bold")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frequency")
        axes[i].axvline(s.mean(), color="red", linestyle="--", alpha=0.8,
                        label=f"mean={s.mean():.1f}")
        axes[i].legend(fontsize=9)

    # Hide the last unused subplot
    axes[-1].set_visible(False)

    plt.suptitle("Numeric Feature Distributions (Train)", fontsize=15, fontweight="bold")
    plt.tight_layout()
    out = FIGS_DIR / "numeric_feature_distributions.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", out)


# ════════════════════════════════════════════════════════════════════════════════
# 9. Duplicate / overlap checks
# ════════════════════════════════════════════════════════════════════════════════

def report_duplicates(train: pd.DataFrame, test: pd.DataFrame) -> None:
    print_section("DUPLICATE & OVERLAP CHECKS")
    print_kv("Train FarmID duplicates:", train["FarmID"].duplicated().sum())
    print_kv("Test  FarmID duplicates:", test["FarmID"].duplicated().sum())
    overlap = set(train["FarmID"].astype(str)) & set(test["FarmID"].astype(str))
    print_kv("Train-Test FarmID overlap:", len(overlap))
    print_kv("Train geometry duplicates:", train["geometry"].astype(str).duplicated().sum())
    print_kv("Test  geometry duplicates:", test["geometry"].astype(str).duplicated().sum())


# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("\n" + "═" * 65)
    print("  Phase 1 — Dataset Analysis")
    print("  Telangana Crop Health Challenge")
    print("═" * 65)

    # 1. Load (without geometry for speed on most analysis steps)
    train, test = load_and_report()

    # 2. Basic info
    report_basic_info(train, test)

    # 3. Crop & category
    report_and_plot_crops(train)
    report_category(train)

    # 4. Dates
    report_and_plot_dates(train, test)

    # 5. Districts & seasons
    report_and_plot_districts(train)

    # 6. Irrigation
    report_and_plot_irrigation(train)

    # 7. Numeric features
    report_and_plot_numeric(train)

    # 8. Duplicates / overlap
    report_duplicates(train, test)

    # 9. Data quality report
    print_section("DATA QUALITY REPORT")
    report = validate(train, test, output_dir=REPORTS_DIR)
    report.print_summary()

    # 10. Geometry analysis (slow — requires shapely projections)
    report_and_plot_geometry()

    print("\n" + "═" * 65)
    print("  ✓  Phase 1 analysis complete.")
    print(f"  Figures saved to : {FIGS_DIR}")
    print(f"  Reports saved to : {REPORTS_DIR}")
    print("═" * 65 + "\n")


if __name__ == "__main__":
    main()
