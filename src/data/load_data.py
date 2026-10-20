"""
src/data/load_data.py
=====================
Data loading utilities for the Telangana Crop Monitoring and Classification dataset.
Note: 'Crop' is the primary ML classification target; 'category' is the secondary crop health target.

Responsibilities:
    - Locate Train.csv and Test.csv relative to a configurable data directory.
    - Parse geometry column from WKT string to Shapely geometry objects.
    - Return clean pandas DataFrames and GeoPandas GeoDataFrames.
    - Parse date columns to proper datetime dtype.
    - Enforce known dtypes for numeric columns.

CRS: EPSG:4326 (WGS84 geographic) — confirmed from official starter notebook.
Geometry format: WKT POLYGON strings stored in the `geometry` column.

DO NOT:
    - Modify the raw CSV files.
    - Fabricate any data.
    - Download satellite data.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import geopandas as gpd
import pandas as pd
from shapely import wkt
from shapely.errors import ShapelyError

# ── Module-level logger ──────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
# Data folder name (relative to project root) — note the three 's' characters.
DATA_FOLDER_NAME: str = "datasss"

# Column name that stores WKT geometry strings.
GEOMETRY_COL: str = "geometry"

# Official CRS confirmed from the GIZ-ADEX starter notebook.
CRS_WGS84: str = "EPSG:4326"

# Date columns and their format as stored in the CSVs.
DATE_COLUMNS: list[str] = ["SDate", "HDate"]
DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"  # e.g. "2023-11-25 0:00:00"

# Numeric columns that should be cast to int64.
INT_COLUMNS: list[str] = [
    "FarmID",
    "CropCoveredArea",
    "CHeight",
    "IrriCount",
    "WaterCov",
    "ExpYield",
]


# ── Internal helpers ─────────────────────────────────────────────────────────

def _resolve_data_dir(data_dir: Optional[Path | str] = None) -> Path:
    """
    Resolve the path to the raw data directory.

    Parameters
    ----------
    data_dir : Path | str | None
        Explicit path to the data directory.  If None, the function
        searches upward from this file for a ``datasss`` folder.

    Returns
    -------
    Path
        Absolute path to the data directory.

    Raises
    ------
    FileNotFoundError
        If the directory cannot be found.
    """
    if data_dir is not None:
        p = Path(data_dir).resolve()
        if not p.is_dir():
            raise FileNotFoundError(f"Supplied data_dir does not exist: {p}")
        return p

    # Walk upward from this file to find the project root containing DATA_FOLDER_NAME.
    here = Path(__file__).resolve()
    for parent in [here.parent, here.parent.parent, here.parent.parent.parent,
                   here.parent.parent.parent.parent]:
        candidate = parent / DATA_FOLDER_NAME
        if candidate.is_dir():
            logger.debug("Auto-detected data directory: %s", candidate)
            return candidate

    raise FileNotFoundError(
        f"Could not auto-detect '{DATA_FOLDER_NAME}' folder. "
        "Pass data_dir= explicitly."
    )


def _parse_geometry_col(series: pd.Series) -> pd.Series:
    """
    Convert a Series of WKT strings to Shapely geometry objects.

    Rows that fail to parse are set to None and logged as warnings.

    Parameters
    ----------
    series : pd.Series
        Series of WKT POLYGON strings.

    Returns
    -------
    pd.Series
        Series of shapely.geometry.Polygon objects (dtype=object).
    """
    def _safe_load(val: str):
        try:
            return wkt.loads(str(val).strip())
        except (ShapelyError, ValueError, TypeError) as exc:
            logger.warning("Failed to parse geometry: %s … (%s)", str(val)[:60], exc)
            return None

    return series.apply(_safe_load)


def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse date columns in-place to pandas datetime dtype.

    The CSV stores dates as "YYYY-MM-DD H:MM:SS" (e.g. "2023-11-25 0:00:00").
    We parse flexibly so that slight format variations do not raise exceptions.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Same dataframe with date columns as datetime64[ns].
    """
    for col in DATE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", format="mixed")
            n_failed = df[col].isna().sum()
            if n_failed:
                logger.warning("Column '%s': %d values could not be parsed as dates.", col, n_failed)
            else:
                logger.debug("Parsed '%s' successfully (%d rows).", col, len(df))
    return df


def _cast_int_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast known integer columns to int64, skipping columns absent from df.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    for col in INT_COLUMNS:
        if col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
            except Exception as exc:
                logger.warning("Could not cast '%s' to Int64: %s", col, exc)
    return df


# ── Public API ───────────────────────────────────────────────────────────────

def load_train(
    data_dir: Optional[Path | str] = None,
    parse_geometry: bool = True,
    as_geodataframe: bool = True,
) -> gpd.GeoDataFrame | pd.DataFrame:
    """
    Load Train.csv and return a clean GeoDataFrame (or DataFrame).

    Parameters
    ----------
    data_dir : Path | str | None
        Path to the raw data folder (``datasss``).  Auto-detected if None.
    parse_geometry : bool
        Whether to parse the WKT geometry column. Default True.
    as_geodataframe : bool
        If True (default) and parse_geometry is True, returns a GeoDataFrame
        with CRS set to EPSG:4326.  Otherwise returns a plain DataFrame.

    Returns
    -------
    gpd.GeoDataFrame or pd.DataFrame
    """
    data_path = _resolve_data_dir(data_dir)
    csv_path = data_path / "Train.csv"

    logger.info("Loading train data from: %s", csv_path)
    df = pd.read_csv(csv_path, low_memory=False)

    df = _parse_dates(df)
    df = _cast_int_columns(df)

    if parse_geometry:
        df[GEOMETRY_COL] = _parse_geometry_col(df[GEOMETRY_COL])
        n_bad = df[GEOMETRY_COL].isna().sum()
        if n_bad:
            logger.warning("Train: %d rows with invalid geometry.", n_bad)

        if as_geodataframe:
            df = gpd.GeoDataFrame(df, geometry=GEOMETRY_COL, crs=CRS_WGS84)
            logger.info("Train GeoDataFrame created: %d rows, CRS=%s", len(df), CRS_WGS84)

    return df


def load_test(
    data_dir: Optional[Path | str] = None,
    parse_geometry: bool = True,
    as_geodataframe: bool = True,
) -> gpd.GeoDataFrame | pd.DataFrame:
    """
    Load Test.csv and return a clean GeoDataFrame (or DataFrame).

    Test.csv contains all farm metadata and 'Crop' type, but does NOT contain
    the 'category' column (crop health status).

    Parameters
    ----------
    data_dir : Path | str | None
        Path to the raw data folder (``datasss``).  Auto-detected if None.
    parse_geometry : bool
        Whether to parse the WKT geometry column. Default True.
    as_geodataframe : bool
        If True (default) and parse_geometry is True, returns a GeoDataFrame
        with CRS set to EPSG:4326.  Otherwise returns a plain DataFrame.

    Returns
    -------
    gpd.GeoDataFrame or pd.DataFrame
    """
    data_path = _resolve_data_dir(data_dir)
    csv_path = data_path / "Test.csv"

    logger.info("Loading test data from: %s", csv_path)
    df = pd.read_csv(csv_path, low_memory=False)

    df = _parse_dates(df)
    df = _cast_int_columns(df)

    if parse_geometry:
        df[GEOMETRY_COL] = _parse_geometry_col(df[GEOMETRY_COL])
        n_bad = df[GEOMETRY_COL].isna().sum()
        if n_bad:
            logger.warning("Test: %d rows with invalid geometry.", n_bad)

        if as_geodataframe:
            df = gpd.GeoDataFrame(df, geometry=GEOMETRY_COL, crs=CRS_WGS84)
            logger.info("Test GeoDataFrame created: %d rows, CRS=%s", len(df), CRS_WGS84)

    return df


def load_both(
    data_dir: Optional[Path | str] = None,
    parse_geometry: bool = True,
    as_geodataframe: bool = True,
) -> Tuple[gpd.GeoDataFrame | pd.DataFrame, gpd.GeoDataFrame | pd.DataFrame]:
    """
    Convenience wrapper: load both Train and Test in one call.

    Returns
    -------
    (train, test) tuple
    """
    train = load_train(data_dir, parse_geometry, as_geodataframe)
    test = load_test(data_dir, parse_geometry, as_geodataframe)
    return train, test


def load_sample_submission(data_dir: Optional[Path | str] = None) -> pd.DataFrame:
    """
    Load SampleSubmission.csv.

    Columns: ID (FarmID), Target (integer class label)
    """
    data_path = _resolve_data_dir(data_dir)
    return pd.read_csv(data_path / "SampleSubmission.csv")


# ── Quick self-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    train, test = load_both()
    print(f"\nTrain shape : {train.shape}")
    print(f"Test  shape : {test.shape}")
    print(f"\nTrain columns:\n  {list(train.columns)}")
    print(f"\nTest  columns:\n  {list(test.columns)}")
    print(f"\nTrain dtypes:\n{train.dtypes}")
    print(f"\nTrain CRS  : {train.crs}")
    print(f"Test  CRS  : {test.crs}")
    print(f"\nTrain Crop counts (primary target):\n{train['Crop'].value_counts()}")
    print(f"\nTrain category counts (secondary health target):\n{train['category'].value_counts()}")
