"""
src/data/validate_data.py
==========================
Data validation utilities for the Telangana Crop Health Challenge dataset.

Responsibilities:
    - Check for missing values across all columns.
    - Check for duplicate FarmIDs within Train and Test.
    - Check for duplicate geometries.
    - Validate date logic (HDate must be after SDate).
    - Check for train/test FarmID overlap (should be zero).
    - Validate that geometry column contains only valid WKT POLYGONs.
    - Return a structured DataQualityReport object.
    - Optionally save a human-readable text/JSON report.

The validator is READ-ONLY — it never mutates the DataFrames.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ── Report data structure ────────────────────────────────────────────────────

@dataclass
class DataQualityReport:
    """Structured container for all validation results."""

    # Basic shape
    train_rows: int = 0
    train_cols: int = 0
    test_rows: int = 0
    test_cols: int = 0

    # Missing values: {column_name: count}
    train_missing: Dict[str, int] = field(default_factory=dict)
    test_missing: Dict[str, int] = field(default_factory=dict)

    # Duplicate checks
    train_dup_farmid: int = 0
    test_dup_farmid: int = 0
    train_dup_geometry: int = 0
    test_dup_geometry: int = 0
    train_test_farmid_overlap: int = 0

    # Date validation
    train_bad_sdate: int = 0    # rows where SDate could not be parsed
    train_bad_hdate: int = 0    # rows where HDate could not be parsed
    train_hdate_before_sdate: int = 0  # rows where HDate <= SDate
    test_bad_sdate: int = 0
    test_bad_hdate: int = 0
    test_hdate_before_sdate: int = 0

    # Geometry validation
    train_bad_geometry: int = 0
    test_bad_geometry: int = 0
    train_non_polygon: int = 0  # geometries that are not POLYGONs
    test_non_polygon: int = 0

    # Column presence
    crop_col_in_train: bool = False
    crop_col_in_test: bool = False
    category_col_in_train: bool = False
    category_col_in_test: bool = False
    target_col_in_train: bool = False
    target_col_in_test: bool = False

    # Warnings list (free-text)
    warnings: List[str] = field(default_factory=list)

    # ── Summary helpers ──────────────────────────────────────────────────────

    def has_critical_issues(self) -> bool:
        """Return True if any integrity or data issue exists in Train or Test."""
        return any([
            sum(self.train_missing.values()) > 0,
            sum(self.test_missing.values()) > 0,
            self.train_dup_farmid > 0,
            self.test_dup_farmid > 0,
            self.train_dup_geometry > 0,
            self.test_dup_geometry > 0,
            self.train_bad_sdate > 0,
            self.train_bad_hdate > 0,
            self.train_hdate_before_sdate > 0,
            self.test_bad_sdate > 0,
            self.test_bad_hdate > 0,
            self.test_hdate_before_sdate > 0,
            self.train_bad_geometry > 0,
            self.test_bad_geometry > 0,
            self.train_non_polygon > 0,
            self.test_non_polygon > 0,
            self.train_test_farmid_overlap > 0,
        ])

    def summary_lines(self) -> List[str]:
        """Return a human-readable summary as a list of strings."""
        lines = [
            "=" * 65,
            "  DATA QUALITY REPORT — Telangana Crop Monitoring Framework",
            "=" * 65,
            f"  Train : {self.train_rows:>6,} rows  ×  {self.train_cols} columns",
            f"  Test  : {self.test_rows:>6,} rows  ×  {self.test_cols} columns",
            "",
            "── Missing Values ──────────────────────────────────────────",
        ]
        if any(v > 0 for v in self.train_missing.values()):
            lines.append("  TRAIN has missing values:")
            for col, cnt in self.train_missing.items():
                if cnt > 0:
                    lines.append(f"    {col}: {cnt}")
        else:
            lines.append("  ✓ Train: NO missing values in any column.")

        if any(v > 0 for v in self.test_missing.values()):
            lines.append("  TEST has missing values:")
            for col, cnt in self.test_missing.items():
                if cnt > 0:
                    lines.append(f"    {col}: {cnt}")
        else:
            lines.append("  ✓ Test : NO missing values in any column.")

        lines += [
            "",
            "── Duplicates ──────────────────────────────────────────────",
            f"  Train FarmID dups   : {self.train_dup_farmid}",
            f"  Test  FarmID dups   : {self.test_dup_farmid}",
            f"  Train geometry dups : {self.train_dup_geometry}",
            f"  Test  geometry dups : {self.test_dup_geometry}",
            f"  Train-Test FarmID overlap : {self.train_test_farmid_overlap}",
            "",
            "── Date Validation ─────────────────────────────────────────",
            f"  Train bad SDate      : {self.train_bad_sdate}",
            f"  Train bad HDate      : {self.train_bad_hdate}",
            f"  Train HDate ≤ SDate  : {self.train_hdate_before_sdate}",
            f"  Test  bad SDate      : {self.test_bad_sdate}",
            f"  Test  bad HDate      : {self.test_bad_hdate}",
            f"  Test  HDate ≤ SDate  : {self.test_hdate_before_sdate}",
            "",
            "── Geometry Validation ─────────────────────────────────────",
            f"  Train bad geometry  : {self.train_bad_geometry}",
            f"  Train non-POLYGON   : {self.train_non_polygon}",
            f"  Test  bad geometry  : {self.test_bad_geometry}",
            f"  Test  non-POLYGON   : {self.test_non_polygon}",
            "",
            "── Column Presence ─────────────────────────────────────────",
            f"  'Crop' in Train (Primary Target)   : {self.crop_col_in_train}",
            f"  'Crop' in Test  (Features/Eval)    : {self.crop_col_in_test}",
            f"  'category' in Train (Sec. Health)  : {self.category_col_in_train}",
            f"  'category' in Test                 : {self.category_col_in_test}  (expected False)",
        ]

        if self.warnings:
            lines += ["", "── Warnings ────────────────────────────────────────────────"]
            for w in self.warnings:
                lines.append(f"  ⚠ {w}")

        lines += [
            "",
            f"  CRITICAL ISSUES DETECTED: {self.has_critical_issues()}",
            "=" * 65,
        ]
        return lines

    def print_summary(self) -> None:
        """Print the summary to stdout."""
        print("\n".join(self.summary_lines()))

    def to_json(self) -> str:
        """Serialise the report to a JSON string."""
        return json.dumps(asdict(self), indent=2)

    def save(self, output_dir: Path, filename: str = "data_quality_report") -> None:
        """
        Save the report as both a .txt and a .json file.

        Parameters
        ----------
        output_dir : Path
            Directory to write the report files.
        filename : str
            Base name (without extension).
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        txt_path = output_dir / f"{filename}.txt"
        txt_path.write_text("\n".join(self.summary_lines()), encoding="utf-8")
        logger.info("Saved text report → %s", txt_path)

        json_path = output_dir / f"{filename}.json"
        json_path.write_text(self.to_json(), encoding="utf-8")
        logger.info("Saved JSON report → %s", json_path)


# ── Validation functions ─────────────────────────────────────────────────────

def _check_missing(df: pd.DataFrame) -> Dict[str, int]:
    """Return a dict of {column: missing_count}."""
    return {col: int(df[col].isna().sum()) for col in df.columns}


def _check_dup_farmid(df: pd.DataFrame) -> int:
    """Return number of duplicate FarmID rows."""
    return int(df["FarmID"].duplicated().sum())


def _check_dup_geometry(df: pd.DataFrame, geom_col: str = "geometry") -> int:
    """Return number of duplicate geometry strings."""
    if geom_col not in df.columns:
        return 0
    return int(df[geom_col].astype(str).duplicated().sum())


def _check_date_validity(df: pd.DataFrame) -> dict:
    """
    Check SDate and HDate for parseability and logical ordering.

    Returns a dict with keys: bad_sdate, bad_hdate, hdate_before_sdate.
    """
    result = {"bad_sdate": 0, "bad_hdate": 0, "hdate_before_sdate": 0}

    sdate = pd.to_datetime(df.get("SDate"), errors="coerce", format="mixed")
    hdate = pd.to_datetime(df.get("HDate"), errors="coerce", format="mixed")

    result["bad_sdate"] = int(sdate.isna().sum())
    result["bad_hdate"] = int(hdate.isna().sum())

    # Only compare rows where both dates are valid
    both_valid = sdate.notna() & hdate.notna()
    result["hdate_before_sdate"] = int((hdate[both_valid] <= sdate[both_valid]).sum())

    return result


def _check_geometry_validity(df: pd.DataFrame, geom_col: str = "geometry") -> dict:
    """
    Validate WKT geometry strings.

    Returns a dict with keys: bad_geometry, non_polygon.
    """
    from shapely import wkt as shp_wkt
    from shapely.errors import ShapelyError

    bad = 0
    non_poly = 0

    if geom_col not in df.columns:
        return {"bad_geometry": 0, "non_polygon": 0}

    for val in df[geom_col]:
        try:
            geom = shp_wkt.loads(str(val).strip())
            if geom.geom_type != "Polygon":
                non_poly += 1
        except (ShapelyError, ValueError, TypeError):
            bad += 1

    return {"bad_geometry": bad, "non_polygon": non_poly}


# ── Public API ───────────────────────────────────────────────────────────────

def validate(
    train: pd.DataFrame,
    test: pd.DataFrame,
    output_dir: Optional[Path] = None,
) -> DataQualityReport:
    """
    Run all validation checks and return a DataQualityReport.

    Parameters
    ----------
    train : pd.DataFrame
        Loaded training DataFrame (raw or with parsed dtypes — not GeoDataFrame required).
    test : pd.DataFrame
        Loaded test DataFrame.
    output_dir : Path | None
        If provided, the report is saved to this directory.

    Returns
    -------
    DataQualityReport
    """
    report = DataQualityReport()

    # ── Basic shape ──────────────────────────────────────────────────────────
    report.train_rows, report.train_cols = train.shape
    report.test_rows, report.test_cols = test.shape

    # ── Missing values ───────────────────────────────────────────────────────
    report.train_missing = _check_missing(train)
    report.test_missing = _check_missing(test)

    # ── Duplicates ───────────────────────────────────────────────────────────
    report.train_dup_farmid = _check_dup_farmid(train)
    report.test_dup_farmid = _check_dup_farmid(test)
    report.train_dup_geometry = _check_dup_geometry(train)
    report.test_dup_geometry = _check_dup_geometry(test)

    # Train-Test FarmID overlap
    overlap = set(train["FarmID"].astype(str)) & set(test["FarmID"].astype(str))
    report.train_test_farmid_overlap = len(overlap)
    if overlap:
        report.warnings.append(
            f"{len(overlap)} FarmIDs appear in both Train and Test: "
            f"{sorted(list(overlap))[:10]} ..."
        )

    # ── Date validation ──────────────────────────────────────────────────────
    td = _check_date_validity(train)
    report.train_bad_sdate = td["bad_sdate"]
    report.train_bad_hdate = td["bad_hdate"]
    report.train_hdate_before_sdate = td["hdate_before_sdate"]

    tsd = _check_date_validity(test)
    report.test_bad_sdate = tsd["bad_sdate"]
    report.test_bad_hdate = tsd["bad_hdate"]
    report.test_hdate_before_sdate = tsd["hdate_before_sdate"]

    # ── Geometry validation ──────────────────────────────────────────────────
    logger.info("Validating train geometries (this may take a moment)…")
    tg = _check_geometry_validity(train)
    report.train_bad_geometry = tg["bad_geometry"]
    report.train_non_polygon = tg["non_polygon"]

    logger.info("Validating test geometries…")
    tesg = _check_geometry_validity(test)
    report.test_bad_geometry = tesg["bad_geometry"]
    report.test_non_polygon = tesg["non_polygon"]

    # ── Column presence ──────────────────────────────────────────────────────
    report.crop_col_in_train = "Crop" in train.columns
    report.crop_col_in_test = "Crop" in test.columns
    report.category_col_in_train = "category" in train.columns
    report.category_col_in_test = "category" in test.columns
    report.target_col_in_train = report.crop_col_in_train
    report.target_col_in_test = report.crop_col_in_test

    if not report.crop_col_in_train:
        report.warnings.append("'Crop' column (primary target) NOT found in Train — unexpected.")
    if not report.crop_col_in_test:
        report.warnings.append("'Crop' column NOT found in Test.")
    if not report.category_col_in_train:
        report.warnings.append("'category' column (secondary target) NOT found in Train.")
    if report.category_col_in_test:
        report.warnings.append("'category' column found in Test — unexpected (present only in Train).")

    # ── Save report if requested ─────────────────────────────────────────────
    if output_dir is not None:
        report.save(output_dir)

    return report


# ── Quick self-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    # Import load_data from sibling module
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.data.load_data import load_both

    train, test = load_both(parse_geometry=False, as_geodataframe=False)
    report = validate(train, test, output_dir=Path("outputs/reports"))
    report.print_summary()
