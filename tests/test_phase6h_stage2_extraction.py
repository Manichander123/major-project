"""
tests/test_phase6h_stage2_extraction.py
=======================================
Phase 6H Stage 2 tests for spatial texture extraction data integrity.
Validates:
  1. Pilot output schema, unique IDs, NaN-preservation, finite-value sanity.
  2. Full extraction cohort coverage, feature counts, reproducibility.
  3. Protected baseline integrity before and after extraction.
  4. No zero-filling of missing texture values.
"""

from __future__ import annotations

import hashlib
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.satellite.s2_spatial_texture import GLCM_FEATURE_NAMES, GLCM_METRICS, GLCM_WINDOWS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PHASE6H_DATA_DIR = PROJECT_ROOT / "outputs" / "data" / "phase6h"
PHASE6H_REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports" / "phase6h"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class TestPhase6HStage2Pilot(unittest.TestCase):
    """Tests against pilot output (runs after pilot extraction)."""

    @classmethod
    def setUpClass(cls):
        cls.pilot_path = PHASE6H_DATA_DIR / "phase6h_pilot_texture.csv"
        cls.pilot_exists = cls.pilot_path.exists()
        if cls.pilot_exists:
            cls.pilot_df = pd.read_csv(cls.pilot_path)
        else:
            cls.pilot_df = None

    def test_01_pilot_file_exists(self):
        """Pilot output file must exist after pilot extraction."""
        if not self.pilot_exists:
            self.skipTest("Pilot not yet run")
        self.assertTrue(self.pilot_path.exists())

    def test_02_pilot_has_all_20_features(self):
        """Pilot output must contain all 20 GLCM feature columns."""
        if self.pilot_df is None:
            self.skipTest("Pilot not yet run")
        for feat in GLCM_FEATURE_NAMES:
            self.assertIn(feat, self.pilot_df.columns, f"Missing feature column: {feat}")

    def test_03_pilot_unique_farm_ids(self):
        """Each farm ID must appear exactly once."""
        if self.pilot_df is None:
            self.skipTest("Pilot not yet run")
        self.assertEqual(self.pilot_df["FarmID"].nunique(), len(self.pilot_df))

    def test_04_pilot_no_artificial_zeros(self):
        """Successful farms must not have ALL 20 features as exactly 0.0 (sign of zero-fill)."""
        if self.pilot_df is None:
            self.skipTest("Pilot not yet run")
        success = self.pilot_df[self.pilot_df["status"] == "success"]
        if success.empty:
            self.skipTest("No successful pilot farms")
        for _, row in success.iterrows():
            vals = [row.get(f, np.nan) for f in GLCM_FEATURE_NAMES]
            # If all finite values are 0.0, that's suspicious
            finite = [v for v in vals if pd.notna(v)]
            if finite:
                self.assertFalse(
                    all(v == 0.0 for v in finite),
                    f"FarmID {row['FarmID']} has all finite texture values = 0.0 (zero-fill suspected)"
                )

    def test_05_pilot_nan_preserved_for_failed(self):
        """Failed farms must have NaN for all texture features, never zeros."""
        if self.pilot_df is None:
            self.skipTest("Pilot not yet run")
        failed = self.pilot_df[self.pilot_df["status"] == "failed"]
        if failed.empty:
            self.skipTest("No failed pilot farms")
        for _, row in failed.iterrows():
            for feat in GLCM_FEATURE_NAMES:
                val = row.get(feat)
                # Failed farms: feature should be NaN (or missing)
                if pd.notna(val):
                    # If present, should not be zero-filled
                    self.assertTrue(
                        np.isnan(val) if isinstance(val, float) else True,
                        f"FarmID {row['FarmID']} feature {feat} should be NaN but got {val}"
                    )

    def test_06_pilot_no_infinite_values(self):
        """No infinite values in texture features."""
        if self.pilot_df is None:
            self.skipTest("Pilot not yet run")
        for feat in GLCM_FEATURE_NAMES:
            if feat in self.pilot_df.columns:
                vals = self.pilot_df[feat].dropna()
                inf_count = np.isinf(vals.values).sum()
                self.assertEqual(inf_count, 0, f"Feature {feat} has {inf_count} infinite values")


class TestPhase6HStage2FullExtraction(unittest.TestCase):
    """Tests against full cohort extraction output."""

    @classmethod
    def setUpClass(cls):
        cls.full_path = PHASE6H_DATA_DIR / "phase6h_texture_all_farms.csv"
        cls.full_exists = cls.full_path.exists()
        if cls.full_exists:
            cls.full_df = pd.read_csv(cls.full_path)
        else:
            cls.full_df = None

    def test_01_full_output_exists(self):
        """Full extraction output must exist."""
        if not self.full_exists:
            self.skipTest("Full extraction not yet run")
        self.assertTrue(self.full_path.exists())

    def test_02_cohort_coverage(self):
        """Full output must cover the 7,843 eligible farm cohort."""
        if self.full_df is None:
            self.skipTest("Full extraction not yet run")
        # Must have at least 7,843 rows (may have more if retries recorded)
        unique_ids = self.full_df["FarmID"].nunique()
        self.assertGreaterEqual(unique_ids, 7843, f"Only {unique_ids} unique farms, expected >= 7,843")

    def test_03_all_20_features_present(self):
        """All 20 GLCM features must be present as columns."""
        if self.full_df is None:
            self.skipTest("Full extraction not yet run")
        for feat in GLCM_FEATURE_NAMES:
            self.assertIn(feat, self.full_df.columns)

    def test_04_unique_farm_ids(self):
        """No duplicate farm IDs after deduplication."""
        if self.full_df is None:
            self.skipTest("Full extraction not yet run")
        self.assertEqual(self.full_df["FarmID"].nunique(), len(self.full_df))

    def test_05_no_artificial_zeros(self):
        """Spot-check: successful farms should not have ALL 20 features = 0.0."""
        if self.full_df is None:
            self.skipTest("Full extraction not yet run")
        success = self.full_df[self.full_df["status"] == "success"]
        if success.empty:
            self.skipTest("No successful farms")
        # Sample up to 100 for speed
        sample = success.sample(n=min(100, len(success)), random_state=42)
        for _, row in sample.iterrows():
            finite = [row[f] for f in GLCM_FEATURE_NAMES if pd.notna(row.get(f))]
            if finite:
                self.assertFalse(
                    all(v == 0.0 for v in finite),
                    f"FarmID {row['FarmID']} has all finite texture = 0.0"
                )

    def test_06_no_infinite_values(self):
        """No infinite values across all texture features."""
        if self.full_df is None:
            self.skipTest("Full extraction not yet run")
        for feat in GLCM_FEATURE_NAMES:
            if feat in self.full_df.columns:
                vals = self.full_df[feat].dropna()
                inf_count = np.isinf(vals.values).sum()
                self.assertEqual(inf_count, 0, f"Feature {feat} has {inf_count} infinite values")

    def test_07_feature_naming_convention(self):
        """All GLCM feature columns follow glcm_{metric}_{window} naming."""
        if self.full_df is None:
            self.skipTest("Full extraction not yet run")
        for feat in GLCM_FEATURE_NAMES:
            parts = feat.split("_")
            self.assertEqual(parts[0], "glcm")
            self.assertIn(parts[1], set(GLCM_METRICS))
            self.assertIn(parts[2], set(GLCM_WINDOWS))


class TestPhase6HBaselineProtection(unittest.TestCase):
    """Verify key protected baselines were not mutated."""

    def test_train_csv_intact(self):
        """Train.csv must not be mutated."""
        path = PROJECT_ROOT / "datasss" / "Train.csv"
        if not path.exists():
            self.skipTest("Train.csv not found")
        expected = "29d31920f5b9aaea8213d7993dedc50c5d19458906f85642d360d785bb07aa06"
        actual = sha256_file(path)
        self.assertEqual(actual, expected, "Train.csv has been corrupted!")

    def test_harmonized_cohort_intact(self):
        """Harmonized cohort must not be mutated."""
        path = PROJECT_ROOT / "outputs" / "data" / "s2_phase6f_harmonized_cohort.csv"
        if not path.exists():
            self.skipTest("Harmonized cohort not found")
        expected = "caed402949eb577425b9d2e2cac0ee2c8462f635ca22b466a7675962ebb5a7af"
        actual = sha256_file(path)
        self.assertEqual(actual, expected, "Harmonized cohort has been corrupted!")


if __name__ == "__main__":
    unittest.main()
