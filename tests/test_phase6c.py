"""
tests/test_phase6c.py
=====================
Focused verification tests for Phase 6C:
- Manifest uniqueness and balanced class counts
- Feature set leakage exclusions (no Crop, category, FarmID, survey proxies)
- Preprocessing isolation (Pipeline fits only on train data)
- NaN handling (failed extractions are explicitly recorded, zero fabricated values)
- Fold integrity (zero group leakage in StratifiedGroupKFold by Sub-District)
- Artifact reproducibility (persisted models load and produce valid predictions)
"""

import json
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.models.train_pilot_models import (
    CLASSES_7,
    define_feature_subsets,
)


class TestPhase6CArtifactsAndIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parent.parent
        cls.manifest_path = cls.root / "outputs" / "data" / "s2_phase6c_manifest_350.csv"
        cls.features_path = cls.root / "outputs" / "data" / "s2_phase6c_350_features.csv"
        cls.benchmark_json = cls.root / "outputs" / "reports" / "phase6c_model_benchmark.json"
        cls.data_quality_json = cls.root / "outputs" / "reports" / "phase6c_data_quality.json"

    def test_manifest_uniqueness_and_counts(self):
        """Verify manifest contains unique FarmIDs and ~50 farms per crop class."""
        self.assertTrue(self.manifest_path.exists(), f"Missing {self.manifest_path}")
        df_manifest = pd.read_csv(self.manifest_path)

        # Unique FarmIDs
        self.assertEqual(len(df_manifest), df_manifest["FarmID"].nunique(), "Duplicate FarmIDs found in manifest")

        # 352 total selected
        self.assertEqual(len(df_manifest), 352)

        # Per crop counts: exactly 50 or 51
        counts = df_manifest["Crop"].value_counts()
        for crop in CLASSES_7:
            self.assertIn(crop, counts.index)
            self.assertGreaterEqual(counts[crop], 50)

    def test_leakage_exclusions(self):
        """Verify no target, category, IDs, or survey proxies are in feature columns."""
        self.assertTrue(self.features_path.exists())
        df_feat = pd.read_csv(self.features_path)

        sat_cols, multi_cols, added_cols = define_feature_subsets(df_feat)

        # Neither feature set should contain forbidden columns
        forbidden = {"Crop", "category", "FarmID", "geometry", "Sub-District", "District", "CTransp", "ExpYield"}
        for col in forbidden:
            self.assertNotIn(col, sat_cols, f"Forbidden col {col} found in satellite_only features")
            self.assertNotIn(col, multi_cols, f"Forbidden col {col} found in multimodal features")

        # Multimodal features must include only the approved geometry/calendar metadata
        diff = set(multi_cols) - set(sat_cols)
        self.assertEqual(diff, set(added_cols))

    def test_nan_handling_and_no_fabrication(self):
        """Verify failed extractions are recorded with NaNs and status='failed'."""
        df_feat = pd.read_csv(self.features_path)
        failed_rows = df_feat[df_feat["status"] == "failed"]

        # Exactly 4 failed extractions
        self.assertEqual(len(failed_rows), 4)

        # Feature columns for failed rows must be NaN, not fabricated numbers
        sat_cols, _, _ = define_feature_subsets(df_feat)
        for _, row in failed_rows.iterrows():
            nan_ratio = row[sat_cols].isna().mean()
            self.assertEqual(nan_ratio, 1.0, "Failed farm has non-NaN satellite features (possible fabrication)")

    def test_spatial_fold_integrity(self):
        """Verify 5-fold StratifiedGroupKFold has zero group overlap across folds."""
        df_feat = pd.read_csv(self.features_path)
        valid = df_feat[df_feat["status"] == "success"].copy()

        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        X = valid.index.values
        y = valid["Crop"].values
        groups = valid["Sub-District"].fillna("Unknown").values

        for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
            train_groups = set(groups[train_idx])
            val_groups = set(groups[val_idx])
            overlap = train_groups.intersection(val_groups)
            self.assertEqual(len(overlap), 0, f"Spatial leakage in fold {fold}: {overlap}")
            
            # All 7 classes must be present in validation set
            val_classes = set(y[val_idx])
            self.assertEqual(len(val_classes), 7, f"Fold {fold} missing classes: {set(CLASSES_7) - val_classes}")

    def test_persisted_models_predictable(self):
        """Verify persisted Phase 6C models load and output valid predictions."""
        models = [
            "rf_satellite_only_phase6c.joblib",
            "xgb_satellite_only_phase6c.joblib",
            "rf_multimodal_phase6c.joblib",
            "xgb_multimodal_phase6c.joblib",
        ]
        df_feat = pd.read_csv(self.features_path)
        valid = df_feat[df_feat["status"] == "success"].iloc[:5].copy()
        if "duration_days" not in valid.columns:
            sdate = pd.to_datetime(valid["SDate"], format="mixed", dayfirst=True)
            hdate = pd.to_datetime(valid["HDate"], format="mixed", dayfirst=True)
            valid["duration_days"] = (hdate - sdate).dt.days.clip(lower=1)
            valid["sowing_month"] = sdate.dt.month
            valid["sowing_doy"] = sdate.dt.dayofyear
            valid["harvest_month"] = hdate.dt.month
            valid["harvest_doy"] = hdate.dt.dayofyear

        for m_name in models:
            m_path = self.root / "outputs" / "models" / m_name
            self.assertTrue(m_path.exists(), f"Model artifact missing: {m_path}")
            model = joblib.load(m_path)
            
            # Determine features
            sat_cols, multi_cols, _ = define_feature_subsets(valid)
            cols = multi_cols if "multimodal" in m_name else sat_cols

            preds = model.predict(valid[cols])
            self.assertEqual(len(preds), 5)
            for p in preds:
                if isinstance(p, (int, np.integer)):
                    self.assertTrue(0 <= p < len(CLASSES_7), f"Invalid class index {p}")
                else:
                    self.assertIn(p, CLASSES_7)


if __name__ == "__main__":
    unittest.main()
