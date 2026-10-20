"""
tests/test_phase6d_audit.py
===========================
Phase 6D Audit Test Suite:
- Validates discrepancy resolution (352 manifest -> 4 failed [2 Chillies, 1 Maize, 1 Paddy] -> 348 clean evaluated)
- Validates that OOF predictions file exists, has exactly 348 rows, and zero NaN predictions
- Validates that recomputing metrics from phase6c_oof_predictions.csv exactly reproduces phase6c_model_benchmark.json
- Validates that Sub-District groups maintain zero fold overlap in the saved OOF predictions
"""

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


class TestPhase6DAuditAndReproducibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parent.parent
        cls.manifest_path = cls.root / "outputs" / "data" / "s2_phase6c_manifest_350.csv"
        cls.features_path = cls.root / "outputs" / "data" / "s2_phase6c_350_features.csv"
        cls.oof_path = cls.root / "outputs" / "data" / "phase6c_oof_predictions.csv"
        cls.benchmark_path = cls.root / "outputs" / "reports" / "phase6c_model_benchmark.json"

    def test_class_support_discrepancy_resolved(self):
        """Verify the 4 failed farms and exact per-class valid sample support."""
        df_feat = pd.read_csv(self.features_path)
        self.assertEqual(len(df_feat), 352)

        # 4 failed extractions
        failed = df_feat[df_feat["status"] == "failed"]
        self.assertEqual(len(failed), 4)
        
        # Verify specific failed crop counts: Chillies: 2, Maize: 1, Paddy: 1
        failed_counts = failed["Crop"].value_counts().to_dict()
        self.assertEqual(failed_counts.get("Chillies", 0), 2)
        self.assertEqual(failed_counts.get("Maize", 0), 1)
        self.assertEqual(failed_counts.get("Paddy", 0), 1)

        # Clean successful farms: 348 total
        clean = df_feat[df_feat["status"] == "success"]
        self.assertEqual(len(clean), 348)
        clean_counts = clean["Crop"].value_counts().to_dict()
        expected_support = {
            "Bengal Gram": 50,
            "Cotton": 50,
            "Groundnut": 50,
            "Maize": 50,
            "Red Gram": 50,
            "Chillies": 49,
            "Paddy": 49,
        }
        self.assertEqual(clean_counts, expected_support)

    def test_oof_predictions_integrity(self):
        """Verify saved OOF predictions file structure and row alignment."""
        self.assertTrue(self.oof_path.exists(), f"Missing {self.oof_path}")
        df_oof = pd.read_csv(self.oof_path)

        self.assertEqual(len(df_oof), 348)
        self.assertEqual(df_oof["FarmID"].nunique(), 348)

        # No nulls in predictions
        pred_cols = [c for c in df_oof.columns if c.startswith("pred_")]
        self.assertGreaterEqual(len(pred_cols), 5)
        for col in pred_cols:
            self.assertEqual(df_oof[col].isna().sum(), 0, f"Nulls found in {col}")

    def test_metric_reproducibility_from_oof(self):
        """Verify all benchmark JSON metrics can be reproduced from saved OOF predictions."""
        df_oof = pd.read_csv(self.oof_path)
        with open(self.benchmark_path, "r", encoding="utf-8") as f:
            bench = json.load(f)

        mapping = [
            ("pred_RF_satellite_only", "RF_satellite_only"),
            ("pred_XGB_satellite_only", "XGB_satellite_only"),
            ("pred_RF_multimodal", "RF_multimodal"),
            ("pred_XGB_multimodal", "XGB_multimodal"),
        ]

        for pred_col, model_key in mapping:
            acc = round(float(accuracy_score(df_oof["Crop"], df_oof[pred_col])), 4)
            macro_f1 = round(float(f1_score(df_oof["Crop"], df_oof[pred_col], average="macro", zero_division=0)), 4)
            weighted_f1 = round(float(f1_score(df_oof["Crop"], df_oof[pred_col], average="weighted", zero_division=0)), 4)

            expected_acc = bench["models"][model_key]["overall_oof_accuracy"]
            expected_macro_f1 = bench["models"][model_key]["overall_oof_macro_f1"]
            expected_weighted_f1 = bench["models"][model_key]["overall_oof_weighted_f1"]

            self.assertAlmostEqual(acc, expected_acc, places=4, msg=f"{model_key} Accuracy mismatch")
            self.assertAlmostEqual(macro_f1, expected_macro_f1, places=4, msg=f"{model_key} Macro-F1 mismatch")
            self.assertAlmostEqual(weighted_f1, expected_weighted_f1, places=4, msg=f"{model_key} Weighted-F1 mismatch")

    def test_zero_spatial_leakage_in_saved_oof(self):
        """Verify that folds in saved OOF predictions strictly contain disjoint Sub-Districts."""
        df_oof = pd.read_csv(self.oof_path)
        for fold in range(5):
            val_groups = set(df_oof[df_oof["fold"] == fold]["Sub-District"])
            train_groups = set(df_oof[df_oof["fold"] != fold]["Sub-District"])
            overlap = val_groups.intersection(train_groups)
            self.assertEqual(len(overlap), 0, f"Spatial leakage found in fold {fold}: {overlap}")


if __name__ == "__main__":
    unittest.main()
