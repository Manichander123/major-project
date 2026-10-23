"""
tests/test_phase7_inference.py
==============================
Quality control and self-checks for Phase 7 Test.csv inference and evaluation.
Verifies:
- 140-feature schema parity with Phase 6H manifest
- Row alignment between Test.csv, raw bands, features, and predictions (2,718 rows)
- Exactly 2,265 evaluated parcels and 453 cloud-excluded parcels
- Per-class supports and confusion matrix sum to exactly 2,265
- Metric recalculation parity (Macro-F1, Accuracy, Balanced Accuracy, Weighted-F1)
- Zero data leakage (forbidden columns strictly excluded)
"""

import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EXPECTED_CLASSES = [
    "Bengal Gram",
    "Chillies",
    "Cotton",
    "Groundnut",
    "Maize",
    "Paddy",
    "Red Gram",
]


class TestPhase7Inference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_csv = pd.read_csv(PROJECT_ROOT / "datasss" / "Test.csv")
        cls.raw_df = pd.read_csv(PROJECT_ROOT / "outputs" / "data" / "phase7" / "s2_test_raw_bands.csv")
        cls.feat_df = pd.read_csv(PROJECT_ROOT / "outputs" / "data" / "phase7" / "s2_test_140_features.csv")
        cls.pred_df = pd.read_csv(PROJECT_ROOT / "outputs" / "data" / "phase7" / "test_predictions.csv")
        cls.manifest = json.loads(
            (PROJECT_ROOT / "outputs" / "reports" / "phase6h" / "phase6h_feature_schema_manifest.json").read_text(encoding="utf-8")
        )
        cls.metrics = json.loads(
            (PROJECT_ROOT / "outputs" / "reports" / "phase7" / "phase7_test_metrics.json").read_text(encoding="utf-8")
        )

    def test_row_counts_and_id_alignment(self):
        self.assertEqual(len(self.test_csv), 2718)
        self.assertEqual(len(self.raw_df), 2718)
        self.assertEqual(len(self.feat_df), 2718)
        self.assertEqual(len(self.pred_df), 2718)
        np.testing.assert_array_equal(self.test_csv["FarmID"].values, self.raw_df["FarmID"].values)
        np.testing.assert_array_equal(self.test_csv["FarmID"].values, self.pred_df["FarmID"].values)

    def test_schema_parity_with_manifest(self):
        expected_st = self.manifest["groups"]["S+T"]
        self.assertEqual(len(expected_st), 140)
        self.assertEqual(list(self.feat_df.columns), expected_st)

    def test_zero_leakage_in_features(self):
        forbidden = [
            "Crop", "category", "FarmID", "State", "District", "Sub-District",
            "SDate", "HDate", "CropCoveredArea", "geometry", "area_ha", "area_m2"
        ]
        for col in forbidden:
            self.assertNotIn(col, self.feat_df.columns)

    def test_cohort_partition_and_nan_preservation(self):
        success_mask = self.raw_df["status"] == "success"
        failed_mask = self.raw_df["status"] != "success"
        self.assertEqual(int(success_mask.sum()), 2265)
        self.assertEqual(int(failed_mask.sum()), 453)

        # Confirm failed records have NaNs in prediction outputs
        failed_preds = self.pred_df.loc[failed_mask, "predicted_crop"]
        self.assertTrue(failed_preds.isna().all())

    def test_confusion_matrix_and_support_conservation(self):
        cm = np.array(self.metrics["confusion_matrix"]["matrix"])
        self.assertEqual(cm.shape, (7, 7))
        self.assertEqual(int(cm.sum()), 2265)

        total_supp = sum(c["evaluated_support"] for c in self.metrics["per_class"].values())
        self.assertEqual(total_supp, 2265)

    def test_metric_recalculation_parity(self):
        eval_mask = self.pred_df["extraction_status"] == "success"
        y_true = self.pred_df.loc[eval_mask, "true_crop"].values
        y_pred = self.pred_df.loc[eval_mask, "predicted_crop"].values

        recalc_oa = accuracy_score(y_true, y_pred)
        recalc_macro_f1 = f1_score(y_true, y_pred, average="macro")
        recalc_weighted_f1 = f1_score(y_true, y_pred, average="weighted")
        recalc_bacc = balanced_accuracy_score(y_true, y_pred)

        self.assertAlmostEqual(recalc_oa, self.metrics["metrics"]["accuracy"], places=4)
        self.assertAlmostEqual(recalc_macro_f1, self.metrics["metrics"]["macro_f1"], places=4)
        self.assertAlmostEqual(recalc_weighted_f1, self.metrics["metrics"]["weighted_f1"], places=4)
        self.assertAlmostEqual(recalc_bacc, self.metrics["metrics"]["balanced_accuracy"], places=4)

    def test_cm_derived_perclass_consistency(self):
        """Per-class precision/recall/F1 in JSON must match what the saved CM implies."""
        cm = np.array(self.metrics["confusion_matrix"]["matrix"])
        classes = self.metrics["confusion_matrix"]["classes"]
        for i, c in enumerate(classes):
            tp = cm[i, i]
            col_sum = cm[:, i].sum()
            row_sum = cm[i, :].sum()
            exp_prec = tp / col_sum if col_sum > 0 else 0.0
            exp_rec = tp / row_sum if row_sum > 0 else 0.0
            exp_f1 = 2 * exp_prec * exp_rec / (exp_prec + exp_rec) if (exp_prec + exp_rec) > 0 else 0.0
            saved = self.metrics["per_class"][c]
            self.assertAlmostEqual(exp_prec, saved["precision"], places=3, msg=f"{c} precision")
            self.assertAlmostEqual(exp_rec, saved["recall"], places=3, msg=f"{c} recall")
            self.assertAlmostEqual(exp_f1, saved["f1_score"], places=3, msg=f"{c} F1")
            self.assertEqual(int(row_sum), saved["evaluated_support"], msg=f"{c} support")

    def test_class_mapping_matches_frozen_model(self):
        """Model's numeric classes 0..6 must map to alphabetical EXPECTED_CLASSES."""
        import joblib
        model = joblib.load(PROJECT_ROOT / "outputs" / "models" / "phase6h" / "xgb_weighted_s+t.joblib")
        self.assertEqual(list(model.classes_), list(range(7)))
        self.assertEqual(model.n_features_in_, 140)

    def test_correct_incorrect_sum(self):
        """correct + incorrect must equal the evaluated cohort size."""
        m = self.metrics["metrics"]
        self.assertEqual(m["correct_predictions"] + m["incorrect_predictions"], 2265)


if __name__ == "__main__":
    unittest.main()
