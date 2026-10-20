"""
tests/test_baseline.py
======================
Comprehensive unit and integration tests for baseline Random Forest:
- Input validation (missing columns, unparseable dates, negative duration, invalid targets)
- Leakage prevention (ensuring survey proxies and secondary targets never enter features)
- Pipeline construction with OrdinalEncoder and SimpleImputer inside sklearn Pipeline
- Handling unseen categories at inference
- Model serialization & persistence verification
- Guaranteed end-to-end execution without silent skips
"""

import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.models.baseline_rf import (
    CLASSES_7,
    EXCLUDED_COLS,
    TARGET_COL,
    build_pipeline,
    prepare_features,
    run_baseline,
    validate_inputs,
)

# Realistic crop cycle durations in days for test generation
CROP_CYCLE_DAYS = {
    "Bengal Gram": 90,
    "Chillies": 180,
    "Cotton": 210,
    "Groundnut": 105,
    "Maize": 120,
    "Paddy": 150,
    "Red Gram": 240,
}


def _create_valid_sample_df(n_per_class: int = 10) -> pd.DataFrame:
    """Helper to generate a clean synthetic DataFrame with all 7 classes."""
    records = []
    classes = list(CLASSES_7)
    sdate = pd.Timestamp("2023-11-01")

    count = 0
    for rep in range(n_per_class):
        for c in classes:
            dur = CROP_CYCLE_DAYS[c]
            hdate = sdate + pd.Timedelta(days=dur)
            records.append({
                "FarmID": 1000 + count,
                "Crop": c,
                "category": count % 4,
                "SDate": sdate.strftime("%Y-%m-%d"),
                "HDate": hdate.strftime("%Y-%m-%d"),
                "Season": "Rabi" if c in ["Bengal Gram", "Groundnut"] else "Kharif",
                "District": f"District_{count % 3}",
                "Sub-District": f"Mandal_{count % 5}",
                "CTransp": "Transplanting" if c == "Paddy" else "None",
                "ExpYield": 20 + (count % 10),
                "CLast": "None",
                "CNext": "None",
                "CropCoveredArea": 85.0,
                "CHeight": 60.0,
                "WaterCov": 10.0,
                "IrriType": "Flood",
                "IrriSource": "Canal",
                "IrriCount": 4,
                "area_ha": 0.5 + (count * 0.05),
                "perimeter_m": 120.0 + count,
                "compactness": 0.75,
                "vertex_count": 5,
                "centroid_lon": 78.1 + (count * 0.01),
                "centroid_lat": 17.5 + (count * 0.01),
                "bbox_min_lon": 78.09 + (count * 0.01),
                "bbox_min_lat": 17.49 + (count * 0.01),
                "bbox_max_lon": 78.11 + (count * 0.01),
                "bbox_max_lat": 17.51 + (count * 0.01),
            })
            count += 1
    return pd.DataFrame(records)


class TestBaselineValidationAndLeakage(unittest.TestCase):

    def setUp(self):
        self.valid_df = _create_valid_sample_df(n_per_class=2)

    def test_target_and_classes(self):
        self.assertEqual(TARGET_COL, "Crop")
        self.assertEqual(len(CLASSES_7), 7)
        self.assertIn("Paddy", CLASSES_7)
        self.assertIn("Cotton", CLASSES_7)

    def test_leakage_exclusion(self):
        """Verify no survey proxies or secondary targets leak into prepared features."""
        X, y, cat_cols, num_cols = prepare_features(self.valid_df, require_target=True)
        for col in EXCLUDED_COLS:
            self.assertNotIn(col, X.columns, f"Leakage column '{col}' found in feature matrix X!")

        self.assertEqual(len(y), len(self.valid_df))
        self.assertIn("duration_days", X.columns)
        self.assertIn("area_ha", X.columns)

    def test_invalid_input_missing_columns(self):
        """Ensure missing required base or geometry columns raise clear ValueError."""
        bad_df = self.valid_df.drop(columns=["SDate"])
        with self.assertRaises(ValueError) as ctx:
            validate_inputs(bad_df)
        self.assertIn("Missing required base column(s)", str(ctx.exception))

        bad_geom_df = self.valid_df.drop(columns=["area_ha"])
        with self.assertRaises(ValueError) as ctx:
            validate_inputs(bad_geom_df)
        self.assertIn("Missing required geometry column(s)", str(ctx.exception))

    def test_invalid_input_unparseable_dates(self):
        """Ensure unparseable dates raise descriptive ValueError."""
        bad_df = self.valid_df.copy()
        bad_df.loc[0, "SDate"] = "not_a_valid_date_xyz"
        with self.assertRaises(ValueError) as ctx:
            validate_inputs(bad_df)
        self.assertIn("Failed to parse 'SDate' or 'HDate'", str(ctx.exception))

    def test_invalid_input_negative_duration(self):
        """Ensure HDate on or before SDate raises descriptive ValueError."""
        bad_df = self.valid_df.copy()
        bad_df.loc[0, "SDate"] = "2024-05-01"
        bad_df.loc[0, "HDate"] = "2024-04-01"
        with self.assertRaises(ValueError) as ctx:
            validate_inputs(bad_df)
        self.assertIn("harvest date (HDate) is on or before sowing date (SDate)", str(ctx.exception))

    def test_invalid_input_nan_or_unknown_target(self):
        """Ensure NaN or unrecognized targets raise descriptive ValueError."""
        bad_df_nan = self.valid_df.copy()
        bad_df_nan.loc[0, "Crop"] = np.nan
        with self.assertRaises(ValueError) as ctx:
            validate_inputs(bad_df_nan, require_target=True)
        self.assertIn("missing/NaN values", str(ctx.exception))

        bad_df_unknown = self.valid_df.copy()
        bad_df_unknown.loc[0, "Crop"] = "Pineapple"
        with self.assertRaises(ValueError) as ctx:
            validate_inputs(bad_df_unknown, require_target=True)
        self.assertIn("Unrecognized crop class(es)", str(ctx.exception))


class TestPipelineAndPersistence(unittest.TestCase):

    def setUp(self):
        self.df = _create_valid_sample_df(n_per_class=10)
        self.X, self.y, self.cat_cols, self.num_cols = prepare_features(self.df, require_target=True)

    def test_unseen_categories_inference(self):
        """Verify pipeline handles unseen categorical levels at test time without failing."""
        pipeline = build_pipeline(
            cat_cols=self.cat_cols,
            num_cols=self.num_cols,
            n_estimators=10,
            random_state=42,
        )
        pipeline.fit(self.X, self.y)

        # Create test row with brand new unseen District and Sub-District
        unseen_test = self.X.iloc[:2].copy()
        unseen_test["District"] = "NeverSeenBeforeDistrict"
        unseen_test["Sub-District"] = "NeverSeenBeforeMandal"

        preds = pipeline.predict(unseen_test)
        self.assertEqual(len(preds), 2)
        for p in preds:
            self.assertIn(p, CLASSES_7)

    def test_pipeline_persistence(self):
        """Verify the full pipeline can be serialized, loaded, and produces identical predictions."""
        pipeline = build_pipeline(
            cat_cols=self.cat_cols,
            num_cols=self.num_cols,
            n_estimators=10,
            random_state=42,
        )
        pipeline.fit(self.X, self.y)
        preds_original = pipeline.predict(self.X)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_pipeline.joblib"
            joblib.dump(pipeline, model_path)
            self.assertTrue(model_path.exists())

            loaded_pipeline = joblib.load(model_path)
            preds_loaded = loaded_pipeline.predict(self.X)
            np.testing.assert_array_equal(preds_original, preds_loaded)

    def test_guaranteed_execution_run_baseline(self):
        """Guaranteed execution test that never silently skips."""
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_csv = Path(tmpdir) / "synth_data.csv"
            self.df.to_csv(temp_csv, index=False)

            results = run_baseline(
                data_path=str(temp_csv),
                test_size=0.2,
                random_state=42,
                save_artifacts=False,
            )

            self.assertIn("random_forest", results)
            self.assertIn("majority_baseline", results)
            self.assertFalse(results["is_satellite_model"])
            self.assertFalse(results["spatial_generalization_claimed"])
            self.assertEqual(results["validation_strategy"], "preliminary_stratified_split")
            self.assertGreater(results["random_forest"]["accuracy"], 0.70)


if __name__ == "__main__":
    unittest.main()
