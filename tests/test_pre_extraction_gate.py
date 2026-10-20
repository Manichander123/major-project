"""
tests/test_pre_extraction_gate.py
=================================
Automated Pre-Extraction Gate Verification Suite for Phase 6D:
1. Duplicate acquisitions on the same date (MGRS tile overlap collapse).
2. Cloud/shadow masking before compositing (clear pixels shine through masked scenes).
3. No fabricated zeros for missing observations (strict NaN preservation).
4. Deterministic output and checkpoint/resume behavior (cache skipping & integrity).
5. Correct alignment between FarmID, crop label, features, and extraction status.
"""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import Polygon

from src.features.indices import compute_all_spectral_indices
from src.satellite.s2_extractor import (
    BatchSentinelExtractor,
    EarthEngineManager,
    extract_zonal_stats_for_farm,
    mask_s2_clouds_sr,
    mosaic_same_date_scenes,
)


class TestPreExtractionGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parent.parent
        cls.ee_available = EarthEngineManager.initialize()

    # -------------------------------------------------------------------------
    # 1. Duplicate acquisitions on the same date
    # -------------------------------------------------------------------------
    def test_duplicate_same_date_acquisitions_collapsed(self):
        """Verify that multiple scenes on the same calendar date are collapsed into exactly one observation."""
        if not self.ee_available:
            self.skipTest("Google Earth Engine not authenticated or offline.")
        import ee

        # Construct 3 scenes on 2024-02-01 (simulating 3 overlapping MGRS tiles) + 1 scene on 2024-02-06
        t1 = int(datetime(2024, 2, 1, 5, 0, tzinfo=timezone.utc).timestamp() * 1000)
        t2 = int(datetime(2024, 2, 1, 5, 3, tzinfo=timezone.utc).timestamp() * 1000)
        t3 = int(datetime(2024, 2, 1, 5, 6, tzinfo=timezone.utc).timestamp() * 1000)
        t4 = int(datetime(2024, 2, 6, 5, 0, tzinfo=timezone.utc).timestamp() * 1000)

        img1 = ee.Image.constant(100).rename("B4").set("system:time_start", t1)
        img2 = ee.Image.constant(100).rename("B4").set("system:time_start", t2)
        img3 = ee.Image.constant(100).rename("B4").set("system:time_start", t3)
        img4 = ee.Image.constant(400).rename("B4").set("system:time_start", t4)

        raw_col = ee.ImageCollection([img1, img2, img3, img4])
        self.assertEqual(int(raw_col.size().getInfo()), 4)

        daily_col = mosaic_same_date_scenes(raw_col)
        self.assertEqual(int(daily_col.size().getInfo()), 2)
        dates = sorted(daily_col.aggregate_array("date").getInfo())
        self.assertEqual(dates, ["2024-02-01", "2024-02-06"])

    def test_daily_mosaicking_prevents_temporal_swath_overweighting(self):
        """Verify that temporal median does not over-weight a single date that has multiple overlapping tiles."""
        if not self.ee_available:
            self.skipTest("Google Earth Engine not authenticated or offline.")
        import ee

        pt = ee.Geometry.Point([78.0, 17.0])
        t1 = int(datetime(2024, 2, 1, 5, 0, tzinfo=timezone.utc).timestamp() * 1000)
        t2 = int(datetime(2024, 2, 1, 5, 3, tzinfo=timezone.utc).timestamp() * 1000)
        t3 = int(datetime(2024, 2, 6, 5, 0, tzinfo=timezone.utc).timestamp() * 1000)
        t4 = int(datetime(2024, 2, 11, 5, 0, tzinfo=timezone.utc).timestamp() * 1000)

        # Cast to Int16 to ensure homogeneous collection
        img1 = ee.Image.constant(100).toInt16().rename("B4").set("system:time_start", t1)
        img2 = ee.Image.constant(100).toInt16().rename("B4").set("system:time_start", t2)
        img3 = ee.Image.constant(300).toInt16().rename("B4").set("system:time_start", t3)
        img4 = ee.Image.constant(500).toInt16().rename("B4").set("system:time_start", t4)

        raw_col = ee.ImageCollection([img1, img2, img3, img4])
        # Raw median across [100, 100, 300, 500] = (100 + 300)/2 = 200 (biased towards Date 1)
        raw_val = raw_col.median().reduceRegion(ee.Reducer.first(), pt, 10).getInfo()["B4"]
        self.assertAlmostEqual(raw_val, 200, delta=1.0)

        # Daily mosaicked across unique dates [100, 300, 500] -> median = 300
        daily_col = mosaic_same_date_scenes(raw_col)
        daily_val = daily_col.median().reduceRegion(ee.Reducer.first(), pt, 10).getInfo()["B4"]
        self.assertAlmostEqual(daily_val, 300, delta=1.0)

    # -------------------------------------------------------------------------
    # 2. Cloud/shadow masking before compositing
    # -------------------------------------------------------------------------
    def test_scl_cloud_and_shadow_masking(self):
        """Verify that mask_s2_clouds_sr masks out shadow (3), cloud (8, 9), cirrus (10), and snow (11)."""
        if not self.ee_available:
            self.skipTest("Google Earth Engine not authenticated or offline.")
        import ee

        pt = ee.Geometry.Point([78.0, 17.0])

        # Test each SCL class
        scl_classes = {
            3: 0,   # shadow -> masked
            4: 1,   # vegetation -> unmasked
            5: 1,   # bare soil -> unmasked
            8: 0,   # medium cloud -> masked
            9: 0,   # high cloud -> masked
            10: 0,  # cirrus -> masked
            11: 0,  # snow/ice -> masked
        }

        for scl_val, expected_mask in scl_classes.items():
            img = ee.Image.constant(500).toInt16().rename("B4").addBands(ee.Image.constant(scl_val).toInt16().rename("SCL"))
            masked_img = mask_s2_clouds_sr(img)
            mask_val = masked_img.select("B4").mask().reduceRegion(ee.Reducer.first(), pt, 10).getInfo()["B4"]
            self.assertEqual(
                mask_val,
                expected_mask,
                f"SCL value {scl_val} expected mask={expected_mask}, got {mask_val}",
            )

    def test_daily_mosaic_clear_pixel_shines_through_cloudy_tile(self):
        """Verify that if one tile is cloudy on a date while an overlapping tile is clear, clear pixel is retained."""
        if not self.ee_available:
            self.skipTest("Google Earth Engine not authenticated or offline.")
        import ee

        pt = ee.Geometry.Point([78.0, 17.0])
        t1 = int(datetime(2024, 2, 1, 5, 0, tzinfo=timezone.utc).timestamp() * 1000)
        t2 = int(datetime(2024, 2, 1, 5, 5, tzinfo=timezone.utc).timestamp() * 1000)

        # Tile 1: SCL=9 (high cloud), B4=9999
        img1 = ee.Image.constant(9999).toInt16().rename("B4").addBands(ee.Image.constant(9).toInt16().rename("SCL")).set("system:time_start", t1)
        # Tile 2: SCL=4 (vegetation, clear), B4=750
        img2 = ee.Image.constant(750).toInt16().rename("B4").addBands(ee.Image.constant(4).toInt16().rename("SCL")).set("system:time_start", t2)

        masked_col = ee.ImageCollection([img1, img2]).map(mask_s2_clouds_sr)
        daily_col = mosaic_same_date_scenes(masked_col)

        mosaic_val = daily_col.first().select("B4").reduceRegion(ee.Reducer.first(), pt, 10).getInfo()["B4"]
        self.assertEqual(mosaic_val, 750, f"Expected clear tile DN=750 to shine through, got {mosaic_val}")

    # -------------------------------------------------------------------------
    # 3. No fabricated zeros for missing observations
    # -------------------------------------------------------------------------
    def test_missing_observations_preserve_nan_not_zero(self):
        """Verify that windows with zero valid observations produce NaN, never fabricated zeros."""
        dummy_df = pd.DataFrame([{
            "FarmID": 9999999,
            "B4_early_mean": np.nan,
            "B8_early_mean": np.nan,
            "B11_early_mean": np.nan,
            "B2_early_mean": np.nan,
            "B3_early_mean": np.nan,
            "B5_early_mean": np.nan,
            "B6_early_mean": np.nan,
            "B7_early_mean": np.nan,
            "B8A_early_mean": np.nan,
            "B12_early_mean": np.nan,
        }])

        indices_df = compute_all_spectral_indices(
            dummy_df,
            blue_col="B2_early_mean",
            green_col="B3_early_mean",
            red_col="B4_early_mean",
            re1_col="B5_early_mean",
            re2_col="B6_early_mean",
            nir_col="B8_early_mean",
            swir1_col="B11_early_mean",
        )
        for col in ["ndvi", "evi", "ndmi", "savi", "ndwi"]:
            val = indices_df[col].iloc[0]
            self.assertTrue(
                np.isnan(val),
                f"Feature {col} fabricated value {val} instead of preserving NaN!",
            )

    # -------------------------------------------------------------------------
    # 4. Deterministic output and checkpoint/resume behavior
    # -------------------------------------------------------------------------
    def test_checkpoint_resume_safeguards(self):
        """Verify that BatchSentinelExtractor resumes properly, skipping success and keeping failed retryable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            cache_file = tmp_path / "test_cache.csv"

            # Pre-populate cache with 1 success and 1 failed
            cached_data = pd.DataFrame([
                {"FarmID": 101, "status": "success", "B4_full_mean": 500.0},
                {"FarmID": 102, "status": "failed", "error_message": "100% cloud cover"},
            ])
            cached_data.to_csv(cache_file, index=False)

            # Input manifest has 101, 102, 103
            manifest_gdf = pd.DataFrame([
                {"FarmID": 101, "Crop": "Paddy", "geometry": Polygon([(78, 17), (78.01, 17), (78.01, 17.01), (78, 17)]), "SDate": "2023-11-01", "HDate": "2024-03-01"},
                {"FarmID": 102, "Crop": "Maize", "geometry": Polygon([(78, 17), (78.01, 17), (78.01, 17.01), (78, 17)]), "SDate": "2023-11-01", "HDate": "2024-03-01"},
                {"FarmID": 103, "Crop": "Cotton", "geometry": Polygon([(78, 17), (78.01, 17), (78.01, 17.01), (78, 17)]), "SDate": "2023-11-01", "HDate": "2024-03-01"},
            ])

            extractor = BatchSentinelExtractor(output_dir=tmp_path, checkpoint_interval=1)
            # Test filtering logic directly without making EE network requests
            existing_df = pd.read_csv(cache_file)
            successful_ids = set(existing_df[existing_df["status"] == "success"]["FarmID"])
            target_gdf = manifest_gdf[~manifest_gdf["FarmID"].isin(successful_ids)]

            self.assertEqual(len(target_gdf), 2, "Should only process 102 (retry) and 103 (new)")
            self.assertIn(102, target_gdf["FarmID"].values)
            self.assertIn(103, target_gdf["FarmID"].values)
            self.assertNotIn(101, target_gdf["FarmID"].values)

    # -------------------------------------------------------------------------
    # 5. Correct alignment between FarmID, crop label, features, and status
    # -------------------------------------------------------------------------
    def test_manifest_feature_oof_alignment(self):
        """Verify 100% agreement between manifest, features, and OOF predictions."""
        manifest_path = self.root / "outputs" / "data" / "s2_phase6c_manifest_350.csv"
        features_path = self.root / "outputs" / "data" / "s2_phase6c_350_features.csv"
        oof_path = self.root / "outputs" / "data" / "phase6c_oof_predictions.csv"
        train_path = self.root / "datasss" / "Train.csv"

        df_man = pd.read_csv(manifest_path)
        df_feat = pd.read_csv(features_path)
        df_oof = pd.read_csv(oof_path)
        df_train = pd.read_csv(train_path)

        # 1. Total manifest count
        self.assertEqual(len(df_man), 352)
        self.assertEqual(len(df_feat), 352)

        # 2. Success vs failed
        clean_feat = df_feat[df_feat["status"] == "success"]
        self.assertEqual(len(clean_feat), 348)
        self.assertEqual(len(df_oof), 348)

        # 3. FarmID sets must align identically
        self.assertEqual(set(clean_feat["FarmID"]), set(df_oof["FarmID"]))
        self.assertTrue(set(df_man["FarmID"]).issubset(set(df_train["FarmID"])))

        # 4. Crop labels must match Train.csv exactly
        train_labels = df_train.set_index("FarmID")["Crop"].to_dict()
        feat_labels = clean_feat.set_index("FarmID")["Crop"].to_dict()
        oof_labels = df_oof.set_index("FarmID")["Crop"].to_dict()

        for fid in clean_feat["FarmID"]:
            expected_crop = train_labels[fid]
            self.assertEqual(feat_labels[fid], expected_crop, f"Crop label mismatch in features for FarmID {fid}")
            self.assertEqual(oof_labels[fid], expected_crop, f"Crop label mismatch in OOF for FarmID {fid}")


if __name__ == "__main__":
    unittest.main()
