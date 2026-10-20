"""
tests/test_s2_mosaic.py
=======================
Unit tests for Sentinel-2 daily mosaicking of overlapping MGRS tiles.
Verifies that multiple same-date scenes are mosaicked before temporal reduction,
preventing duplicate weighting from overlapping swaths.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timezone

from src.satellite.s2_extractor import EarthEngineManager, mosaic_same_date_scenes


class TestDailyMosaicking(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ee_available = EarthEngineManager.initialize()

    def test_mosaic_same_date_scenes_reduces_to_unique_dates(self):
        """Verify that an ImageCollection with overlapping same-date scenes is collapsed to unique dates."""
        if not self.ee_available:
            self.skipTest("Google Earth Engine not authenticated or offline.")

        import ee

        # Create 4 synthetic constant images with known dates
        # Two on 2024-01-10 (simulating 2 overlapping MGRS tiles)
        t1 = int(datetime(2024, 1, 10, 5, 0, tzinfo=timezone.utc).timestamp() * 1000)
        t2 = int(datetime(2024, 1, 10, 5, 5, tzinfo=timezone.utc).timestamp() * 1000)
        t3 = int(datetime(2024, 1, 15, 5, 0, tzinfo=timezone.utc).timestamp() * 1000)
        t4 = int(datetime(2024, 1, 20, 5, 0, tzinfo=timezone.utc).timestamp() * 1000)

        img1 = ee.Image.constant(100).rename("B4").set("system:time_start", t1)
        img2 = ee.Image.constant(200).rename("B4").set("system:time_start", t2)
        img3 = ee.Image.constant(300).rename("B4").set("system:time_start", t3)
        img4 = ee.Image.constant(400).rename("B4").set("system:time_start", t4)

        raw_col = ee.ImageCollection([img1, img2, img3, img4])
        self.assertEqual(int(raw_col.size().getInfo()), 4)

        # Apply daily mosaicking
        daily_col = mosaic_same_date_scenes(raw_col)
        
        # Resulting collection must have exactly 3 images (3 unique dates)
        n_daily = int(daily_col.size().getInfo())
        self.assertEqual(n_daily, 3, f"Expected 3 daily images, got {n_daily}")

        # Check the extracted dates
        daily_dates = daily_col.aggregate_array("date").getInfo()
        self.assertEqual(sorted(daily_dates), ["2024-01-10", "2024-01-15", "2024-01-20"])

    def test_temporal_median_aggregates_across_dates_not_tiles(self):
        """Verify that temporal median operates over unique daily mosaics, not duplicate tiles."""
        if not self.ee_available:
            self.skipTest("Google Earth Engine not authenticated or offline.")

        import ee

        pt = ee.Geometry.Point([78.0, 17.0])

        # Date 1 (2024-01-10): two tiles with value 100 and 100
        t1 = int(datetime(2024, 1, 10, 5, 0, tzinfo=timezone.utc).timestamp() * 1000)
        t2 = int(datetime(2024, 1, 10, 5, 5, tzinfo=timezone.utc).timestamp() * 1000)
        # Date 2 (2024-01-15): value 200
        t3 = int(datetime(2024, 1, 15, 5, 0, tzinfo=timezone.utc).timestamp() * 1000)
        # Date 3 (2024-01-20): value 300
        t4 = int(datetime(2024, 1, 20, 5, 0, tzinfo=timezone.utc).timestamp() * 1000)

        img1 = ee.Image.constant(100).toFloat().rename("B4").set("system:time_start", t1)
        img2 = ee.Image.constant(100).toFloat().rename("B4").set("system:time_start", t2)
        img3 = ee.Image.constant(200).toFloat().rename("B4").set("system:time_start", t3)
        img4 = ee.Image.constant(300).toFloat().rename("B4").set("system:time_start", t4)

        # Raw collection: [100, 100, 200, 300] -> median = (100 + 200) / 2 = 150
        raw_col = ee.ImageCollection([img1, img2, img3, img4])
        raw_median = raw_col.median().reduceRegion(ee.Reducer.first(), pt, 10).getInfo()["B4"]
        self.assertAlmostEqual(raw_median, 150, delta=1.0)

        # Daily collection: dates [100, 200, 300] -> median = 200
        daily_col = mosaic_same_date_scenes(raw_col)
        daily_median = daily_col.median().reduceRegion(ee.Reducer.first(), pt, 10).getInfo()["B4"]
        self.assertAlmostEqual(daily_median, 200, delta=1.0)


if __name__ == "__main__":
    unittest.main()
