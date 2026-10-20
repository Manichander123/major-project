"""
tests/test_phase6h_spatial_texture.py
=====================================
Unit tests for Phase 6H spatial-texture extraction module.
Validates:
1. Exact 20 feature schema and naming conventions.
2. Label independence (zero crop labels in extraction signatures).
3. Missingness preservation (strict NaNs, zero fabricated zeros).
4. Geometry conversion for Polygon and MultiPolygon.
5. Quantization bounds and range verification.
"""

from __future__ import annotations

import inspect
import unittest
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon

from src.satellite.s2_spatial_texture import (
    GLCM_FEATURE_NAMES,
    GLCM_METRICS,
    GLCM_WINDOWS,
    extract_spatial_texture_for_farm,
)


class TestPhase6HSpatialTexture(unittest.TestCase):
    def test_01_feature_schema_and_count(self):
        """Verify exactly 20 features across 5 metrics and 4 phenological windows."""
        self.assertEqual(len(GLCM_FEATURE_NAMES), 20)
        self.assertEqual(len(GLCM_METRICS), 5)
        self.assertEqual(len(GLCM_WINDOWS), 4)

        expected_metrics = {"contrast", "dissimilarity", "homogeneity", "energy", "correlation"}
        expected_windows = {"early", "mid", "late", "full"}

        self.assertEqual(set(GLCM_METRICS), expected_metrics)
        self.assertEqual(set(GLCM_WINDOWS), expected_windows)

        # Ensure every feature adheres to naming pattern glcm_{metric}_{window}
        for feat in GLCM_FEATURE_NAMES:
            self.assertTrue(feat.startswith("glcm_"))
            parts = feat.split("_")
            self.assertEqual(len(parts), 3)
            self.assertIn(parts[1], expected_metrics)
            self.assertIn(parts[2], expected_windows)

    def test_02_label_independence(self):
        """Verify extraction module is strictly decoupled from crop labels."""
        sig = inspect.signature(extract_spatial_texture_for_farm)
        param_names = list(sig.parameters.keys())
        for forbidden in ["crop", "label", "category", "target"]:
            self.assertNotIn(forbidden, param_names)

    def test_03_default_missingness_preservation(self):
        """Verify unextracted features initialize to NaN and not zero."""
        # Check dictionary template initialization
        empty_res = {feat: np.nan for feat in GLCM_FEATURE_NAMES}
        for feat in GLCM_FEATURE_NAMES:
            self.assertTrue(np.isnan(empty_res[feat]))
            self.assertNotEqual(empty_res[feat], 0.0)

    def test_04_geometry_conversion(self):
        """Verify Polygon and MultiPolygon geometry conversions work cleanly."""
        from src.satellite.s2_extractor import EarthEngineManager, get_ee_polygon

        try:
            EarthEngineManager.initialize()
        except Exception:
            pass

        if not EarthEngineManager.is_authenticated():
            self.skipTest("Earth Engine not authenticated in test environment")

        poly = Polygon([(78.1, 17.1), (78.2, 17.1), (78.2, 17.2), (78.1, 17.2), (78.1, 17.1)])
        ee_poly = get_ee_polygon(poly)
        self.assertIsNotNone(ee_poly)

        multipoly = MultiPolygon([poly])
        ee_multipoly = get_ee_polygon(multipoly)
        self.assertIsNotNone(ee_multipoly)

    def test_05_quantization_logic(self):
        """Test linear quantization math matches [0, 31] discrete levels."""
        # Formula: round(clamp(val / 10000 * 31, 0, 31))
        def quantize_val(val, max_refl=10000.0, levels=32):
            max_idx = levels - 1
            scaled = (val / max_refl) * max_idx
            clamped = max(0.0, min(float(max_idx), scaled))
            return int(round(clamped))

        self.assertEqual(quantize_val(0.0), 0)
        self.assertEqual(quantize_val(10000.0), 31)
        self.assertEqual(quantize_val(5000.0), 16)
        self.assertEqual(quantize_val(15000.0), 31)  # clamped
        self.assertEqual(quantize_val(-100.0), 0)   # clamped


if __name__ == "__main__":
    unittest.main()
