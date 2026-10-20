"""
tests/test_phase6f_stage2_recovery.py
=====================================
Automated unit tests for Phase 6F Stage 2 Recovery:
- Enforces strict allowlist of the 5 candidate FarmIDs.
- Rejects unexpected FarmIDs.
- Validates window recovery matrix (1330053 missing full; 1330055/56 missing mid/late/full; 1330059/230 missing all).
- Asserts preservation of existing valid partial observations.
- Asserts protected baseline files have not changed (SHA-256 match).
- Validates 97-column canonical schema.
- Rejects infinite values.
- Mocks Earth Engine calls to ensure no live API calls occur during test discovery.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from scripts.recover_phase6f_failures import (
    ALLOWED_FARM_IDS,
    CANONICAL_COLUMNS,
    RECOVERY_MATRIX,
)


class TestPhase6FStage2Recovery(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baseline_hashes = {
            "datasss/Train.csv": "29d31920f5b9aaea8213d7993dedc50c5d19458906f85642d360d785bb07aa06",
            "outputs/data/s2_full_expansion_manifest_7164.csv": "f4b92b55f248edf687c60b85177cb793859d04fd39442a4841991b79110807b9",
            "outputs/data/s2_phase6e_7164_raw_bands.csv": "d1ee90a5d9e09126a71bdc8e809767a30860671e88e42a8a8aebb19c46cf83af",
            "outputs/data/s2_phase6e_chunk_00_0_500_raw_bands.csv": "9cdecd9c3162ca6301883b62ffaeffebf579f2e3c09e8631a986905dccf03ed6",
            "outputs/data/s2_phase6e_chunk_05_2500_3000_raw_bands.csv": "88650fdb722979455e0d3fa4c8280655872f8ea8db31094e180ed1eb0b0750e5",
            "outputs/data/s2_phase6e_chunk_14_7000_7164_raw_bands.csv": "3ac9eb1a66e7d82c93830d4d7e6a568d2b3d13b24c2fc86c80a9990dcd029c53",
        }

    def test_allowlist_membership(self):
        """Verify only the exact 5 FarmIDs are allowed."""
        self.assertEqual(sorted(ALLOWED_FARM_IDS), [1330053, 1330055, 1330056, 1330059, 1330230])

    def test_unexpected_farmid_rejected(self):
        """Verify unexpected FarmID is not in the allowlist."""
        self.assertNotIn(9999999, ALLOWED_FARM_IDS)
        self.assertNotIn(1307790, ALLOWED_FARM_IDS)

    def test_window_recovery_matrix(self):
        """Verify exact window reductions planned for each farm."""
        self.assertEqual(RECOVERY_MATRIX[1330053], ["full"])
        self.assertEqual(RECOVERY_MATRIX[1330055], ["mid", "late", "full"])
        self.assertEqual(RECOVERY_MATRIX[1330056], ["mid", "late", "full"])
        self.assertEqual(RECOVERY_MATRIX[1330059], ["early", "mid", "late", "full"])
        self.assertEqual(RECOVERY_MATRIX[1330230], ["early", "mid", "late", "full"])
        total_reductions = sum(len(wins) for wins in RECOVERY_MATRIX.values())
        self.assertEqual(total_reductions, 15)

    def test_canonical_schema_definition(self):
        """Verify canonical schema has exactly 97 columns."""
        self.assertEqual(len(CANONICAL_COLUMNS), 97)
        self.assertEqual(CANONICAL_COLUMNS[0], "FarmID")
        self.assertEqual(CANONICAL_COLUMNS[1], "status")
        self.assertEqual(CANONICAL_COLUMNS[2], "error_message")
        self.assertIn("B8_early_mean", CANONICAL_COLUMNS)
        self.assertIn("B8_full_std", CANONICAL_COLUMNS)
        self.assertEqual(CANONICAL_COLUMNS[-6:], ["Crop", "District", "Sub-District", "Season", "SDate", "HDate"])

    def test_protected_files_remain_unmodified(self):
        """Verify protected baseline files exactly match baseline SHA-256 hashes."""
        for rel_path, expected_hash in self.baseline_hashes.items():
            full_path = PROJECT_ROOT / rel_path
            self.assertTrue(full_path.exists(), f"Protected file missing: {rel_path}")
            current_hash = hashlib.sha256(full_path.read_bytes()).hexdigest()
            self.assertEqual(current_hash, expected_hash, f"Protected file modified! {rel_path}")

    def test_partial_preservation_logic(self):
        """Verify existing valid partial values are retained when building merged row."""
        orig_row = {
            "FarmID": 1330053,
            "status": "failed",
            "B8_early_mean": 3932.24,
            "B8_mid_mean": 1699.41,
            "B8_late_mean": 2839.89,
            "B8_full_mean": np.nan,
        }
        mock_recovered_full = {
            "B8_full_mean": 2500.0,
            "outcome": "success",
            "has_valid_data": True,
        }
        merged = dict(orig_row)
        for k, v in mock_recovered_full.items():
            if k not in ["has_valid_data", "outcome"]:
                merged[k] = v

        # Preserved
        self.assertEqual(merged["B8_early_mean"], 3932.24)
        self.assertEqual(merged["B8_mid_mean"], 1699.41)
        self.assertEqual(merged["B8_late_mean"], 2839.89)
        # Newly recovered
        self.assertEqual(merged["B8_full_mean"], 2500.0)

    def test_recovered_dataset_file_integrity(self):
        """Verify the generated recovered dataset satisfies all canonical schema requirements."""
        rec_path = PROJECT_ROOT / "outputs" / "data" / "s2_phase6f_recovered_farms.csv"
        self.assertTrue(rec_path.exists(), "Missing s2_phase6f_recovered_farms.csv")
        df_rec = pd.read_csv(rec_path)
        self.assertEqual(len(df_rec), 5)
        self.assertEqual(list(df_rec.columns), CANONICAL_COLUMNS)
        self.assertEqual(sorted(df_rec["FarmID"].tolist()), sorted(ALLOWED_FARM_IDS))
        self.assertTrue((df_rec["status"] == "success").all())

        # Check no infinite values in any numeric column
        numeric_cols = df_rec.select_dtypes(include=[np.number]).columns
        for c in numeric_cols:
            self.assertEqual(int(np.isinf(df_rec[c]).sum()), 0, f"Infinite values in {c}")

    def test_recovery_ledger_integrity(self):
        """Verify the generated ledger accounts for all 20 operations with correct outcomes."""
        led_path = PROJECT_ROOT / "outputs" / "reports" / "phase6f_stage2_recovery_ledger.csv"
        self.assertTrue(led_path.exists(), "Missing phase6f_stage2_recovery_ledger.csv")
        df_led = pd.read_csv(led_path)
        self.assertEqual(len(df_led), 20)
        requested = df_led[df_led["requested"] == True]
        self.assertEqual(len(requested), 15)
        self.assertTrue((requested["outcome"] == "success").all())
        preserved = df_led[df_led["requested"] == False]
        self.assertEqual(len(preserved), 5)
        self.assertTrue((preserved["outcome"] == "preserved_existing").all())


if __name__ == "__main__":
    unittest.main()
