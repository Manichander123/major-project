"""
tests/test_phase6f_stage1_audit.py
==================================
Unit tests asserting Phase 6F Stage 1 Gate A reconciliation:
- Manifest (7,164) vs Consolidated (7,164) exact 1:1 FarmID match.
- Zero duplicate FarmIDs in manifest or consolidated data.
- All 15 chunks sum to exactly 7,164 rows with zero inter-chunk duplicate FarmIDs.
- Zero crop label mismatches against Train.csv.
- Exact failure counts: 7,124 success, 40 failed.
- Zero infinite spectral values and zero illegal negative reflectance values.
"""

from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestPhase6FStage1Audit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report_path = PROJECT_ROOT / "outputs" / "reports" / "phase6f_dataset_audit.json"
        with open(cls.report_path, "r", encoding="utf-8") as f:
            cls.audit = json.load(f)

    def test_gate_a_verdict_pass(self):
        self.assertEqual(self.audit["gate_a_verdict"], "PASS")

    def test_coverage_and_duplicates(self):
        cov = self.audit["coverage"]
        self.assertEqual(cov["manifest_total_rows"], 7164)
        self.assertEqual(cov["manifest_unique_farmids"], 7164)
        self.assertEqual(cov["manifest_duplicate_count"], 0)
        self.assertEqual(cov["consolidated_total_rows"], 7164)
        self.assertEqual(cov["consolidated_unique_farmids"], 7164)
        self.assertEqual(cov["consolidated_duplicate_count"], 0)
        self.assertEqual(len(cov["missing_in_consolidated"]), 0)
        self.assertEqual(len(cov["unexpected_in_consolidated"]), 0)

    def test_chunk_reconciliation(self):
        chunks = self.audit["chunk_reconciliation"]
        self.assertEqual(chunks["total_chunks_found"], 15)
        self.assertEqual(chunks["total_rows_across_chunks"], 7164)
        self.assertEqual(chunks["inter_chunk_duplicates_count"], 0)
        self.assertTrue(chunks["chunks_vs_consolidated_row_match"])

    def test_label_alignment_against_train(self):
        lbl = self.audit["label_alignment"]
        self.assertEqual(lbl["farms_matched_in_train"], 7164)
        self.assertEqual(lbl["crop_label_mismatches_count"], 0)
        self.assertEqual(lbl["district_mismatches_count"], 0)
        self.assertEqual(lbl["subdistrict_mismatches_count"], 0)
        self.assertEqual(lbl["season_mismatches_count"], 0)

    def test_failure_accounting(self):
        fail = self.audit["failure_analysis"]
        self.assertEqual(fail["total_success"], 7124)
        self.assertEqual(fail["total_failed"], 40)
        modes = fail["failure_modes_distribution"]
        self.assertEqual(modes.get("No valid cloud-free imagery found across any phenological window", 0), 35)
        self.assertEqual(modes.get("Computation timed out.", 0), 4)
        self.assertEqual(modes.get("Earth Engine is not authenticated. Please run `earthengine authenticate`.", 0), 1)

    def test_spectral_sanity(self):
        spec = self.audit["spectral_sanity"]
        self.assertEqual(spec["total_infinite_values"], 0)
        self.assertEqual(spec["total_negative_reflectance_values"], 0)


if __name__ == "__main__":
    unittest.main()
