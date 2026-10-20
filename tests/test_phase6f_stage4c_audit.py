"""
Unit tests for Phase 6F Stage 4C: Scientific Integrity Audit & Verification.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROTECTED_BASELINE_HASHES = {
    "datasss/Train.csv": "29d31920f5b9aaea8213d7993dedc50c5d19458906f85642d360d785bb07aa06",
    "outputs/data/s2_full_expansion_manifest_7164.csv": "f4b92b55f248edf687c60b85177cb793859d04fd39442a4841991b79110807b9",
    "outputs/data/s2_phase6e_7164_raw_bands.csv": "d1ee90a5d9e09126a71bdc8e809767a30860671e88e42a8a8aebb19c46cf83af",
    "outputs/data/s2_phase6d_manifest_724.csv": "326207f8ca1b0848533bdc4ff673db24b18ff46e30b558952444555c20824214",
    "outputs/data/s2_phase6d_724_raw_bands.csv": "cdd3cf8a8a6ac0dde6c0e13d262011a379e129598e8f667a38fab747f1f73826",
    "outputs/data/s2_phase6d_724_features.csv": "98ac03841ea85219f3df1f90d038741aa6faa65e212de6a4d12fd02f0e7bfc75",
    "outputs/data/phase6d_oof_predictions.csv": "dfa09e53ec961edd08c3ba917364e8d2a666e23d4357d7d46864974a53ee49a4",
    "outputs/data/s2_phase6f_recovered_farms.csv": "235ca39d953be9b0ef3984f9042fecda4c3fb3e0324de1e94f0af04ccebdc32f",
    "outputs/reports/phase6f_stage2_recovery_ledger.csv": "ced6944ce6db7ee02841c3de11b6c8edc48ea27f2f2dfaa013a42054f299e272",
    "outputs/data/s2_phase6e_chunk_00_0_500_raw_bands.csv": "9cdecd9c3162ca6301883b62ffaeffebf579f2e3c09e8631a986905dccf03ed6",
    "outputs/data/s2_phase6e_chunk_01_500_1000_raw_bands.csv": "e4f2027ba78cef3267f54287809f9379e6d0dc77b6ab67ae3a059b0297ef9dc5",
    "outputs/data/s2_phase6e_chunk_02_1000_1500_raw_bands.csv": "ef428abd732902e3625f9be2161f96f2a80742022616452c205ef57b1f202219",
    "outputs/data/s2_phase6e_chunk_03_1500_2000_raw_bands.csv": "5f03c44197ee3e0a397e56a869cab07f3dafae76736821e1988730d8aff0e30a",
    "outputs/data/s2_phase6e_chunk_04_2000_2500_raw_bands.csv": "a291487191587bcc1680729e45026740008ee81b89aae55b99b923c7fa24453f",
    "outputs/data/s2_phase6e_chunk_05_2500_3000_raw_bands.csv": "88650fdb722979455e0d3fa4c8280655872f8ea8db31094e180ed1eb0b0750e5",
    "outputs/data/s2_phase6e_chunk_06_3000_3500_raw_bands.csv": "32015d7d6733c4b9b2684f4c9a8e05f42b7d29c01a09986ab69fa3368f6e6ce4",
    "outputs/data/s2_phase6e_chunk_07_3500_4000_raw_bands.csv": "572d96fddf458bebb864da59eb44f27e6e2cec505bb855ccabf75855154a557e",
    "outputs/data/s2_phase6e_chunk_08_4000_4500_raw_bands.csv": "9db8b9bd2fdd8e2968543fe83ddcfe6a9a5e0b0a317c5d2145fda454f6f10f66",
    "outputs/data/s2_phase6e_chunk_09_4500_5000_raw_bands.csv": "b9c9e0a53197d115e218501d219e26570c5d7d203097031e57b6e35e01814d6d",
    "outputs/data/s2_phase6e_chunk_10_5000_5500_raw_bands.csv": "6877114a7dc422d0b8539704bac2d777bcbd6b4fcb8c81a632bd1c13c64f7454",
    "outputs/data/s2_phase6e_chunk_11_5500_6000_raw_bands.csv": "4fa33f6ecdebfd6b02deba3634832362b56b9af695cb18fb39c68341884640fb",
    "outputs/data/s2_phase6e_chunk_12_6000_6500_raw_bands.csv": "2458256baddcb4f915ee299dd33e2f2811fd8e57d0f81f78435657804bf5a144",
    "outputs/data/s2_phase6e_chunk_13_6500_7000_raw_bands.csv": "b26e7e8fa202a7c76252df06df11b3189722a8c0fdc36dabbb7d042a1f040ca3",
    "outputs/data/s2_phase6e_chunk_14_7000_7164_raw_bands.csv": "3ac9eb1a66e7d82c93830d4d7e6a568d2b3d13b24c2fc86c80a9990dcd029c53",
}

DISCREPANT_8_FARMS = {1304367, 1311790, 1317244, 1317978, 1320213, 1325016, 1331153, 1331755}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class TestPhase6FStage4CAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit_json = PROJECT_ROOT / "outputs/reports/phase6f_stage4c_integrity_audit.json"
        cls.audit_md = PROJECT_ROOT / "outputs/reports/phase6f_stage4c_integrity_audit.md"
        cls.features_csv = PROJECT_ROOT / "outputs/data/s2_phase6f_features.csv"
        cls.manifest_json = PROJECT_ROOT / "outputs/reports/phase6f_feature_schema_manifest.json"

        with open(cls.audit_json, "r") as f:
            cls.audit_data = json.load(f)

    def test_01_audit_reports_exist(self):
        """Audit JSON and Markdown reports must exist and have content."""
        self.assertTrue(self.audit_json.exists())
        self.assertTrue(self.audit_md.exists())
        self.assertEqual(self.audit_data["overall_status"], "PASS WITH CRITICAL SCIENTIFIC ADDENDA")

    def test_02_cohort_accounting_exactness(self):
        """Cohort accounting must reflect 7,888 total, 7,843 eligible, 45 excluded."""
        ca = self.audit_data["cohort_accounting"]
        self.assertEqual(ca["total_farms"], 7888)
        self.assertEqual(ca["unique_farms"], 7888)
        self.assertEqual(ca["cloud_failures_excluded"], 45)
        self.assertEqual(ca["eligible_modeling_farms"], 7843)
        self.assertEqual(ca["status"], "PASS")

    def test_03_trajectory_reconciliation_exactness(self):
        """The 8 discrepant farms between 7,107 and 7,099 must match audited IDs."""
        tr = self.audit_data["trajectory_count_reconciliation"]
        self.assertEqual(tr["metadata_image_valid_4windows_count"], 7107)
        self.assertEqual(tr["strictly_zero_nan_140_features_count"], 7099)
        self.assertEqual(tr["discrepancy_count"], 8)
        audited_fids = {r["FarmID"] for r in tr["discrepant_farms"]}
        self.assertEqual(audited_fids, DISCREPANT_8_FARMS)

    def test_04_metric_reproduction_completeness(self):
        """All 15 models across 6 runs must reproduce reported metrics exactly."""
        mr = self.audit_data["metric_reproduction"]
        self.assertTrue(mr["all_15_models_reproduced_exactly"])

        for run_id, models in mr["runs"].items():
            for m_name, stats in models.items():
                self.assertTrue(stats["matches_report"], f"Mismatch in {run_id} {m_name}")
                self.assertEqual(stats["nan_predictions"], 0)
                self.assertEqual(stats["sample_count"], stats["expected_sample_count"])
                self.assertEqual(stats["unique_farms"], stats["expected_sample_count"])

    def test_05_red_gram_variability_attribution(self):
        """Audit must report actual Red Gram F1 std dev distinct from overall macro-F1 std dev."""
        rga = self.audit_data["red_gram_variability_audit"]
        exp_c_xgb = rga["Exp_C_xgb"]

        # Pooled stats
        self.assertEqual(exp_c_xgb["pooled_support"], 221)
        self.assertAlmostEqual(exp_c_xgb["pooled_f1"], 0.3333, delta=1e-4)

        # Actual Red Gram std dev is ~0.1171 (ddof=0) and ~0.1309 (ddof=1), distinct from macro-F1 std dev ~0.0230
        self.assertAlmostEqual(exp_c_xgb["actual_red_gram_fold_f1_std_ddof0"], 0.1171, delta=1e-3)
        self.assertAlmostEqual(exp_c_xgb["actual_red_gram_fold_f1_std_ddof1"], 0.1309, delta=1e-3)
        self.assertAlmostEqual(exp_c_xgb["previously_misattributed_macro_f1_std"], 0.0230, delta=1e-3)
        self.assertNotEqual(
            exp_c_xgb["actual_red_gram_fold_f1_std_ddof0"],
            exp_c_xgb["previously_misattributed_macro_f1_std"],
        )

    def test_06_experiment_b_rank_reversal(self):
        """Audit must document the rank reversal in Exp B and explain the population shift."""
        eba = self.audit_data["experiment_b_sensitivity_audit"]
        self.assertTrue(eba["ranking_reversal"])
        self.assertEqual(eba["primary_sample_count"], 7843)
        self.assertEqual(eba["complete_case_sample_count"], 7298)
        self.assertEqual(eba["excluded_sample_count"], 545)
        # Red Gram exclusion rate must be documented (~31.7%)
        self.assertIn("70 / 221", eba["class_distribution_comparison"]["red_gram_exclusion_rate"])

    def test_07_protected_baselines_immutability(self):
        """All 24 protected baseline files must remain strictly unmodified."""
        pb = self.audit_data["protected_baseline_hashes"]
        self.assertTrue(pb["all_passed"])
        for rel_path, expected_hash in PROTECTED_BASELINE_HASHES.items():
            full_path = PROJECT_ROOT / rel_path
            self.assertTrue(full_path.exists(), f"Protected file missing: {rel_path}")
            actual_hash = sha256_file(full_path)
            self.assertEqual(actual_hash, expected_hash, f"CORRUPTED: {rel_path}")


if __name__ == "__main__":
    unittest.main()
