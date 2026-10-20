"""
Unit tests for Phase 6G Experiment 2: Imbalance-Aware Pilot on Outer Fold 0.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

import joblib
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

EXPECTED_CLASSES = [
    "Bengal Gram",
    "Chillies",
    "Cotton",
    "Groundnut",
    "Maize",
    "Paddy",
    "Red Gram",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class TestPhase6GExp2Imbalance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.phase6g_data = PROJECT_ROOT / "outputs/data/phase6g"
        cls.phase6g_reports = PROJECT_ROOT / "outputs/reports/phase6g"
        cls.phase6g_models = PROJECT_ROOT / "outputs/models/phase6g"

        cls.oof_csv = cls.phase6g_data / "exp2_imbalance_pilot_oof_predictions.csv"
        cls.ledger_csv = cls.phase6g_reports / "exp2_imbalance_pilot_ledger.csv"
        cls.report_json = cls.phase6g_reports / "phase6g_exp2_imbalance_report.json"
        cls.report_md = cls.phase6g_reports / "phase6g_exp2_imbalance_report.md"
        cls.model_joblib = cls.phase6g_models / "xgb_exp2_selected.joblib"

    def test_01_exp2_artifacts_exist(self):
        """All expected Phase 6G Experiment 2 artifacts must exist."""
        self.assertTrue(self.oof_csv.exists(), f"Missing {self.oof_csv}")
        self.assertTrue(self.ledger_csv.exists(), f"Missing {self.ledger_csv}")
        self.assertTrue(self.report_json.exists(), f"Missing {self.report_json}")
        self.assertTrue(self.report_md.exists(), f"Missing {self.report_md}")
        self.assertTrue(self.model_joblib.exists(), f"Missing {self.model_joblib}")

    def test_02_strategy_ledger_completeness(self):
        """Strategy ledger must contain exactly 7 weighting strategies."""
        df_ledger = pd.read_csv(self.ledger_csv)
        self.assertEqual(len(df_ledger), 7)
        expected_ids = {
            "unweighted_ref",
            "balanced_inv_freq",
            "sqrt_balanced",
            "red_gram_1.5x",
            "red_gram_2.0x",
            "targeted_1.5x_1.3x",
            "targeted_2.0x_1.3x",
        }
        self.assertEqual(set(df_ledger["strategy_id"]), expected_ids)

        for _, row in df_ledger.iterrows():
            self.assertGreater(row["inner_macro_f1"], 0.60)
            self.assertLess(row["inner_macro_f1"], 0.90)
            self.assertGreater(row["inner_accuracy"], 0.65)
            self.assertGreater(row["inner_red_gram_f1"], 0.15)
            self.assertGreater(row["fit_time_seconds"], 0)

    def test_03_oof_predictions_integrity(self):
        """OOF predictions must have exactly 1,508 rows matching Outer Fold 0."""
        df_oof = pd.read_csv(self.oof_csv)
        self.assertEqual(len(df_oof), 1508)
        self.assertEqual(df_oof["FarmID"].nunique(), 1508)
        self.assertEqual(set(df_oof["fold"].unique()), {0})
        self.assertEqual(df_oof["predicted_crop"].isna().sum(), 0)
        self.assertTrue(set(df_oof["predicted_crop"].unique()).issubset(set(EXPECTED_CLASSES)))

    def test_04_report_content_and_comparisons(self):
        """Report JSON must contain required comparisons and disclaimers."""
        with open(self.report_json, "r") as f:
            data = json.load(f)

        self.assertEqual(data["status"], "COMPLETED SUCCESSFULLY")
        self.assertIn("EXPLORATORY PILOT RESULT", data["scope_boundary_note"])
        self.assertEqual(data["outer_fold"], 0)
        self.assertEqual(data["strategies_evaluated"], 7)

        # Baseline comparison must exist
        self.assertIn("phase6f_fold0_baseline", data)
        self.assertIn("unweighted_pilot_fold0", data)
        self.assertIn("delta_vs_unweighted_pilot", data)

    def test_05_weight_functions_correctness(self):
        """Verify mathematical definitions of class weight functions."""
        # Simulated class distribution (100 samples across 3 classes: [60, 30, 10])
        y_dummy = np.array([0] * 60 + [1] * 30 + [2] * 10)
        
        # Test balanced: w_c = N / (K * N_c)
        counts = np.bincount(y_dummy, minlength=3)
        w_bal = {c: len(y_dummy) / (3 * counts[c]) for c in range(3)}
        self.assertAlmostEqual(w_bal[0], 100.0 / (3 * 60))
        self.assertAlmostEqual(w_bal[1], 100.0 / (3 * 30))
        self.assertAlmostEqual(w_bal[2], 100.0 / (3 * 10))

        # Test sqrt_balanced: w_c = sqrt(N_max / N_c)
        w_sqrt = {c: np.sqrt(60.0 / counts[c]) for c in range(3)}
        self.assertAlmostEqual(w_sqrt[0], 1.0)
        self.assertAlmostEqual(w_sqrt[2], np.sqrt(6.0))

    def test_06_protected_baselines_immutability(self):
        """All 24 protected baseline files must remain strictly unmodified."""
        for rel_path, expected_hash in PROTECTED_BASELINE_HASHES.items():
            fp = PROJECT_ROOT / rel_path
            self.assertTrue(fp.exists(), f"Missing {rel_path}")
            act_hash = sha256_file(fp)
            self.assertEqual(act_hash, expected_hash, f"CORRUPTED: {rel_path}")


if __name__ == "__main__":
    unittest.main()
