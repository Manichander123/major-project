"""
Unit tests for Phase 6F Stage 4B: Controlled Model Benchmarking and Scientific Evaluation.
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

EXPECTED_CLASSES = {
    "Paddy",
    "Cotton",
    "Maize",
    "Bengal Gram",
    "Groundnut",
    "Chillies",
    "Red Gram",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class TestPhase6FStage4Benchmark(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.features_csv = PROJECT_ROOT / "outputs/data/s2_phase6f_features.csv"
        cls.schema_json = PROJECT_ROOT / "outputs/reports/phase6f_feature_schema_manifest.json"
        cls.cv_folds_csv = PROJECT_ROOT / "outputs/data/phase6f_cv_folds.csv"
        cls.cv_audit_json = PROJECT_ROOT / "outputs/reports/phase6f_cv_folds_audit.json"
        cls.oof_csv = PROJECT_ROOT / "outputs/data/phase6f_oof_predictions.csv"
        cls.bench_json = PROJECT_ROOT / "outputs/reports/phase6f_stage4_benchmark_report.json"
        cls.bench_md = PROJECT_ROOT / "outputs/reports/phase6f_stage4_benchmark_report.md"

    def test_01_feature_matrix_dimensions_and_population(self):
        """Feature matrix must have 7,888 rows, 165 columns, and 7,843 eligible farms."""
        self.assertTrue(self.features_csv.exists(), f"Missing {self.features_csv}")
        df = pd.read_csv(self.features_csv)
        self.assertEqual(len(df), 7888)
        self.assertEqual(len(df.columns), 165)
        self.assertEqual(df["FarmID"].nunique(), 7888)

        # Check eligibility split via cloud_obscured_failure
        eligible = df[df["cloud_obscured_failure"] == False]
        self.assertEqual(len(eligible), 7843)
        excluded = df[df["cloud_obscured_failure"] == True]
        self.assertEqual(len(excluded), 45)

        # Labels must match expected 7 classes exactly
        self.assertEqual(set(eligible["Crop"].unique()), EXPECTED_CLASSES)

    def test_02_feature_schema_manifest_dimensions(self):
        """Schema manifest must define exact feature dimensions for all 4 experiments."""
        self.assertTrue(self.schema_json.exists(), f"Missing {self.schema_json}")
        with open(self.schema_json, "r") as f:
            manifest = json.load(f)

        experiments = manifest.get("experiments", {})
        self.assertEqual(experiments.get("experiment_a_full_season_retrospective", {}).get("feature_count"), 34)
        self.assertEqual(experiments.get("experiment_b_early_season_in_season", {}).get("feature_count"), 34)
        self.assertEqual(experiments.get("experiment_c_multitemporal_trajectory", {}).get("feature_count"), 140)
        self.assertEqual(experiments.get("experiment_d_multimodal_benchmark", {}).get("feature_count"), 146)

    def test_03_cv_fold_leakage_and_class_representation(self):
        """Folds must have 0 Sub-District overlap, all 7 classes, and no missing assignments."""
        self.assertTrue(self.cv_folds_csv.exists(), f"Missing {self.cv_folds_csv}")
        df_folds = pd.read_csv(self.cv_folds_csv)
        self.assertEqual(len(df_folds), 7843)
        self.assertEqual(df_folds["FarmID"].nunique(), 7843)
        self.assertEqual(set(df_folds["fold"].unique()), {0, 1, 2, 3, 4})

        # No Sub-District leakage across train/val in any fold
        for fold in range(5):
            val_mandals = set(df_folds[df_folds["fold"] == fold]["Sub-District"].unique())
            train_mandals = set(df_folds[df_folds["fold"] != fold]["Sub-District"].unique())
            overlap = val_mandals.intersection(train_mandals)
            self.assertEqual(len(overlap), 0, f"Spatial leakage in fold {fold}: {overlap}")

            # All 7 classes represented in validation split of each fold
            val_crops = set(df_folds[df_folds["fold"] == fold]["Crop"].unique())
            self.assertEqual(val_crops, EXPECTED_CLASSES, f"Fold {fold} missing classes: {EXPECTED_CLASSES - val_crops}")

            # Red Gram has positive support
            red_gram_count = len(df_folds[(df_folds["fold"] == fold) & (df_folds["Crop"] == "Red Gram")])
            self.assertGreater(red_gram_count, 0, f"Fold {fold} has 0 Red Gram samples")

    def test_04_oof_predictions_integrity(self):
        """Each evaluated farm must have exactly one OOF prediction per (run, model)."""
        self.assertTrue(self.oof_csv.exists(), f"Missing {self.oof_csv}")
        df_oof = pd.read_csv(self.oof_csv)
        self.assertGreater(len(df_oof), 0)

        expected_runs = [
            ("Exp_A", "primary", ["dummy", "rf", "xgb"], 7843),
            ("Exp_B", "primary", ["dummy", "rf", "xgb"], 7843),
            ("Exp_B", "sensitivity_complete_case", ["rf", "xgb"], 7298),
            ("Exp_C", "primary", ["dummy", "rf", "xgb"], 7843),
            ("Exp_C", "sensitivity_complete_trajectory", ["rf", "xgb"], 7099),
            ("Exp_D", "primary", ["dummy", "rf", "xgb"], 7843),
        ]

        for exp, run_type, models, expected_n in expected_runs:
            for model_name in models:
                subset = df_oof[
                    (df_oof["experiment"] == exp)
                    & (df_oof["run_type"] == run_type)
                    & (df_oof["model_name"] == model_name)
                ]
                self.assertEqual(
                    len(subset),
                    expected_n,
                    f"Mismatch in count for ({exp}, {run_type}, {model_name}): got {len(subset)}, expected {expected_n}",
                )
                self.assertEqual(
                    subset["FarmID"].nunique(),
                    expected_n,
                    f"Duplicate FarmID in OOF predictions for ({exp}, {run_type}, {model_name})",
                )
                # No NaNs in predicted crop
                self.assertEqual(subset["predicted_crop"].isna().sum(), 0)
                # True crop matches expected classes
                self.assertTrue(set(subset["true_crop"].unique()).issubset(EXPECTED_CLASSES))

    def test_05_benchmark_report_metrics(self):
        """Benchmark report must contain valid metrics and distinguish pooled vs fold-mean."""
        self.assertTrue(self.bench_json.exists(), f"Missing {self.bench_json}")
        with open(self.bench_json, "r") as f:
            data = json.load(f)

        self.assertIn("Exp_A_primary", data)
        self.assertIn("Exp_C_primary", data)
        self.assertIn("Exp_D_primary", data)

        for run_id, run_data in data.items():
            models = run_data.get("models", {})
            for model_name, m_res in models.items():
                pooled_macro = m_res.get("pooled_oof_macro_f1")
                self.assertIsNotNone(pooled_macro)
                self.assertGreaterEqual(pooled_macro, 0.0)
                self.assertLessEqual(pooled_macro, 1.0)

                fold_mean_macro = m_res.get("fold_macro_f1_mean")
                self.assertIsNotNone(fold_mean_macro)

                # Dummy classifier must be ~most frequent baseline (~0.0639 macro F1)
                if model_name == "dummy":
                    self.assertAlmostEqual(pooled_macro, 0.0639, delta=0.01)
                else:
                    # ML models must beat dummy baseline
                    self.assertGreater(pooled_macro, 0.10)

                # Per-class metrics must exist for all 7 crops
                per_class = m_res.get("per_class", {})
                self.assertEqual(set(per_class.keys()), EXPECTED_CLASSES)

    def test_06_protected_baselines_immutability(self):
        """All 24 protected baseline files must remain strictly unmodified."""
        for rel_path, expected_hash in PROTECTED_BASELINE_HASHES.items():
            full_path = PROJECT_ROOT / rel_path
            self.assertTrue(full_path.exists(), f"Protected file missing: {rel_path}")
            actual_hash = sha256_file(full_path)
            self.assertEqual(
                actual_hash,
                expected_hash,
                f"PROTECTED FILE CORRUPTED/MODIFIED: {rel_path}\nExpected: {expected_hash}\nActual:   {actual_hash}",
            )


if __name__ == "__main__":
    unittest.main()
