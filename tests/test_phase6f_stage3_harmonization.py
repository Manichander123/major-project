"""
Unit tests for Phase 6F Stage 3: Harmonized Cohort & Gate B Feature Readiness.
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

APPROVED_RECOVERY_IDS = {1330053, 1330055, 1330056, 1330059, 1330230}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class TestPhase6FStage3Harmonization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.csv_path = PROJECT_ROOT / "outputs/data/s2_phase6f_harmonized_cohort.csv"
        cls.json_path = PROJECT_ROOT / "outputs/reports/phase6f_stage3_harmonization_report.json"
        cls.md_path = PROJECT_ROOT / "outputs/reports/phase6f_stage3_harmonization_report.md"
        cls.train_path = PROJECT_ROOT / "datasss/Train.csv"

        if cls.csv_path.exists():
            cls.df = pd.read_csv(cls.csv_path)
        else:
            cls.df = None

        if cls.train_path.exists():
            cls.train_df = pd.read_csv(cls.train_path)
        else:
            cls.train_df = None

    def test_01_target_artifacts_exist(self):
        self.assertTrue(self.csv_path.exists(), f"Harmonized CSV missing: {self.csv_path}")
        self.assertTrue(self.json_path.exists(), f"Harmonized JSON report missing: {self.json_path}")
        self.assertTrue(self.md_path.exists(), f"Harmonized MD report missing: {self.md_path}")
        self.assertGreater(self.csv_path.stat().st_size, 10_000_000)
        self.assertGreater(self.json_path.stat().st_size, 500)
        self.assertGreater(self.md_path.stat().st_size, 1000)

    def test_02_cohort_dimensions_and_uniqueness(self):
        self.assertIsNotNone(self.df)
        self.assertEqual(len(self.df), 7888)
        self.assertEqual(self.df["FarmID"].nunique(), 7888)
        self.assertFalse(self.df["FarmID"].duplicated().any())

    def test_03_train_csv_id_equivalence(self):
        self.assertIsNotNone(self.df)
        self.assertIsNotNone(self.train_df)
        harmonized_ids = set(self.df["FarmID"].astype(int))
        train_ids = set(self.train_df["FarmID"].astype(int))
        self.assertEqual(harmonized_ids, train_ids)

    def test_04_cohort_group_breakdown(self):
        self.assertIsNotNone(self.df)
        succ_mask = self.df["extraction_status"] == "success"
        fail_mask = self.df["extraction_status"] == "failed"

        self.assertEqual(succ_mask.sum(), 7843)
        self.assertEqual(fail_mask.sum(), 45)
        self.assertEqual(self.df["cloud_obscured_failure"].sum(), 45)
        self.assertTrue((self.df["cloud_obscured_failure"] == fail_mask).all())
        self.assertTrue((self.df["status"] == self.df["extraction_status"]).all())

        sources = self.df["source_phase"].value_counts().to_dict()
        expected_sources = {
            "phase6e": 7124,
            "phase6d": 714,
            "phase6e_cloud_failed": 35,
            "phase6d_cloud_failed": 10,
            "phase6f_recovered": 5,
        }
        self.assertEqual(sources, expected_sources)

    def test_05_schema_integrity(self):
        self.assertIsNotNone(self.df)
        self.assertEqual(len(self.df.columns), 100)

        # Check canonical 97 extraction columns
        df_6e = pd.read_csv(PROJECT_ROOT / "outputs/data/s2_phase6e_7164_raw_bands.csv", nrows=1)
        canonical_cols = list(df_6e.columns)
        self.assertEqual(list(self.df.columns[:97]), canonical_cols)

        # Check trailing audit columns
        self.assertEqual(
            list(self.df.columns[97:]),
            ["source_phase", "extraction_status", "cloud_obscured_failure"],
        )

    def test_06_recovered_farms_fidelity(self):
        self.assertIsNotNone(self.df)
        rec_farms = self.df[self.df["FarmID"].isin(APPROVED_RECOVERY_IDS)]
        self.assertEqual(len(rec_farms), 5)
        self.assertTrue((rec_farms["source_phase"] == "phase6f_recovered").all())
        self.assertTrue((rec_farms["status"] == "success").all())
        self.assertTrue((rec_farms["extraction_status"] == "success").all())
        self.assertFalse(rec_farms["cloud_obscured_failure"].any())

        # Check that spectral bands for recovered farms are non-NaN
        band_cols = [c for c in rec_farms.columns if "_mean" in c or "_std" in c]
        self.assertEqual(len(band_cols), 80)
        self.assertEqual(rec_farms[band_cols].isna().sum().sum(), 0)

    def test_07_ground_truth_metadata_consistency(self):
        self.assertIsNotNone(self.df)
        self.assertIsNotNone(self.train_df)
        train_indexed = self.train_df.set_index("FarmID")

        for col in ["Crop", "District", "Sub-District", "Season"]:
            diff = (self.df[col] != train_indexed.loc[self.df["FarmID"], col].values).sum()
            self.assertEqual(diff, 0, f"Mismatch in {col}")

        # Date consistency
        sdate_diff = (
            pd.to_datetime(self.df["SDate"], format="mixed")
            != pd.to_datetime(train_indexed.loc[self.df["FarmID"], "SDate"].values, format="mixed")
        ).sum()
        hdate_diff = (
            pd.to_datetime(self.df["HDate"], format="mixed")
            != pd.to_datetime(train_indexed.loc[self.df["FarmID"], "HDate"].values, format="mixed")
        ).sum()
        self.assertEqual(sdate_diff, 0)
        self.assertEqual(hdate_diff, 0)

    def test_08_crop_distribution_exactness(self):
        self.assertIsNotNone(self.df)
        self.assertIsNotNone(self.train_df)
        self.assertEqual(
            self.df["Crop"].value_counts().to_dict(),
            self.train_df["Crop"].value_counts().to_dict(),
        )

    def test_09_numerical_invariants_and_missingness(self):
        self.assertIsNotNone(self.df)
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        self.assertEqual(np.isinf(self.df[num_cols].to_numpy()).sum(), 0)

        # 45 cloud-failed farms must have 100% NaN for spectral values
        band_cols = [c for c in self.df.columns if "_mean" in c or "_std" in c]
        failed_df = self.df[self.df["cloud_obscured_failure"]]
        self.assertEqual(len(failed_df), 45)
        self.assertEqual(failed_df[band_cols].notna().sum().sum(), 0)

        # 7,843 successful farms must have valid full window observations
        success_df = self.df[~self.df["cloud_obscured_failure"]]
        self.assertEqual(len(success_df), 7843)
        self.assertTrue((success_df["n_images_valid_full"] > 0).all())
        self.assertTrue(success_df["B2_full_mean"].notna().all())

    def test_10_protected_files_immutability(self):
        for rel_path, expected_hash in PROTECTED_BASELINE_HASHES.items():
            abs_path = PROJECT_ROOT / rel_path
            self.assertTrue(abs_path.exists(), f"Protected file missing: {rel_path}")
            actual_hash = sha256_file(abs_path)
            self.assertEqual(
                actual_hash,
                expected_hash,
                f"MUTATION DETECTED in protected baseline file: {rel_path}",
            )


if __name__ == "__main__":
    unittest.main()
