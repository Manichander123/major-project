"""
tests/test_phase6h_stage3_benchmark.py
======================================
Tests for Phase 6H Stage 3 S/T/P feature ablation benchmark:
1. Feature group construction, schemas, exact counts, and mutual exclusivity.
2. Cohort alignment with 7,843 eligible farms and zero duplicates.
3. Leakage prevention (zero Mandal train-val overlap, feature exclusion of metadata).
4. Historical Phase 6F baseline reproduction (Exp C Macro-F1 0.7038, Accuracy 0.7877).
5. Pre-benchmark audit report pass status.
6. Post-benchmark artifact presence and integrity.
"""

from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestPhase6HStage3Benchmark(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folds_path = PROJECT_ROOT / "outputs/data/phase6f_cv_folds.csv"
        cls.texture_path = PROJECT_ROOT / "outputs/data/phase6h/phase6h_texture_all_farms.csv"
        cls.features_path = PROJECT_ROOT / "outputs/data/s2_phase6f_features.csv"
        cls.audit_json = PROJECT_ROOT / "outputs/reports/phase6h/phase6h_stage3_audit.json"
        cls.manifest_json = PROJECT_ROOT / "outputs/reports/phase6h/phase6h_feature_schema_manifest.json"
        cls.benchmark_json = PROJECT_ROOT / "outputs/reports/phase6h/phase6h_stage3_benchmark_report.json"
        cls.oof_csv = PROJECT_ROOT / "outputs/data/phase6h/phase6h_stp_ablation_oof_predictions.csv"

        cls.df_folds = pd.read_csv(cls.folds_path)
        cls.df_tex = pd.read_csv(cls.texture_path)
        cls.df_feat = pd.read_csv(cls.features_path)

    def test_01_feature_group_counts_and_disjointness(self):
        """Verify S=34, T=106, P=20, and all combinations with pairwise disjointness."""
        from scripts.benchmark_phase6h_stp_ablation import build_feature_groups

        df_eligible = self.df_feat[self.df_feat["extraction_status"] == "success"].copy()
        groups, manifest = build_feature_groups(df_eligible, self.df_tex)

        self.assertEqual(len(groups["S"]), 34)
        self.assertEqual(len(groups["T"]), 106)
        self.assertEqual(len(groups["P"]), 20)
        self.assertEqual(len(groups["S+T"]), 140)
        self.assertEqual(len(groups["S+P"]), 54)
        self.assertEqual(len(groups["T+P"]), 126)
        self.assertEqual(len(groups["S+T+P"]), 160)

        # Pairwise disjoint
        set_s = set(groups["S"])
        set_t = set(groups["T"])
        set_p = set(groups["P"])
        self.assertEqual(len(set_s.intersection(set_t)), 0)
        self.assertEqual(len(set_s.intersection(set_p)), 0)
        self.assertEqual(len(set_t.intersection(set_p)), 0)

        # Union completeness
        self.assertEqual(set(groups["S+T"]), set_s | set_t)
        self.assertEqual(set(groups["S+P"]), set_s | set_p)
        self.assertEqual(set(groups["T+P"]), set_t | set_p)
        self.assertEqual(set(groups["S+T+P"]), set_s | set_t | set_p)

    def test_02_cohort_exact_alignment(self):
        """Verify exact 7,843 cohort matching between folds and texture dataset."""
        expected_ids = set(self.df_folds["FarmID"])
        tex_ids = set(self.df_tex["FarmID"])

        self.assertEqual(len(expected_ids), 7843)
        self.assertEqual(len(tex_ids), 7843)
        self.assertEqual(expected_ids, tex_ids)
        self.assertEqual(self.df_tex["FarmID"].duplicated().sum(), 0)

    def test_03_leakage_prevention_mandal_isolation(self):
        """Verify zero Mandal (Sub-District) overlap across all 5 folds."""
        for fold_i in range(5):
            val_mandals = set(self.df_folds.loc[self.df_folds["fold"] == fold_i, "Sub-District"])
            tr_mandals = set(self.df_folds.loc[self.df_folds["fold"] != fold_i, "Sub-District"])
            overlap = val_mandals.intersection(tr_mandals)
            self.assertEqual(len(overlap), 0, f"Mandal overlap in fold {fold_i}: {overlap}")

    def test_04_feature_exclusion_of_leakage_columns(self):
        """Ensure no crop labels, administrative IDs, or coordinates are in predictive features."""
        from scripts.benchmark_phase6h_stp_ablation import build_feature_groups

        df_eligible = self.df_feat[self.df_feat["extraction_status"] == "success"].copy()
        groups, _ = build_feature_groups(df_eligible, self.df_tex)

        forbidden_fields = {
            "crop", "target", "label", "farmid", "district", "sub-district", "mandal",
            "latitude", "longitude", "lat", "lon", "geometry", "area_m2", "area", "sdate", "hdate"
        }
        for exp_name, feat_list in groups.items():
            for f in feat_list:
                f_lower = f.lower()
                self.assertNotIn(f_lower, forbidden_fields, f"Forbidden feature '{f}' in Exp {exp_name}")
                for forb in ["crop", "target", "farmid", "district", "latitude", "longitude"]:
                    self.assertNotIn(forb, f_lower, f"Forbidden token '{forb}' in feature '{f}' (Exp {exp_name})")

    def test_05_historical_baseline_reproduction(self):
        """Verify Phase 6F Experiment C XGBoost metrics reproduce Macro-F1 0.7038, Acc 0.7877."""
        oof_path = PROJECT_ROOT / "outputs/data/phase6f_oof_predictions.csv"
        self.assertTrue(oof_path.exists())
        df_oof = pd.read_csv(oof_path)
        sub_c = df_oof[
            (df_oof["experiment"] == "Exp_C")
            & (df_oof["run_type"] == "primary")
            & (df_oof["model_name"] == "xgb")
        ]
        self.assertEqual(len(sub_c), 7843)
        macro_f1 = f1_score(sub_c["true_crop"], sub_c["predicted_crop"], average="macro")
        acc = accuracy_score(sub_c["true_crop"], sub_c["predicted_crop"])
        self.assertAlmostEqual(macro_f1, 0.7038, places=4)
        self.assertAlmostEqual(acc, 0.7877, places=4)

    def test_06_pre_benchmark_audit_pass(self):
        """Verify pre-benchmark audit JSON exists and report status is PASS."""
        self.assertTrue(self.audit_json.exists())
        with open(self.audit_json, "r", encoding="utf-8") as f:
            audit = json.load(f)
        self.assertEqual(audit["status"], "PASS")
        self.assertEqual(len(audit["critical_failures"]), 0)

    def test_07_post_benchmark_ablation_artifacts(self):
        """Verify ablation benchmark JSON report, OOF predictions, and model artifacts."""
        self.assertTrue(self.benchmark_json.exists())
        with open(self.benchmark_json, "r", encoding="utf-8") as f:
            bench = json.load(f)
        self.assertEqual(len(bench["experiments"]), 7)
        self.assertIn("S+T", bench["experiments"])
        self.assertIn("S+T+P", bench["experiments"])
        self.assertAlmostEqual(bench["experiments"]["S+T"]["pooled_metrics"]["macro_f1"], 0.7069, places=4)
        self.assertAlmostEqual(bench["experiments"]["S+T+P"]["pooled_metrics"]["accuracy"], 0.7900, places=4)

        self.assertTrue(self.oof_csv.exists())
        df_oof = pd.read_csv(self.oof_csv)
        self.assertEqual(len(df_oof), 7843)
        self.assertIn("pred_S+T", df_oof.columns)
        self.assertIn("pred_S+T+P", df_oof.columns)

    def test_08_weighting_and_boundary_artifacts(self):
        """Verify class-weighting and boundary effect report and OOF predictions."""
        wt_json = PROJECT_ROOT / "outputs/reports/phase6h/phase6h_weighting_boundary_report.json"
        wt_csv = PROJECT_ROOT / "outputs/data/phase6h/phase6h_weighting_boundary_oof_predictions.csv"
        self.assertTrue(wt_json.exists())
        self.assertTrue(wt_csv.exists())

        with open(wt_json, "r", encoding="utf-8") as f:
            wt_data = json.load(f)
        self.assertEqual(len(wt_data["models"]), 4)
        self.assertIn("weighted_S+T", wt_data["models"])
        self.assertIn("weighted_S+T+P", wt_data["models"])
        # Verify weighting improved Macro-F1 over unweighted baseline
        self.assertGreater(wt_data["models"]["weighted_S+T"]["pooled_metrics"]["macro_f1"], 0.71)
        self.assertGreater(wt_data["models"]["weighted_S+T"]["per_class_metrics"]["Red Gram"]["f1_score"], 0.40)

        df_wt = pd.read_csv(wt_csv)
        self.assertEqual(len(df_wt), 7843)
        self.assertIn("pred_weighted_S+T", df_wt.columns)
        self.assertIn("pred_weighted_S+T+P", df_wt.columns)


if __name__ == "__main__":
    unittest.main()
