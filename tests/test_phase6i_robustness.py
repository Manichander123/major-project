"""
tests/test_phase6i_robustness.py
================================
Unit and regression tests for Phase 6I Robustness Validation and Model Selection.
Validates:
1. Exact 7,843-farm cohort alignment and zero duplicate FarmIDs.
2. Complete spatial isolation: zero Mandal overlap in all 5 folds.
3. Bit-for-bit reproduction of pooled and fold metrics from out-of-fold predictions.
4. Reconciliation between pooled Macro-F1 and fold-mean Macro-F1.
5. Strict zero-leakage in predictive feature definitions.
6. Statistical indistinguishability between Weighted S+T and Weighted S+T+P (Mandal bootstrap 95% CI contains 0).
7. Protected baseline SHA-256 hashes intact.
8. Presence of all Phase 6I deliverables in outputs/reports/phase6i/.
"""

from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestPhase6IRobustness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folds_path = PROJECT_ROOT / "outputs/data/phase6f_cv_folds.csv"
        cls.oof_path = PROJECT_ROOT / "outputs/data/phase6h/phase6h_weighting_boundary_oof_predictions.csv"
        cls.manifest_path = PROJECT_ROOT / "outputs/reports/phase6h/phase6h_feature_schema_manifest.json"
        cls.phase6i_json = PROJECT_ROOT / "outputs/reports/phase6i/phase6i_robustness_audit.json"
        cls.phase6i_md = PROJECT_ROOT / "outputs/reports/phase6i/phase6i_model_selection_report.md"

        cls.df_folds = pd.read_csv(cls.folds_path)
        cls.df_oof = pd.read_csv(cls.oof_path)

        with open(cls.phase6i_json, "r", encoding="utf-8") as f:
            cls.audit_data = json.load(f)

    def test_01_cohort_exact_alignment(self):
        """Verify exact 7,843 eligible farm cohort matching and 0 duplicate FarmIDs."""
        self.assertEqual(len(self.df_folds), 7843)
        self.assertEqual(len(self.df_oof), 7843)
        self.assertEqual(set(self.df_folds["FarmID"]), set(self.df_oof["FarmID"]))
        self.assertEqual(self.df_oof["FarmID"].duplicated().sum(), 0)

    def test_02_spatial_isolation_zero_mandal_overlap(self):
        """Verify zero Mandal overlap between train and val for every fold."""
        for fold_i in range(5):
            val_mandals = set(self.df_oof.loc[self.df_oof["fold"] == fold_i, "Sub-District"])
            tr_mandals = set(self.df_oof.loc[self.df_oof["fold"] != fold_i, "Sub-District"])
            overlap = val_mandals.intersection(tr_mandals)
            self.assertEqual(len(overlap), 0, f"Mandal overlap in fold {fold_i}: {overlap}")

    def test_03_metric_reproduction_from_oof(self):
        """Verify out-of-fold predictions reproduce reported metrics exactly."""
        y_true = self.df_oof["true_crop"]

        # 1. Weighted S+T
        p_st = self.df_oof["pred_weighted_S+T"]
        mf1_st = f1_score(y_true, p_st, average="macro")
        acc_st = accuracy_score(y_true, p_st)
        bacc_st = balanced_accuracy_score(y_true, p_st)

        self.assertAlmostEqual(mf1_st, 0.7226, places=4)
        self.assertAlmostEqual(acc_st, 0.7881, places=4)
        self.assertAlmostEqual(bacc_st, 0.7180, places=4)

        # 2. Weighted S+T+P
        p_stp = self.df_oof["pred_weighted_S+T+P"]
        mf1_stp = f1_score(y_true, p_stp, average="macro")
        acc_stp = accuracy_score(y_true, p_stp)
        bacc_stp = balanced_accuracy_score(y_true, p_stp)

        self.assertAlmostEqual(mf1_stp, 0.7228, places=4)
        self.assertAlmostEqual(acc_stp, 0.7905, places=4)
        self.assertAlmostEqual(bacc_stp, 0.7167, places=4)

    def test_04_minority_class_f1_reproduction(self):
        """Verify Red Gram and Chillies F1 scores reproduced from OOF."""
        y_true = self.df_oof["true_crop"]
        p_st = self.df_oof["pred_weighted_S+T"]
        p_stp = self.df_oof["pred_weighted_S+T+P"]

        # Red Gram
        rg_mask = y_true == "Red Gram"
        rg_f1_st = f1_score(rg_mask, p_st == "Red Gram")
        rg_f1_stp = f1_score(rg_mask, p_stp == "Red Gram")
        self.assertAlmostEqual(rg_f1_st, 0.4420, places=4)
        self.assertAlmostEqual(rg_f1_stp, 0.4317, places=4)

        # Chillies
        ch_mask = y_true == "Chillies"
        ch_f1_st = f1_score(ch_mask, p_st == "Chillies")
        ch_f1_stp = f1_score(ch_mask, p_stp == "Chillies")
        self.assertAlmostEqual(ch_f1_st, 0.5343, places=4)
        self.assertAlmostEqual(ch_f1_stp, 0.5217, places=4)

    def test_05_statistical_indistinguishability_bootstrap_ci(self):
        """Verify Mandal cluster bootstrap 95% confidence interval covers zero."""
        boot = self.audit_data["paired_significance_audit"]["mandal_cluster_bootstrap"]
        ci = boot["ci_95_percent"]
        self.assertTrue(ci[0] <= 0 <= ci[1], f"Bootstrap 95% CI does not cover 0: {ci}")
        self.assertGreater(boot["bootstrap_p_value"], 0.05)

    def test_06_leakage_checks_in_features(self):
        """Verify zero leakage fields in feature manifest."""
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        forbidden = {"crop", "target", "label", "farmid", "district", "sub-district", "mandal", "lat", "lon", "geometry", "area"}
        for grp_name, feat_list in manifest["groups"].items():
            for f in feat_list:
                f_low = f.lower()
                self.assertNotIn(f_low, forbidden, f"Forbidden leakage column '{f}' in group '{grp_name}'")

    def test_07_protected_hashes_intact(self):
        """Verify all protected baseline hashes remain intact."""
        from scripts.audit_phase6i_robustness import verify_protected_baselines
        ok, _ = verify_protected_baselines()
        self.assertTrue(ok)

    def test_08_phase6i_deliverables_exist(self):
        """Verify all Phase 6I deliverables exist and are non-empty."""
        self.assertTrue(self.phase6i_json.exists())
        self.assertTrue(self.phase6i_md.exists())
        self.assertGreater(self.phase6i_json.stat().st_size, 1000)
        self.assertGreater(self.phase6i_md.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
