"""
scripts/audit_phase7_metrics.py
===============================
Phase 7 Plan 3 — Metric Consistency Audit.

Recomputes every metric from the raw predictions CSV, compares with saved
artifacts, identifies root cause of any discrepancies, and regenerates
corrected reports, JSON, and confusion-matrix figure.

Usage:  python scripts/audit_phase7_metrics.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PHASE7_DATA = PROJECT_ROOT / "outputs" / "data" / "phase7"
PHASE7_REPORTS = PROJECT_ROOT / "outputs" / "reports" / "phase7"

CLASSES = [
    "Bengal Gram",
    "Chillies",
    "Cotton",
    "Groundnut",
    "Maize",
    "Paddy",
    "Red Gram",
]

discrepancies: list[str] = []


def load_predictions() -> pd.DataFrame:
    df = pd.read_csv(PHASE7_DATA / "test_predictions.csv")
    assert len(df) == 2718, f"Expected 2718 total rows, got {len(df)}"
    return df


def load_saved_metrics() -> dict:
    with open(PHASE7_REPORTS / "phase7_test_metrics.json") as f:
        return json.load(f)


def audit():
    print("=" * 72)
    print("Phase 7 Plan 3 — Metric Consistency Audit")
    print("=" * 72)

    df = load_predictions()
    saved = load_saved_metrics()

    # ── 1. Filter to evaluated cohort ──────────────────────────────
    eval_df = df[df["extraction_status"] == "success"].copy()
    excl_df = df[df["extraction_status"] != "success"]
    n_eval = len(eval_df)
    n_excl = len(excl_df)
    print(f"\nCohort: {n_eval} evaluated, {n_excl} cloud-excluded, {len(df)} total")
    assert n_eval == 2265, f"Expected 2265 evaluable, got {n_eval}"
    assert n_excl == 453, f"Expected 453 excluded, got {n_excl}"

    y_true = eval_df["true_crop"].values
    y_pred = eval_df["predicted_crop"].values

    # Verify all classes present
    true_set = set(y_true)
    pred_set = set(y_pred)
    for c in CLASSES:
        assert c in true_set, f"Class '{c}' missing from ground truth"
    print(f"All 7 classes present in ground truth: YES")
    print(f"Classes in predictions: {sorted(pred_set)}")

    # ── 2. Recompute confusion matrix ─────────────────────────────
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    cm_saved = np.array(saved["confusion_matrix"]["matrix"])
    saved_classes = saved["confusion_matrix"]["classes"]

    print(f"\n--- Confusion Matrix Comparison ---")
    print(f"Saved class order: {saved_classes}")
    print(f"Audit class order: {CLASSES}")
    assert saved_classes == CLASSES, "Class order mismatch!"

    cm_match = np.array_equal(cm, cm_saved)
    print(f"Confusion matrices identical: {cm_match}")
    if not cm_match:
        discrepancies.append("CONFUSION MATRIX: Saved CM differs from recomputed CM")
        print("Saved CM:")
        print(cm_saved)
        print("Recomputed CM:")
        print(cm)
        print("Difference:")
        print(cm - cm_saved)

    # Matrix totals
    assert cm.sum() == n_eval, f"CM total {cm.sum()} != evaluated count {n_eval}"
    print(f"CM total: {cm.sum()} == {n_eval} ✓")

    # ── 3. Recompute per-class metrics ────────────────────────────
    print(f"\n--- Per-Class Metrics (sklearn.metrics.classification_report) ---")
    report_dict = classification_report(
        y_true, y_pred, labels=CLASSES, output_dict=True, zero_division=0
    )
    print(classification_report(y_true, y_pred, labels=CLASSES, zero_division=0))

    # Also manual computation from the CM to triple-check
    print(f"\n--- Manual CM-derived vs sklearn vs Saved JSON ---")
    print(f"{'Class':>14s}  {'CM_prec':>8s} {'SK_prec':>8s} {'Saved_prec':>10s} {'Match':>6s} | {'CM_rec':>8s} {'SK_rec':>8s} {'Saved_rec':>10s} {'Match':>6s} | {'CM_f1':>8s} {'SK_f1':>8s} {'Saved_f1':>10s} {'Match':>6s} | Support")
    print("-" * 150)

    for i, c in enumerate(CLASSES):
        tp = cm[i, i]
        col_sum = cm[:, i].sum()  # total predicted as class i
        row_sum = cm[i, :].sum()  # support (true class i)

        cm_prec = tp / col_sum if col_sum > 0 else 0.0
        cm_rec = tp / row_sum if row_sum > 0 else 0.0
        cm_f1 = 2 * cm_prec * cm_rec / (cm_prec + cm_rec) if (cm_prec + cm_rec) > 0 else 0.0

        sk_prec = report_dict[c]["precision"]
        sk_rec = report_dict[c]["recall"]
        sk_f1 = report_dict[c]["f1-score"]
        sk_sup = int(report_dict[c]["support"])

        sv = saved["per_class"][c]
        sv_prec = sv["precision"]
        sv_rec = sv["recall"]
        sv_f1 = sv["f1_score"]
        sv_sup = sv["evaluated_support"]

        # Check CM vs sklearn match (should be identical)
        assert abs(cm_prec - sk_prec) < 1e-6, f"CM vs sklearn precision mismatch for {c}"
        assert abs(cm_rec - sk_rec) < 1e-6, f"CM vs sklearn recall mismatch for {c}"
        assert abs(cm_f1 - sk_f1) < 1e-6, f"CM vs sklearn F1 mismatch for {c}"
        assert sk_sup == row_sum, f"Support mismatch for {c}"

        # Check saved vs recomputed
        p_ok = abs(cm_prec - sv_prec) < 0.001
        r_ok = abs(cm_rec - sv_rec) < 0.001
        f_ok = abs(cm_f1 - sv_f1) < 0.001
        s_ok = sk_sup == sv_sup

        if not p_ok:
            discrepancies.append(f"{c} precision: saved={sv_prec:.4f}, recomputed={cm_prec:.4f}")
        if not r_ok:
            discrepancies.append(f"{c} recall: saved={sv_rec:.4f}, recomputed={cm_rec:.4f}")
        if not f_ok:
            discrepancies.append(f"{c} F1: saved={sv_f1:.4f}, recomputed={cm_f1:.4f}")
        if not s_ok:
            discrepancies.append(f"{c} support: saved={sv_sup}, recomputed={sk_sup}")

        print(f"{c:>14s}  {cm_prec:8.4f} {sk_prec:8.4f} {sv_prec:10.4f} {'  ✓' if p_ok else ' ✗✗':>6s} | "
              f"{cm_rec:8.4f} {sk_rec:8.4f} {sv_rec:10.4f} {'  ✓' if r_ok else ' ✗✗':>6s} | "
              f"{cm_f1:8.4f} {sk_f1:8.4f} {sv_f1:10.4f} {'  ✓' if f_ok else ' ✗✗':>6s} | {sk_sup}")

    # ── 4. Recompute headline metrics ────────────────────────────
    print(f"\n--- Headline Metrics ---")
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=CLASSES, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, labels=CLASSES, average="weighted")
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    correct = int((y_true == y_pred).sum())
    incorrect = n_eval - correct

    sv_m = saved["metrics"]
    metrics_checks = {
        "accuracy": (acc, sv_m["accuracy"]),
        "macro_f1": (macro_f1, sv_m["macro_f1"]),
        "weighted_f1": (weighted_f1, sv_m["weighted_f1"]),
        "balanced_accuracy": (bal_acc, sv_m["balanced_accuracy"]),
        "correct_predictions": (correct, sv_m["correct_predictions"]),
        "incorrect_predictions": (incorrect, sv_m["incorrect_predictions"]),
    }

    for name, (recomp, sv_val) in metrics_checks.items():
        tol = 0.0005 if isinstance(recomp, float) else 0
        ok = abs(recomp - sv_val) <= tol
        status = "✓" if ok else "✗✗ MISMATCH"
        print(f"  {name:>25s}:  recomputed={recomp}  saved={sv_val}  {status}")
        if not ok:
            discrepancies.append(f"Headline {name}: saved={sv_val}, recomputed={recomp}")

    # ── 5. Root cause analysis ────────────────────────────────────
    print(f"\n{'=' * 72}")
    print(f"DISCREPANCIES FOUND: {len(discrepancies)}")
    print(f"{'=' * 72}")
    for d in discrepancies:
        print(f"  ✗ {d}")

    if discrepancies:
        print("\n--- ROOT CAUSE ANALYSIS ---")
        # Check if the original script used numeric labels vs string labels
        # The original script maps y_true_str -> y_true_num via crop_to_idx,
        # then calls sklearn with numeric labels. sklearn's confusion_matrix
        # with labels=list(range(7)) should produce the same ordering.
        # But the per-class metrics were computed using numeric arrays and
        # precision_score(average=None), which should match.
        # Let's check if there's a label encoding mismatch.
        print("Checking if original numeric encoding matches string-based computation...")
        crop_to_idx = {c: i for i, c in enumerate(CLASSES)}
        y_true_num = np.array([crop_to_idx[c] for c in y_true])
        y_pred_num = np.array([crop_to_idx[c] for c in y_pred])
        cm_num = confusion_matrix(y_true_num, y_pred_num, labels=list(range(7)))
        print(f"Numeric CM == String CM: {np.array_equal(cm_num, cm)}")
        prec_num = precision_score(y_true_num, y_pred_num, average=None, zero_division=0)
        rec_num = recall_score(y_true_num, y_pred_num, average=None, zero_division=0)
        f1_num = f1_score(y_true_num, y_pred_num, average=None, zero_division=0)
        for i, c in enumerate(CLASSES):
            sk = report_dict[c]
            if abs(prec_num[i] - sk["precision"]) > 1e-6:
                print(f"  {c}: numeric prec={prec_num[i]:.6f} vs string prec={sk['precision']:.6f} — DIFFERS")
            if abs(rec_num[i] - sk["recall"]) > 1e-6:
                print(f"  {c}: numeric rec={rec_num[i]:.6f} vs string rec={sk['recall']:.6f} — DIFFERS")
            if abs(f1_num[i] - sk["f1-score"]) > 1e-6:
                print(f"  {c}: numeric f1={f1_num[i]:.6f} vs string f1={sk['f1-score']:.6f} — DIFFERS")
        # If all match, the issue is that the saved JSON was generated from a different run
        # or the CM in the JSON doesn't match the metrics in the JSON.
        print("\nChecking if saved CM is internally consistent with saved per-class metrics...")
        for i, c in enumerate(CLASSES):
            tp_sv = cm_saved[i, i]
            col_sv = cm_saved[:, i].sum()
            row_sv = cm_saved[i, :].sum()
            prec_from_sv_cm = tp_sv / col_sv if col_sv > 0 else 0
            rec_from_sv_cm = tp_sv / row_sv if row_sv > 0 else 0
            f1_from_sv_cm = 2 * prec_from_sv_cm * rec_from_sv_cm / (prec_from_sv_cm + rec_from_sv_cm) if (prec_from_sv_cm + rec_from_sv_cm) > 0 else 0
            sv = saved["per_class"][c]
            p_ok = abs(prec_from_sv_cm - sv["precision"]) < 0.001
            r_ok = abs(rec_from_sv_cm - sv["recall"]) < 0.001
            f_ok = abs(f1_from_sv_cm - sv["f1_score"]) < 0.001
            if not (p_ok and r_ok and f_ok):
                print(f"  {c}: SAVED JSON IS INTERNALLY INCONSISTENT")
                print(f"    CM implies: prec={prec_from_sv_cm:.4f} rec={rec_from_sv_cm:.4f} f1={f1_from_sv_cm:.4f}")
                print(f"    JSON says:  prec={sv['precision']:.4f} rec={sv['recall']:.4f} f1={sv['f1_score']:.4f}")

    # ── 6. Regenerate corrected artifacts ─────────────────────────
    if discrepancies:
        print(f"\n--- REGENERATING CORRECTED ARTIFACTS ---")
        regenerate_artifacts(y_true, y_pred, eval_df, df, cm, report_dict, acc, macro_f1, weighted_f1, bal_acc, correct, incorrect, n_eval, n_excl)
    else:
        print("\nAll metrics consistent — no regeneration needed.")

    # ── 7. Final self-check assertions ────────────────────────────
    print(f"\n--- FINAL SELF-CHECK ASSERTIONS ---")
    # Reload the (possibly regenerated) JSON
    final_json = load_saved_metrics()
    final_cm = np.array(final_json["confusion_matrix"]["matrix"])

    # Assertion 1: CM sums to evaluated count
    assert final_cm.sum() == n_eval, f"FAIL: CM sum {final_cm.sum()} != {n_eval}"
    print(f"  ✓ CM sum ({final_cm.sum()}) == evaluated count ({n_eval})")

    # Assertion 2: per-class supports sum to n_eval
    total_support = sum(final_json["per_class"][c]["evaluated_support"] for c in CLASSES)
    assert total_support == n_eval, f"FAIL: support sum {total_support} != {n_eval}"
    print(f"  ✓ Per-class support sum ({total_support}) == {n_eval}")

    # Assertion 3: CM row sums match supports
    for i, c in enumerate(CLASSES):
        row_s = final_cm[i].sum()
        json_s = final_json["per_class"][c]["evaluated_support"]
        assert row_s == json_s, f"FAIL: {c} CM row sum {row_s} != JSON support {json_s}"
    print(f"  ✓ All CM row sums match per-class supports")

    # Assertion 4: CM-derived precision/recall/F1 match JSON per-class
    for i, c in enumerate(CLASSES):
        tp = final_cm[i, i]
        col_s = final_cm[:, i].sum()
        row_s = final_cm[i, :].sum()
        exp_prec = tp / col_s if col_s > 0 else 0
        exp_rec = tp / row_s if row_s > 0 else 0
        exp_f1 = 2 * exp_prec * exp_rec / (exp_prec + exp_rec) if (exp_prec + exp_rec) > 0 else 0
        j = final_json["per_class"][c]
        assert abs(exp_prec - j["precision"]) < 0.001, f"FAIL: {c} prec CM={exp_prec:.4f} JSON={j['precision']}"
        assert abs(exp_rec - j["recall"]) < 0.001, f"FAIL: {c} rec CM={exp_rec:.4f} JSON={j['recall']}"
        assert abs(exp_f1 - j["f1_score"]) < 0.001, f"FAIL: {c} f1 CM={exp_f1:.4f} JSON={j['f1_score']}"
    print(f"  ✓ All per-class metrics consistent with CM")

    # Assertion 5: headline metrics match
    assert abs(acc - final_json["metrics"]["accuracy"]) < 0.001
    assert abs(macro_f1 - final_json["metrics"]["macro_f1"]) < 0.001
    assert abs(weighted_f1 - final_json["metrics"]["weighted_f1"]) < 0.001
    assert abs(bal_acc - final_json["metrics"]["balanced_accuracy"]) < 0.001
    print(f"  ✓ Headline metrics match recomputed values")

    # Assertion 6: correct + incorrect == n_eval
    assert final_json["metrics"]["correct_predictions"] + final_json["metrics"]["incorrect_predictions"] == n_eval
    print(f"  ✓ correct + incorrect == {n_eval}")

    print(f"\n{'=' * 72}")
    if discrepancies:
        print(f"AUDIT COMPLETE: {len(discrepancies)} discrepancies found and corrected.")
    else:
        print("AUDIT COMPLETE: All metrics verified consistent.")
    print(f"{'=' * 72}")

    return len(discrepancies)


def regenerate_artifacts(y_true, y_pred, eval_df, full_df, cm, report_dict,
                          acc, macro_f1, weighted_f1, bal_acc, correct, incorrect,
                          n_eval, n_excl):
    """Regenerate all Phase 7 reports/JSON/figure from verified predictions."""
    test_csv = pd.read_csv(PROJECT_ROOT / "datasss" / "Test.csv")

    # ── Per-class results ──
    per_class = {}
    for c in CLASSES:
        total_in_test = int((test_csv["Crop"] == c).sum())
        ev_support = int(report_dict[c]["support"])
        per_class[c] = {
            "total_test_support": total_in_test,
            "evaluated_support": ev_support,
            "cloud_excluded": total_in_test - ev_support,
            "extraction_coverage_pct": round(ev_support / total_in_test * 100, 2),
            "precision": round(report_dict[c]["precision"], 4),
            "recall": round(report_dict[c]["recall"], 4),
            "f1_score": round(report_dict[c]["f1-score"], 4),
        }

    # ── Metrics JSON ──
    metrics_data = {
        "evaluation_name": "Phase 7 Test.csv Supervised Independent Evaluation",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "audit_note": "Regenerated by audit_phase7_metrics.py after metric consistency audit",
        "model_artifact": "outputs\\models\\phase6h\\xgb_weighted_s+t.joblib",
        "cohort": {
            "total_test_farms": len(full_df),
            "evaluated_farms": n_eval,
            "cloud_excluded_farms": n_excl,
            "extraction_coverage_pct": round(n_eval / len(full_df) * 100, 2),
            "test_mandals_count": 35,
            "mandal_overlap_with_train": 0,
        },
        "metrics": {
            "macro_f1": round(float(macro_f1), 4),
            "accuracy": round(float(acc), 4),
            "accuracy_pct": round(float(acc * 100), 2),
            "balanced_accuracy": round(float(bal_acc), 4),
            "weighted_f1": round(float(weighted_f1), 4),
            "correct_predictions": correct,
            "incorrect_predictions": incorrect,
        },
        "per_class": per_class,
        "confusion_matrix": {
            "classes": CLASSES,
            "matrix": cm.tolist(),
        },
        "comparison_with_spatial_cv": {
            "cv_macro_f1": 0.7226,
            "test_macro_f1": round(float(macro_f1), 4),
            "macro_f1_delta": round(float(macro_f1 - 0.7226), 4),
            "cv_accuracy": 0.7881,
            "test_accuracy": round(float(acc), 4),
            "accuracy_delta": round(float(acc - 0.7881), 4),
        },
    }
    json_path = PHASE7_REPORTS / "phase7_test_metrics.json"
    json_path.write_text(json.dumps(metrics_data, indent=2), encoding="utf-8")
    print(f"  ✓ Saved corrected {json_path.name}")

    # ── Confusion matrix figure ──
    fig, ax = plt.subplots(figsize=(9, 7.5), dpi=300)
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(
        f"Figure 7: Confusion Matrix — Independent Test Set (N={n_eval})\n"
        f"35 Unseen Mandals, Telangana",
        fontsize=12, fontweight="bold", pad=12,
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ticks = np.arange(len(CLASSES))
    ax.set_xticks(ticks)
    ax.set_xticklabels(CLASSES, rotation=40, ha="right", fontsize=9.5)
    ax.set_yticks(ticks)
    ax.set_yticklabels(CLASSES, fontsize=9.5)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            color = "white" if val > thresh else "black"
            ax.text(j, i, f"{val}", ha="center", va="center",
                    color=color, fontsize=10, fontweight="bold")

    ax.set_ylabel("Ground Truth Crop Label", fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted Crop Label", fontsize=11, fontweight="bold")
    fig.tight_layout()
    cm_path = PHASE7_REPORTS / "phase7_confusion_matrix.png"
    fig.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved corrected {cm_path.name}")

    # ── Evaluation report markdown ──
    cm_text_rows = []
    for i, c in enumerate(CLASSES):
        abbr = {"Bengal Gram": "BG", "Chillies": "CH", "Cotton": "COT",
                "Groundnut": "GN", "Maize": "MZ", "Paddy": "PAD", "Red Gram": "RG"}[c]
        vals = "   ".join(f"{cm[i, j]:3d}" for j in range(7))
        cm_text_rows.append(f"{c:15s} {vals} |     {cm[i].sum():3d}")
    pred_totals = "   ".join(f"{cm[:, j].sum():3d}" for j in range(7))

    per_class_table = []
    for c in CLASSES:
        pc = per_class[c]
        per_class_table.append(
            f"| **{c}** | {pc['total_test_support']} | {pc['evaluated_support']} | "
            f"{pc['cloud_excluded']} | {pc['precision']:.4f} | {pc['recall']:.4f} | "
            f"**{pc['f1_score']:.4f}** |"
        )

    report_md = f"""# Phase 7 — Test.csv Supervised Independent Evaluation Report

**Evaluation Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}
**Audit Note:** Metrics regenerated after consistency audit by `audit_phase7_metrics.py`
**Evaluated Cohort:** $N = {n_eval}$ farm parcels across 35 unseen Mandals in Telangana
**Excluded Cloud Cohort:** $N = {n_excl}$ parcels ({n_excl / len(full_df) * 100:.2f}%) with persistent cloud obscuration
**Model Evaluated:** Frozen `Weighted S+T` (140 Spectral-Temporal features, inverse-frequency weighted regularized XGBoost from Phase 6I)
**Model Path:** `outputs\\models\\phase6h\\xgb_weighted_s+t.joblib`

---

## 1. Executive Summary & Master Test Metrics

The frozen Phase 6I champion model was evaluated against ground-truth crop observations on the independent `Test.csv` cohort with zero hyperparameter tuning, zero threshold recalibration, and zero test-feature selection:

| Metric | Test Set ($N={n_eval}$) | 5-Fold Spatial CV ($N=7,843$) | Delta | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Macro-F1** | **{macro_f1:.4f}** | **0.7226** | **{macro_f1 - 0.7226:+.4f}** | Comparable |
| **Overall Accuracy** | **{acc:.4f} ({acc*100:.2f}%)** | **0.7881 (78.81%)** | **{acc - 0.7881:+.4f}** | Comparable |
| **Balanced Accuracy** | **{bal_acc:.4f}** | **0.7180** | **{bal_acc - 0.7180:+.4f}** | Comparable |
| **Weighted-F1** | **{weighted_f1:.4f}** | **0.7857** | **{weighted_f1 - 0.7857:+.4f}** | Comparable |
| **Correct Predictions** | **{correct:,d} / {n_eval:,d}** | — | — | {acc*100:.2f}% Hit Rate |
| **Incorrect Predictions** | **{incorrect:,d} / {n_eval:,d}** | — | — | {(1-acc)*100:.2f}% Error Rate |

*Note:* These metrics reflect the {n_eval} cloud-filtered evaluable farms from 35 Mandals absent from training. The difference in cohort and evaluation design means that observed performance differences may reflect class-distribution changes and cloud-related selection effects in addition to geographic shift. The {n_eval} evaluated farms do not represent all {len(full_df)} Test.csv farms.

---

## 2. Per-Class Independent Performance Matrix

| Crop Species | Total Test Support | Evaluated Support ($N$) | Cloud Excluded ($N$) | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
{chr(10).join(per_class_table)}

---

## 3. 7×7 Confusion Matrix (rows = true label, columns = predicted label)

```
Predicted ->
True Crop        BG    CH   COT    GN    MZ   PAD    RG | Support
------------------------------------------------------------------
{chr(10).join(cm_text_rows)}
------------------------------------------------------------------
Total Predicted {pred_totals} |    {cm.sum():4d}
```

---

## 4. Nuanced Comparison: Test Cohort vs Spatial Cross-Validation

1. **Geographic Distribution:**
   - Spatial CV evaluated across 116 training Mandals.
   - Test evaluation was conducted across **35 distinct, geographically isolated Mandals** with **zero training overlap**.
2. **Class Distribution Shift:**
   - In the training set, Maize represented 17.6% of farms; in the test set, Maize represents {631/n_eval*100:.1f}% of evaluated farms.
   - Bengal Gram dropped from 12.0% in training to {46/n_eval*100:.1f}% in the evaluated test set.
   - Chillies increased from 7.6% in training to {311/n_eval*100:.1f}% in the test set.
3. **Cloud Contamination Differences:**
   - In `Train.csv`, 45 farms (0.57%) were completely cloud-obscured.
   - In `Test.csv`, {n_excl} farms ({n_excl/len(full_df)*100:.2f}%) were cloud-obscured, predominantly during monsoon Paddy cultivation.
   - The reported test metrics reflect the **usable, cloud-filtered cohort ($N={n_eval}$)**, not all {len(full_df)} farms.

---

## 5. Deliverable Paths

- **Test Predictions:** `outputs\\data\\phase7\\test_predictions.csv`
- **Evaluation Metrics JSON:** `outputs\\reports\\phase7\\phase7_test_metrics.json`
- **Confusion Matrix Graphic:** `outputs\\reports\\phase7\\phase7_confusion_matrix.png`
"""
    report_path = PHASE7_REPORTS / "phase7_test_evaluation_report.md"
    report_path.write_text(report_md, encoding="utf-8")
    print(f"  ✓ Saved corrected {report_path.name}")

    # ── Plan 3 summary ──
    summary_md = f"""# Phase 7 — Plan 3 Execution Summary (Post-Audit)

- **Evaluated Farms:** {n_eval} / {len(full_df)} ({n_eval/len(full_df)*100:.2f}% coverage)
- **Cloud Excluded:** {n_excl} farms ({n_excl/len(full_df)*100:.2f}%)
- **Supervised Metrics (verified by audit):**
  - Macro-F1: **{macro_f1:.4f}** (Spatial CV baseline: 0.7226, delta: {macro_f1 - 0.7226:+.4f})
  - Overall Accuracy: **{acc:.4f} ({acc*100:.2f}%)** (Spatial CV baseline: 0.7881, delta: {acc - 0.7881:+.4f})
  - Balanced Accuracy: **{bal_acc:.4f}** (Spatial CV baseline: 0.7180)
  - Weighted-F1: **{weighted_f1:.4f}** (Spatial CV baseline: 0.7857)
- **Audit Status:** Metrics recomputed from raw predictions and verified internally consistent.
- **Deliverables:**
  - `outputs\\data\\phase7\\test_predictions.csv`
  - `outputs\\reports\\phase7\\phase7_test_metrics.json`
  - `outputs\\reports\\phase7\\phase7_confusion_matrix.png`
  - `outputs\\reports\\phase7\\phase7_test_evaluation_report.md`
"""
    summary_path = PHASE7_REPORTS / "PHASE7_PLAN3_SUMMARY.md"
    summary_path.write_text(summary_md, encoding="utf-8")
    print(f"  ✓ Saved corrected {summary_path.name}")


if __name__ == "__main__":
    n = audit()
    sys.exit(0)
