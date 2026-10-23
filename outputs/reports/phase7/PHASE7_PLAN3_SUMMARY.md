# Phase 7 — Plan 3 Execution Summary

- **Evaluated Farms:** 2265 / 2,718 (83.33% coverage)
- **Cloud Excluded:** 453 farms (16.67%)
- **Supervised Metrics:**
  - Macro-F1: **0.7308** (Spatial CV baseline: 0.7226, delta: +0.0082)
  - Overall Accuracy: **0.8155 (81.55%)** (Spatial CV baseline: 0.7881, delta: +0.0274)
  - Balanced Accuracy: **0.7326** (Spatial CV baseline: 0.7180)
  - Weighted-F1: **0.8137** (Spatial CV baseline: 0.7857)
- **Zero Leakage:** Strictly confirmed (no labels, coordinates, or proxies entered model).
- **Deliverables Generated:**
  - `outputs\data\phase7\test_predictions.csv`
  - `outputs\reports\phase7\phase7_test_metrics.json`
  - `outputs\reports\phase7\phase7_confusion_matrix.png`
  - `outputs\reports\phase7\phase7_test_evaluation_report.md`
