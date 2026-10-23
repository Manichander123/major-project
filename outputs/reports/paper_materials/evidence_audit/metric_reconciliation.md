# Metric Reconciliation Audit

**Project:** AI-Based Remote Sensing Framework — Seven-Class Crop Classification, Telangana  
**Scope:** Resolution of all reported quantitative discrepancies (A1–A6, B)  
**Cohort:** N = 7,843 eligible farms, 116 Mandals, 5-Fold StratifiedGroupKFold  
**Primary authoritative source:** `outputs/reports/phase6i/phase6i_robustness_audit.json`  
**Audit date:** 2026-10-23 (read-only inspection; no files modified)

---

## 1. Discrepancy A1 — Unweighted S+T Macro-F1: 0.7069 vs 0.7038

### Finding: RESOLVED — Two distinct models, not the same model

| Attribute | Unweighted S+T | Unweighted S+T+P |
|---|---|---|
| **Pooled macro-F1** | **0.7069** | **0.7038** |
| Feature count | 140 | 160 |
| Sample weighting | None | None |
| Cohort | 7,843 | 7,843 |
| Fold-mean macro-F1 ± std | 0.6997 ± 0.0228 | 0.6965 ± 0.0173 |
| Balanced accuracy | 0.6914 | 0.6895 |
| Overall accuracy | 0.7873 | 0.7900 |
| Weighted-F1 | 0.7796 | 0.7818 |
| Source key | `reconciled_models.unweighted_S+T.pooled_metrics` | `reconciled_models.unweighted_S+T+P.pooled_metrics` |
| Source file | `phase6i_robustness_audit.json` | `phase6i_robustness_audit.json` |

Root cause: secondary documents quoted the 160-feature S+T+P figure (0.7038) where the 140-feature S+T figure (0.7069) was intended, or vice versa.

**Paper recommendation:** Qualify every occurrence by model name. Use 0.7069 for unweighted 140-feature S+T and 0.7038 for unweighted 160-feature S+T+P.

---

## 2. Discrepancy A2 — Weighting Gain: +1.57 pp?

### Finding: VERIFIED

| Pair | Unweighted macro-F1 | Weighted macro-F1 | Delta |
|---|---|---|---|
| S+T (140 features) | 0.7069 | 0.7226 | **+0.0157 (+1.57 pp)** |
| S+T+P (160 features) | 0.7038 | 0.7228 | +0.0190 (+1.90 pp) |

Source: `phase6i_robustness_audit.json`, keys `reconciled_models.*.pooled_metrics.macro_f1`.  
Same cohort (N=7,843), same folds, same features; only sample_weight argument differs.

**Paper recommendation:** The +1.57 pp figure is correctly attributed to the S+T pair. It is safe to write in the paper.

---

## 3. Discrepancy A3 — Red Gram F1 "0.3478": origin and validity

### Finding: CORRECTED — 0.3478 is a precision value from a different model/cohort; correct F1 is 0.3683

**Authoritative source for Red Gram in unweighted S+T (`phase6i_robustness_audit.json`, key `reconciled_models.unweighted_S+T.per_class_metrics.Red Gram`):**

| Metric | Value |
|---|---|
| Precision | 0.6170 |
| Recall | 0.2624 |
| **F1-score** | **0.3683** |
| Support | 221 |

The value 0.3478 appears in `outputs/reports/pilot_spatial_cv_benchmark.json` line 442 as the *precision* of a Random Forest model on the 173-farm Phase 6B pilot — not an F1-score and not from the 7,843-farm Phase 6H/6I benchmark.

Occurrences claiming "Red Gram F1 collapsed to 0.3478" (found in `results_and_discussion.md` line 73, `research_gaps_and_contributions.md` line 54, `related_work_matrix.md` line 43, `paper_outline.md` line 16) are **incorrect citations**. No Phase 6H/6I JSON key resolves to this value as an F1.

Correct weighting gain for Red Gram F1: 0.4420 − 0.3683 = **+0.0737 (+7.37 pp)**, not +9.42 pp.

**Required action:** Replace every occurrence of "Red Gram F1 = 0.3478" in paper materials with **0.3683**. Update all downstream gain percentages.

---

## 4. Discrepancy A4 — Weighted-F1: 0.7847 vs 0.7857

### Finding: RESOLVED — 0.7857 is canonical

Source: `phase6i_robustness_audit.json`, key `reconciled_models.weighted_S+T.pooled_metrics.weighted_f1` = **0.7857**.  
Independently confirmed in `phase6i_model_selection_report.md` line 30.

The value 0.7847 does not appear in any Phase 6H or 6I JSON artifact. It originated from a rounding or transcription error in an earlier draft.

**Paper recommendation:** Use **0.7857** as the canonical weighted-F1 for the primary model.

---

## 5. Discrepancy A5 — Pooled OOF metrics vs fold-mean metrics

### Finding: VERIFIED — mechanistically explained in Phase 6I audit

From `phase6i_robustness_audit.json`, key `reconciliation_audit`:

> Pooled metrics aggregate all 7,843 OOF predictions into one global confusion matrix.  
> Fold-mean takes the unweighted arithmetic mean of five independently-computed fold-level metrics.  
> Because minority class counts vary across folds (Red Gram ranges from ~30 to ~65 per fold depending on Adilabad mandal assignment), the non-linear harmonic averaging in F1 produces divergence.

| Model | Pooled macro-F1 | Fold-mean macro-F1 | Divergence |
|---|---|---|---|
| Weighted S+T | 0.7226 | 0.7117 | 0.0109 |
| Weighted S+T+P | 0.7228 | 0.7131 | 0.0097 |

**Paper recommendation:** Report pooled macro-F1 = 0.7226 as the primary metric. Report fold-mean ± std = 0.7117 ± 0.0181 as the spatial stability indicator. Explicitly state both in the same sentence with their definitions.

---

## 6. Discrepancy A6 — Confusion matrix denominators and class ordering

### Finding: VERIFIED

Row order (true class) = alphabetical: Bengal Gram, Chillies, Cotton, Groundnut, Maize, Paddy, Red Gram.  
Row sums match per-class support exactly (938+601+1679+768+1376+2260+221 = 7843).  
Column sums for weighted S+T: 1027+623+1747+777+1236+2283+150 = 7843.  
Source: `phase6i_robustness_audit.json` keys `reconciled_models.weighted_S+T.confusion_matrix` and `per_class_metrics`.

**Paper recommendation:** Add "rows = true class, columns = predicted class" to confusion matrix caption. Report class order explicitly.

---

## 7. Summary Table

| ID | Issue | Status | Canonical Value | Action |
|---|---|---|---|---|
| A1 | 0.7069 vs 0.7038 | RESOLVED | 0.7069 = unweighted S+T; 0.7038 = unweighted S+T+P | Qualify by model name |
| A2 | Weighting gain +1.57 pp | VERIFIED | +0.0157 for S+T pair | Safe to write |
| A3 | Red Gram F1 = 0.3478 | **CORRECTED** | Unweighted S+T Red Gram F1 = **0.3683** | Replace everywhere |
| A4 | Weighted-F1 0.7847 vs 0.7857 | RESOLVED | **0.7857** | Replace all 0.7847 occurrences |
| A5 | Pooled vs fold-mean divergence | VERIFIED | Explained mechanistically | Add clarifying sentence |
| A6 | Confusion matrix denominators | VERIFIED | Rows = true, alphabetical | Add caption note |
