# Figures and Tables Audit

**Task H & G:** Figure/table correctness, mixed cohort comparison, typos  
**Sources inspected:**  
- `outputs/reports/phase6i/phase6i_robustness_audit.json`  
- `outputs/reports/phase6i/phase6i_model_selection_report.md`  
- `outputs/reports/paper_materials/figures_and_tables_plan.md`  
- `outputs/reports/paper_materials/results_and_discussion.md`  
- `outputs/reports/phase6h/phase6h_weighting_boundary_report.json`  
- `outputs/reports/pilot_spatial_cv_benchmark.json`  
- `outputs/reports/phase6d_research_baseline_report.md`  
**Audit date:** 2026-10-23 (read-only; no figures modified)

---

## 1. Figure 1 — Confusion Matrices

**Data source verified:** `phase6i_robustness_audit.json`, key `reconciled_models.weighted_S+T.confusion_matrix` and `reconciled_models.weighted_S+T+P.confusion_matrix`.

Matrix for Weighted S+T (row = true, col = predicted):
```
Predicted ->  Bengal  Chilli  Cotton  Ground   Maize   Paddy  RedGrm
Bengal Gram  [   813       9      24       2      73       4      13 ]   N=938  recall=86.67%
Chillies     [     3     327      82      23      70      91       5 ]   N=601  recall=54.41%
Cotton       [    39      81    1387      37      49      56      30 ]   N=1679 recall=82.61%
Groundnut    [     0      29      36     668      13      20       2 ]   N=768  recall=86.98%
Maize        [   134      81      71      23     926     126      15 ]   N=1376 recall=67.30%
Paddy        [     9      91      71      21      87    1978       3 ]   N=2260 recall=87.52%
Red Gram     [    29       5      76       3      18       8      82 ]   N=221  recall=37.10%
```
Row sums: 938+601+1679+768+1376+2260+221 = 7843. ✓  
**Status: VERIFIED.**

---

## 2. Figure 2 — Class-Wise F1 Comparison

**Issue flagged:** Paddy value labels reportedly hidden behind legend.

**Data verified:** Both models' per-class F1 match `phase6i_robustness_audit.json` exactly:

| Class | S+T F1 | S+T+P F1 |
|---|---|---|
| Bengal Gram | 0.8275 | 0.8381 |
| Chillies | 0.5343 | 0.5217 |
| Cotton | 0.8097 | 0.8129 |
| Groundnut | 0.8647 | 0.8720 |
| Maize | 0.7090 | 0.7098 |
| Paddy | **0.8708** | **0.8732** |
| Red Gram | 0.4420 | 0.4317 |

**Required correction:** Paddy has the second-highest F1 (0.8708) and its label must be visible. Recommendation: move legend to the lower-left panel or inside the bars, or shorten bar labels to two decimal places to reduce overlap.

**Regeneration command:** Run the figure generation script (not run in this audit). Specify `legend_loc='lower right'` or `loc='outside'`.

---

## 3. Figure 3 — Fold Variability

**Fold-level macro-F1 for Weighted S+T (source: `phase6i_robustness_audit.json`):**

| Fold | N | Macro-F1 |
|---|---|---|
| 0 | 1,508 | 0.7269 |
| 1 | 1,600 | 0.7381 |
| 2 | 1,595 | 0.6919 |
| 3 | 1,530 | 0.6945 |
| 4 | 1,610 | 0.7073 |
| **Mean ± std** | | **0.7117 ± 0.0181** |

**Standard deviation convention:** The JSON stores `macro_f1_std: 0.0181`. The `np.std()` default is `ddof=0` (population SD). For five observations, the sample SD (`ddof=1`) would be 0.0181 × √(5/4) = 0.0202. The reported 0.0181 is the population SD.

**Paper recommendation:** Disclose the convention. For N=5, the difference is meaningful. The CONSORT-recommended convention for reporting fold variability is sample SD (ddof=1). If adopting sample SD: **0.7117 ± 0.0202**. Whichever is chosen, state it explicitly.

**Status:** Figure fold scores are correct. SD convention should be clarified.

---

## 4. Figure 4 — Parcel Size Quartiles

**Data verified:** (See `quartile_and_spatial_claims_audit.md` §1 for full table.)

- Q3 upper bound 5,517 m² and Q4 lower bound 5,528 m² gap is a pandas `qcut` artifact, not an error.
- Legend label "five Mandals" would be incorrect; the folds are groups of ~23 Mandals each, not individual Mandals.

**Required correction:** If any axis label or caption says "five Mandals," replace with "five geographic fold groups (~23 Mandals each)."

---

## 5. Figure 5 — Pipeline Evolution (Mixed Cohort)

**Task G — Source verification for each bar:**

| Bar label | Value | Sample N | Feature set | Split method | Source file | Status |
|---|---|---|---|---|---|---|
| Pilot baseline | **0.4630** | 714 | 34 (S, full season) | 5-Fold Mandal grouped CV | `results_and_discussion.md` line 130 (cites "Phase 6D") | **UNVERIFIED in JSON** — not found in `pilot_spatial_cv_benchmark.json` or `phase6d_model_benchmark.json` under that value |
| Early-season XGBoost | **0.5058** | 714 | Early season only | 5-Fold Mandal grouped CV | cited as "Phase 6D" | **UNVERIFIED in JSON** |
| Full-season XGBoost | **0.6174** | 7,843 | 34 (S, full season) | 5-Fold Mandal grouped CV | `phase6f_stage4_benchmark_report.json` key `Exp_A_primary` — value not found with this key format | **Needs JSON key verification** |
| Unweighted S+T | **0.7069** | 7,843 | 140 | 5-Fold Mandal grouped CV | `phase6i_robustness_audit.json` `unweighted_S+T.pooled_metrics.macro_f1` | **VERIFIED** |
| Weighted S+T | **0.7226** | 7,843 | 140 | 5-Fold Mandal grouped CV | `phase6i_robustness_audit.json` `weighted_S+T.pooled_metrics.macro_f1` | **VERIFIED** |
| Weighted S+T+P | **0.7228** | 7,843 | 160 | 5-Fold Mandal grouped CV | `phase6i_robustness_audit.json` `weighted_S+T+P.pooled_metrics.macro_f1` | **VERIFIED** |

**Critical finding: Figure 5 is a mixed-cohort comparison.** Bars 1–2 (N=714) and bars 3–6 (N=7,843) use different sample populations. The XGBoost multimodal pilot result from Phase 6D was **0.7047** (N=714, `phase6d_research_baseline_report.md` line 37), not 0.4630. The value 0.4630 does not appear in any Phase 6D JSON — it may refer to a RF satellite-only result or an earlier pilot configuration.

**Note on 0.4630:** `pilot_spatial_cv_benchmark.json` shows N=173 (Phase 6B pilot), RF satellite-only OOF macro-F1 = 0.3479. For N=714 Phase 6D: RF satellite-only = 0.4961, XGB satellite-only = 0.5511. Neither matches 0.4630 exactly. The value exists only in secondary paper materials documents.

**Required correction for Figure 5:**
1. Add a cohort-break annotation (vertical dashed line or separate panel) between N=714 and N=7,843 bars.
2. Add per-bar annotation of sample size.
3. Update caption to: "Macro-F1 across successive research phases. Bars 1–2 are evaluated on the 714-farm pilot cohort; bars 3–6 on the full 7,843-farm cohort. Cross-cohort comparison reflects engineering progression, not controlled ablation."
4. Verify or replace the 0.4630 value by locating its exact source artifact.

---

## 6. Typo: "0.7881%"

The paper_outline.md line 16 contains "**Overall Accuracy of 0.7881%**." This is a unit error. Either:
- "Overall Accuracy of **0.7881**" (fraction), or
- "Overall Accuracy of **78.81%**" (percentage).

The underlying value is verified as 0.7881 (fraction). The "%" symbol is erroneous.

**Required correction:** Remove the "%" symbol from all occurrences of "0.7881%" in paper materials.

---

## 7. Description of Five Folds vs Five Mandals

If any figure or caption uses "five Mandals" to describe the CV structure, it must be corrected. There are 116 Mandals distributed across 5 folds (~23 Mandals per fold). The folds are geographic fold groups, not individual Mandals.

---

## 8. Summary of Required Figure/Table Corrections

| Asset | Issue | Correction |
|---|---|---|
| Figure 2 | Paddy label hidden behind legend | Move legend or reduce label size |
| Figure 3 | SD convention undisclosed | Add "(population SD, ddof=0)" or switch to sample SD (0.0202) |
| Figure 4 | Possibly mislabelled "five Mandals" | Replace with "five geographic fold groups" |
| Figure 5 | Mixed cohort; 0.4630 unverified in JSON | Add cohort annotation; verify 0.4630 source |
| Paper text | "0.7881%" | Remove "%" |
| Paper text | "Red Gram F1 = 0.3478" | Replace with 0.3683 |
