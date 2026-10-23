# Quartile and Spatial Claims Audit

**Task F:** Parcel-area quartile and spatial-boundary claims  
**Primary sources:**  
- `outputs/reports/phase6i/phase6i_robustness_audit.json` (quartile metrics)  
- `outputs/reports/phase6i/phase6i_model_selection_report.md` §5  
- `outputs/reports/phase6h/phase6h_weighting_boundary_report.json` (boundary contamination)  
- `outputs/reports/phase6f_cv_folds_audit.json` (fold grouping)  
**Audit date:** 2026-10-23 (read-only)

---

## 1. Verified Area Quartile Metrics

Source: `phase6i_robustness_audit.json`, key `reconciled_models.weighted_S+T.area_quartile_metrics`  
and `reconciled_models.weighted_S+T+P.area_quartile_metrics`.

| Quartile | Area range (m²) | N | Weighted S+T macro-F1 | Weighted S+T+P macro-F1 | Delta |
|---|---|---|---|---|---|
| Q1_small | 54 – 1,257 | 1,961 | **0.6598** | **0.6465** | −0.0133 |
| Q2_med_low | 1,257 – 2,783 | 1,961 | **0.7332** | **0.7184** | −0.0148 |
| Q3_med_high | 2,784 – 5,517 | 1,960 | **0.7400** | **0.7530** | +0.0130 |
| Q4_large | 5,528 – 69,491 | 1,961 | **0.7253** | **0.7219** | −0.0034 |

All values match the paper materials documents exactly.

### Cut-point gap: 5,517 vs 5,528 m²

The Q3 upper bound is listed as 5,517 m² and Q4 lower bound as 5,528 m² in `phase6i_model_selection_report.md` §5. This gap of 11 m² represents farms whose area falls between the Q3 maximum (exclusive) and Q4 minimum (inclusive) in the actual sorted data, arising from a 1-based versus 0-based quartile definition in pandas `qcut`. The 1,960 Q3 count vs 1,961 for other quartiles is consistent with this: one farm with area = exactly the Q3/Q4 boundary value was assigned to Q4. This is a standard `qcut` artifact, not an error or data exclusion.

---

## 2. Sample Count Balance

Four bins contain 1961, 1961, 1960, 1961 farms = 7,843 total. The bins are near-equal but not exactly equal because equal-frequency binning (`qcut`) rounds to whole-farm boundaries. No farms are excluded.

---

## 3. Statistical Uncertainty

**No confidence intervals, repeated resampling, or statistical tests exist for the quartile-level metrics.** The quartile breakdown is a post-hoc descriptive analysis on out-of-fold predictions, not a preregistered hypothesis test.

The paired significance tests in Phase 6I (paired t, Wilcoxon, cluster bootstrap) test the global S+T vs S+T+P difference only. No separate tests for the per-quartile deltas are present in any artifact.

**Paper recommendation:** Describe the quartile results as exploratory and descriptive. Do not claim statistical significance for per-quartile differences. Use cautious language:
> "Parcel-area stratification revealed an association between field size and GLCM effectiveness: smaller parcels (Q1, < 1,257 m²) showed a –0.0133 macro-F1 reduction when texture features were added, whereas larger consolidated parcels (Q3, 2,784–5,517 m²) showed a +0.0130 gain. These patterns are consistent with the physical boundary contamination mechanism but have not been independently tested with per-quartile confidence intervals."

---

## 4. Boundary Contamination: Observed Association, Not Proven Causal Mechanism

Source: `phase6h_weighting_boundary_report.json`, key `boundary_investigation`.

### Physical GLCM kernel facts (verified)

- Kernel dimension: 3×3 Sentinel-2 pixels = 30 m × 30 m = 900 m²
- Kernel radius: 15 m
- Farms < 900 m²: 1,353 (17.25% of cohort)
- Farms < 1,800 m² (kernel area × 2): 2,748 (35.04%)
- Mean estimated boundary contamination fraction: 0.6493 (64.9%)

These are geometric facts derived from farm polygon areas. However:
- The experiments do **not** directly measure out-of-parcel pixel sampling.
- GLCM was computed on masked parcel pixels, not on a sliding window that necessarily crosses boundaries. Whether Earth Engine GLCM reducers respect the polygon mask or use a moving-window that extends beyond it depends on the implementation — this is **not confirmed** by any reviewed artifact.
- "Edge contamination" must therefore be described as a plausible physical explanation consistent with the observed size-stratified association, not a proven causal mechanism.

**Paper recommendation:** Write "consistent with boundary contamination" or "attributable to edge-pixel mixing under the 30 m × 30 m kernel footprint," not "proven by" or "directly demonstrates."

---

## 5. Spatial Grouping: Mandal Groups, Not Contiguous Zones

The five folds are **groups of Sub-District (Mandal) identifiers** assigned by StratifiedGroupKFold, which minimises Mandal overlap while approximately balancing class distribution across folds. They are NOT five contiguous rectangular geographic blocks.

Source: `project_evidence_and_discrepancies.md` §4; `phase6i_robustness_audit.json`, `zero_mandal_overlap_confirmed: true`.

What the validation guarantees:
- Zero Mandal overlap between training and validation in any fold.
- No farm from any Mandal appears in both training and validation in the same fold.
- Mandals from the same district may be split across folds.

What it does NOT guarantee:
- Contiguous geographic separation (farms in different folds may be geographically adjacent if they belong to different Mandals).
- Elimination of spatial autocorrelation between adjacent-Mandal farms in training and validation sets.

**Paper recommendation:** State "grouped by Sub-District (Mandal) boundaries" instead of "geographically disjoint zones." Describe the validation as "reducing spatial autocorrelation leakage by ensuring no farm from a tested Mandal appears in the training set for that fold."

---

## 6. "First Systematic Quartile Ablation" Novelty Claim

**This priority claim is NOT defensible without a literature search.** No literature search was conducted as part of this audit. The related work matrix cites Vuolo et al. (2018) and Lebourgeois et al. (2017) as related work but does not document whether any of the 10 cited papers performed parcel-area quartile ablation of GLCM features.

**Paper recommendation:** Remove or heavily qualify the novelty claim. Replace with: "To our knowledge, this is the first parcel-area quartile evaluation of GLCM texture features under Mandal-grouped spatial cross-validation in the Telangana smallholder context" — which is geographically scoped and defensible even without exhaustive literature search.
