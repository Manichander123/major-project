# Feature Schema Audit

**Task D & E:** GLCM texture features (160 vs 140) and 34-feature baseline schema  
**Primary source:** `outputs/reports/phase6h/phase6h_feature_schema_manifest.json`  
**Secondary source:** `outputs/reports/phase6h/phase6h_stage3_benchmark_report.json`  
**Audit date:** 2026-10-23 (read-only)

---

## 1. Confirmed Group Counts

Source: `phase6h_feature_schema_manifest.json` key `group_counts`:

| Group | Feature count |
|---|---|
| S (spectral, full season) | 34 |
| T (temporal windows) | 106 |
| P (GLCM texture) | 20 |
| S+T (primary model) | **140** |
| S+P | 54 |
| T+P | 126 |
| S+T+P (secondary model) | **160** |

All counts are verified: 34 + 106 = 140; 140 + 20 = 160. No discrepancy.

---

## 2. Task E — The 34-Feature S (Spectral) Baseline Schema

Source: `phase6h_feature_schema_manifest.json` key `groups.S` and `phase6h_stage3_benchmark_report.json` key `experiments.S.features`.

The documentation claim "10 bands + 14 indices + 10 aggregations" is **incorrect**. The actual composition is:

### 2.1 Band features (20 features = 10 bands × 2 aggregations)

Each band produces one mean and one std over the **full season composite** (Jun–Dec).

| Band | Sentinel-2 name | Central wavelength | Features |
|---|---|---|---|
| B2 | Blue | 490 nm | `B2_full_mean`, `B2_full_std` |
| B3 | Green | 560 nm | `B3_full_mean`, `B3_full_std` |
| B4 | Red | 665 nm | `B4_full_mean`, `B4_full_std` |
| B5 | Red Edge 1 | 705 nm | `B5_full_mean`, `B5_full_std` |
| B6 | Red Edge 2 | 740 nm | `B6_full_mean`, `B6_full_std` |
| B7 | Red Edge 3 | 783 nm | `B7_full_mean`, `B7_full_std` |
| B8 | NIR | 842 nm | `B8_full_mean`, `B8_full_std` |
| B8A | Narrow NIR | 865 nm | `B8A_full_mean`, `B8A_full_std` |
| B11 | SWIR 1 | 1610 nm | `B11_full_mean`, `B11_full_std` |
| B12 | SWIR 2 | 2190 nm | `B12_full_mean`, `B12_full_std` |

**10 bands × 2 aggregations (mean + std) = 20 features**

### 2.2 Spectral index features (14 features — one scalar per farm from full composite)

| Index | Feature name | Formula / bands |
|---|---|---|
| NDVI | `ndvi_full` | (B8−B4)/(B8+B4) |
| EVI | `evi_full` | 2.5×(B8−B4)/(B8+6B4−7.5B2+1) |
| GNDVI | `gndvi_full` | (B8−B3)/(B8+B3) |
| SAVI | `savi_full` | (B8−B4)/(B8+B4+0.5)×1.5 |
| MSAVI | `msavi_full` | (2B8+1−√((2B8+1)²−8(B8−B4)))/2 |
| NDMI | `ndmi_full` | (B8−B11)/(B8+B11) |
| NDWI | `ndwi_full` | (B3−B8)/(B3+B8) |
| NDRe1 | `ndre1_full` | (B8−B5)/(B8+B5) |
| NDRe2 | `ndre2_full` | (B8−B6)/(B8+B6) |
| PSRI | `psri_full` | (B4−B2)/B6 |
| BSI | `bsi_full` | ((B11+B4)−(B8+B2))/((B11+B4)+(B8+B2)) |
| MSI | `msi_full` | B11/B8 |
| CIre | `cire_full` | (B7/B5)−1 |
| CIg | `cig_full` | (B8/B3)−1 |

**14 spectral index scalars.**

### 2.3 Correct composition

- **20 band statistics** (10 bands × mean + std)  
- **14 spectral index scalars**  
- **Total = 34** ✓

The documentation description "10 bands + 14 indices + 10 aggregations" conflates the 10 bands and the 2 per-band aggregations (20 total) into a misleading framing. There is no separate "10 aggregation" group — the aggregations are _within_ the band features.

**NDBI / −NDMI equivalence check:** The implemented NDMI formula is `(B8−B11)/(B8+B11)`. NDBI would be `(B11−B8)/(B11+B8) = −NDMI`. The manifest uses `ndmi_full` only; NDBI is not present. These two are algebraically identical with opposite sign, so no redundancy exists (only one is included).

---

## 3. Task D — The 20 GLCM Texture Features (Group P)

Source: `phase6h_feature_schema_manifest.json` key `groups.P`.

The documentation claim "5 GLCM metrics × 2 kernels = 10 combinations" is **wrong**. The actual structure is:

**5 GLCM metrics × 4 temporal windows = 20 features**

There is only ONE kernel/window size documented (3×3 = 900 m² footprint on 10 m Sentinel-2 B8). The temporal dimension (early, mid, late, full) provides the factor of 4.

### Complete GLCM feature table

| Feature name | GLCM metric | Source band | Kernel/window | Temporal window | Aggregation | Source |
|---|---|---|---|---|---|---|
| `glcm_contrast_early` | Contrast | B8 (NIR, 842 nm) | 3×3 pixels (30 m × 30 m) | Early (Jun–Jul) | Mean over parcel pixels | `phase6h_feature_schema_manifest.json` |
| `glcm_contrast_mid` | Contrast | B8 | 3×3 | Mid (Aug–Sep) | Mean | ibid |
| `glcm_contrast_late` | Contrast | B8 | 3×3 | Late (Oct–Nov) | Mean | ibid |
| `glcm_contrast_full` | Contrast | B8 | 3×3 | Full (Jun–Dec) | Mean | ibid |
| `glcm_dissimilarity_early` | Dissimilarity | B8 | 3×3 | Early | Mean | ibid |
| `glcm_dissimilarity_mid` | Dissimilarity | B8 | 3×3 | Mid | Mean | ibid |
| `glcm_dissimilarity_late` | Dissimilarity | B8 | 3×3 | Late | Mean | ibid |
| `glcm_dissimilarity_full` | Dissimilarity | B8 | 3×3 | Full | Mean | ibid |
| `glcm_homogeneity_early` | Homogeneity | B8 | 3×3 | Early | Mean | ibid |
| `glcm_homogeneity_mid` | Homogeneity | B8 | 3×3 | Mid | Mean | ibid |
| `glcm_homogeneity_late` | Homogeneity | B8 | 3×3 | Late | Mean | ibid |
| `glcm_homogeneity_full` | Homogeneity | B8 | 3×3 | Full | Mean | ibid |
| `glcm_energy_early` | Energy (ASM) | B8 | 3×3 | Early | Mean | ibid |
| `glcm_energy_mid` | Energy (ASM) | B8 | 3×3 | Mid | Mean | ibid |
| `glcm_energy_late` | Energy (ASM) | B8 | 3×3 | Late | Mean | ibid |
| `glcm_energy_full` | Energy (ASM) | B8 | 3×3 | Full | Mean | ibid |
| `glcm_correlation_early` | Correlation | B8 | 3×3 | Early | Mean | ibid |
| `glcm_correlation_mid` | Correlation | B8 | 3×3 | Mid | Mean | ibid |
| `glcm_correlation_late` | Correlation | B8 | 3×3 | Late | Mean | ibid |
| `glcm_correlation_full` | Correlation | B8 | 3×3 | Full | Mean | ibid |

**5 metrics × 4 temporal windows = 20 features. Single source band (B8). Single kernel (3×3).**

The documentation's claim of "5 metrics × 2 kernels" must be corrected. There is no evidence of a second kernel size in any manifest or extraction report.

---

## 4. T Group — 106 Temporal Features

Composition: 10 bands × 2 stats × 3 temporal windows (early, mid, late) = 60 band-temporal features, plus 14 indices × 3 windows = 42 index-temporal features, plus 4 dynamic phenological deltas = 106 total.

The 4 deltas: `ndvi_growth_delta`, `peak_to_early_ratio`, `ndvi_senescence_delta`, `moisture_loss_delta`.

---

## 5. Corrections Required

| Claim | Incorrect wording | Correct wording |
|---|---|---|
| S baseline composition | "10 bands + 14 indices + 10 aggregations" | "10 bands (20 features: mean+std each) + 14 spectral indices = 34 features" |
| GLCM dimensionality | "5 metrics × 2 kernels = 10" | "5 metrics × 4 temporal windows = 20 features; single kernel (3×3)" |
