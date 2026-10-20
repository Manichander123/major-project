# Phase 6F Stage 3 — Final Cohort Harmonization & Gate B Report

## Executive Summary
Phase 6F Stage 3 has completed the deterministic harmonization of the full 7,888-farm survey population defined in `datasss/Train.csv`. By uniting the Phase 6D historical cohort (724 farms), Phase 6E campaign extractions (7,164 farms), and Phase 6F Stage 2 bounded recoveries (5 farms), the project establishes the final authoritative raw bands dataset (`s2_phase6f_harmonized_cohort.csv`) with zero data leakage, zero synthetic values, and 100% protected-file immutability.

---

## 1. Harmonized Dataset Specifications
- **Artifact Path:** `outputs/data/s2_phase6f_harmonized_cohort.csv`
- **SHA-256 Hash:** `caed402949eb577425b9d2e2cac0ee2c8462f635ca22b466a7675962ebb5a7af`
- **Dimensions:** 7,888 rows $\times$ 100 columns
- **Population Coverage:** 100.0% of `datasss/Train.csv` (7,888 / 7,888 unique FarmIDs)
- **Duplicate FarmIDs:** 0

### Schema Layout (100 Columns)
1. **Canonical Extraction Columns (1–97):**
   - `FarmID`
   - `status` (`success` / `failed`)
   - `error_message`
   - 88 Seasonal Window Spectral Columns: 4 windows (`early`, `mid`, `late`, `full`) $\times$ [2 count fields + 20 band statistics (`mean`, `std`)]
   - 6 Authoritative Metadata Columns: `Crop`, `District`, `Sub-District`, `Season`, `SDate`, `HDate`
2. **Standardized Audit & Provenance Columns (98–100):**
   - `source_phase`: Provenance tag (`phase6e`, `phase6d`, `phase6f_recovered`, `phase6e_cloud_failed`, `phase6d_cloud_failed`)
   - `extraction_status`: Standardized extraction flag (`success` / `failed`)
   - `cloud_obscured_failure`: Boolean flag (`True` for 45 cloud-failed farms, `False` for 7,843 successful farms)

---

## 2. Definitive Cohort Accounting

| Cohort Group | Source File | Records | Status | Description |
| :--- | :--- | :---: | :---: | :--- |
| **Phase 6E Successful** | `s2_phase6e_7164_raw_bands.csv` | 7,124 | `success` | Valid multi-temporal Sentinel-2 observations |
| **Phase 6D Successful** | `s2_phase6d_724_raw_bands.csv` | 714 | `success` | Benchmark historical cohort |
| **Phase 6F Stage 2 Recovered** | `s2_phase6f_recovered_farms.csv` | 5 | `success` | Concurrency-1 recovered timeouts & EE errors |
| **Eligible Modeling Cohort** | **Combined Successful** | **7,843** | **`success`** | **99.43% complete feature matrix for Stage 4** |
| **Phase 6E Persistent Cloud** | `s2_phase6e_7164_raw_bands.csv` | 35 | `failed` | Persistent cloud obscuration (100% NaN) |
| **Phase 6D Persistent Cloud** | `s2_phase6d_724_raw_bands.csv` | 10 | `failed` | Persistent cloud obscuration (100% NaN) |
| **Cloud-Obscured Failures** | **Combined Failed** | **45** | **`failed`** | **0.57% honest physical missingness (NaN)** |
| **Total Population** | **Train.csv Full Survey** | **7,888** | — | **100% accounted, 0 duplicates, 0 missing** |

---

## 3. Window-Level Completeness (Successful Cohort: N=7,843)

Full-season window (`full`) completeness is **100.0%** across all 7,843 successful farms. Phenological sub-windows reflect honest physical monsoon cloud dynamics in Telangana:

| Window | Definition | Valid Farms | Valid % | Unobserved (NaN) | Interpretation |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **`early`** | Initial vegetative phase | 7,302 | 93.10% | 541 | Monsoon cloud cover peak (July–August) |
| **`mid`** | Reproductive / flowering | 7,722 | 98.46% | 121 | Post-monsoon clearing |
| **`late`** | Maturation / pre-harvest | 7,651 | 97.55% | 192 | Dry season clear skies |
| **`full`** | Entire crop duration | **7,843** | **100.00%** | **0** | **Complete aggregate observation for every farm** |

---

## 4. Ground Truth & Metadata Concordance
- **Crop Label Matches:** 7,888 / 7,888 (100.0%, 0 mismatches)
- **District Matches:** 7,888 / 7,888 (100.0%, 0 mismatches)
- **Sub-District Matches:** 7,888 / 7,888 (100.0%, 0 mismatches)
- **Season Matches:** 7,888 / 7,888 (100.0%, 0 mismatches)
- **SDate / HDate Matches:** 7,888 / 7,888 (100.0% date concordance)

### Class Distribution Verification
| Crop | Harmonized Count | Train.csv Count | Difference |
| :--- | :---: | :---: | :---: |
| Paddy | 2,268 | 2,268 | 0 |
| Cotton | 1,692 | 1,692 | 0 |
| Maize | 1,384 | 1,384 | 0 |
| Bengal Gram | 938 | 938 | 0 |
| Groundnut | 768 | 768 | 0 |
| Chillies | 614 | 614 | 0 |
| Red Gram | 224 | 224 | 0 |
| **Total** | **7,888** | **7,888** | **0** |

---

## 5. Numerical Integrity & Safeguards
- **Infinities:** 0
- **Fabricated Zeros:** 0 (all unobserved spectral values preserved as strict NaNs)
- **Missingness in 45 Cloud Failures:** 100.0% NaN across all 80 spectral band statistics
- **Protected File Immutability:** All 24 baseline files verified with bit-for-bit SHA-256 identity before and after execution.

---

## 6. Gate B Verdict: PASS
The harmonized cohort dataset independently satisfies all criteria for Gate B Feature Readiness:
- [x] Complete population accounting (7,888 farms)
- [x] Zero duplicate FarmIDs
- [x] Zero label or metadata mismatches against `Train.csv`
- [x] Canonical 97-column extraction schema strictly preserved
- [x] Audit/provenance flags cleanly appended as separate columns
- [x] Honest missingness preserved without synthetic data or fabricated zeros
- [x] All 24 protected baseline files remain 100% unmodified
- [x] Primary modeling cohort (7,843 farms) clearly demarcated for Stage 4
