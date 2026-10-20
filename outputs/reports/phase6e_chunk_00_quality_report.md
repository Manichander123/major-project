# Phase 6E — Chunk 0 Extraction Quality Report

**Execution Scope:** Chunk 0 (Farms 0 to 500 of 7,164 candidates)  
**Manifest:** `outputs/data/s2_full_expansion_manifest_7164.csv`  
**Output Artifact:** `outputs/data/s2_phase6e_chunk_00_0_500_raw_bands.csv`  
**Quality Verdict:** **`CHUNK_0_VERIFIED_PASS`**

---

## 1. Accounting and Extraction Yield

- **Total Chunk Candidates:** 500
- **Total Processed Records:** 500
- **Unique FarmIDs:** 500 (0 duplicates)
- **Manifest FarmID Alignment:** 100% agreement (500/500 matching `s2_full_expansion_manifest_7164.csv[0:500]`)
- **Crop Label Alignment:** 0 mismatches between manifest and extracted records
- **Successful Extractions:** **498 farms (99.6%)**
- **Failed Extractions:** **2 farms (0.4%)**

### Crop Breakdown by Status
| Crop Class | Candidate Farms | Successfully Extracted | Failed | Success Rate |
| :--- | :---: | :---: | :---: | :---: |
| **Cotton** | 261 | 259 | 2 | 99.23% |
| **Paddy** | 87 | 87 | 0 | 100.0% |
| **Maize** | 85 | 85 | 0 | 100.0% |
| **Bengal Gram** | 67 | 67 | 0 | 100.0% |
| **Total** | **500** | **498** | **2** | **99.60%** |

---

## 2. Failure Investigation and Scientific Audit

Both failed farms were analyzed in detail:
1. **FarmID `1326947` (Cotton):** SDate `2023-12-30`, HDate `2024-01-02` (recorded duration: 3 days). Sentinel-2 has a 5-day revisit cycle over Telangana. No satellite pass occurred over that tile in that 3-day window (`0 images found`).
2. **FarmID `1326994` (Cotton):** SDate `2023-12-27`, HDate `2024-01-03` (recorded duration: 7 days). Exactly 1 satellite pass occurred, which was 100% cloud-obscured (`unmasked_pixels = 0`).

**Audit Finding:** These failures are caused by physical archive absence and cloud cover on extreme short-duration survey entries. Both failed farms strictly preserve `NaN` across all spectral band columns (zero artificial zero filling, zero corruption).

---

## 3. Spectral Value and Numerical Integrity Checks

- **Infinite Values (`Inf`/`-Inf`):** 0 across all 40 mean/std band columns.
- **Artificial Zeros:** 0 across all band columns.
- **Red Band Range (`B4_full_mean`):** Min = `276.02`, Max = `1978.60`, Mean = `881.12` (realistic surface reflectance $\times 10^4$).
- **NIR Band Range (`B8_full_mean`):** Min = `1181.06`, Max = `4471.49`, Mean = `2471.56` (healthy vegetation reflectance response).
- **Daily Mosaicking:** Multi-scene date aggregation verified; no double-counting of overlapping MGRS swaths.
- **Cloud/Shadow Masking:** SCL-based masking successfully removed cloud and cloud-shadow pixels.

---

## 4. Checkpoint / Resume Verification

- **Resume Capability:** Tested and proven in production. When resumed with 497 cached successful farms, the extraction engine skipped all 497 already-completed farms in 0.1 seconds and evaluated only the pending records.
- **Atomic File Writing:** Implemented via `.tmp` file + atomic replace, guaranteeing zero file corruption during checkpoints.

---

## 5. Performance and Throughput

- **Effective Extraction Rate:** **6.5 farms per minute** (with 4 concurrent workers).
- **Chunk 0 Wall-Clock Runtime:** ~77 minutes.
- **Projected Full Campaign (Remaining 6,664 farms):**
  - ~1,025 minutes (**~17.1 hours**) at 6.5 farms/min with 4 workers.
  - Across 14 remaining chunks (13 chunks of 500 farms + 1 chunk of 164 farms).

---

## 6. Next Step Recommendation

**Chunk 0 validation has passed all quality gates with zero blocking defects.**  
Ready to proceed with processing the remaining 14 chunks (`chunk_idx 1` through `14`) using the same validated configuration.
