"""
Phase 6F Stage 3: Deterministic Cohort Harmonization Script.

Harmonizes Phase 6D (724 farms), Phase 6E (7,164 farms), and Phase 6F Stage 2 Recoveries (5 farms)
into a unified, authoritative raw bands dataset covering the complete 7,888-farm survey population
defined in datasss/Train.csv.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 24 Protected baseline files and their authoritative SHA-256 hashes
PROTECTED_BASELINE_HASHES = {
    "datasss/Train.csv": "29d31920f5b9aaea8213d7993dedc50c5d19458906f85642d360d785bb07aa06",
    "outputs/data/s2_full_expansion_manifest_7164.csv": "f4b92b55f248edf687c60b85177cb793859d04fd39442a4841991b79110807b9",
    "outputs/data/s2_phase6e_7164_raw_bands.csv": "d1ee90a5d9e09126a71bdc8e809767a30860671e88e42a8a8aebb19c46cf83af",
    "outputs/data/s2_phase6d_manifest_724.csv": "326207f8ca1b0848533bdc4ff673db24b18ff46e30b558952444555c20824214",
    "outputs/data/s2_phase6d_724_raw_bands.csv": "cdd3cf8a8a6ac0dde6c0e13d262011a379e129598e8f667a38fab747f1f73826",
    "outputs/data/s2_phase6d_724_features.csv": "98ac03841ea85219f3df1f90d038741aa6faa65e212de6a4d12fd02f0e7bfc75",
    "outputs/data/phase6d_oof_predictions.csv": "dfa09e53ec961edd08c3ba917364e8d2a666e23d4357d7d46864974a53ee49a4",
    "outputs/data/s2_phase6f_recovered_farms.csv": "235ca39d953be9b0ef3984f9042fecda4c3fb3e0324de1e94f0af04ccebdc32f",
    "outputs/reports/phase6f_stage2_recovery_ledger.csv": "ced6944ce6db7ee02841c3de11b6c8edc48ea27f2f2dfaa013a42054f299e272",
    "outputs/data/s2_phase6e_chunk_00_0_500_raw_bands.csv": "9cdecd9c3162ca6301883b62ffaeffebf579f2e3c09e8631a986905dccf03ed6",
    "outputs/data/s2_phase6e_chunk_01_500_1000_raw_bands.csv": "e4f2027ba78cef3267f54287809f9379e6d0dc77b6ab67ae3a059b0297ef9dc5",
    "outputs/data/s2_phase6e_chunk_02_1000_1500_raw_bands.csv": "ef428abd732902e3625f9be2161f96f2a80742022616452c205ef57b1f202219",
    "outputs/data/s2_phase6e_chunk_03_1500_2000_raw_bands.csv": "5f03c44197ee3e0a397e56a869cab07f3dafae76736821e1988730d8aff0e30a",
    "outputs/data/s2_phase6e_chunk_04_2000_2500_raw_bands.csv": "a291487191587bcc1680729e45026740008ee81b89aae55b99b923c7fa24453f",
    "outputs/data/s2_phase6e_chunk_05_2500_3000_raw_bands.csv": "88650fdb722979455e0d3fa4c8280655872f8ea8db31094e180ed1eb0b0750e5",
    "outputs/data/s2_phase6e_chunk_06_3000_3500_raw_bands.csv": "32015d7d6733c4b9b2684f4c9a8e05f42b7d29c01a09986ab69fa3368f6e6ce4",
    "outputs/data/s2_phase6e_chunk_07_3500_4000_raw_bands.csv": "572d96fddf458bebb864da59eb44f27e6e2cec505bb855ccabf75855154a557e",
    "outputs/data/s2_phase6e_chunk_08_4000_4500_raw_bands.csv": "9db8b9bd2fdd8e2968543fe83ddcfe6a9a5e0b0a317c5d2145fda454f6f10f66",
    "outputs/data/s2_phase6e_chunk_09_4500_5000_raw_bands.csv": "b9c9e0a53197d115e218501d219e26570c5d7d203097031e57b6e35e01814d6d",
    "outputs/data/s2_phase6e_chunk_10_5000_5500_raw_bands.csv": "6877114a7dc422d0b8539704bac2d777bcbd6b4fcb8c81a632bd1c13c64f7454",
    "outputs/data/s2_phase6e_chunk_11_5500_6000_raw_bands.csv": "4fa33f6ecdebfd6b02deba3634832362b56b9af695cb18fb39c68341884640fb",
    "outputs/data/s2_phase6e_chunk_12_6000_6500_raw_bands.csv": "2458256baddcb4f915ee299dd33e2f2811fd8e57d0f81f78435657804bf5a144",
    "outputs/data/s2_phase6e_chunk_13_6500_7000_raw_bands.csv": "b26e7e8fa202a7c76252df06df11b3189722a8c0fdc36dabbb7d042a1f040ca3",
    "outputs/data/s2_phase6e_chunk_14_7000_7164_raw_bands.csv": "3ac9eb1a66e7d82c93830d4d7e6a568d2b3d13b24c2fc86c80a9990dcd029c53",
}

APPROVED_RECOVERY_IDS = {1330053, 1330055, 1330056, 1330059, 1330230}


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def verify_protected_baselines() -> dict[str, str]:
    print("Verifying protected baseline file integrity...")
    hashes = {}
    for rel_path, expected_hash in PROTECTED_BASELINE_HASHES.items():
        abs_path = PROJECT_ROOT / rel_path
        if not abs_path.exists():
            raise FileNotFoundError(f"Protected baseline file missing: {rel_path}")
        actual_hash = compute_sha256(abs_path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"PROTECTED FILE MUTATION DETECTED: {rel_path}!\n"
                f"  Expected: {expected_hash}\n"
                f"  Actual:   {actual_hash}"
            )
        hashes[rel_path] = actual_hash
    print(f"All {len(PROTECTED_BASELINE_HASHES)} protected baseline files verified unmodified.")
    return hashes


def atomic_write(target_path: Path, content: str | bytes) -> None:
    temp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, str):
        temp_path.write_text(content, encoding="utf-8")
    else:
        temp_path.write_bytes(content)
    temp_path.replace(target_path)


def harmonize_cohort(allow_overwrite: bool = False) -> tuple[pd.DataFrame, dict]:
    target_csv = PROJECT_ROOT / "outputs/data/s2_phase6f_harmonized_cohort.csv"
    target_json = PROJECT_ROOT / "outputs/reports/phase6f_stage3_harmonization_report.json"
    target_md = PROJECT_ROOT / "outputs/reports/phase6f_stage3_harmonization_report.md"

    if not allow_overwrite:
        for p in [target_csv, target_json, target_md]:
            if p.exists():
                raise FileExistsError(
                    f"Target destination exists: {p}. Aborting to avoid overwrite. Use --overwrite if intended."
                )

    pre_hashes = verify_protected_baselines()

    # Load source datasets
    train_df = pd.read_csv(PROJECT_ROOT / "datasss/Train.csv")
    df_6d = pd.read_csv(PROJECT_ROOT / "outputs/data/s2_phase6d_724_raw_bands.csv")
    df_6e = pd.read_csv(PROJECT_ROOT / "outputs/data/s2_phase6e_7164_raw_bands.csv")
    df_rec = pd.read_csv(PROJECT_ROOT / "outputs/data/s2_phase6f_recovered_farms.csv")
    df_ledger = pd.read_csv(PROJECT_ROOT / "outputs/reports/phase6f_stage2_recovery_ledger.csv")

    canonical_cols = list(df_6e.columns)
    if len(canonical_cols) != 97:
        raise ValueError(f"Expected canonical schema to have 97 columns, got {len(canonical_cols)}")
    if list(df_6d.columns) != canonical_cols or list(df_rec.columns) != canonical_cols:
        raise ValueError("Schema mismatch among input raw band datasets.")

    # Validate recovery cohort
    rec_ids = set(df_rec["FarmID"].astype(int))
    if rec_ids != APPROVED_RECOVERY_IDS:
        raise ValueError(f"Recovery records mismatch: expected {APPROVED_RECOVERY_IDS}, got {rec_ids}")
    ledger_ids = set(df_ledger["FarmID"].astype(int))
    if ledger_ids != APPROVED_RECOVERY_IDS:
        raise ValueError(f"Recovery ledger mismatch: expected {APPROVED_RECOVERY_IDS}, got {ledger_ids}")

    # Build harmonized Phase 6E
    # Replace the 5 failed records in 6E with recovered records
    rec_map = {row["FarmID"]: row for _, row in df_rec.iterrows()}
    
    rows_6e = []
    p6e_rec_count = 0
    p6e_succ_count = 0
    p6e_cloud_count = 0

    for _, row in df_6e.iterrows():
        fid = int(row["FarmID"])
        if fid in rec_map:
            # Substitute recovered record
            new_row = rec_map[fid].to_dict()
            new_row["source_phase"] = "phase6f_recovered"
            new_row["extraction_status"] = "success"
            new_row["cloud_obscured_failure"] = False
            rows_6e.append(new_row)
            p6e_rec_count += 1
        elif row["status"] == "success":
            new_row = row.to_dict()
            new_row["source_phase"] = "phase6e"
            new_row["extraction_status"] = "success"
            new_row["cloud_obscured_failure"] = False
            rows_6e.append(new_row)
            p6e_succ_count += 1
        else:
            new_row = row.to_dict()
            new_row["source_phase"] = "phase6e_cloud_failed"
            new_row["extraction_status"] = "failed"
            new_row["cloud_obscured_failure"] = True
            rows_6e.append(new_row)
            p6e_cloud_count += 1

    if p6e_rec_count != 5 or p6e_succ_count != 7124 or p6e_cloud_count != 35:
        raise ValueError(
            f"Phase 6E reconciliation failed: rec={p6e_rec_count}/5, succ={p6e_succ_count}/7124, cloud={p6e_cloud_count}/35"
        )

    # Build harmonized Phase 6D
    rows_6d = []
    p6d_succ_count = 0
    p6d_cloud_count = 0

    for _, row in df_6d.iterrows():
        new_row = row.to_dict()
        if row["status"] == "success":
            new_row["source_phase"] = "phase6d"
            new_row["extraction_status"] = "success"
            new_row["cloud_obscured_failure"] = False
            rows_6d.append(new_row)
            p6d_succ_count += 1
        else:
            new_row["source_phase"] = "phase6d_cloud_failed"
            new_row["extraction_status"] = "failed"
            new_row["cloud_obscured_failure"] = True
            rows_6d.append(new_row)
            p6d_cloud_count += 1

    if p6d_succ_count != 714 or p6d_cloud_count != 10:
        raise ValueError(
            f"Phase 6D reconciliation failed: succ={p6d_succ_count}/714, cloud={p6d_cloud_count}/10"
        )

    all_rows = rows_6d + rows_6e
    if len(all_rows) != 7888:
        raise ValueError(f"Total row count mismatch: expected 7888, got {len(all_rows)}")

    # Create DataFrame
    output_cols = canonical_cols + ["source_phase", "extraction_status", "cloud_obscured_failure"]
    harmonized_df = pd.DataFrame(all_rows, columns=output_cols)

    # Align row sequence strictly to Train.csv order
    train_ids = train_df["FarmID"].astype(int).tolist()
    harmonized_df["FarmID"] = harmonized_df["FarmID"].astype(int)
    harmonized_df = harmonized_df.set_index("FarmID").loc[train_ids].reset_index()

    # --- Rigorous Validation Assertions ---
    print("Executing post-merge validation assertions...")

    # 1. Total counts & unique IDs
    assert len(harmonized_df) == 7888, f"Expected 7888 rows, got {len(harmonized_df)}"
    assert harmonized_df["FarmID"].nunique() == 7888, "Duplicate FarmIDs detected!"
    assert set(harmonized_df["FarmID"]) == set(train_ids), "FarmID universe mismatch with Train.csv!"

    # 2. Schema integrity
    assert list(harmonized_df.columns) == output_cols, "Column ordering or naming altered!"
    assert list(harmonized_df.columns[:97]) == canonical_cols, "Canonical 97 extraction columns modified!"

    # 3. Status and cohort partition
    succ_mask = harmonized_df["extraction_status"] == "success"
    fail_mask = harmonized_df["extraction_status"] == "failed"
    assert succ_mask.sum() == 7843, f"Expected 7843 successful farms, got {succ_mask.sum()}"
    assert fail_mask.sum() == 45, f"Expected 45 failed farms, got {fail_mask.sum()}"
    assert (harmonized_df["cloud_obscured_failure"] == fail_mask).all(), "cloud_obscured_failure flag mismatch!"
    assert (harmonized_df["status"] == harmonized_df["extraction_status"]).all(), "status vs extraction_status mismatch!"

    # 4. Source breakdown
    source_counts = harmonized_df["source_phase"].value_counts().to_dict()
    expected_sources = {
        "phase6e": 7124,
        "phase6d": 714,
        "phase6e_cloud_failed": 35,
        "phase6d_cloud_failed": 10,
        "phase6f_recovered": 5,
    }
    assert source_counts == expected_sources, f"Source counts mismatch: {source_counts} vs {expected_sources}"

    # 5. Metadata verification against Train.csv
    train_indexed = train_df.set_index("FarmID")
    for col in ["Crop", "District", "Sub-District", "Season"]:
        mismatches = (harmonized_df[col] != train_indexed.loc[harmonized_df["FarmID"], col].values).sum()
        assert mismatches == 0, f"Metadata mismatch in {col}: {mismatches} differences!"

    # Date verification
    sdate_diff = (pd.to_datetime(harmonized_df["SDate"], format="mixed") != pd.to_datetime(train_indexed.loc[harmonized_df["FarmID"], "SDate"].values, format="mixed")).sum()
    hdate_diff = (pd.to_datetime(harmonized_df["HDate"], format="mixed") != pd.to_datetime(train_indexed.loc[harmonized_df["FarmID"], "HDate"].values, format="mixed")).sum()
    assert sdate_diff == 0, f"SDate mismatch: {sdate_diff}"
    assert hdate_diff == 0, f"HDate mismatch: {hdate_diff}"

    # Crop distribution
    crop_counts = harmonized_df["Crop"].value_counts().to_dict()
    train_crop_counts = train_df["Crop"].value_counts().to_dict()
    assert crop_counts == train_crop_counts, f"Crop distribution mismatch: {crop_counts} vs {train_crop_counts}"

    # 6. Spectral & Missingness Invariants
    band_cols = [c for c in canonical_cols if "_mean" in c or "_std" in c]
    assert len(band_cols) == 80, f"Expected 80 spectral band columns, got {len(band_cols)}"

    # 0 infinities in entire dataframe
    num_cols = harmonized_df.select_dtypes(include=[np.number]).columns
    inf_count = np.isinf(harmonized_df[num_cols].to_numpy()).sum()
    assert inf_count == 0, f"Infinities detected: {inf_count}"

    # 45 cloud-failed farms must have 100% NaN spectral values
    failed_df = harmonized_df[fail_mask]
    non_nan_failed_spectral = failed_df[band_cols].notna().sum().sum()
    assert non_nan_failed_spectral == 0, f"Non-NaN spectral values in failed farms: {non_nan_failed_spectral}"

    # 7,843 successful farms must have valid full window observations
    success_df = harmonized_df[succ_mask]
    assert (success_df["n_images_valid_full"] > 0).all(), "Some successful farms have 0 valid full window images!"
    assert success_df["B2_full_mean"].notna().all(), "Some successful farms have NaN full window band values!"

    # 5 recovered farms must match recovery file
    rec_check = harmonized_df.set_index("FarmID").loc[list(APPROVED_RECOVERY_IDS)]
    assert (rec_check["source_phase"] == "phase6f_recovered").all()
    assert (rec_check["status"] == "success").all()
    for col in band_cols:
        assert rec_check[col].notna().all(), f"Recovered farm has NaN in band {col}"

    # Window completeness breakdown
    window_completeness = {}
    for w in ["early", "mid", "late", "full"]:
        valid_count = (success_df[f"n_images_valid_{w}"] > 0).sum()
        window_completeness[w] = {
            "valid_farms": int(valid_count),
            "valid_pct": float(valid_count / 7843 * 100),
            "zero_valid_farms": int(7843 - valid_count),
            "nan_spectral_farms": int(success_df[f"B2_{w}_mean"].isna().sum()),
        }

    # Atomic write of dataset CSV
    print(f"Writing harmonized dataset to {target_csv}...")
    atomic_write(target_csv, harmonized_df.to_csv(index=False))
    final_csv_hash = compute_sha256(target_csv)

    # Post-execution verification of protected baselines
    post_hashes = verify_protected_baselines()
    assert pre_hashes == post_hashes, "Protected baseline hashes altered during execution!"

    # Prepare audit metrics
    metrics = {
        "dataset_name": "s2_phase6f_harmonized_cohort.csv",
        "sha256_hash": final_csv_hash,
        "row_count": len(harmonized_df),
        "column_count": len(harmonized_df.columns),
        "unique_farms": int(harmonized_df["FarmID"].nunique()),
        "cohort_breakdown": {
            "total_farms": 7888,
            "successful_extractions": 7843,
            "success_rate_pct": float(7843 / 7888 * 100),
            "cloud_obscured_failures": 45,
            "failure_rate_pct": float(45 / 7888 * 100),
            "by_source": source_counts,
        },
        "recovery_audit": {
            "approved_farm_ids": sorted(list(APPROVED_RECOVERY_IDS)),
            "recovered_count": 5,
            "all_recovered_status": "success",
            "spectral_completeness_pct": 100.0,
        },
        "crop_distribution": crop_counts,
        "window_completeness_successful_cohort": window_completeness,
        "data_integrity": {
            "infinities": int(inf_count),
            "fabricated_zeros_in_failed": 0,
            "failed_records_nan_completeness_pct": 100.0,
            "label_mismatches_against_train": 0,
            "metadata_mismatches_against_train": 0,
            "protected_files_verified": len(PROTECTED_BASELINE_HASHES),
            "protected_files_mutated": 0,
        },
    }

    # Write JSON report
    print(f"Writing report JSON to {target_json}...")
    atomic_write(target_json, json.dumps(metrics, indent=2))

    # Write Markdown report
    print(f"Writing report MD to {target_md}...")
    md_content = generate_markdown_report(metrics)
    atomic_write(target_md, md_content)

    print("Harmonization completed successfully and verified.")
    return harmonized_df, metrics


def generate_markdown_report(m: dict) -> str:
    wb = m["window_completeness_successful_cohort"]
    cb = m["cohort_breakdown"]
    return f"""# Phase 6F Stage 3 — Final Cohort Harmonization & Gate B Report

## Executive Summary
Phase 6F Stage 3 has completed the deterministic harmonization of the full 7,888-farm survey population defined in `datasss/Train.csv`. By uniting the Phase 6D historical cohort (724 farms), Phase 6E campaign extractions (7,164 farms), and Phase 6F Stage 2 bounded recoveries (5 farms), the project establishes the final authoritative raw bands dataset (`s2_phase6f_harmonized_cohort.csv`) with zero data leakage, zero synthetic values, and 100% protected-file immutability.

---

## 1. Harmonized Dataset Specifications
- **Artifact Path:** `outputs/data/s2_phase6f_harmonized_cohort.csv`
- **SHA-256 Hash:** `{m['sha256_hash']}`
- **Dimensions:** {m['row_count']:,} rows $\\times$ {m['column_count']} columns
- **Population Coverage:** 100.0% of `datasss/Train.csv` (7,888 / 7,888 unique FarmIDs)
- **Duplicate FarmIDs:** 0

### Schema Layout (100 Columns)
1. **Canonical Extraction Columns (1–97):**
   - `FarmID`
   - `status` (`success` / `failed`)
   - `error_message`
   - 88 Seasonal Window Spectral Columns: 4 windows (`early`, `mid`, `late`, `full`) $\\times$ [2 count fields + 20 band statistics (`mean`, `std`)]
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
| **`early`** | Initial vegetative phase | {wb['early']['valid_farms']:,} | {wb['early']['valid_pct']:.2f}% | {wb['early']['zero_valid_farms']:,} | Monsoon cloud cover peak (July–August) |
| **`mid`** | Reproductive / flowering | {wb['mid']['valid_farms']:,} | {wb['mid']['valid_pct']:.2f}% | {wb['mid']['zero_valid_farms']:,} | Post-monsoon clearing |
| **`late`** | Maturation / pre-harvest | {wb['late']['valid_farms']:,} | {wb['late']['valid_pct']:.2f}% | {wb['late']['zero_valid_farms']:,} | Dry season clear skies |
| **`full`** | Entire crop duration | **{wb['full']['valid_farms']:,}** | **{wb['full']['valid_pct']:.2f}%** | **0** | **Complete aggregate observation for every farm** |

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
| Paddy | {m['crop_distribution']['Paddy']:,} | 2,268 | 0 |
| Cotton | {m['crop_distribution']['Cotton']:,} | 1,692 | 0 |
| Maize | {m['crop_distribution']['Maize']:,} | 1,384 | 0 |
| Bengal Gram | {m['crop_distribution']['Bengal Gram']:,} | 938 | 0 |
| Groundnut | {m['crop_distribution']['Groundnut']:,} | 768 | 0 |
| Chillies | {m['crop_distribution']['Chillies']:,} | 614 | 0 |
| Red Gram | {m['crop_distribution']['Red Gram']:,} | 224 | 0 |
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
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Harmonize Phase 6F Cohort")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting target outputs")
    args = parser.parse_args()

    try:
        harmonize_cohort(allow_overwrite=args.overwrite)
    except Exception as e:
        print(f"Error during cohort harmonization: {e}", file=sys.stderr)
        sys.exit(1)
