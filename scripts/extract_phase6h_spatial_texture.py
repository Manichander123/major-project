"""
scripts/extract_phase6h_spatial_texture.py
==========================================
Phase 6H Stage 2: Spatial Texture Extraction Campaign.

Extracts 20 GLCM features (5 Haralick metrics x 4 phenological windows) from
Sentinel-2 B8 (NIR 10m) using server-side GEE glcmTexture().

Execution modes:
  --pilot   Run pilot on 20 representative farms (small + median parcels across
            early/mid/late/full windows) and validate before full extraction.
  (default) Run chunked extraction for the full 7,843 eligible farm cohort.

Safety:
  - Verifies all protected baseline SHA-256 hashes before and after execution.
  - Atomic writes (.tmp -> .csv) with checkpoint resume.
  - Cloud-masked/unavailable values preserved as NaN (never zero-filled).
  - Outputs only to outputs/data/phase6h/, outputs/reports/phase6h/.
  - Never overwrites Phase 6F/6G artifacts.
"""

from __future__ import annotations

import argparse
from concurrent.futures import as_completed, ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from shapely import wkt

# UTF-8 stdout for Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.satellite.s2_extractor import (
    DEFAULT_CLOUD_COVER_THRESHOLD,
    EarthEngineManager,
    S2_SR_COLLECTION,
)
from src.satellite.s2_spatial_texture import (
    GLCM_FEATURE_NAMES,
    extract_spatial_texture_for_farm,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PHASE6H_DATA_DIR = PROJECT_ROOT / "outputs" / "data" / "phase6h"
PHASE6H_REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports" / "phase6h"
PHASE6H_DATA_DIR.mkdir(parents=True, exist_ok=True)
PHASE6H_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Protected baselines — verified before AND after extraction.
PROTECTED_BASELINE_HASHES = {
    "datasss/Train.csv": "29d31920f5b9aaea8213d7993dedc50c5d19458906f85642d360d785bb07aa06",
    "outputs/data/s2_full_expansion_manifest_7164.csv": "f4b92b55f248edf687c60b85177cb793859d04fd39442a4841991b79110807b9",
    "outputs/data/s2_phase6e_7164_raw_bands.csv": "d1ee90a5d9e09126a71bdc8e809767a30860671e88e42a8a8aebb19c46cf83af",
    "outputs/data/phase6f_cv_folds.csv": "sha256_deferred",  # populated at runtime
    "outputs/data/s2_phase6f_harmonized_cohort.csv": "caed402949eb577425b9d2e2cac0ee2c8462f635ca22b466a7675962ebb5a7af",
    "outputs/data/s2_phase6f_features.csv": "sha256_deferred",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def verify_protected_baselines(label: str = "pre") -> Dict[str, str]:
    """Verify protected baselines, returning actual hashes. Raises on mismatch for fixed hashes."""
    logger.info("Verifying protected baselines (%s-execution)...", label)
    actual_hashes = {}
    for rel_path, expected_hash in PROTECTED_BASELINE_HASHES.items():
        fp = PROJECT_ROOT / rel_path
        if not fp.exists():
            raise FileNotFoundError(f"Missing protected baseline: {rel_path}")
        actual = sha256_file(fp)
        actual_hashes[rel_path] = actual
        if expected_hash != "sha256_deferred" and actual != expected_hash:
            raise ValueError(f"CORRUPTED BASELINE ({label}): {rel_path}\n  expected: {expected_hash}\n  actual:   {actual}")
    logger.info("All protected baselines verified intact (%s-execution).", label)
    return actual_hashes


def atomic_save_dataframe(df: pd.DataFrame, target_path: Path) -> None:
    """Atomic write: tmp -> target."""
    tmp_path = target_path.with_suffix(target_path.suffix + f".tmp.{os.getpid()}")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(tmp_path, index=False)
    try:
        os.replace(tmp_path, target_path)
    except OSError:
        time.sleep(0.1)
        os.replace(tmp_path, target_path)


def _extract_single_texture_with_retry(
    row_dict: Dict[str, Any],
    cloud_cover_threshold: float,
    max_retries: int = 3,
    initial_delay: float = 2.0,
) -> Dict[str, Any]:
    """Extract GLCM texture for a single farm with exponential backoff on transient errors."""
    farm_id = int(row_dict["FarmID"])
    sdate = str(row_dict["SDate"])[:10]
    hdate = str(row_dict["HDate"])[:10]
    geom = wkt.loads(str(row_dict["geometry"]))

    last_exc: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            res = extract_spatial_texture_for_farm(
                farm_id=farm_id,
                geometry=geom,
                sdate_str=sdate,
                hdate_str=hdate,
                collection_name=S2_SR_COLLECTION,
                cloud_cover_threshold=cloud_cover_threshold,
            )
            # Attach metadata for traceability
            res["Crop"] = row_dict.get("Crop", "")
            res["District"] = row_dict.get("District", "")
            res["Sub-District"] = row_dict.get("Sub-District", "")
            res["Season"] = row_dict.get("Season", "")
            res["SDate"] = sdate
            res["HDate"] = hdate
            return res
        except Exception as exc:
            last_exc = exc
            err_str = str(exc)
            if attempt < max_retries and any(
                t in err_str.lower()
                for t in ["429", "503", "quota", "rate limit", "timeout", "unavailable"]
            ):
                backoff = initial_delay * (2 ** (attempt - 1)) + random.uniform(0.1, 1.0)
                logger.warning(
                    "FarmID %d attempt %d/%d transient error: %s. Backing off %.2fs...",
                    farm_id, attempt, max_retries, err_str, backoff,
                )
                time.sleep(backoff)
            else:
                break

    err_msg = str(last_exc) if last_exc else "Unknown error"
    logger.error("Extraction failed for FarmID %d after %d attempts: %s", farm_id, max_retries, err_msg)
    res = {
        "FarmID": farm_id,
        "Crop": row_dict.get("Crop", ""),
        "District": row_dict.get("District", ""),
        "Sub-District": row_dict.get("Sub-District", ""),
        "Season": row_dict.get("Season", ""),
        "SDate": sdate,
        "HDate": hdate,
        "status": "failed",
        "error_message": err_msg,
    }
    for feat_name in GLCM_FEATURE_NAMES:
        res[feat_name] = np.nan
    return res


def run_texture_extraction_chunk(
    manifest_chunk: pd.DataFrame,
    checkpoint_file: Path,
    cloud_cover_threshold: float = DEFAULT_CLOUD_COVER_THRESHOLD,
    max_workers: int = 4,
    checkpoint_interval: int = 10,
) -> pd.DataFrame:
    """Run GLCM texture extraction for a chunk with deduplicated resume and atomic checkpoints."""
    cached_records: Dict[int, Dict[str, Any]] = {}
    successful_ids = set()

    if checkpoint_file.exists():
        try:
            cached_df = pd.read_csv(checkpoint_file)
            for rec in cached_df.to_dict("records"):
                fid = int(rec["FarmID"])
                if rec.get("status") == "success":
                    successful_ids.add(fid)
                    cached_records[fid] = rec
            logger.info("Loaded %d previously successful farms from %s", len(successful_ids), checkpoint_file.name)
        except Exception as e:
            logger.warning("Could not read checkpoint %s: %s", checkpoint_file, e)

    pending_df = manifest_chunk[~manifest_chunk["FarmID"].isin(successful_ids)]
    logger.info(
        "Chunk total: %d | Already successful: %d | Pending: %d",
        len(manifest_chunk), len(successful_ids), len(pending_df),
    )

    all_records: Dict[int, Dict[str, Any]] = dict(cached_records)
    if pending_df.empty:
        logger.info("All chunk farms already successfully extracted.")
        return pd.DataFrame(list(all_records.values()))

    row_dicts = pending_df.to_dict("records")
    completed = 0
    t0 = time.time()

    logger.info("Launching GLCM texture extraction with %d workers...", max_workers)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(_extract_single_texture_with_retry, r, cloud_cover_threshold): r["FarmID"]
            for r in row_dicts
        }

        for future in as_completed(future_map):
            completed += 1
            fid = future_map[future]
            try:
                res = future.result()
            except Exception as e:
                res = {"FarmID": fid, "status": "failed", "error_message": str(e)}

            all_records[fid] = res
            status = res.get("status", "unknown")
            elapsed = time.time() - t0
            rate = completed / max(elapsed, 1.0) * 60.0

            logger.info(
                "Progress [%d/%d] (%.1f farms/min) FarmID %d | Status: %s",
                completed, len(row_dicts), rate, fid, status,
            )

            if completed % checkpoint_interval == 0 or completed == len(row_dicts):
                chk_df = pd.DataFrame(list(all_records.values()))
                atomic_save_dataframe(chk_df, checkpoint_file)
                logger.info("Checkpoint saved: %d records -> %s", len(chk_df), checkpoint_file.name)

    final_df = pd.DataFrame(list(all_records.values()))
    atomic_save_dataframe(final_df, checkpoint_file)
    logger.info("Chunk complete: %d records -> %s", len(final_df), checkpoint_file)
    return final_df


def select_pilot_farms(cohort: pd.DataFrame, n_pilot: int = 20) -> pd.DataFrame:
    """Select representative pilot farms: mix of small (bottom 10%), median, and large parcels."""
    if "area_m2" in cohort.columns:
        merged = cohort.copy()
    else:
        geom_path = PROJECT_ROOT / "outputs" / "data" / "train_enriched_geometry.csv"
        df_geom = pd.read_csv(geom_path)
        merged = cohort.merge(df_geom[["FarmID", "area_m2"]], on="FarmID", how="left")

    merged = merged.dropna(subset=["area_m2"])

    # Stratify: 7 small (<p10), 6 median (p40-p60), 7 large (>p90)
    p10 = merged["area_m2"].quantile(0.10)
    p40 = merged["area_m2"].quantile(0.40)
    p60 = merged["area_m2"].quantile(0.60)
    p90 = merged["area_m2"].quantile(0.90)

    small = merged[merged["area_m2"] <= p10]
    median_grp = merged[(merged["area_m2"] > p40) & (merged["area_m2"] <= p60)]
    large = merged[merged["area_m2"] >= p90]

    rng = np.random.RandomState(42)
    picks = []
    for group, n in [(small, 7), (median_grp, 6), (large, 7)]:
        if len(group) >= n:
            picks.append(group.sample(n=n, random_state=rng))
        else:
            picks.append(group)

    pilot = pd.concat(picks, ignore_index=True).drop_duplicates(subset="FarmID")
    return pilot.head(n_pilot).drop(columns=["area_m2"], errors="ignore")


def validate_pilot_output(pilot_df: pd.DataFrame) -> Dict[str, Any]:
    """Validate pilot extraction output. Returns validation dict; raises on stop-conditions."""
    n_total = len(pilot_df)
    n_success = int((pilot_df["status"] == "success").sum())
    n_failed = n_total - n_success

    report = {
        "n_pilot_farms": n_total,
        "n_success": n_success,
        "n_failed": n_failed,
        "success_rate": round(n_success / max(n_total, 1) * 100, 2),
    }

    # Check GLCM feature columns exist
    found_features = [c for c in GLCM_FEATURE_NAMES if c in pilot_df.columns]
    missing_features = [c for c in GLCM_FEATURE_NAMES if c not in pilot_df.columns]
    report["found_features"] = len(found_features)
    report["missing_features"] = missing_features

    if missing_features:
        raise ValueError(f"STOP: Missing GLCM feature columns in pilot output: {missing_features}")

    # Verify no zeros replacing NaN (missingness preservation)
    success_rows = pilot_df[pilot_df["status"] == "success"]
    if len(success_rows) > 0:
        texture_values = success_rows[found_features]

        # Check finite-value rates
        finite_rates = {}
        for feat in found_features:
            finite_count = texture_values[feat].notna().sum()
            finite_rates[feat] = round(finite_count / len(success_rows) * 100, 2)
        report["finite_value_rates_pct"] = finite_rates

        # Verify no artificially zero-filled NaNs:
        # If a farm has ALL 20 features as exactly 0.0, that's suspicious
        all_zero_rows = (texture_values.fillna(-999) == 0.0).all(axis=1).sum()
        report["suspicious_all_zero_rows"] = int(all_zero_rows)
        if all_zero_rows > 0:
            logger.warning("STOP CONDITION CHECK: %d rows have ALL 20 texture features = 0.0", all_zero_rows)

        # Mean missingness across features
        mean_miss = float(texture_values.isna().mean().mean() * 100)
        report["mean_missingness_pct"] = round(mean_miss, 2)

        # Check for Inf values
        inf_count = int(np.isinf(texture_values.select_dtypes(include=[np.number]).values).sum())
        report["infinite_values_count"] = inf_count
        if inf_count > 0:
            raise ValueError(f"STOP: {inf_count} infinite values found in pilot texture output")

    # Unique farm IDs
    unique_ids = pilot_df["FarmID"].nunique()
    report["unique_farm_ids"] = unique_ids
    if unique_ids != n_total:
        raise ValueError(f"STOP: Duplicate FarmIDs in pilot output: {n_total} rows but {unique_ids} unique IDs")

    report["validation_passed"] = True
    return report


def main():
    parser = argparse.ArgumentParser(description="Phase 6H Stage 2: Spatial Texture Extraction")
    parser.add_argument("--pilot", action="store_true", help="Run pilot on 20 representative farms first")
    parser.add_argument("--pilot_only", action="store_true", help="Run pilot and stop (don't proceed to full extraction)")
    parser.add_argument("--chunk_size", type=int, default=500, help="Farms per extraction chunk")
    parser.add_argument("--chunk_idx", type=int, default=None, help="Process only specific 0-indexed chunk")
    parser.add_argument("--max_workers", type=int, default=4, help="Concurrent worker threads")
    parser.add_argument("--checkpoint_interval", type=int, default=10, help="Farms between checkpoints")
    parser.add_argument("--cloud_threshold", type=float, default=80.0, help="S2 CLOUDY_PIXEL_PERCENTAGE threshold")
    parser.add_argument("--stop_at", type=str, default=None, help="Stop before next chunk if time >= HH:MM")
    args = parser.parse_args()

    # Prevent Windows sleep
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
            logger.info("Windows sleep prevention activated.")
        except Exception:
            pass

    # Step 0: Verify protected baselines BEFORE execution
    pre_hashes = verify_protected_baselines("pre")

    # Step 1: Load eligible cohort (7,843 farms from phase6f_cv_folds.csv)
    cohort_path = PROJECT_ROOT / "outputs" / "data" / "phase6f_cv_folds.csv"
    df_cohort = pd.read_csv(cohort_path)
    logger.info("Loaded eligible cohort: %d farms from %s", len(df_cohort), cohort_path.name)

    # Ensure geometry column has WKT strings from enriched geometry
    geom_path = PROJECT_ROOT / "outputs" / "data" / "train_enriched_geometry.csv"
    df_geom = pd.read_csv(geom_path)
    cohort_with_geom = df_cohort.merge(
        df_geom[["FarmID", "geometry", "area_m2"]],
        on="FarmID",
        how="left",
    )

    harmonized_path = PROJECT_ROOT / "outputs" / "data" / "s2_phase6f_harmonized_cohort.csv"
    df_harm = pd.read_csv(harmonized_path)
    # Merge metadata + dates from harmonized cohort
    needed_cols = ["FarmID", "SDate", "HDate", "Crop", "District", "Sub-District", "Season"]
    available_cols = [c for c in needed_cols if c in df_harm.columns]
    cohort_with_geom = cohort_with_geom.merge(df_harm[available_cols], on="FarmID", how="left", suffixes=("", "_harm"))
    # Prefer the harmonized columns
    for col in ["SDate", "HDate", "Crop", "District", "Sub-District", "Season"]:
        harm_col = f"{col}_harm"
        if harm_col in cohort_with_geom.columns:
            cohort_with_geom[col] = cohort_with_geom[harm_col].fillna(cohort_with_geom.get(col, pd.NA))
            cohort_with_geom.drop(columns=[harm_col], inplace=True)

    # Initialize EE
    EarthEngineManager.initialize()
    if not EarthEngineManager.is_authenticated():
        logger.error("Earth Engine not authenticated. Aborting.")
        sys.exit(1)

    # Step 2: PILOT
    if args.pilot or args.pilot_only:
        logger.info("=" * 60)
        logger.info("PHASE 6H STAGE 2: PILOT EXTRACTION (20 representative farms)")
        logger.info("=" * 60)

        pilot_farms = select_pilot_farms(cohort_with_geom, n_pilot=20)
        pilot_checkpoint = PHASE6H_DATA_DIR / "phase6h_pilot_texture.csv"

        pilot_df = run_texture_extraction_chunk(
            manifest_chunk=pilot_farms,
            checkpoint_file=pilot_checkpoint,
            cloud_cover_threshold=args.cloud_threshold,
            max_workers=min(args.max_workers, 2),  # conservative for pilot
            checkpoint_interval=5,
        )

        # Validate pilot
        pilot_report = validate_pilot_output(pilot_df)
        pilot_report["timestamp"] = datetime.now().isoformat()
        pilot_report["pilot_farm_ids"] = sorted(pilot_df["FarmID"].astype(int).tolist())

        pilot_report_path = PHASE6H_REPORTS_DIR / "phase6h_stage2_pilot_validation.json"
        with open(pilot_report_path, "w", encoding="utf-8") as f:
            json.dump(pilot_report, f, indent=2)
        logger.info("Pilot validation report saved to %s", pilot_report_path)

        if not pilot_report["validation_passed"]:
            logger.error("PILOT VALIDATION FAILED. Stopping before full extraction.")
            sys.exit(1)

        logger.info("PILOT PASSED: %d/%d successful, %d features, %.1f%% mean missingness",
                     pilot_report["n_success"], pilot_report["n_pilot_farms"],
                     pilot_report["found_features"], pilot_report.get("mean_missingness_pct", 0))

        if args.pilot_only:
            logger.info("--pilot_only flag set. Stopping after pilot.")
            verify_protected_baselines("post-pilot")
            sys.exit(0)

    # Step 3: FULL COHORT EXTRACTION
    logger.info("=" * 60)
    logger.info("PHASE 6H STAGE 2: FULL COHORT TEXTURE EXTRACTION (%d farms)", len(cohort_with_geom))
    logger.info("=" * 60)

    total_farms = len(cohort_with_geom)
    chunk_size = args.chunk_size
    num_chunks = int(np.ceil(total_farms / chunk_size))
    logger.info("Total farms: %d | Chunk size: %d | Chunks: %d", total_farms, chunk_size, num_chunks)

    if args.chunk_idx is not None:
        c_idx = args.chunk_idx
        if c_idx < 0 or c_idx >= num_chunks:
            logger.error("Invalid chunk_idx %d (must be 0 to %d)", c_idx, num_chunks - 1)
            sys.exit(1)
        start_idx = c_idx * chunk_size
        end_idx = min(start_idx + chunk_size, total_farms)
        chunk_manifest = cohort_with_geom.iloc[start_idx:end_idx].copy().reset_index(drop=True)
        chunk_file = PHASE6H_DATA_DIR / f"phase6h_texture_chunk_{c_idx:02d}_{start_idx}_{end_idx}.csv"
        logger.info("Executing Chunk %d/%d (Farms %d to %d)...", c_idx + 1, num_chunks, start_idx, end_idx)
        run_texture_extraction_chunk(
            manifest_chunk=chunk_manifest,
            checkpoint_file=chunk_file,
            cloud_cover_threshold=args.cloud_threshold,
            max_workers=args.max_workers,
            checkpoint_interval=args.checkpoint_interval,
        )
    else:
        all_chunk_dfs = []
        for c_idx in range(num_chunks):
            start_idx = c_idx * chunk_size
            end_idx = min(start_idx + chunk_size, total_farms)
            chunk_file = PHASE6H_DATA_DIR / f"phase6h_texture_chunk_{c_idx:02d}_{start_idx}_{end_idx}.csv"

            if args.stop_at:
                now_hm = time.strftime("%H:%M")
                if now_hm >= args.stop_at:
                    logger.info("Reached stop_at deadline (%s >= %s). Halting before Chunk %d.", now_hm, args.stop_at, c_idx + 1)
                    break

            # Fast-forward complete chunks
            if chunk_file.exists():
                try:
                    existing_chunk = pd.read_csv(chunk_file)
                    if len(existing_chunk) >= (end_idx - start_idx):
                        logger.info("Chunk %d/%d already complete (%d records). Skipping.", c_idx + 1, num_chunks, len(existing_chunk))
                        all_chunk_dfs.append(existing_chunk)
                        continue
                except Exception:
                    pass

            chunk_manifest = cohort_with_geom.iloc[start_idx:end_idx].copy().reset_index(drop=True)
            logger.info("=== Processing Chunk %d/%d (Farms %d to %d) ===", c_idx + 1, num_chunks, start_idx, end_idx)
            chunk_df = run_texture_extraction_chunk(
                manifest_chunk=chunk_manifest,
                checkpoint_file=chunk_file,
                cloud_cover_threshold=args.cloud_threshold,
                max_workers=args.max_workers,
                checkpoint_interval=args.checkpoint_interval,
            )
            all_chunk_dfs.append(chunk_df)

        if all_chunk_dfs:
            merged_df = pd.concat(all_chunk_dfs, ignore_index=True)
            merged_df = merged_df.drop_duplicates(subset="FarmID", keep="last")
            merged_path = PHASE6H_DATA_DIR / "phase6h_texture_all_farms.csv"
            atomic_save_dataframe(merged_df, merged_path)
            logger.info("Merged texture output: %d farms -> %s", len(merged_df), merged_path)

            # Generate Stage 2 cohort coverage report
            _write_stage2_report(merged_df, len(cohort_with_geom))

    # Step 4: Post-execution baseline verification
    post_hashes = verify_protected_baselines("post")
    for rel_path in pre_hashes:
        if pre_hashes[rel_path] != post_hashes[rel_path]:
            raise ValueError(f"CRITICAL: Protected file mutated during execution: {rel_path}")

    logger.info("=" * 60)
    logger.info("PHASE 6H STAGE 2 EXECUTION COMPLETED SUCCESSFULLY.")
    logger.info("=" * 60)


def _write_stage2_report(merged_df: pd.DataFrame, expected_cohort: int) -> None:
    """Write Stage 2 completion report."""
    n_total = len(merged_df)
    n_success = int((merged_df["status"] == "success").sum())
    n_failed = n_total - n_success

    # Feature completeness
    found_features = [c for c in GLCM_FEATURE_NAMES if c in merged_df.columns]
    success_rows = merged_df[merged_df["status"] == "success"]

    feature_stats = {}
    if len(success_rows) > 0 and found_features:
        for feat in found_features:
            vals = success_rows[feat]
            feature_stats[feat] = {
                "finite_count": int(vals.notna().sum()),
                "finite_rate_pct": round(vals.notna().mean() * 100, 2),
                "nan_count": int(vals.isna().sum()),
                "mean": round(float(vals.mean()), 6) if vals.notna().any() else None,
                "std": round(float(vals.std()), 6) if vals.notna().any() else None,
                "min": round(float(vals.min()), 6) if vals.notna().any() else None,
                "max": round(float(vals.max()), 6) if vals.notna().any() else None,
            }

    report = {
        "phase": "Phase 6H Stage 2: Spatial Texture Extraction",
        "timestamp": datetime.now().isoformat(),
        "expected_cohort_size": expected_cohort,
        "extracted_farms": n_total,
        "successful_farms": n_success,
        "failed_farms": n_failed,
        "success_rate_pct": round(n_success / max(n_total, 1) * 100, 2),
        "cohort_coverage_pct": round(n_total / max(expected_cohort, 1) * 100, 2),
        "unique_farm_ids": int(merged_df["FarmID"].nunique()),
        "glcm_features_count": len(found_features),
        "glcm_feature_names": found_features,
        "feature_statistics": feature_stats,
    }

    json_path = PHASE6H_REPORTS_DIR / "phase6h_stage2_extraction_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Stage 2 extraction report saved to %s", json_path)

    # Markdown report
    md_lines = [
        "# Phase 6H Stage 2: Spatial Texture Extraction Report",
        "",
        f"**Timestamp:** {report['timestamp']}  ",
        f"**Expected Cohort:** {expected_cohort} farms  ",
        f"**Extracted:** {n_total} farms ({report['cohort_coverage_pct']}% coverage)  ",
        f"**Successful:** {n_success} ({report['success_rate_pct']}%)  ",
        f"**Failed:** {n_failed}  ",
        f"**Unique FarmIDs:** {report['unique_farm_ids']}  ",
        f"**GLCM Features:** {len(found_features)}  ",
        "",
        "## Feature Statistics (Successful Farms)",
        "",
        "| Feature | Finite Rate (%) | Mean | Std | Min | Max |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for feat, stats in feature_stats.items():
        md_lines.append(
            f"| `{feat}` | {stats['finite_rate_pct']} | {stats.get('mean', 'N/A')} | {stats.get('std', 'N/A')} | {stats.get('min', 'N/A')} | {stats.get('max', 'N/A')} |"
        )

    md_lines += [
        "",
        "## Next Steps",
        "",
        "- [ ] Verify cohort coverage matches 7,843 eligible farms",
        "- [ ] Confirm all 20 GLCM features present and finite-value rates acceptable",
        "- [ ] Run `test_phase6h_spatial_texture.py` and `test_phase6h_stage2_extraction.py`",
        "- [ ] Await authorization for Stage 3: Seven model-ablation experiments",
    ]

    md_path = PHASE6H_REPORTS_DIR / "phase6h_stage2_extraction_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    logger.info("Stage 2 Markdown report saved to %s", md_path)


if __name__ == "__main__":
    main()
