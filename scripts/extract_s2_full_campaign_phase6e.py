"""
scripts/extract_s2_full_campaign_phase6e.py
============================================
Phase 6E: Resumable Full-Dataset Sentinel-2 Extraction Campaign.
Processes candidate farms from outputs/data/s2_full_expansion_manifest_7164.csv
in operational chunks with daily mosaicking, cloud masking, atomic checkpointing,
and exponential backoff retry.

Safety Guarantees:
- Never overwrites Phase 6C or Phase 6D files.
- Daily mosaicking before temporal median composite (no MGRS tile duplicate weighting).
- SCL cloud and cloud-shadow masking.
- Missing observations strictly preserve NaN (0 artificial zeros).
- Atomic disk writes via temporary file replacement (.tmp -> .csv).
- Checkpoints every N farms with FarmID deduplication.
- Transient network/GEE errors retried with bounded exponential backoff.
- Resumable: can run chunk by chunk or full run.
"""

from __future__ import annotations

import argparse
from concurrent.futures import as_completed, ThreadPoolExecutor
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

# UTF-8 stdout configuration for Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.indices import compute_all_spectral_indices, compute_temporal_dynamics
from src.satellite.s2_extractor import (
    DEFAULT_CLOUD_COVER_THRESHOLD,
    EarthEngineManager,
    S2_SR_COLLECTION,
    extract_zonal_stats_for_farm,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CLASSES_7: List[str] = [
    "Bengal Gram",
    "Chillies",
    "Cotton",
    "Groundnut",
    "Maize",
    "Paddy",
    "Red Gram",
]

WINDOWS: List[str] = ["early", "mid", "late", "full"]

OUTPUTS_DATA_DIR = PROJECT_ROOT / "outputs" / "data"
OUTPUTS_REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
OUTPUTS_DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _extract_single_farm_with_retry(
    row_dict: Dict[str, Any],
    cloud_cover_threshold: float,
    daily_mosaic: bool = True,
    max_retries: int = 3,
    initial_delay: float = 2.0,
) -> Dict[str, Any]:
    """Extract zonal statistics for a single farm with exponential backoff on transient errors."""
    farm_id = int(row_dict["FarmID"])
    crop = row_dict["Crop"]
    district = row_dict.get("District", "")
    sub_district = row_dict.get("Sub-District", "")
    season = row_dict.get("Season", "")
    sdate = str(row_dict["SDate"])[:10]
    hdate = str(row_dict["HDate"])[:10]
    geom = wkt.loads(str(row_dict["geometry"]))

    last_exc: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            res = extract_zonal_stats_for_farm(
                farm_id=farm_id,
                geometry=geom,
                sdate_str=sdate,
                hdate_str=hdate,
                collection_name=S2_SR_COLLECTION,
                cloud_cover_threshold=cloud_cover_threshold,
                daily_mosaic=daily_mosaic,
            )
            res["Crop"] = crop
            res["District"] = district
            res["Sub-District"] = sub_district
            res["Season"] = season
            res["SDate"] = sdate
            res["HDate"] = hdate
            return res
        except Exception as exc:
            last_exc = exc
            err_str = str(exc)
            # If rate-limited or transient GEE error, back off
            if attempt < max_retries and any(t in err_str.lower() for t in ["429", "503", "quota", "rate limit", "timeout", "unavailable"]):
                backoff = initial_delay * (2 ** (attempt - 1)) + random.uniform(0.1, 1.0)
                logger.warning(
                    "FarmID %d attempt %d/%d encountered transient error: %s. Backing off for %.2fs...",
                    farm_id, attempt, max_retries, err_str, backoff
                )
                time.sleep(backoff)
            else:
                break

    err_msg = str(last_exc) if last_exc else "Unknown error"
    logger.error("Extraction failed for FarmID %d after %d attempts: %s", farm_id, max_retries, err_msg)
    res = {
        "FarmID": farm_id,
        "Crop": crop,
        "District": district,
        "Sub-District": sub_district,
        "Season": season,
        "SDate": sdate,
        "HDate": hdate,
        "status": "failed",
        "error_message": err_msg,
    }
    return res


def atomic_save_dataframe(df: pd.DataFrame, target_path: Path) -> None:
    """Save dataframe atomically using a temporary file and atomic replace."""
    tmp_path = target_path.with_suffix(target_path.suffix + f".tmp.{os.getpid()}")
    df.to_csv(tmp_path, index=False)
    if target_path.exists():
        try:
            os.replace(tmp_path, target_path)
        except OSError:
            # Fallback for Windows if destination locked
            time.sleep(0.1)
            os.replace(tmp_path, target_path)
    else:
        tmp_path.rename(target_path)


def run_extraction_chunk(
    manifest_chunk: pd.DataFrame,
    checkpoint_file: Path,
    cloud_cover_threshold: float = DEFAULT_CLOUD_COVER_THRESHOLD,
    max_workers: int = 4,
    checkpoint_interval: int = 10,
    daily_mosaic: bool = True,
) -> pd.DataFrame:
    """Run extraction for a chunk with deduplicated resume and atomic checkpoints."""
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
    logger.info("Chunk total: %d | Already successful: %d | Pending: %d", len(manifest_chunk), len(successful_ids), len(pending_df))

    all_records: Dict[int, Dict[str, Any]] = dict(cached_records)
    if pending_df.empty:
        logger.info("All chunk farms already successfully extracted.")
        return pd.DataFrame(list(all_records.values()))

    row_dicts = pending_df.to_dict("records")
    completed = 0
    t0 = time.time()

    logger.info("Launching chunk extraction with %d workers (daily_mosaic=%s)...", max_workers, daily_mosaic)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(_extract_single_farm_with_retry, r, cloud_cover_threshold, daily_mosaic): r["FarmID"]
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
            crop = res.get("Crop", "unknown")
            elapsed = time.time() - t0
            rate = completed / max(elapsed, 1.0) * 60.0

            logger.info(
                "Progress [%d/%d] (%.1f farms/min) FarmID %d | Crop: %-12s | Status: %s",
                completed, len(row_dicts), rate, fid, crop, status
            )

            if completed % checkpoint_interval == 0 or completed == len(row_dicts):
                chk_df = pd.DataFrame(list(all_records.values()))
                atomic_save_dataframe(chk_df, checkpoint_file)
                logger.info("Atomic checkpoint saved: %d records -> %s", len(chk_df), checkpoint_file.name)

    final_df = pd.DataFrame(list(all_records.values()))
    atomic_save_dataframe(final_df, checkpoint_file)
    logger.info("Chunk complete: %d records saved to %s", len(final_df), checkpoint_file)
    return final_df


def main():
    parser = argparse.ArgumentParser(description="Phase 6E: Resumable Full Sentinel-2 Extraction Campaign")
    parser.add_argument("--manifest", type=str, default=str(OUTPUTS_DATA_DIR / "s2_full_expansion_manifest_7164.csv"))
    parser.add_argument("--output_file", type=str, default=str(OUTPUTS_DATA_DIR / "s2_phase6e_7164_raw_bands.csv"))
    parser.add_argument("--chunk_size", type=int, default=500, help="Number of farms per chunk")
    parser.add_argument("--chunk_idx", type=int, default=None, help="Process only specific 0-indexed chunk (0 to N-1)")
    parser.add_argument("--max_workers", type=int, default=4, help="Concurrent worker threads")
    parser.add_argument("--checkpoint_interval", type=int, default=10, help="Farms between checkpoints")
    parser.add_argument("--cloud_threshold", type=float, default=80.0, help="S2 cloudy pixel percentage threshold")
    parser.add_argument("--no_daily_mosaic", action="store_true", help="Disable daily mosaicking guard")
    parser.add_argument("--stop_at", type=str, default=None, help="Stop before launching next chunk if local time >= HH:MM (e.g. 06:55)")
    args = parser.parse_args()

    # Prevent Windows from entering system sleep while extraction is active
    if sys.platform == "win32":
        try:
            import ctypes
            # ES_CONTINUOUS (0x80000000) | ES_SYSTEM_REQUIRED (0x00000001)
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
            logger.info("Windows sleep prevention activated (ES_SYSTEM_REQUIRED).")
        except Exception as e:
            logger.warning("Could not set thread execution state: %s", e)

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        logger.error("Manifest not found: %s", manifest_path)
        sys.exit(1)

    EarthEngineManager.initialize()
    if not EarthEngineManager.is_authenticated():
        logger.error("Earth Engine not authenticated. Aborting.")
        sys.exit(1)

    manifest = pd.read_csv(manifest_path)
    total_farms = len(manifest)
    chunk_size = args.chunk_size
    num_chunks = int(np.ceil(total_farms / chunk_size))
    logger.info("Total manifest farms: %d | Chunk size: %d | Total chunks: %d", total_farms, chunk_size, num_chunks)

    output_path = Path(args.output_file)
    daily_mosaic = not args.no_daily_mosaic

    if args.chunk_idx is not None:
        c_idx = args.chunk_idx
        if c_idx < 0 or c_idx >= num_chunks:
            logger.error("Invalid chunk_idx %d (must be 0 to %d)", c_idx, num_chunks - 1)
            sys.exit(1)
        start_idx = c_idx * chunk_size
        end_idx = min(start_idx + chunk_size, total_farms)
        chunk_manifest = manifest.iloc[start_idx:end_idx].copy().reset_index(drop=True)
        chunk_file = OUTPUTS_DATA_DIR / f"s2_phase6e_chunk_{c_idx:02d}_{start_idx}_{end_idx}_raw_bands.csv"
        logger.info("Executing Chunk %d/%d (Farms %d to %d)...", c_idx + 1, num_chunks, start_idx, end_idx)
        run_extraction_chunk(
            manifest_chunk=chunk_manifest,
            checkpoint_file=chunk_file,
            cloud_cover_threshold=args.cloud_threshold,
            max_workers=args.max_workers,
            checkpoint_interval=args.checkpoint_interval,
            daily_mosaic=daily_mosaic,
        )
    else:
        logger.info("Executing full campaign sequentially across all %d chunks...", num_chunks)
        all_chunk_dfs = []
        for c_idx in range(num_chunks):
            start_idx = c_idx * chunk_size
            end_idx = min(start_idx + chunk_size, total_farms)
            chunk_file = OUTPUTS_DATA_DIR / f"s2_phase6e_chunk_{c_idx:02d}_{start_idx}_{end_idx}_raw_bands.csv"

            # Check stop_at deadline before launching new chunk
            if args.stop_at:
                now_hm = time.strftime("%H:%M")
                if now_hm >= args.stop_at:
                    logger.info("Reached stop_at deadline (%s >= %s). Halting campaign before Chunk %d.", now_hm, args.stop_at, c_idx + 1)
                    break

            # Fast-forward if chunk already fully completed
            if chunk_file.exists():
                try:
                    existing_chunk = pd.read_csv(chunk_file)
                    if len(existing_chunk) >= (end_idx - start_idx):
                        logger.info("Chunk %d/%d already complete (%d records in %s). Skipping.", c_idx + 1, num_chunks, len(existing_chunk), chunk_file.name)
                        all_chunk_dfs.append(existing_chunk)
                        continue
                except Exception as e:
                    logger.warning("Error reading existing chunk %s: %s", chunk_file, e)

            chunk_manifest = manifest.iloc[start_idx:end_idx].copy().reset_index(drop=True)
            logger.info("=== Processing Chunk %d/%d (Farms %d to %d) ===", c_idx + 1, num_chunks, start_idx, end_idx)
            chunk_df = run_extraction_chunk(
                manifest_chunk=chunk_manifest,
                checkpoint_file=chunk_file,
                cloud_cover_threshold=args.cloud_threshold,
                max_workers=args.max_workers,
                checkpoint_interval=args.checkpoint_interval,
                daily_mosaic=daily_mosaic,
            )
            all_chunk_dfs.append(chunk_df)

        if all_chunk_dfs:
            merged_df = pd.concat(all_chunk_dfs, ignore_index=True)
            merged_df = merged_df.drop_duplicates(subset="FarmID", keep="last")
            atomic_save_dataframe(merged_df, output_path)
            logger.info("Saved merged progress: %d total farms -> %s", len(merged_df), output_path)


if __name__ == "__main__":
    main()
