"""Pipeline orchestrator for Business Entity Resolution preprocessing.

Executes end-to-end preprocessing, chunked normalization, leakage-free splitting,
audit reporting, and artifact generation.
"""

import argparse
import gc
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Reconfigure stdout for UTF-8 compatibility on Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.audit import audit_dataframe, format_audit_report, verify_leakage
from src.preprocessing.load_data import load_tsv, save_tsv, stream_tsv_chunks
from src.preprocessing.normalize import normalize_dataframe
from src.preprocessing.split import split_source1_and_ground_truth

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def process_stream_file(
    input_path: Path,
    output_path: Path,
    chunksize: int = 250000,
    sample_size: Optional[int] = None,
) -> int:
    """Process a large TSV file chunk-by-chunk, normalize, and write to output_path.

    Returns:
        Total number of rows processed.
    """
    logger.info("Processing stream: %s -> %s", input_path.name, output_path.name)
    total_processed = 0
    first_chunk = True

    if sample_size is not None:
        # Load sample directly
        df_sample = load_tsv(input_path, nrows=sample_size)
        norm_df = normalize_dataframe(df_sample)
        save_tsv(norm_df, output_path, mode="w", header=True)
        return len(norm_df)

    # Full streaming mode
    for chunk in stream_tsv_chunks(input_path, chunksize=chunksize):
        norm_chunk = normalize_dataframe(chunk)
        mode = "w" if first_chunk else "a"
        header = first_chunk
        save_tsv(norm_chunk, output_path, mode=mode, header=header)
        total_processed += len(norm_chunk)
        first_chunk = False
        gc.collect()

    logger.info("Finished %s: %s rows written.", output_path.name, f"{total_processed:,}")
    return total_processed


def run_pipeline(
    train_dir: Path,
    test_dir: Path,
    output_dir: Path,
    val_ratio: float = 0.20,
    seed: int = 42,
    chunksize: int = 250000,
    sample_size: Optional[int] = None,
) -> None:
    """Execute complete preprocessing workflow."""
    start_time = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve train folder automatically
    if not train_dir.exists():
        for alt in [
            PROJECT_ROOT / "data" / "train",
            train_dir.parent / "data" / "train",
            train_dir.parent / "Train",
            train_dir.parent / "train",
            PROJECT_ROOT / "Train",
            PROJECT_ROOT / "train",
        ]:
            if alt.exists():
                train_dir = alt
                break

    # Resolve test folder automatically
    if not test_dir.exists():
        for alt in [
            PROJECT_ROOT / "data" / "test",
            test_dir.parent / "data" / "test",
            test_dir.parent / "Test",
            test_dir.parent / "test",
            PROJECT_ROOT / "Test",
            PROJECT_ROOT / "test",
        ]:
            if alt.exists():
                test_dir = alt
                break

    logger.info("Starting Preprocessing Pipeline")
    logger.info("Train directory: %s", train_dir)
    logger.info("Test directory:  %s", test_dir)
    logger.info("Output directory: %s", output_dir)
    logger.info("Mode: %s", f"SAMPLE ({sample_size:,} rows)" if sample_size else "FULL DATASET")

    # Paths
    train_s1_path = train_dir / "train_source1.tsv"
    train_s2_path = train_dir / "train_source2.tsv"
    train_s3_path = train_dir / "train_source3.tsv"
    train_gt_path = train_dir / "train_ground_truth.tsv"

    test_s1_path = test_dir / "test_source1.tsv"
    test_s2_path = test_dir / "test_source2.tsv"
    test_s3_path = test_dir / "test_source3.tsv"

    audits = []

    # ---------------------------------------------------------
    # 1. PROCESS SOURCE 1 & GROUND TRUTH (TRAIN / VAL SPLIT)
    # ---------------------------------------------------------
    logger.info("Loading and normalizing Source 1 (Train)...")
    s1_df = load_tsv(train_s1_path, nrows=sample_size)
    s1_norm = normalize_dataframe(s1_df)

    logger.info("Loading Ground Truth...")
    gt_df = load_tsv(train_gt_path, nrows=sample_size)

    # Filter GT if in sample mode
    if sample_size is not None:
        gt_df = gt_df[gt_df["source1_entity_id"].isin(set(s1_norm["entity_id"]))].copy()

    logger.info("Executing leakage-safe train/validation split (val_ratio=%.2f, seed=%d)...", val_ratio, seed)
    train_s1, val_s1, train_gt, val_gt = split_source1_and_ground_truth(
        s1_norm, gt_df, val_ratio=val_ratio, seed=seed
    )

    # Save split artifacts
    save_tsv(s1_norm, output_dir / "source1_processed.tsv")
    save_tsv(train_s1, output_dir / "train_source1_processed.tsv")
    save_tsv(val_s1, output_dir / "val_source1_processed.tsv")
    save_tsv(train_gt, output_dir / "train_ground_truth_train.tsv")
    save_tsv(val_gt, output_dir / "train_ground_truth_val.tsv")

    # Leakage verification
    leakage_info = verify_leakage(train_s1["entity_id"], val_s1["entity_id"])
    assert not leakage_info["leakage_detected"], "CRITICAL: Train/Val leakage detected!"

    audits.append(audit_dataframe(s1_norm, "Train Source 1 (Full Normalized)"))
    audits.append(audit_dataframe(train_s1, "Train Source 1 (Train Split 80%)"))
    audits.append(audit_dataframe(val_s1, "Train Source 1 (Validation Split 20%)"))
    audits.append(audit_dataframe(train_gt, "Ground Truth (Train Split)"))
    audits.append(audit_dataframe(val_gt, "Ground Truth (Validation Split)"))

    # Free memory
    del s1_df, s1_norm, gt_df, train_s1, val_s1, train_gt, val_gt
    gc.collect()

    # ---------------------------------------------------------
    # 2. PROCESS TRAIN SOURCE 2 & SOURCE 3
    # ---------------------------------------------------------
    process_stream_file(train_s2_path, output_dir / "source2_processed.tsv", chunksize=chunksize, sample_size=sample_size)
    process_stream_file(train_s3_path, output_dir / "source3_processed.tsv", chunksize=chunksize, sample_size=sample_size)

    # Quick audit on head of processed S2 & S3
    audits.append(audit_dataframe(load_tsv(output_dir / "source2_processed.tsv", nrows=10000), "Train Source 2 (Sample Profile)"))
    audits.append(audit_dataframe(load_tsv(output_dir / "source3_processed.tsv", nrows=10000), "Train Source 3 (Sample Profile)"))

    # ---------------------------------------------------------
    # 3. PROCESS TEST SOURCES (1, 2, 3)
    # ---------------------------------------------------------
    logger.info("Processing Test Sources (completely isolated)...")
    process_stream_file(test_s1_path, output_dir / "test_source1_processed.tsv", chunksize=chunksize, sample_size=sample_size)
    process_stream_file(test_s2_path, output_dir / "test_source2_processed.tsv", chunksize=chunksize, sample_size=sample_size)
    process_stream_file(test_s3_path, output_dir / "test_source3_processed.tsv", chunksize=chunksize, sample_size=sample_size)

    audits.append(audit_dataframe(load_tsv(output_dir / "test_source1_processed.tsv", nrows=10000), "Test Source 1 (Sample Profile)"))
    audits.append(audit_dataframe(load_tsv(output_dir / "test_source2_processed.tsv", nrows=10000), "Test Source 2 (Sample Profile)"))
    audits.append(audit_dataframe(load_tsv(output_dir / "test_source3_processed.tsv", nrows=10000), "Test Source 3 (Sample Profile)"))

    # ---------------------------------------------------------
    # 4. GENERATE AUDIT REPORT
    # ---------------------------------------------------------
    report_text = format_audit_report(audits, leakage_info=leakage_info, seed=seed)
    report_path = output_dir / "preprocessing_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    elapsed = time.time() - start_time
    logger.info("Pipeline completed successfully in %.1f seconds.", elapsed)
    logger.info("Preprocessing report saved to: %s", report_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Preprocessing Pipeline")
    parser.add_argument("--train-dir", type=Path, default=PROJECT_ROOT / "data" / "train", help="Path to raw train data directory")
    parser.add_argument("--test-dir", type=Path, default=PROJECT_ROOT / "data" / "test", help="Path to raw test data directory")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/preprocessing"), help="Output directory")
    parser.add_argument("--val-ratio", type=float, default=0.20, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    parser.add_argument("--chunksize", type=int, default=250000, help="Chunk size for streaming")
    parser.add_argument("--sample-size", type=int, default=None, help="Sample size for testing (default: None for full run)")

    args = parser.parse_args()
    run_pipeline(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
        chunksize=args.chunksize,
        sample_size=args.sample_size,
    )
