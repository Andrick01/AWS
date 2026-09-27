"""End-to-End Test Inference Engine for Business Entity Resolution.

Executes test set inference end-to-end:
1. Loads test sources (test_source1.tsv, test_source2.tsv, test_source3.tsv)
2. Normalizes entity attributes
3. Generates candidate blocking pairs
4. Extracts pairwise features
5. Applies trained ML model (LightGBM/XGBoost)
6. Outputs competition submission files:
   - output/matching_results.tsv
   - output/candidate_pairs.tsv
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# UTF-8 stdout support for Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.blocking.generate_candidates import generate_candidate_pairs, save_candidate_pairs
from src.features.build_features import build_pair_features, load_entity_attributes
from src.modeling.predict import generate_submission_files, validate_outputs
from src.preprocessing.load_data import load_tsv
from src.preprocessing.normalize import normalize_dataframe

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_test_inference(
    test_dir: Path,
    model_path: Path,
    output_dir: Path,
    threshold: float = 0.55,
    sample_size: int = None,
) -> None:
    """Run full end-to-end inference on test dataset."""
    logger.info("==================================================")
    logger.info("  STARTING END-TO-END TEST INFERENCE PIPELINE     ")
    logger.info("==================================================")

    test_s1_path = test_dir / "test_source1.tsv"
    test_s2_path = test_dir / "test_source2.tsv"
    test_s3_path = test_dir / "test_source3.tsv"

    # Step 1: Preprocess & Normalize
    logger.info("Step 1: Normalizing test datasets...")
    s1_df = normalize_dataframe(load_tsv(test_s1_path, nrows=sample_size))
    s2_df = normalize_dataframe(load_tsv(test_s2_path, nrows=sample_size)) if test_s2_path.exists() else None
    s3_df = normalize_dataframe(load_tsv(test_s3_path, nrows=sample_size)) if test_s3_path.exists() else None

    cands_list = [df for df in (s2_df, s3_df) if df is not None]
    if not cands_list:
        raise FileNotFoundError(f"Neither test Source 2 nor Source 3 found in {test_dir}")

    candidates_df = pd.concat(cands_list, ignore_index=True)

    # Step 2: Blocking Candidate Generation
    logger.info("Step 2: Generating candidate pairs via Lexical + TF-IDF blocking...")
    candidate_pairs_df = generate_candidate_pairs(s1_df, candidates_df)
    temp_cand_tsv = output_dir / "test_candidate_pairs.tsv"
    save_candidate_pairs(candidate_pairs_df, temp_cand_tsv)

    # Step 3: Extract Features
    logger.info("Step 3: Extracting pairwise similarity features...")
    s1_lookup = load_entity_attributes(s1_df)
    cand_lookup = load_entity_attributes(candidates_df)

    # Flatten candidate pairs for feature extraction
    flattened_pairs_tsv = output_dir / "test_pair_flat.tsv"
    with open(temp_cand_tsv, "r", encoding="utf-8") as in_f, \
         open(flattened_pairs_tsv, "w", encoding="utf-8", newline="") as out_f:
        import csv
        reader = csv.DictReader(in_f, delimiter="\t")
        writer = csv.writer(out_f, delimiter="\t")
        writer.writerow(["source1_entity_id", "candidate_entity_id"])
        for row in reader:
            s1_id = row["source1_entity_id"]
            cands_raw = row.get("candidate_entity_ids", "").strip()
            if cands_raw:
                for c in cands_raw.split(","):
                    if c.strip():
                        writer.writerow([s1_id, c.strip()])

    test_features_tsv = output_dir / "test_features.tsv"
    build_pair_features(
        pairs_tsv=flattened_pairs_tsv,
        s1_lookup=s1_lookup,
        cand_lookup=cand_lookup,
        output_tsv=test_features_tsv,
        has_label=False,
    )

    # Step 4: Submission Formatting & Validation
    logger.info("Step 4: Generating final submission files & auditing format...")
    m_out, c_out = generate_submission_files(
        test_s1_path=test_s1_path,
        candidate_pairs_path=temp_cand_tsv,
        test_features_path=test_features_tsv,
        model_path=model_path,
        output_dir=output_dir,
        threshold=threshold,
    )

    validate_outputs(m_out, c_out, test_s1_path)
    logger.info("Inference completed successfully!")


def main():
    parser = argparse.ArgumentParser(description="Run End-to-End Test Inference")
    parser.add_argument("--test-dir", type=Path, default=PROJECT_ROOT / "data" / "test")
    parser.add_argument("--model-path", type=Path, default=PROJECT_ROOT / "models" / "final_model" / "model.pkl")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "output")
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--sample-size", type=int, default=None)
    args = parser.parse_args()

    run_test_inference(args.test_dir, args.model_path, args.output_dir, args.threshold, args.sample_size)


if __name__ == "__main__":
    main()
