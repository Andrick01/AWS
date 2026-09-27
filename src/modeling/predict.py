"""Inference & Submission Formatting Module for Business Entity Resolution.

Applies trained model to pair features, applies optimal F_0.5 decision threshold,
and generates competition-compliant submission artifacts:
1. output/matching_results.tsv
2. output/candidate_pairs.tsv

Strictly enforces competition rules:
- Exactly 1 row per Source 1 test entity
- Singletons (no match) represented as empty matched_entity_ids column
- Subtab-separated with comma-separated ID lists
- No duplicate IDs within list
- Only valid Source 2 (S2-) and Source 3 (S3-) IDs
- All matched IDs in matching_results.tsv are a strict subset of candidate_pairs.tsv
"""

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
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

from src.modeling.train_model import FEATURE_COLS, load_model, predict_scores
from src.preprocessing.load_data import load_tsv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_submission_files(
    test_s1_path: Path,
    candidate_pairs_path: Path,
    test_features_path: Path,
    model_path: Optional[Path],
    output_dir: Path,
    threshold: float = 0.55,
) -> Tuple[Path, Path]:
    """Generate matching_results.tsv and candidate_pairs.tsv adhering to official competition specs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    matching_output = output_dir / "matching_results.tsv"
    candidate_output = output_dir / "candidate_pairs.tsv"

    logger.info("Loading Test Source 1 entity list from %s...", test_s1_path)
    test_s1_df = load_tsv(test_s1_path)
    s1_ids = test_s1_df["entity_id"].astype(str).tolist()
    logger.info("Total Source 1 test entities: %d", len(s1_ids))

    # Index candidate pairs from blocking stage
    logger.info("Reading candidate pairs from %s...", candidate_pairs_path)
    s1_candidates_map: Dict[str, List[str]] = {sid: [] for sid in s1_ids}
    
    if candidate_pairs_path.exists():
        with open(candidate_pairs_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                sid = row.get("source1_entity_id", "").strip()
                if sid in s1_candidates_map:
                    cands_raw = row.get("candidate_entity_ids", row.get("candidate_entity_id", "")).strip()
                    if cands_raw:
                        cands = [c.strip() for c in cands_raw.split(",") if c.strip()]
                        # Deduplicate while preserving order, filter only S2 and S3 IDs
                        seen = set()
                        clean_cands = []
                        for c in cands:
                            if (c.startswith("S2-") or c.startswith("S3-")) and c not in seen:
                                seen.add(c)
                                clean_cands.append(c)
                        s1_candidates_map[sid] = clean_cands

    # Compute matches using model if features & model available
    s1_matches_map: Dict[str, List[str]] = {sid: [] for sid in s1_ids}

    if model_path and model_path.exists() and test_features_path.exists():
        logger.info("Loading trained model from %s...", model_path)
        model = load_model(model_path)

        logger.info("Loading test pairwise features from %s...", test_features_path)
        df_feat = pd.read_csv(test_features_path, sep="\t")
        
        if not df_feat.empty:
            X_test = df_feat[FEATURE_COLS].fillna(0.0).values.astype(np.float32)
            scores = predict_scores(model, X_test)

            logger.info("Applying optimal decision threshold (threshold=%.3f)...", threshold)
            s1_pair_scores: Dict[str, List[Tuple[str, float]]] = {}

            for idx, row in df_feat.iterrows():
                s1_id = str(row["source1_entity_id"]).strip()
                c_id = str(row["candidate_entity_id"]).strip()
                score = float(scores[idx])

                if score >= threshold:
                    if s1_id not in s1_pair_scores:
                        s1_pair_scores[s1_id] = []
                    s1_pair_scores[s1_id].append((c_id, score))

            for s1_id, pair_list in s1_pair_scores.items():
                if s1_id in s1_matches_map:
                    # Sort matches by score descending
                    pair_list.sort(key=lambda x: -x[1])
                    s1_matches_map[s1_id] = [c_id for c_id, _ in pair_list]

    # Write output/matching_results.tsv
    logger.info("Writing %s...", matching_output)
    with open(matching_output, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["source1_entity_id", "matched_entity_ids"])
        for sid in s1_ids:
            matches = s1_matches_map.get(sid, [])
            # Guarantee candidate subset constraint
            cand_set = set(s1_candidates_map.get(sid, []))
            valid_matches = [m for m in matches if m in cand_set or not s1_candidates_map.get(sid)]
            writer.writerow([sid, ",".join(valid_matches)])

    # Write output/candidate_pairs.tsv
    logger.info("Writing %s...", candidate_output)
    with open(candidate_output, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["source1_entity_id", "candidate_entity_ids"])
        for sid in s1_ids:
            cands = s1_candidates_map.get(sid, [])
            writer.writerow([sid, ",".join(cands)])

    logger.info("Successfully generated submission outputs:")
    logger.info("  1. %s", matching_output)
    logger.info("  2. %s", candidate_output)

    return matching_output, candidate_output


def validate_outputs(matching_tsv: Path, candidate_tsv: Path, test_s1_path: Path) -> bool:
    """Audit submission files against official competition constraints."""
    logger.info("Auditing generated submission files...")

    s1_test_ids = set(load_tsv(test_s1_path)["entity_id"].astype(str))

    # Audit matching_results.tsv
    matching_rows = 0
    matched_ids_seen = set()
    s1_matched_seen = set()

    with open(matching_tsv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            matching_rows += 1
            sid = row["source1_entity_id"]
            if sid in s1_matched_seen:
                logger.error("DUPLICATE source1_entity_id in matching_results: %s", sid)
                return False
            s1_matched_seen.add(sid)

            raw_m = row.get("matched_entity_ids", "").strip()
            if raw_m:
                m_list = [x.strip() for x in raw_m.split(",")]
                if len(m_list) != len(set(m_list)):
                    logger.error("DUPLICATE matched IDs in row for %s", sid)
                    return False
                for m in m_list:
                    if not (m.startswith("S2-") or m.startswith("S3-")):
                        logger.error("INVALID matched ID prefix (must be S2- or S3-): %s", m)
                        return False
                    matched_ids_seen.add((sid, m))

    if len(s1_matched_seen) != len(s1_test_ids):
        logger.error(
            "MISMATCH in row count for matching_results: expected %d, got %d",
            len(s1_test_ids), len(s1_matched_seen)
        )
        return False

    # Audit candidate_pairs.tsv
    candidate_rows = 0
    s1_cand_seen = set()
    cand_pairs_set = set()

    with open(candidate_tsv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            candidate_rows += 1
            sid = row["source1_entity_id"]
            if sid in s1_cand_seen:
                logger.error("DUPLICATE source1_entity_id in candidate_pairs: %s", sid)
                return False
            s1_cand_seen.add(sid)

            raw_c = row.get("candidate_entity_ids", "").strip()
            if raw_c:
                c_list = [x.strip() for x in raw_c.split(",")]
                if len(c_list) != len(set(c_list)):
                    logger.error("DUPLICATE candidate IDs in row for %s", sid)
                    return False
                for c in c_list:
                    cand_pairs_set.add((sid, c))

    if len(s1_cand_seen) != len(s1_test_ids):
        logger.error(
            "MISMATCH in row count for candidate_pairs: expected %d, got %d",
            len(s1_test_ids), len(s1_cand_seen)
        )
        return False

    # Verify subset rule: Every match in matching_results MUST appear in candidate_pairs
    for sid, m in matched_ids_seen:
        if (sid, m) not in cand_pairs_set:
            logger.error("SUBSET VIOLATION: matched pair (%s, %s) not in candidate_pairs!", sid, m)
            return False

    logger.info("AUDIT PASSED: All submission rules & constraints strictly satisfied!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Generate & Audit Competition Submission Files")
    parser.add_argument("--test-s1", type=Path, default=PROJECT_ROOT / "data" / "test" / "test_source1.tsv")
    parser.add_argument("--candidate-pairs", type=Path, default=PROJECT_ROOT / "artifacts" / "test_candidate_pairs.tsv")
    parser.add_argument("--test-features", type=Path, default=PROJECT_ROOT / "artifacts" / "test_pair_features.tsv")
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "models" / "final_model" / "model.pkl")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "output")
    parser.add_argument("--threshold", type=float, default=0.55)
    args = parser.parse_args()

    m_out, c_out = generate_submission_files(
        test_s1_path=args.test_s1,
        candidate_pairs_path=args.candidate_pairs,
        test_features_path=args.test_features,
        model_path=args.model,
        output_dir=args.output_dir,
        threshold=args.threshold,
    )

    if args.test_s1.exists():
        validate_outputs(m_out, c_out, args.test_s1)


if __name__ == "__main__":
    main()
