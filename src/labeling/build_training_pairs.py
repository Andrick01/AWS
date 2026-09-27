"""Training Pairs Builder for Business Entity Resolution.

Cross-references candidate pairs against ground truth to construct labeled
pairwise datasets (label=1 for true match, label=0 for non-match) for training
and evaluation of matchers and classifiers.

Includes:
- Ground-truth matching
- Hard negative generation from blocking candidates
- Configurable negative-to-positive ratio subsampling
- Memory-efficient streaming / batching for large scale
"""

import argparse
import csv
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

# UTF-8 stdout support for Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_ground_truth(gt_path: Path) -> Dict[str, Set[str]]:
    """Load ground truth into a lookup map: source1_id -> set of matched_entity_ids."""
    logger.info("Loading ground truth from %s...", gt_path)
    gt_map: Dict[str, Set[str]] = {}
    with open(gt_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            s1_id = row.get("source1_entity_id", "").strip()
            raw_matches = row.get("matched_entity_ids", "").strip()
            if not s1_id:
                continue
            if raw_matches:
                matches = set(m.strip() for m in raw_matches.split(",") if m.strip())
            else:
                matches = set()
            gt_map[s1_id] = matches
    logger.info("Ground truth loaded: %d entities.", len(gt_map))
    return gt_map


def build_labeled_pairs(
    candidate_pairs_path: Path,
    ground_truth_map: Dict[str, Set[str]],
    output_path: Path,
    neg_to_pos_ratio: int = 5,
    random_seed: int = 42,
    max_rows: Optional[int] = None,
) -> Dict[str, int]:
    """Stream candidate_pairs.tsv, assign labels, subsample negatives, and write to output TSV."""
    random.seed(random_seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Building labeled training pairs -> %s", output_path)
    logger.info("Negative-to-Positive ratio: %d:1", neg_to_pos_ratio)

    pos_count = 0
    neg_count = 0
    total_s1 = 0

    with open(candidate_pairs_path, "r", encoding="utf-8", errors="replace") as in_f, \
         open(output_path, "w", encoding="utf-8", newline="") as out_f:

        reader = csv.DictReader(in_f, delimiter="\t")
        writer = csv.writer(out_f, delimiter="\t")
        writer.writerow(["source1_entity_id", "candidate_entity_id", "label"])

        for row in reader:
            s1_id = row.get("source1_entity_id", "").strip()
            cand_raw = row.get("candidate_entity_ids", "").strip()
            if not s1_id:
                continue

            total_s1 += 1
            if max_rows and total_s1 > max_rows:
                break

            gt_matches = ground_truth_map.get(s1_id, set())
            cands = [c.strip() for c in cand_raw.split(",") if c.strip()] if cand_raw else []

            # Separate into true matches (positives) and blocker hard negatives
            positives = [c for c in cands if c in gt_matches]
            negatives = [c for c in cands if c not in gt_matches]

            # Also include ground truth positives if blocker missed them (ensure positive coverage)
            for m in gt_matches:
                if m not in positives:
                    positives.append(m)

            # Subsample negatives to maintain stable ratio
            if negatives and neg_to_pos_ratio > 0:
                target_neg_count = max(len(positives) * neg_to_pos_ratio, 1 if not positives else 0)
                if len(negatives) > target_neg_count:
                    negatives = random.sample(negatives, target_neg_count)

            # Write positive pairs
            for p in positives:
                writer.writerow([s1_id, p, 1])
                pos_count += 1

            # Write negative pairs
            for n in negatives:
                writer.writerow([s1_id, n, 0])
                neg_count += 1

            if total_s1 % 50000 == 0:
                logger.info(
                    "Processed %d entities | Positives: %d | Negatives: %d | Ratio: %.1f:1",
                    total_s1,
                    pos_count,
                    neg_count,
                    neg_count / max(pos_count, 1),
                )

    stats = {
        "total_source1_entities": total_s1,
        "positive_pairs": pos_count,
        "negative_pairs": neg_count,
        "total_pairs": pos_count + neg_count,
    }
    logger.info("Pair labeling complete: %s", stats)
    return stats


def main():
    parser = argparse.ArgumentParser(description="Build Labeled Training Pairs from Candidates")
    parser.add_argument(
        "--candidates",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "candidate_pairs.tsv",
        help="Path to candidate_pairs.tsv",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=PROJECT_ROOT / "data" / "train" / "train_ground_truth.tsv",
        help="Path to ground truth TSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "training_pairs.tsv",
        help="Path to save labeled training pairs",
    )
    parser.add_argument(
        "--neg-ratio",
        type=int,
        default=5,
        help="Number of hard negatives per positive pair",
    )
    parser.add_argument(
        "--max-entities",
        type=int,
        default=None,
        help="Limit number of Source 1 entities for quick testing",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for negative subsampling",
    )

    args = parser.parse_args()

    gt_map = load_ground_truth(args.ground_truth)
    build_labeled_pairs(
        candidate_pairs_path=args.candidates,
        ground_truth_map=gt_map,
        output_path=args.output,
        neg_to_pos_ratio=args.neg_ratio,
        random_seed=args.seed,
        max_rows=args.max_entities,
    )


if __name__ == "__main__":
    main()
