"""Macro-Averaged F_0.5 Evaluator for Business Entity Resolution.

Computes exact competition evaluation metric:
F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)

Macro-averaged per Source 1 entity:
- Singletons (no match in ground truth) score 1.0 when predicted empty list
- False merges on singletons or multi-matches are heavily penalized (precision weighted 2x over recall)
"""

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.load_data import load_tsv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_f05_single_entity(pred_set: Set[str], gt_set: Set[str]) -> float:
    """Compute F_0.5 score for a single Source 1 entity."""
    # Singleton case: entity has no true matches
    if not gt_set:
        return 1.0 if not pred_set else 0.0

    # Entity has ground truth matches but model predicted empty list
    if not pred_set:
        return 0.0

    tp = len(pred_set.intersection(gt_set))
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if 0.25 * precision + recall == 0:
        return 0.0

    return (1.25 * precision * recall) / (0.25 * precision + recall)


def evaluate_f05_score(predictions_tsv: Path, ground_truth_tsv: Path) -> Dict[str, float]:
    """Compute macro-averaged F_0.5 score across all Source 1 entities."""
    logger.info("Loading ground truth from %s...", ground_truth_tsv)
    gt_map: Dict[str, Set[str]] = {}
    gt_df = load_tsv(ground_truth_tsv)

    for _, row in gt_df.iterrows():
        s1_id = str(row["source1_entity_id"]).strip()
        matched_raw = str(row.get("matched_entity_ids", "")).strip()
        gt_set = set()
        if matched_raw and matched_raw.lower() != "nan":
            for m in matched_raw.split(","):
                m_clean = m.strip()
                if m_clean:
                    gt_set.add(m_clean)
        gt_map[s1_id] = gt_set

    logger.info("Reading predicted matches from %s...", predictions_tsv)
    pred_map: Dict[str, Set[str]] = {}
    with open(predictions_tsv, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            s1_id = str(row.get("source1_entity_id", "")).strip()
            matched_raw = str(row.get("matched_entity_ids", "")).strip()
            pred_set = set()
            if matched_raw and matched_raw.lower() != "nan":
                for m in matched_raw.split(","):
                    m_clean = m.strip()
                    if m_clean:
                        pred_set.add(m_clean)
            pred_map[s1_id] = pred_set

    # Compute macro-average over all ground-truth S1 entities
    scores: List[float] = []
    singleton_count = 0
    singleton_correct = 0

    for s1_id, gt_set in gt_map.items():
        pred_set = pred_map.get(s1_id, set())
        f05 = compute_f05_single_entity(pred_set, gt_set)
        scores.append(f05)

        if not gt_set:
            singleton_count += 1
            if not pred_set:
                singleton_correct += 1

    macro_f05 = sum(scores) / len(scores) if scores else 0.0
    singleton_acc = singleton_correct / singleton_count if singleton_count > 0 else 0.0

    metrics = {
        "macro_f05_score": round(macro_f05, 4),
        "total_entities_evaluated": len(scores),
        "total_singletons": singleton_count,
        "singleton_accuracy": round(singleton_acc, 4),
    }

    logger.info("=" * 60)
    logger.info("  MACRO-AVERAGED F_0.5 SCORE: %.4f", macro_f05)
    logger.info("  Total Entities Evaluated:   %d", len(scores))
    logger.info("  Singleton Accuracy:         %.4f (%d / %d)", singleton_acc, singleton_correct, singleton_count)
    logger.info("=" * 60)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate Macro-Averaged F_0.5 Score")
    parser.add_argument("--predictions", type=Path, default=PROJECT_ROOT / "output" / "matching_results.tsv")
    parser.add_argument("--ground-truth", type=Path, default=PROJECT_ROOT / "data" / "train" / "train_ground_truth.tsv")
    args = parser.parse_args()

    evaluate_f05_score(args.predictions, args.ground_truth)


if __name__ == "__main__":
    main()
