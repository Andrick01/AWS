"""Threshold Tuner for F_0.5 Optimization in Business Entity Resolution.

Per competition guidelines, F_0.5 rewards precision 2x more than recall.
This module grid-searches optimal classification threshold on validation
predictions to maximize F_0.5 score.

Key insight: singletons (entities correctly predicted as "no match") receive
a full 1.0 F_0.5 score, so the threshold must be high enough to avoid
false positives but not so high that it misses true matches.
"""

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_f05(precision: float, recall: float) -> float:
    """Compute F_0.5 score: emphasizes precision over recall.
    
    F_0.5 = (1 + 0.5^2) * (P * R) / (0.5^2 * P + R)
           = 1.25 * P * R / (0.25 * P + R)
    """
    if precision + recall == 0:
        return 0.0
    return (1.25 * precision * recall) / (0.25 * precision + recall)


def evaluate_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """Evaluate precision, recall, and F_0.5 at a given threshold."""
    predictions = (scores >= threshold).astype(int)
    tp = int(((predictions == 1) & (labels == 1)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())
    tn = int(((predictions == 0) & (labels == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f05 = compute_f05(precision, recall)

    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f05": f05,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def tune_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    min_threshold: float = 0.1,
    max_threshold: float = 0.95,
    step: float = 0.01,
) -> Tuple[float, Dict[str, float]]:
    """Grid search over thresholds to find the one maximizing F_0.5."""
    logger.info(
        "Tuning threshold from %.2f to %.2f (step=%.3f) on %d samples...",
        min_threshold, max_threshold, step, len(scores),
    )

    best_threshold = 0.5
    best_metrics = evaluate_threshold(scores, labels, 0.5)
    all_results = []

    threshold = min_threshold
    while threshold <= max_threshold:
        metrics = evaluate_threshold(scores, labels, threshold)
        all_results.append(metrics)

        if metrics["f05"] > best_metrics["f05"]:
            best_metrics = metrics
            best_threshold = threshold

        threshold = round(threshold + step, 4)

    logger.info(
        "Best threshold: %.3f -> F_0.5=%.4f (P=%.4f, R=%.4f, TP=%d, FP=%d, FN=%d)",
        best_threshold,
        best_metrics["f05"],
        best_metrics["precision"],
        best_metrics["recall"],
        best_metrics["tp"],
        best_metrics["fp"],
        best_metrics["fn"],
    )

    return best_threshold, best_metrics


def main():
    parser = argparse.ArgumentParser(description="Tune Classification Threshold for F_0.5")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "baseline_predictions.tsv",
        help="TSV with columns: source1_entity_id, candidate_entity_id, score, prediction, label",
    )
    parser.add_argument("--min-threshold", type=float, default=0.1)
    parser.add_argument("--max-threshold", type=float, default=0.95)
    parser.add_argument("--step", type=float, default=0.01)
    args = parser.parse_args()

    # Load predictions
    logger.info("Loading predictions from %s...", args.predictions)
    scores_list = []
    labels_list = []
    with open(args.predictions, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                scores_list.append(float(row.get("score", 0.0)))
                labels_list.append(int(row.get("label", 0)))
            except (ValueError, TypeError):
                continue

    scores = np.array(scores_list, dtype=np.float32)
    labels = np.array(labels_list, dtype=np.int32)
    logger.info("Loaded %d prediction pairs.", len(scores))

    best_thresh, best_metrics = tune_threshold(
        scores, labels,
        min_threshold=args.min_threshold,
        max_threshold=args.max_threshold,
        step=args.step,
    )

    print(f"\n{'='*60}")
    print(f"  OPTIMAL THRESHOLD: {best_thresh:.3f}")
    print(f"  F_0.5 Score:       {best_metrics['f05']:.4f}")
    print(f"  Precision:         {best_metrics['precision']:.4f}")
    print(f"  Recall:            {best_metrics['recall']:.4f}")
    print(f"  TP={best_metrics['tp']}, FP={best_metrics['fp']}, FN={best_metrics['fn']}, TN={best_metrics['tn']}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
