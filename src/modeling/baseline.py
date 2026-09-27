"""Rule-based baseline matcher for Business Entity Resolution.

Uses handcrafted thresholds on name/address similarity features to produce
baseline match predictions. Useful for sanity-checking data quality before
training ML models.
"""

import csv
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def baseline_score(features: Dict[str, float]) -> float:
    """Compute a simple weighted composite score from pairwise features.

    This is a hand-tuned heuristic baseline, NOT a trained model.
    Weights emphasize name similarity (most discriminative) and address numeric overlap.
    """
    score = (
        0.30 * features.get("name_jaro_winkler", 0.0)
        + 0.20 * features.get("name_token_jaccard", 0.0)
        + 0.15 * features.get("name_levenshtein_sim", 0.0)
        + 0.10 * features.get("addr_token_jaccard", 0.0)
        + 0.10 * features.get("addr_numeric_overlap_ratio", 0.0)
        + 0.05 * features.get("name_prefix_match_5", 0.0)
        + 0.05 * features.get("country_match", 0.0)
        + 0.05 * features.get("addr_seq_ratio", 0.0)
    )
    return score


def run_baseline(
    features_tsv: Path,
    output_tsv: Path,
    threshold: float = 0.55,
    max_rows: Optional[int] = None,
) -> Dict[str, int]:
    """Apply baseline scoring to a feature matrix and write predictions."""
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Running baseline matcher: threshold=%.2f", threshold)

    tp, fp, tn, fn = 0, 0, 0, 0
    total = 0

    with open(features_tsv, "r", encoding="utf-8", errors="replace") as in_f, \
         open(output_tsv, "w", encoding="utf-8", newline="") as out_f:

        reader = csv.DictReader(in_f, delimiter="\t")
        writer = csv.writer(out_f, delimiter="\t")
        writer.writerow(["source1_entity_id", "candidate_entity_id", "score", "prediction", "label"])

        for row in reader:
            total += 1
            if max_rows and total > max_rows:
                break

            features = {}
            for key, val in row.items():
                if key not in ("source1_entity_id", "candidate_entity_id", "label"):
                    try:
                        features[key] = float(val)
                    except (ValueError, TypeError):
                        features[key] = 0.0

            score = baseline_score(features)
            pred = 1 if score >= threshold else 0
            label = int(row.get("label", 0))

            if pred == 1 and label == 1:
                tp += 1
            elif pred == 1 and label == 0:
                fp += 1
            elif pred == 0 and label == 1:
                fn += 1
            else:
                tn += 1

            writer.writerow([
                row.get("source1_entity_id", ""),
                row.get("candidate_entity_id", ""),
                round(score, 4),
                pred,
                label,
            ])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f05 = (1.25 * precision * recall) / (0.25 * precision + recall) if (0.25 * precision + recall) > 0 else 0.0

    stats = {
        "total_pairs": total,
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f05_score": round(f05, 4),
    }
    logger.info("Baseline results: %s", stats)
    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Baseline Matcher")
    parser.add_argument("--features", type=Path, default=PROJECT_ROOT / "artifacts" / "test_pair_features.tsv")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts" / "baseline_predictions.tsv")
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    run_baseline(args.features, args.output, threshold=args.threshold, max_rows=args.max_rows)
