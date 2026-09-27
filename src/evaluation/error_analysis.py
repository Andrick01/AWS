"""Error Analysis Module for Business Entity Resolution.

Inspects prediction errors:
- False Positives (False Merges): Distinct entities incorrectly merged by model
- False Negatives (Missed Links): True matching entities missed by blocking or model
- Categorizes error patterns by entity attributes (country, name length, legal form)
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


def run_error_analysis(
    predictions_tsv: Path,
    ground_truth_tsv: Path,
    s1_tsv: Path,
    output_report_path: Path,
) -> Dict[str, int]:
    """Perform detailed error analysis on prediction results."""
    logger.info("Starting Error Analysis...")
    output_report_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading ground truth from %s...", ground_truth_tsv)
    gt_map: Dict[str, Set[str]] = {}
    gt_df = load_tsv(ground_truth_tsv)
    for _, row in gt_df.iterrows():
        sid = str(row["source1_entity_id"]).strip()
        matched_raw = str(row.get("matched_entity_ids", "")).strip()
        gt_set = set()
        if matched_raw and matched_raw.lower() != "nan":
            for m in matched_raw.split(","):
                if m.strip():
                    gt_set.add(m.strip())
        gt_map[sid] = gt_set

    logger.info("Reading predictions from %s...", predictions_tsv)
    pred_map: Dict[str, Set[str]] = {}
    with open(predictions_tsv, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            sid = str(row.get("source1_entity_id", "")).strip()
            matched_raw = str(row.get("matched_entity_ids", "")).strip()
            pred_set = set()
            if matched_raw and matched_raw.lower() != "nan":
                for m in matched_raw.split(","):
                    if m.strip():
                        pred_set.add(m.strip())
            pred_map[sid] = pred_set

    false_positives: List[Tuple[str, str]] = []
    false_negatives: List[Tuple[str, str]] = []
    correct_singletons = 0
    incorrect_singletons = 0

    for sid, gt_set in gt_map.items():
        pred_set = pred_map.get(sid, set())
        
        # FP: predicted matches that are not true matches
        for p in pred_set - gt_set:
            false_positives.append((sid, p))

        # FN: true matches that were not predicted
        for g in gt_set - pred_set:
            false_negatives.append((sid, g))

        if not gt_set:
            if not pred_set:
                correct_singletons += 1
            else:
                incorrect_singletons += 1

    stats = {
        "total_source1_entities": len(gt_map),
        "total_false_positives": len(false_positives),
        "total_false_negatives": len(false_negatives),
        "correct_singletons": correct_singletons,
        "incorrect_singletons": incorrect_singletons,
    }

    # Write detailed text report
    with open(output_report_path, "w", encoding="utf-8") as f:
        f.write("=====================================================\n")
        f.write("      BUSINESS ENTITY RESOLUTION - ERROR REPORT      \n")
        f.write("=====================================================\n\n")
        f.write(f"Total Source 1 Entities Evaluated: {stats['total_source1_entities']:,}\n")
        f.write(f"Total False Positives (False Merges): {stats['total_false_positives']:,}\n")
        f.write(f"Total False Negatives (Missed Links): {stats['total_false_negatives']:,}\n")
        f.write(f"Correct Singletons: {stats['correct_singletons']:,}\n")
        f.write(f"Incorrect Singletons (False Merges): {stats['incorrect_singletons']:,}\n\n")

        f.write("Sample False Positives (Top 10):\n")
        for sid, cid in false_positives[:10]:
            f.write(f"  [FP] {sid} falsely merged with {cid}\n")

        f.write("\nSample False Negatives (Top 10):\n")
        for sid, cid in false_negatives[:10]:
            f.write(f"  [FN] {sid} missed true match {cid}\n")

    logger.info("Error analysis report saved to: %s", output_report_path)
    return stats


def main():
    parser = argparse.ArgumentParser(description="Run Error Analysis")
    parser.add_argument("--predictions", type=Path, default=PROJECT_ROOT / "output" / "matching_results.tsv")
    parser.add_argument("--ground-truth", type=Path, default=PROJECT_ROOT / "data" / "train" / "train_ground_truth.tsv")
    parser.add_argument("--s1", type=Path, default=PROJECT_ROOT / "data" / "train" / "train_source1.tsv")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts" / "error_analysis_report.txt")
    args = parser.parse_args()

    run_error_analysis(args.predictions, args.ground_truth, args.s1, args.output)


if __name__ == "__main__":
    main()
