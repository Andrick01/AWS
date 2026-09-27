"""Feature Vector Construction Orchestrator for Business Entity Resolution.

Joins candidate pairs with preprocessed entity attributes (Source 1, Source 2, Source 3),
computes comprehensive pairwise similarity features, and outputs training/evaluation feature tables.
"""

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
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

from src.features.address_features import compute_address_features
from src.features.country_features import compute_country_features
from src.features.name_features import compute_name_features
from src.preprocessing.load_data import load_tsv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_entity_attributes(df: pd.DataFrame) -> Dict[str, Tuple[str, str, str]]:
    """Index entity attributes into a fast lookup: id -> (name_norm, addr_norm, country_norm)."""
    lookup: Dict[str, Tuple[str, str, str]] = {}
    ids = df["entity_id"].astype(str).tolist()
    names = df["business_name_normalized"].fillna("").astype(str).tolist()
    addrs = df["business_address_normalized"].fillna("").astype(str).tolist() if "business_address_normalized" in df.columns else [""] * len(ids)
    countries = df["country_normalized"].fillna("").astype(str).tolist() if "country_normalized" in df.columns else [""] * len(ids)

    for eid, n, a, c in zip(ids, names, addrs, countries):
        lookup[eid] = (n, a, c)
    return lookup


def build_pair_features(
    pairs_tsv: Path,
    s1_lookup: Dict[str, Tuple[str, str, str]],
    cand_lookup: Dict[str, Tuple[str, str, str]],
    output_tsv: Path,
    batch_size: int = 50000,
    has_label: bool = True,
    max_pairs: Optional[int] = None,
) -> int:
    """Stream pairwise dataset, compute all feature metrics, and write feature matrix to TSV."""
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Extracting pairwise features: %s -> %s", pairs_tsv, output_tsv)

    feature_names = [
        # Name features
        "name_exact_match", "name_seq_ratio", "name_jaro_winkler",
        "name_levenshtein_sim", "name_token_jaccard", "name_token_overlap_ratio",
        "name_prefix_match_3", "name_prefix_match_5", "name_len_diff_ratio",
        # Address features
        "addr_exact_match", "addr_seq_ratio", "addr_token_jaccard",
        "addr_token_overlap_ratio", "addr_numeric_exact_match", "addr_numeric_overlap_ratio",
        "addr_has_both_address",
        # Country / Source features
        "country_match", "country_mismatch", "is_us", "is_india", "is_france",
        "is_source2", "is_source3",
    ]

    header = ["source1_entity_id", "candidate_entity_id"] + feature_names
    if has_label:
        header.append("label")

    total_pairs = 0

    with open(pairs_tsv, "r", encoding="utf-8", errors="replace") as in_f, \
         open(output_tsv, "w", encoding="utf-8", newline="") as out_f:

        reader = csv.DictReader(in_f, delimiter="\t")
        writer = csv.writer(out_f, delimiter="\t")
        writer.writerow(header)

        for row in reader:
            s1_id = row.get("source1_entity_id", "").strip()
            c_id = row.get("candidate_entity_id", "").strip()
            if not s1_id or not c_id:
                continue

            total_pairs += 1
            if max_pairs and total_pairs > max_pairs:
                break

            s1_attr = s1_lookup.get(s1_id, ("", "", ""))
            c_attr = cand_lookup.get(c_id, ("", "", ""))

            n_feats = compute_name_features(s1_attr[0], c_attr[0])
            a_feats = compute_address_features(s1_attr[1], c_attr[1])
            c_feats = compute_country_features(s1_attr[2], c_attr[2], candidate_id=c_id)

            row_values = [s1_id, c_id]
            for f in feature_names:
                if f in n_feats:
                    row_values.append(round(n_feats[f], 4))
                elif f in a_feats:
                    row_values.append(round(a_feats[f], 4))
                elif f in c_feats:
                    row_values.append(round(c_feats[f], 4))
                else:
                    row_values.append(0.0)

            if has_label:
                row_values.append(int(row.get("label", 0)))

            writer.writerow(row_values)

            if total_pairs % batch_size == 0:
                logger.info("Extracted features for %d pairs...", total_pairs)

    logger.info("Feature extraction complete: %d total pairs -> %s", total_pairs, output_tsv)
    return total_pairs


def main():
    parser = argparse.ArgumentParser(description="Feature Extraction Pipeline")
    parser.add_argument(
        "--pairs",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "test_training_pairs.tsv",
        help="Input pairs TSV (source1_entity_id, candidate_entity_id, [label])",
    )
    parser.add_argument(
        "--source1",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "preprocessing" / "val_source1_processed.tsv",
        help="Source 1 processed entities TSV",
    )
    parser.add_argument(
        "--source2",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "preprocessing" / "source2_processed.tsv",
        help="Source 2 processed entities TSV",
    )
    parser.add_argument(
        "--source3",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "preprocessing" / "source3_processed.tsv",
        help="Source 3 processed entities TSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "pair_features.tsv",
        help="Output features TSV path",
    )
    parser.add_argument(
        "--no-label",
        action="store_true",
        help="Flag if input pairs has no label column (e.g. test inference)",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Limit number of pairs to extract for testing",
    )

    args = parser.parse_args()

    logger.info("Loading entity attribute tables for feature extraction...")
    s1_df = load_tsv(args.source1)
    s1_lookup = load_entity_attributes(s1_df)
    del s1_df

    cand_lookup = {}
    if args.source2.exists():
        s2_df = load_tsv(args.source2)
        cand_lookup.update(load_entity_attributes(s2_df))
        del s2_df

    if args.source3.exists():
        s3_df = load_tsv(args.source3)
        cand_lookup.update(load_entity_attributes(s3_df))
        del s3_df

    build_pair_features(
        pairs_tsv=args.pairs,
        s1_lookup=s1_lookup,
        cand_lookup=cand_lookup,
        output_tsv=args.output,
        has_label=not args.no_label,
        max_pairs=args.max_pairs,
    )


if __name__ == "__main__":
    main()
