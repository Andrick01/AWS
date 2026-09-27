"""Candidate generation orchestrator.

Combines lexical and TF-IDF blocking to produce candidate_pairs.tsv.
Ensures:
- Exactly 1 row per Source 1 entity in query set
- Empty candidate_entity_ids for entities with no candidates
- No duplicate candidate IDs
- Only Source 2 (S2-) and Source 3 (S3-) IDs
"""

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd

# UTF-8 terminal support for Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.blocking.lexical_block import LexicalBlocker
from src.blocking.tfidf_block import TFIDFBlocker
from src.preprocessing.load_data import load_tsv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_candidate_pairs(
    source1_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
    tfidf_top_k: int = 25,
    lexical_max_k: int = 50,
    tfidf_min_similarity: float = 0.30,
    use_tfidf: bool = True,
) -> pd.DataFrame:
    """Generate union of candidates from Lexical and TF-IDF blocking engines."""
    logger.info("Starting candidate generation...")
    logger.info("Source 1 count: %d, Candidate pool count: %d", len(source1_df), len(candidates_df))

    # 1. Lexical blocking
    lexical = LexicalBlocker(prefix_len=4, min_token_len=3)
    lexical.fit(candidates_df)
    lex_candidates = lexical.query(source1_df, max_candidates_per_query=lexical_max_k)

    # 2. TF-IDF fuzzy blocking (optional)
    tfidf_candidates = {}
    if use_tfidf:
        tfidf = TFIDFBlocker(top_k=tfidf_top_k, min_similarity=tfidf_min_similarity)
        tfidf.fit(candidates_df)
        tfidf_candidates = tfidf.query_batch(source1_df)

    # 3. Union candidates per Source 1 entity
    s1_ids = source1_df["entity_id"].astype(str).tolist()
    rows = []

    for s1_id in s1_ids:
        c1 = lex_candidates.get(s1_id, set())
        c2 = tfidf_candidates.get(s1_id, set())
        combined = c1.union(c2)

        # Ensure only S2 and S3 IDs and deduplicated
        clean_cand_ids = sorted([
            cid for cid in combined
            if cid.startswith("S2-") or cid.startswith("S3-")
        ])

        rows.append({
            "source1_entity_id": s1_id,
            "candidate_entity_ids": ",".join(clean_cand_ids),
        })

    result_df = pd.DataFrame(rows)
    logger.info("Generated candidate pairs for %d Source 1 entities.", len(result_df))
    return result_df


def save_candidate_pairs(df: pd.DataFrame, output_path: Path) -> None:
    """Save candidates to candidate_pairs.tsv in exact competition format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        output_path,
        sep="\t",
        index=False,
        quoting=csv.QUOTE_NONE,
        encoding="utf-8",
    )
    logger.info("Candidate pairs saved to: %s", output_path)


def run_demo(output_path: Path) -> pd.DataFrame:
    """Run candidate generation on benchmark demo dataset from video slide 2."""
    from src.preprocessing.normalize import normalize_dataframe

    logger.info("Running candidate generation in DEMO mode...")
    s1 = pd.DataFrame({
        "entity_id": ["S1-732914", "S1-889301", "S1-410562", "S1-205774"],
        "business_name": ["Acme Robotics Inc", "Delta Foods", "Bright Cafe LLC", "Zen Traders"],
        "business_address": ["500 Market St, San Jose", "8 Oak Ave, Austin", "22 Pine St, Reno", "4 Hill Rd, Boise"],
        "country": ["US", "US", "US", "US"],
    })

    candidates = pd.DataFrame({
        "entity_id": [
            "S2-118820", "S2-540221", "S3-905477", "S3-063118",
            "S2-397155", "S3-651230", "S2-663049", "S3-472088",
        ],
        "business_name": [
            "Acme Robotics Inc", "Acme Robotix", "Acme Robotics", "Acme Bakery",
            "Delta Foods Co", "Delta Foods Ltd", "Bright Cafe", "Kappa Motors",
        ],
        "business_address": [
            "500 Market Street, San Jose", "12 Elm Rd, San Jose", "Nr City Hall, San Jose", "500 Market St, San Jose",
            "8 Oak Avenue, Austin", "Oak Ave, Austin", "22 Pine Street, Reno", "90 Lake Dr, Fargo",
        ],
        "country": ["US", "US", "US", "US", "US", "US", "US", "US"],
    })

    s1_norm = normalize_dataframe(s1)
    cand_norm = normalize_dataframe(candidates)

    pairs_df = generate_candidate_pairs(
        s1_norm,
        cand_norm,
        tfidf_top_k=10,
        lexical_max_k=10,
        tfidf_min_similarity=0.35,
    )

    save_candidate_pairs(pairs_df, output_path)
    return pairs_df


def main():
    parser = argparse.ArgumentParser(description="Candidate Pair Generation (Blocking)")
    parser.add_argument("--source1", type=Path, default=None, help="Path to Source 1 TSV file")
    parser.add_argument("--source2", type=Path, default=None, help="Path to Source 2 TSV file")
    parser.add_argument("--source3", type=Path, default=None, help="Path to Source 3 TSV file")
    parser.add_argument("--output", type=Path, default=Path("artifacts/candidate_pairs.tsv"), help="Output TSV path")
    parser.add_argument("--tfidf-top-k", type=int, default=25, help="Top K candidates for TF-IDF blocker")
    parser.add_argument("--lexical-max-k", type=int, default=50, help="Max candidates per query for Lexical blocker")
    parser.add_argument("--min-similarity", type=float, default=0.35, help="Minimum TF-IDF similarity threshold")
    parser.add_argument("--sample-size", type=int, default=None, help="Limit rows for testing")
    parser.add_argument("--skip-tfidf", action="store_true", help="Skip TF-IDF blocker and use lexical blocking only")
    parser.add_argument("--demo", action="store_true", help="Run demonstration benchmark mode")

    args = parser.parse_args()

    # Search for default input files if none specified
    s1_path = args.source1
    if s1_path is None:
        candidates_to_check = [
            PROJECT_ROOT / "artifacts" / "preprocessing" / "val_source1_processed.tsv",
            PROJECT_ROOT / "artifacts" / "preprocessing" / "train_source1_processed.tsv",
            PROJECT_ROOT / "artifacts" / "preprocessing" / "source1_processed.tsv",
            PROJECT_ROOT / "data" / "train" / "train_source1.tsv",
            PROJECT_ROOT / "Train" / "train_source1.tsv",
            PROJECT_ROOT / "train" / "train_source1.tsv",
        ]
        for p in candidates_to_check:
            if p.exists():
                s1_path = p
                break

    if args.demo or s1_path is None or not s1_path.exists():
        if not args.demo:
            logger.info("No source dataset found at standard paths. Running verification demo...")
        pairs_df = run_demo(args.output)
        print("\nCandidate Pairs Output:")
        for _, row in pairs_df.iterrows():
            print(f"  {row['source1_entity_id']} -> {row['candidate_entity_ids'] if row['candidate_entity_ids'] else '(singleton - no match)'}")
        print(f"\nSaved candidates to: {args.output}\n")
        return

    logger.info("Loading Source 1 from: %s", s1_path)
    s1_df = load_tsv(s1_path, nrows=args.sample_size)

    # Resolve Source 2 & 3
    s2_path = args.source2 or (PROJECT_ROOT / "artifacts" / "preprocessing" / "source2_processed.tsv")
    if not s2_path.exists():
        s2_path = PROJECT_ROOT / "data" / "train" / "train_source2.tsv"
    if not s2_path.exists():
        s2_path = PROJECT_ROOT / "Train" / "train_source2.tsv"
    if not s2_path.exists():
        s2_path = PROJECT_ROOT / "train" / "train_source2.tsv"

    s3_path = args.source3 or (PROJECT_ROOT / "artifacts" / "preprocessing" / "source3_processed.tsv")
    if not s3_path.exists():
        s3_path = PROJECT_ROOT / "data" / "train" / "train_source3.tsv"
    if not s3_path.exists():
        s3_path = PROJECT_ROOT / "Train" / "train_source3.tsv"
    if not s3_path.exists():
        s3_path = PROJECT_ROOT / "train" / "train_source3.tsv"

    candidate_dfs = []
    if s2_path.exists():
        logger.info("Loading Source 2 from: %s", s2_path)
        candidate_dfs.append(load_tsv(s2_path, nrows=args.sample_size))
    if s3_path.exists():
        logger.info("Loading Source 3 from: %s", s3_path)
        candidate_dfs.append(load_tsv(s3_path, nrows=args.sample_size))

    if not candidate_dfs:
        raise FileNotFoundError(f"Neither Source 2 nor Source 3 could be found at {s2_path} / {s3_path}")

    candidates_df = pd.concat(candidate_dfs, ignore_index=True)

    pairs_df = generate_candidate_pairs(
        source1_df=s1_df,
        candidates_df=candidates_df,
        tfidf_top_k=args.tfidf_top_k,
        lexical_max_k=args.lexical_max_k,
        tfidf_min_similarity=args.min_similarity,
        use_tfidf=not args.skip_tfidf,
    )

    save_candidate_pairs(pairs_df, args.output)
    print("\nSample Output (first 5 rows):")
    print(pairs_df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()

