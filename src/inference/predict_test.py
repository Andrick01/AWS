"""End-to-End Test Inference Engine for Business Entity Resolution.

Memory-aware pipeline:
1. Loads preprocessed (or raw+normalized) test sources
2. Partitions by country to keep peak RAM manageable
3. Generates candidates (lexical + optional TF-IDF) in S1 chunks
4. Scores pairs with the trained model
5. Writes competition submission files:
   - output/matching_results.tsv
   - output/candidate_pairs.tsv
"""

from __future__ import annotations

import argparse
import csv
import gc
import logging
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.blocking.lexical_block import LexicalBlocker
from src.blocking.tfidf_block import TFIDFBlocker
from src.features.build_features import build_pair_features, load_entity_attributes
from src.modeling.predict import validate_outputs
from src.modeling.train_model import FEATURE_COLS, load_model, predict_scores
from src.preprocessing.load_data import load_tsv
from src.preprocessing.normalize import normalize_dataframe

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

NEEDED_COLS = [
    "entity_id",
    "business_name_normalized",
    "business_address_normalized",
    "country_normalized",
    "combined_text_normalized",
]


def _load_processed_or_raw(path: Path, nrows: Optional[int] = None) -> pd.DataFrame:
    """Load a TSV; normalize if normalized columns are missing."""
    df = load_tsv(path, nrows=nrows)
    if "business_name_normalized" not in df.columns:
        logger.info("Normalizing %s...", path.name)
        df = normalize_dataframe(df)
    keep = [c for c in NEEDED_COLS if c in df.columns]
    return df[keep].copy()


def _iter_filtered_chunks(
    path: Path,
    country: Optional[str],
    chunksize: int = 250_000,
) -> Sequence[pd.DataFrame]:
    """Yield country-filtered chunks from a processed TSV (columns subset only)."""
    usecols = NEEDED_COLS
    # If raw file without normalized cols, fall back to full load+normalize (sample / small only)
    header = pd.read_csv(path, sep="\t", nrows=0)
    if "business_name_normalized" not in header.columns:
        df = _load_processed_or_raw(path)
        if country:
            df = df[df["country_normalized"].astype(str) == country]
        return [df]

    frames = []
    for chunk in pd.read_csv(
        path,
        sep="\t",
        usecols=[c for c in usecols if c in header.columns],
        chunksize=chunksize,
        dtype=str,
    ):
        if country:
            chunk = chunk[chunk["country_normalized"].fillna("").astype(str) == country]
        if not chunk.empty:
            frames.append(chunk)
    return frames


def _load_candidates_for_country(
    s2_path: Path,
    s3_path: Path,
    country: Optional[str],
    sample_size: Optional[int] = None,
) -> pd.DataFrame:
    """Load S2+S3 candidates for one country without keeping other countries in RAM."""
    parts: List[pd.DataFrame] = []
    for path in (s2_path, s3_path):
        if not path.exists():
            continue
        if sample_size is not None:
            df = _load_processed_or_raw(path, nrows=sample_size)
            if country:
                df = df[df["country_normalized"].astype(str) == country]
            if not df.empty:
                parts.append(df)
        else:
            parts.extend(_iter_filtered_chunks(path, country))
    if not parts:
        return pd.DataFrame(columns=NEEDED_COLS)
    out = pd.concat(parts, ignore_index=True)
    del parts
    gc.collect()
    return out

def _score_feature_file(
    features_tsv: Path,
    model,
    threshold: float,
) -> Dict[str, List[Tuple[str, float]]]:
    """Score pairwise features in streaming batches; return s1 -> [(cand, score), ...]."""
    matches: Dict[str, List[Tuple[str, float]]] = {}
    if not features_tsv.exists() or features_tsv.stat().st_size == 0:
        return matches

    chunk_iter = pd.read_csv(features_tsv, sep="\t", chunksize=100_000)
    for chunk in chunk_iter:
        if chunk.empty:
            continue
        X = chunk[FEATURE_COLS].fillna(0.0).to_numpy(dtype=np.float32)
        scores = predict_scores(model, X)
        for s1_id, c_id, score in zip(
            chunk["source1_entity_id"].astype(str),
            chunk["candidate_entity_id"].astype(str),
            scores,
        ):
            if float(score) < threshold:
                continue
            matches.setdefault(s1_id, []).append((c_id, float(score)))
    return matches


def _block_chunk(
    chunk: pd.DataFrame,
    lexical: LexicalBlocker,
    tfidf: Optional[TFIDFBlocker],
    lexical_max_k: int = 50,
) -> Dict[str, List[str]]:
    """Query pre-fit blockers for one S1 chunk; return sid -> candidate id list."""
    lex = lexical.query(chunk, max_candidates_per_query=lexical_max_k)
    tfidf_hits: Dict[str, set] = {}
    if tfidf is not None:
        tfidf_hits = tfidf.query_batch(chunk)

    out: Dict[str, List[str]] = {}
    for sid in chunk["entity_id"].astype(str):
        combined = set(lex.get(sid, set())) | set(tfidf_hits.get(sid, set()))
        clean = sorted(c for c in combined if c.startswith("S2-") or c.startswith("S3-"))
        out[sid] = clean
    return out


def _process_partition(
    s1_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
    model,
    threshold: float,
    s1_chunk_size: int,
    use_tfidf: bool,
    work_dir: Path,
    partition_name: str,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Block + feature + score one country partition; return cand/match maps."""
    cand_map: Dict[str, List[str]] = {}
    match_map: Dict[str, List[str]] = {}

    if s1_df.empty or candidates_df.empty:
        for sid in s1_df["entity_id"].astype(str):
            cand_map[sid] = []
            match_map[sid] = []
        return cand_map, match_map

    logger.info(
        "[%s] S1=%d candidates=%d (tfidf=%s)",
        partition_name,
        len(s1_df),
        len(candidates_df),
        use_tfidf,
    )

    # Fit blockers ONCE per country partition (critical for runtime)
    lexical = LexicalBlocker(prefix_len=4, min_token_len=3)
    lexical.fit(candidates_df)
    tfidf: Optional[TFIDFBlocker] = None
    if use_tfidf:
        tfidf = TFIDFBlocker(top_k=25, min_similarity=0.30)
        tfidf.fit(candidates_df)

    cand_lookup = load_entity_attributes(candidates_df)
    s1_ids = s1_df["entity_id"].astype(str).tolist()

    for start in range(0, len(s1_df), s1_chunk_size):
        chunk = s1_df.iloc[start : start + s1_chunk_size]
        logger.info(
            "[%s] chunk %d-%d / %d",
            partition_name,
            start,
            start + len(chunk),
            len(s1_df),
        )

        chunk_cands = _block_chunk(chunk, lexical, tfidf)

        flat_path = work_dir / f"{partition_name}_flat_{start}.tsv"
        feat_path = work_dir / f"{partition_name}_feat_{start}.tsv"

        with open(flat_path, "w", encoding="utf-8", newline="") as out_f:
            writer = csv.writer(out_f, delimiter="\t")
            writer.writerow(["source1_entity_id", "candidate_entity_id"])
            for sid, cands in chunk_cands.items():
                cand_map[sid] = cands
                for c in cands:
                    writer.writerow([sid, c])

        s1_lookup = load_entity_attributes(chunk)
        build_pair_features(
            pairs_tsv=flat_path,
            s1_lookup=s1_lookup,
            cand_lookup=cand_lookup,
            output_tsv=feat_path,
            has_label=False,
        )

        scored = _score_feature_file(feat_path, model, threshold)
        for sid, pair_list in scored.items():
            pair_list.sort(key=lambda x: -x[1])
            allowed = set(cand_map.get(sid, []))
            match_map[sid] = [c for c, _ in pair_list if c in allowed]

        for sid in chunk["entity_id"].astype(str):
            cand_map.setdefault(sid, [])
            match_map.setdefault(sid, [])

        for p in (flat_path, feat_path):
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass
        gc.collect()

    for sid in s1_ids:
        cand_map.setdefault(sid, [])
        match_map.setdefault(sid, [])

    del lexical, tfidf, cand_lookup
    gc.collect()
    return cand_map, match_map


def run_test_inference(
    test_dir: Path,
    model_path: Path,
    output_dir: Path,
    threshold: float = 0.55,
    sample_size: Optional[int] = None,
    preprocessed_dir: Optional[Path] = None,
    s1_chunk_size: int = 25_000,
    use_tfidf: bool = True,
) -> None:
    """Run memory-aware end-to-end inference on the test dataset."""
    logger.info("==================================================")
    logger.info("  STARTING END-TO-END TEST INFERENCE PIPELINE     ")
    logger.info("==================================================")

    output_dir.mkdir(parents=True, exist_ok=True)
    prep = preprocessed_dir or (PROJECT_ROOT / "artifacts" / "preprocessing")

    s1_path = prep / "test_source1_processed.tsv"
    s2_path = prep / "test_source2_processed.tsv"
    s3_path = prep / "test_source3_processed.tsv"
    raw_s1 = test_dir / "test_source1.tsv"

    if not s1_path.exists():
        s1_path = raw_s1
        s2_path = test_dir / "test_source2.tsv"
        s3_path = test_dir / "test_source3.tsv"

    logger.info("Loading model from %s...", model_path)
    model = load_model(model_path)

    logger.info("Loading Source 1 from %s...", s1_path)
    s1_df = _load_processed_or_raw(s1_path, nrows=sample_size)
    s1_order = s1_df["entity_id"].astype(str).tolist()
    logger.info("Source 1 entities: %d", len(s1_order))

    countries = sorted(
        {c for c in s1_df["country_normalized"].fillna("").astype(str).unique() if c}
    )
    if not countries:
        countries = [""]
    logger.info("Country partitions: %s", countries)

    if not s2_path.exists() and not s3_path.exists():
        raise FileNotFoundError("No Source 2 / Source 3 candidate files found")

    all_cands: Dict[str, List[str]] = {sid: [] for sid in s1_order}
    all_matches: Dict[str, List[str]] = {sid: [] for sid in s1_order}

    with tempfile.TemporaryDirectory(prefix="ber_infer_", dir=str(output_dir)) as tmp:
        work_dir = Path(tmp)
        for country in countries:
            if country:
                s1_part = s1_df[s1_df["country_normalized"].astype(str) == country]
            else:
                s1_part = s1_df

            logger.info("Loading candidates for country=%s...", country or "all")
            cand_part = _load_candidates_for_country(
                s2_path, s3_path, country if country else None, sample_size=sample_size
            )
            logger.info("Candidates loaded for %s: %d", country or "all", len(cand_part))

            cand_map, match_map = _process_partition(
                s1_df=s1_part,
                candidates_df=cand_part,
                model=model,
                threshold=threshold,
                s1_chunk_size=s1_chunk_size,
                use_tfidf=use_tfidf,
                work_dir=work_dir,
                partition_name=country or "all",
            )
            all_cands.update(cand_map)
            all_matches.update(match_map)
            del s1_part, cand_part, cand_map, match_map
            gc.collect()

    del s1_df
    gc.collect()

    matching_output = output_dir / "matching_results.tsv"
    candidate_output = output_dir / "candidate_pairs.tsv"

    logger.info("Writing %s (%d rows)...", matching_output, len(s1_order))
    with open(matching_output, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["source1_entity_id", "matched_entity_ids"])
        for sid in s1_order:
            writer.writerow([sid, ",".join(all_matches.get(sid, []))])

    logger.info("Writing %s...", candidate_output)
    with open(candidate_output, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["source1_entity_id", "candidate_entity_ids"])
        for sid in s1_order:
            writer.writerow([sid, ",".join(all_cands.get(sid, []))])

    # Validate against full raw test S1 when sample_size is unset
    validate_path = raw_s1 if sample_size is None and raw_s1.exists() else None
    if validate_path is not None:
        validate_outputs(matching_output, candidate_output, validate_path)
    else:
        logger.info("Skipping full-set audit (sample run). Row count=%d", len(s1_order))

    logger.info("Inference completed successfully!")
    logger.info("  1. %s", matching_output)
    logger.info("  2. %s", candidate_output)


def main():
    parser = argparse.ArgumentParser(description="Run End-to-End Test Inference")
    parser.add_argument("--test-dir", type=Path, default=PROJECT_ROOT / "data" / "test")
    parser.add_argument("--model-path", type=Path, default=PROJECT_ROOT / "models" / "final_model" / "model.pkl")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "output")
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument(
        "--preprocessed-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "preprocessing",
    )
    parser.add_argument("--s1-chunk-size", type=int, default=25_000)
    parser.add_argument(
        "--lexical-only",
        action="store_true",
        help="Disable TF-IDF blocking to reduce peak memory",
    )
    args = parser.parse_args()

    run_test_inference(
        test_dir=args.test_dir,
        model_path=args.model_path,
        output_dir=args.output_dir,
        threshold=args.threshold,
        sample_size=args.sample_size,
        preprocessed_dir=args.preprocessed_dir,
        s1_chunk_size=args.s1_chunk_size,
        use_tfidf=not args.lexical_only,
    )


if __name__ == "__main__":
    main()
