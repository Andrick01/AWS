"""TF-IDF and Character N-Gram Fuzzy Blocking for Business Entity Resolution.

Uses character 3-grams and sparse matrix cosine similarity to retrieve
top-K candidate matches tolerant to typos, word order flips, and slight misspellings.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


class TFIDFBlocker:
    """Performs scalable fuzzy candidate retrieval using character n-gram TF-IDF and sparse cosine similarity."""

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 1,
        top_k: int = 25,
        min_similarity: float = 0.35,
    ):
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.top_k = top_k
        self.min_similarity = min_similarity

        # Common high-frequency address tokens that should not trigger false-positive candidate retrieval alone
        stop_words = ["st", "rd", "ave", "dr", "ln", "blvd", "apt", "ste", "fl", "us", "india", "france"]
        self.vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=self.ngram_range,
            min_df=max(self.min_df, 2),
            max_features=250000,
            stop_words=stop_words,
            dtype=np.float32,
            norm="l2",
        )
        self.candidate_ids: List[str] = []
        self.candidate_matrix: Optional[csr_matrix] = None
        self.candidate_countries: List[str] = []

    def fit(
        self,
        candidate_df: pd.DataFrame,
        id_col: str = "entity_id",
        text_col: str = "combined_text_normalized",
        country_col: str = "country_normalized",
    ) -> "TFIDFBlocker":
        """Fit TF-IDF vectorizer and transform candidate texts (Source 2 and Source 3)."""
        logger.info("Fitting TF-IDF on %d candidate records...", len(candidate_df))
        self.candidate_ids = candidate_df[id_col].astype(str).tolist()
        self.candidate_countries = candidate_df[country_col].fillna("").astype(str).tolist()

        texts = candidate_df[text_col].fillna("").astype(str).tolist()
        # candidate_matrix in CSC format allows fast multiplication: candidate_matrix.dot(q_T)
        self.candidate_matrix = self.vectorizer.fit_transform(texts).tocsc()
        logger.info(
            "TF-IDF matrix built: %d records, %d n-gram features.",
            self.candidate_matrix.shape[0],
            self.candidate_matrix.shape[1],
        )
        return self

    def query_batch(
        self,
        query_df: pd.DataFrame,
        id_col: str = "entity_id",
        text_col: str = "combined_text_normalized",
        country_col: str = "country_normalized",
        batch_size: int = 200,
    ) -> Dict[str, Set[str]]:
        """Query top-K candidates for query records (Source 1) with safe memory footprint."""
        if self.candidate_matrix is None:
            raise ValueError("TFIDFBlocker must be fit before querying.")

        logger.info("Querying TF-IDF candidates for %d records (top_k=%d)...", len(query_df), self.top_k)
        candidates_per_query: Dict[str, Set[str]] = {}

        query_ids = query_df[id_col].astype(str).tolist()
        query_texts = query_df[text_col].fillna("").astype(str).tolist()
        query_countries = query_df[country_col].fillna("").astype(str).tolist()
        n_queries = len(query_df)

        candidate_ids_arr = np.array(self.candidate_ids)
        candidate_countries_arr = np.array(self.candidate_countries)

        for i in range(0, n_queries, batch_size):
            if i > 0 and i % 25000 == 0:
                logger.info("  TF-IDF query progress: %d / %d (%.1f%%)", i, n_queries, i / n_queries * 100)
            b_ids = query_ids[i : i + batch_size]
            b_texts = query_texts[i : i + batch_size]
            b_countries = query_countries[i : i + batch_size]

            q_matrix = self.vectorizer.transform(b_texts)
            # cand (N, V) dot q.T (V, B) -> sim_csc (N, B)
            sim_csc = self.candidate_matrix.dot(q_matrix.T).tocsc()

            # For each column j corresponding to query entity j in batch
            for j, (q_id, q_country) in enumerate(zip(b_ids, b_countries)):
                start = sim_csc.indptr[j]
                end = sim_csc.indptr[j + 1]
                if start == end:
                    candidates_per_query[q_id] = set()
                    continue

                col_indices = sim_csc.indices[start:end]
                similarities = sim_csc.data[start:end]

                # Filter by similarity threshold
                valid_mask = similarities >= self.min_similarity
                if not np.any(valid_mask):
                    candidates_per_query[q_id] = set()
                    continue

                col_indices = col_indices[valid_mask]
                similarities = similarities[valid_mask]

                # Filter by matching country if provided
                if q_country and q_country != "":
                    cand_c = candidate_countries_arr[col_indices]
                    country_mask = (cand_c == "") | (cand_c == q_country)
                    col_indices = col_indices[country_mask]
                    similarities = similarities[country_mask]

                if len(col_indices) == 0:
                    candidates_per_query[q_id] = set()
                    continue

                # Select top-k highest similarity candidates
                if len(similarities) > self.top_k:
                    top_idx = np.argpartition(similarities, -self.top_k)[-self.top_k :]
                    selected_cand_indices = col_indices[top_idx]
                else:
                    selected_cand_indices = col_indices

                candidates_per_query[q_id] = set(candidate_ids_arr[selected_cand_indices])

        logger.info("TF-IDF query complete: %d queries processed.", n_queries)
        return candidates_per_query
