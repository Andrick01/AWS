"""Lexical and Rule-Based Blocking for Business Entity Resolution.

Generates candidate pairs using deterministic, hash-based indexing:
- First-word token keys (e.g. 'acme')
- Brand 3-character prefixes (e.g. 'acm')
- Exact name keys
- Country-partitioned candidate filtering
"""

from collections import defaultdict
import logging
import re
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# Stopwords and noisy terms that are too frequent for high-quality single-token blocking
COMMON_BLOCKING_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "in", "at", "by", "for", "with",
    "inc", "corp", "corporation", "incorporated", "ltd", "limited", "pvt", "llc", "co", "company"
}


def extract_blocking_tokens(
    name: str,
    address: str = "",
    min_token_len: int = 3,
    prefix_len: int = 4,
) -> Set[str]:
    """Generate multiple candidate blocking keys for a normalized entity name and address.
    
    Keys generated:
    1. Entire clean name (for exact matches)
    2. Tokens for first 3 words
    3. Prefixes for tokens (>= 4 chars)
    4. Address number + first token prefix combination (high recall & precision)
    """
    keys = set()
    if not name or not isinstance(name, str):
        return keys

    clean_name = name.strip()
    if not clean_name:
        return keys

    # Exact normalized name key
    keys.add(f"exact:{clean_name}")

    tokens = [tok for tok in clean_name.split() if tok not in COMMON_BLOCKING_STOPWORDS and len(tok) >= min_token_len]
    for idx, tok in enumerate(tokens[:3]):
        keys.add(f"tok:{tok}")
        if len(tok) >= prefix_len:
            keys.add(f"pref:{tok[:prefix_len]}")

    # Address number feature
    if address and isinstance(address, str):
        clean_addr = address.strip()
        nums = re.findall(r"\b\d{2,6}\b", clean_addr)
        if nums and tokens:
            keys.add(f"num_name:{nums[0]}_{tokens[0][:prefix_len]}")

    return keys


class LexicalBlocker:
    """Builds a country-partitioned inverted index of candidate entities and retrieves lexical matches."""

    def __init__(self, prefix_len: int = 4, min_token_len: int = 3):
        self.prefix_len = prefix_len
        self.min_token_len = min_token_len
        # Maps country -> blocking_key -> set of candidate entity_ids
        self.country_index: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))

    def fit(
        self,
        candidate_df: pd.DataFrame,
        id_col: str = "entity_id",
        name_col: str = "business_name_normalized",
        addr_col: str = "business_address_normalized",
        country_col: str = "country_normalized",
        max_bucket_size: int = 5000,
    ) -> "LexicalBlocker":
        """Index candidate records (Source 2 and Source 3) partitioned by country."""
        logger.info("Building lexical index for %d candidate records...", len(candidate_df))
        self.country_index.clear()

        ids = candidate_df[id_col].astype(str).tolist()
        names = candidate_df[name_col].fillna("").astype(str).tolist()
        addrs = candidate_df[addr_col].fillna("").astype(str).tolist() if addr_col in candidate_df.columns else [""] * len(ids)
        countries = candidate_df[country_col].fillna("").astype(str).tolist()

        for c_id, name, addr, country in zip(ids, names, addrs, countries):
            c_key = country.strip().lower() if country else "ALL"
            keys = extract_blocking_tokens(
                name,
                address=addr,
                min_token_len=self.min_token_len,
                prefix_len=self.prefix_len,
            )
            for k in keys:
                self.country_index[c_key][k].add(c_id)

        # Prune hot keys per country
        total_keys = 0
        for c_key, idx in self.country_index.items():
            hot_keys = [k for k, v in idx.items() if len(v) > max_bucket_size]
            for k in hot_keys:
                del idx[k]
            total_keys += len(idx)

        logger.info("Lexical index built with %d unique keys across %d country partitions.", total_keys, len(self.country_index))
        return self

    def query(
        self,
        query_df: pd.DataFrame,
        id_col: str = "entity_id",
        name_col: str = "business_name_normalized",
        addr_col: str = "business_address_normalized",
        country_col: str = "country_normalized",
        max_candidates_per_query: int = 50,
    ) -> Dict[str, Set[str]]:
        """Retrieve candidate IDs for each query record (Source 1)."""
        total = len(query_df)
        logger.info("Querying lexical index for %d query records...", total)
        candidates_per_query: Dict[str, Set[str]] = {}

        q_ids = query_df[id_col].astype(str).tolist()
        q_names = query_df[name_col].fillna("").astype(str).tolist()
        q_addrs = query_df[addr_col].fillna("").astype(str).tolist() if addr_col in query_df.columns else [""] * len(q_ids)
        q_countries = query_df[country_col].fillna("").astype(str).tolist()

        for i, (q_id, name, addr, country) in enumerate(zip(q_ids, q_names, q_addrs, q_countries)):
            if i > 0 and i % 250000 == 0:
                logger.info("  Lexical query progress: %d / %d (%.1f%%)", i, total, i / total * 100)

            keys = extract_blocking_tokens(
                name,
                address=addr,
                min_token_len=self.min_token_len,
                prefix_len=self.prefix_len,
            )

            c_key = country.strip().lower() if country else "ALL"
            target_idx = self.country_index.get(c_key, {})
            all_idx = self.country_index.get("ALL", {})

            cand_scores: Dict[str, int] = defaultdict(int)
            for k in keys:
                weight = 10 if k.startswith("exact:") else (5 if k.startswith("num_name:") else (3 if k.startswith("tok:") else 1))
                if k in target_idx and len(target_idx[k]) <= 1000:
                    for cid in target_idx[k]:
                        cand_scores[cid] += weight
                if k in all_idx and c_key != "ALL" and len(all_idx[k]) <= 1000:
                    for cid in all_idx[k]:
                        cand_scores[cid] += weight

            if max_candidates_per_query and len(cand_scores) > max_candidates_per_query:
                top_cands = sorted(cand_scores.keys(), key=lambda c: -cand_scores[c])[:max_candidates_per_query]
                candidates_per_query[q_id] = set(top_cands)
            else:
                candidates_per_query[q_id] = set(cand_scores.keys())

        logger.info("Lexical query complete: %d queries processed.", total)
        return candidates_per_query
