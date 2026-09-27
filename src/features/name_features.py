"""Business name similarity feature extractors for Business Entity Resolution.

Computes exact, fuzzy, token-based, and character n-gram similarities
between Source 1 and candidate business names.
"""

from difflib import SequenceMatcher
from typing import Dict, List, Set

from rapidfuzz.distance import Levenshtein, JaroWinkler


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Compute fast C-accelerated Levenshtein edit distance using rapidfuzz."""
    return Levenshtein.distance(s1, s2)


def _jaro_winkler_similarity(s1: str, s2: str, p: float = 0.1, max_l: int = 4) -> float:
    """Compute fast C-accelerated Jaro-Winkler similarity using rapidfuzz."""
    return JaroWinkler.similarity(s1, s2, prefix_weight=p)


def compute_name_features(name1: str, name2: str) -> Dict[str, float]:
    """Compute all name-level similarity metrics for a pair of entities."""
    s1 = str(name1).strip().lower() if name1 else ""
    s2 = str(name2).strip().lower() if name2 else ""

    # Handle empty cases
    if not s1 or not s2:
        return {
            "name_exact_match": 1.0 if s1 == s2 and s1 != "" else 0.0,
            "name_seq_ratio": 0.0,
            "name_jaro_winkler": 0.0,
            "name_levenshtein_sim": 0.0,
            "name_token_jaccard": 0.0,
            "name_token_overlap_ratio": 0.0,
            "name_prefix_match_3": 0.0,
            "name_prefix_match_5": 0.0,
            "name_len_diff_ratio": 0.0,
        }

    # 1. Exact match
    exact_match = 1.0 if s1 == s2 else 0.0

    # 2. SequenceMatcher ratio (Gestalt Pattern Matching)
    seq_ratio = SequenceMatcher(None, s1, s2).ratio()

    # 3. Jaro-Winkler similarity
    jw_sim = _jaro_winkler_similarity(s1, s2)

    # 4. Normalized Levenshtein similarity
    max_len = max(len(s1), len(s2))
    lev_dist = _levenshtein_distance(s1, s2)
    lev_sim = 1.0 - (lev_dist / max_len) if max_len > 0 else 0.0

    # 5. Token Jaccard & Token Overlap
    tokens1 = set(s1.split())
    tokens2 = set(s2.split())
    intersection = tokens1 & tokens2
    union = tokens1 | tokens2
    token_jaccard = len(intersection) / len(union) if union else 0.0
    min_tokens = min(len(tokens1), len(tokens2))
    token_overlap = len(intersection) / min_tokens if min_tokens > 0 else 0.0

    # 6. Prefix matches
    prefix_3 = 1.0 if len(s1) >= 3 and len(s2) >= 3 and s1[:3] == s2[:3] else 0.0
    prefix_5 = 1.0 if len(s1) >= 5 and len(s2) >= 5 and s1[:5] == s2[:5] else 0.0

    # 7. Length difference ratio
    len_diff_ratio = abs(len(s1) - len(s2)) / max_len if max_len > 0 else 0.0

    return {
        "name_exact_match": exact_match,
        "name_seq_ratio": float(seq_ratio),
        "name_jaro_winkler": float(jw_sim),
        "name_levenshtein_sim": float(lev_sim),
        "name_token_jaccard": float(token_jaccard),
        "name_token_overlap_ratio": float(token_overlap),
        "name_prefix_match_3": prefix_3,
        "name_prefix_match_5": prefix_5,
        "name_len_diff_ratio": float(len_diff_ratio),
    }
