"""Business name similarity feature extractors for Business Entity Resolution.

Computes exact, fuzzy, token-based, and character n-gram similarities
between Source 1 and candidate business names.
"""

from difflib import SequenceMatcher
from typing import Dict, List, Set


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Compute standard Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def _jaro_distance(s1: str, s2: str) -> float:
    """Compute Jaro distance between two strings."""
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0

    max_dist = max(len1, len2) // 2 - 1
    match1 = [False] * len1
    match2 = [False] * len2
    matches = 0

    for i in range(len1):
        start = max(0, i - max_dist)
        end = min(i + max_dist + 1, len2)
        for j in range(start, end):
            if match2[j] or s1[i] != s2[j]:
                continue
            match1[i] = True
            match2[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    transpositions = 0
    k = 0
    for i in range(len1):
        if not match1[i]:
            continue
        while not match2[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    transpositions //= 2
    return (matches / len1 + matches / len2 + (matches - transpositions) / matches) / 3.0


def _jaro_winkler_similarity(s1: str, s2: str, p: float = 0.1, max_l: int = 4) -> float:
    """Compute Jaro-Winkler similarity."""
    j_dist = _jaro_distance(s1, s2)
    prefix_len = 0
    for c1, c2 in zip(s1[:max_l], s2[:max_l]):
        if c1 == c2:
            prefix_len += 1
        else:
            break
    return j_dist + prefix_len * p * (1.0 - j_dist)


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
