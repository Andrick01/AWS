"""Business address similarity feature extractors for Business Entity Resolution.

Computes street number overlap, token Jaccard, fuzzy sequence ratio, and length features.
"""

import re
from typing import Dict, List, Set
from rapidfuzz import fuzz


def _extract_numbers(text: str) -> Set[str]:
    """Extract all numeric digits/tokens (e.g. street numbers, unit numbers, zip codes)."""
    return set(re.findall(r"\b\d+\b", str(text)))


def compute_address_features(addr1: str, addr2: str) -> Dict[str, float]:
    """Compute all address-level similarity metrics for a pair of entities."""
    a1 = str(addr1).strip().lower() if addr1 else ""
    a2 = str(addr2).strip().lower() if addr2 else ""

    if not a1 or not a2:
        return {
            "addr_exact_match": 1.0 if a1 == a2 and a1 != "" else 0.0,
            "addr_seq_ratio": 0.0,
            "addr_token_jaccard": 0.0,
            "addr_token_overlap_ratio": 0.0,
            "addr_numeric_exact_match": 0.0,
            "addr_numeric_overlap_ratio": 0.0,
            "addr_has_both_address": 0.0 if not (a1 and a2) else 1.0,
        }

    # 1. Exact match
    exact_match = 1.0 if a1 == a2 else 0.0

    # 2. Sequence ratio using fast C++ rapidfuzz
    seq_ratio = fuzz.ratio(a1, a2) / 100.0

    # 3. Token Jaccard & Token Overlap
    tokens1 = set(a1.split())
    tokens2 = set(a2.split())
    intersection = tokens1 & tokens2
    union = tokens1 | tokens2
    token_jaccard = len(intersection) / len(union) if union else 0.0
    min_tokens = min(len(tokens1), len(tokens2))
    token_overlap = len(intersection) / min_tokens if min_tokens > 0 else 0.0

    # 4. Numeric / Street number features
    nums1 = _extract_numbers(a1)
    nums2 = _extract_numbers(a2)
    num_exact = 1.0 if nums1 and nums2 and nums1 == nums2 else 0.0
    num_inter = nums1 & nums2
    min_nums = min(len(nums1), len(nums2))
    num_overlap = len(num_inter) / min_nums if min_nums > 0 else (1.0 if not nums1 and not nums2 else 0.0)

    return {
        "addr_exact_match": exact_match,
        "addr_seq_ratio": float(seq_ratio),
        "addr_token_jaccard": float(token_jaccard),
        "addr_token_overlap_ratio": float(token_overlap),
        "addr_numeric_exact_match": num_exact,
        "addr_numeric_overlap_ratio": float(num_overlap),
        "addr_has_both_address": 1.0,
    }
