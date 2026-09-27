"""Country compatibility feature extractors for Business Entity Resolution.

Computes country match, cross-country flags, and candidate source indicators.
"""

from typing import Dict


def compute_country_features(country1: str, country2: str, candidate_id: str = "") -> Dict[str, float]:
    """Compute country match and candidate source features."""
    c1 = str(country1).strip().lower() if country1 else ""
    c2 = str(country2).strip().lower() if country2 else ""
    cand_id = str(candidate_id).strip()

    # 1. Exact country match
    if c1 and c2:
        country_match = 1.0 if c1 == c2 else 0.0
        country_mismatch = 1.0 if c1 != c2 else 0.0
    else:
        # One or both missing
        country_match = 0.5
        country_mismatch = 0.0

    # 2. Specific country indicators
    is_us = 1.0 if c1 == "us" or c2 == "us" else 0.0
    is_india = 1.0 if c1 == "india" or c2 == "india" else 0.0
    is_france = 1.0 if c1 == "france" or c2 == "france" else 0.0

    # 3. Source indicators (Source 2 vs Source 3)
    is_source2 = 1.0 if cand_id.startswith("S2-") else 0.0
    is_source3 = 1.0 if cand_id.startswith("S3-") else 0.0

    return {
        "country_match": country_match,
        "country_mismatch": country_mismatch,
        "is_us": is_us,
        "is_india": is_india,
        "is_france": is_france,
        "is_source2": is_source2,
        "is_source3": is_source3,
    }
