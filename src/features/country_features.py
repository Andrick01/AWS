"""Country compatibility feature extractors for Business Entity Resolution.

Computes country match/mismatch and candidate source indicators.

Design: country is treated as an arbitrary string — no hardcoded country names.
The only comparison performed is generic equality (same country vs. different country).
This works correctly for any country that appears in the data, including countries
that were absent during training (open-set requirement).
"""

from typing import Dict


def compute_country_features(country1: str, country2: str, candidate_id: str = "") -> Dict[str, float]:
    """Compute generic country match and candidate source features.

    Country values are compared as plain strings after normalisation (lowercase,
    stripped).  No specific country name is referenced anywhere in this function,
    so the feature set is identical regardless of which countries are present in
    the data.

    Args:
        country1: Normalised country of the Source-1 entity.
        country2: Normalised country of the candidate (Source-2 or Source-3).
        candidate_id: entity_id of the candidate, used to derive source indicators.

    Returns:
        Dict with keys:
          - country_match      : 1.0 if both non-empty and equal, 0.5 if either
                                 is missing, 0.0 otherwise.
          - country_mismatch   : 1.0 if both non-empty and unequal, 0.0 otherwise.
          - is_source2         : 1.0 if candidate_id starts with "S2-".
          - is_source3         : 1.0 if candidate_id starts with "S3-".
    """
    c1 = str(country1).strip().lower() if country1 else ""
    c2 = str(country2).strip().lower() if country2 else ""
    cand_id = str(candidate_id).strip()

    # Generic equality comparison — no country names referenced
    if c1 and c2:
        country_match = 1.0 if c1 == c2 else 0.0
        country_mismatch = 1.0 if c1 != c2 else 0.0
    else:
        # One or both values missing — treat as uncertain
        country_match = 0.5
        country_mismatch = 0.0

    # Source indicators derived from ID prefix only — fully generic
    is_source2 = 1.0 if cand_id.startswith("S2-") else 0.0
    is_source3 = 1.0 if cand_id.startswith("S3-") else 0.0

    return {
        "country_match": country_match,
        "country_mismatch": country_mismatch,
        "is_source2": is_source2,
        "is_source3": is_source3,
    }
