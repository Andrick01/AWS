"""Normalization module for Business Entity Resolution.

Provides ultra-fast, Unicode-aware, script-preserving text normalization.
Safely handles:
- Lowercase conversion
- Whitespace stripping and collapsing
- Punctuation and noise symbol removal
- Preserves all Unicode Letters (L*), Numbers (N*), and Combining Marks (M* e.g. Devanagari/Kannada matras)
- Preserves French accented characters (e.g. é, è, ô, ç)
- Retains legal entity designations (pvt, ltd, inc, corp, llc)
- Cleans website noise (www., .com, .org, .net, etc.) from business names
- Standardizes common address tokens (road -> rd, street -> st, ave, dr, apt, fl, etc.)
- Creates combined_text_normalized field for blocking / candidate generation
- Preserves original columns and appends *_normalized columns
"""

import logging
import re
import unicodedata
from typing import Optional

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Precompute Unicode translation table for character code points up to 0x10000
# Replace any character that is NOT Letter (L), Number (N), or Mark (M) with a single space.
# Characters with category L, N, M are strictly preserved.
_PUNCT_TABLE = {}
for code_point in range(0x10000):
    cat = unicodedata.category(chr(code_point))
    if not cat.startswith(("L", "N", "M")):
        _PUNCT_TABLE[code_point] = 32  # ASCII space ord(' ')

# Regex for stripping domain extensions & web noise from business names
_RE_WWW = re.compile(r"\bwww\.", flags=re.IGNORECASE)
_RE_DOMAINS = re.compile(r"\.(com|org|net|co|in|io|ai|biz|info)\b", flags=re.IGNORECASE)

# Fast token mapping for standardizing common address words
_ADDRESS_MAP = {
    "street": "st",
    "road": "rd",
    "avenue": "ave",
    "boulevard": "blvd",
    "drive": "dr",
    "lane": "ln",
    "court": "ct",
    "circle": "cir",
    "highway": "hwy",
    "parkway": "pkwy",
    "apartment": "apt",
    "apts": "apt",
    "suite": "ste",
    "floor": "fl",
    "building": "bldg",
    "north": "n",
    "south": "s",
    "east": "e",
    "west": "w",
}


def normalize_text(text: Optional[str]) -> str:
    """Normalize a single string safely.

    1. Checks for null/empty/NaN.
    2. Applies Unicode NFKC normalization.
    3. Converts to lowercase.
    4. Replaces noise symbols and punctuation with space using precomputed table.
    5. Collapses repeated whitespace into a single space.
    """
    if text is None or not isinstance(text, str) or not text.strip():
        return ""
    normalized = unicodedata.normalize("NFKC", text).lower()
    cleaned = normalized.translate(_PUNCT_TABLE)
    return " ".join(cleaned.split())


def clean_name_string(text: Optional[str]) -> str:
    """Normalize a business name string, removing domain noise while preserving legal terms."""
    if text is None or not isinstance(text, str) or not text.strip():
        return ""
    # Strip web prefix / domain extension
    s = _RE_WWW.sub("", text)
    s = _RE_DOMAINS.sub("", s)
    # Apply standard text normalization (preserves legal terms and non-Latin scripts)
    return normalize_text(s)


def clean_address_string(text: Optional[str]) -> str:
    """Normalize an address string and standardize common street/unit tokens."""
    cleaned = normalize_text(text)
    if not cleaned:
        return ""
    tokens = cleaned.split()
    standardized = [_ADDRESS_MAP.get(tok, tok) for tok in tokens]
    return " ".join(standardized)


def normalize_name(series: pd.Series) -> pd.Series:
    """Normalize business name while removing domain noise and preserving business terms."""
    vals = series.fillna("").astype(str).tolist()
    norm_vals = [clean_name_string(s) for s in vals]
    return pd.Series(norm_vals, index=series.index, dtype=str)


def normalize_address(series: pd.Series) -> pd.Series:
    """Normalize address while standardizing street/unit abbreviations."""
    vals = series.fillna("").astype(str).tolist()
    norm_vals = [clean_address_string(s) for s in vals]
    return pd.Series(norm_vals, index=series.index, dtype=str)


def normalize_country(series: pd.Series) -> pd.Series:
    """Normalize country field safely (lowercase, stripped)."""
    vals = series.fillna("").astype(str).tolist()
    return pd.Series([s.strip().lower() for s in vals], index=series.index, dtype=str)


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize entity DataFrame while strictly preserving original columns.

    Appends:
    - business_name_normalized
    - business_address_normalized
    - country_normalized
    - combined_text_normalized

    Leaves original entity_id, business_name, business_address, country untouched.
    """
    result = df.copy()

    # Compute normalized columns
    result["business_name_normalized"] = normalize_name(result["business_name"])
    result["business_address_normalized"] = normalize_address(result["business_address"])
    result["country_normalized"] = normalize_country(result["country"])

    # Compute combined text for easy candidate generation/blocking
    names = result["business_name_normalized"].tolist()
    addrs = result["business_address_normalized"].tolist()
    countries = result["country_normalized"].tolist()

    combined = [
        " ".join(filter(None, [n, a, c]))
        for n, a, c in zip(names, addrs, countries)
    ]
    result["combined_text_normalized"] = pd.Series(combined, index=result.index, dtype=str)

    return result
