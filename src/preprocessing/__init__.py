"""Preprocessing package for Business Entity Resolution.

Exports data loaders, normalization functions, entity-level splitters,
and audit utilities.
"""

from src.preprocessing.audit import audit_dataframe, format_audit_report, verify_leakage
from src.preprocessing.load_data import load_tsv, save_tsv, stream_tsv_chunks
from src.preprocessing.normalize import (
    clean_address_string,
    clean_name_string,
    normalize_address,
    normalize_country,
    normalize_dataframe,
    normalize_name,
    normalize_text,
)
from src.preprocessing.split import partition_entity_ids, split_source1_and_ground_truth

__all__ = [
    "load_tsv",
    "save_tsv",
    "stream_tsv_chunks",
    "normalize_text",
    "clean_name_string",
    "clean_address_string",
    "normalize_name",
    "normalize_address",
    "normalize_country",
    "normalize_dataframe",
    "partition_entity_ids",
    "split_source1_and_ground_truth",
    "audit_dataframe",
    "verify_leakage",
    "format_audit_report",
]
