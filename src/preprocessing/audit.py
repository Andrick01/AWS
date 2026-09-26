"""Data audit and leakage verification module for Business Entity Resolution.

Profiles raw and processed datasets, verifies split integrity, and generates reports.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def audit_dataframe(df: pd.DataFrame, dataset_name: str = "Dataset") -> Dict:
    """Perform a comprehensive audit of a DataFrame.

    Returns dictionary of metrics:
    - total_rows
    - total_cols
    - column_names
    - missing_values_per_col
    - unique_ids
    - duplicate_ids
    - country_distribution
    """
    total_rows = len(df)
    columns = list(df.columns)

    missing_counts = {}
    for col in columns:
        # Check both NaN and empty string
        is_missing = df[col].isna() | (df[col].astype(str).str.strip() == "")
        missing_counts[col] = int(is_missing.sum())

    id_col = "entity_id" if "entity_id" in df.columns else ("source1_entity_id" if "source1_entity_id" in df.columns else None)
    unique_ids = int(df[id_col].nunique()) if id_col else total_rows
    duplicate_ids = total_rows - unique_ids if id_col else 0

    country_dist = {}
    if "country" in df.columns:
        country_dist = df["country"].value_counts(dropna=False).to_dict()
        country_dist = {str(k): int(v) for k, v in country_dist.items()}

    audit_result = {
        "dataset_name": dataset_name,
        "total_rows": total_rows,
        "total_cols": len(columns),
        "column_names": columns,
        "missing_counts": missing_counts,
        "unique_ids": unique_ids,
        "duplicate_ids": duplicate_ids,
        "country_distribution": country_dist,
    }
    return audit_result


def verify_leakage(train_ids: Union[Set[str], List[str]], val_ids: Union[Set[str], List[str]]) -> Dict:
    """Verify that there is ZERO intersection between train and validation ID sets."""
    set_train = set(train_ids)
    set_val = set(val_ids)
    intersection = set_train.intersection(set_val)
    leakage_detected = len(intersection) > 0

    return {
        "train_count": len(set_train),
        "val_count": len(set_val),
        "overlap_count": len(intersection),
        "leakage_detected": leakage_detected,
        "sample_leaked_ids": list(intersection)[:5] if leakage_detected else [],
    }


def format_audit_report(audits: List[Dict], leakage_info: Optional[Dict] = None, seed: Optional[int] = None) -> str:
    """Format audit results into a readable text report."""
    lines = [
        "=" * 80,
        "           AWS ML CHALLENGE 2026 - PREPROCESSING AUDIT REPORT",
        "=" * 80,
        "",
    ]

    if seed is not None:
        lines.append(f"Random Seed for Splitting: {seed}")
        lines.append("")

    if leakage_info:
        lines.extend([
            "-" * 80,
            "TRAIN / VALIDATION LEAKAGE VERIFICATION",
            "-" * 80,
            f"Train S1 Entities:     {leakage_info['train_count']:,}",
            f"Validation S1 Entities: {leakage_info['val_count']:,}",
            f"Overlapping Entities:   {leakage_info['overlap_count']:,}",
            f"Leakage Status:         {'FAILED - LEAKAGE DETECTED!' if leakage_info['leakage_detected'] else 'PASSED - ZERO OVERLAP (100% LEAK-FREE)'}",
            "",
        ])

    lines.extend([
        "-" * 80,
        "DATASET AUDIT DETAILS",
        "-" * 80,
    ])

    for a in audits:
        lines.append(f"\n[{a['dataset_name']}]")
        lines.append(f"  Shape: {a['total_rows']:,} rows x {a['total_cols']} columns")
        lines.append(f"  Columns: {', '.join(a['column_names'])}")
        lines.append(f"  Unique Entity IDs: {a['unique_ids']:,} (Duplicates: {a['duplicate_ids']:,})")
        lines.append("  Missing Values:")
        for col, count in a["missing_counts"].items():
            pct = (count / a["total_rows"] * 100) if a["total_rows"] > 0 else 0
            lines.append(f"    - {col}: {count:,} ({pct:0.2f}%)")
        if a["country_distribution"]:
            lines.append("  Country Distribution:")
            for country, count in a["country_distribution"].items():
                pct = (count / a["total_rows"] * 100) if a["total_rows"] > 0 else 0
                lines.append(f"    - {country}: {count:,} ({pct:0.2f}%)")

    lines.extend([
        "",
        "=" * 80,
        "END OF PREPROCESSING REPORT",
        "=" * 80,
    ])

    return "\n".join(lines)
