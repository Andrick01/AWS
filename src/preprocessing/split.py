"""Train/Validation split module for Business Entity Resolution.

Ensures strict entity-level partitioning on Source 1 IDs to prevent data leakage.
Source 1 entities appearing in the training split will NEVER appear in the validation split.
"""

import logging
from pathlib import Path
from typing import Set, Tuple, Union

import numpy as np
import pandas as pd

from src.preprocessing.load_data import load_tsv, save_tsv

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def partition_entity_ids(
    entity_ids: pd.Series,
    val_ratio: float = 0.20,
    seed: int = 42,
) -> Tuple[Set[str], Set[str]]:
    """Partition unique Source 1 entity IDs into disjoint train and validation sets.

    Args:
        entity_ids: Series or array of Source 1 entity IDs.
        val_ratio: Fraction of IDs to reserve for validation (default: 0.20).
        seed: Random seed for reproducible splitting.

    Returns:
        Tuple of (train_id_set, val_id_set).
    """
    unique_ids = np.array(list(dict.fromkeys(entity_ids.tolist())))
    total_entities = len(unique_ids)

    # Use reproducible random number generator
    rng = np.random.default_rng(seed=seed)
    shuffled_indices = rng.permutation(total_entities)

    n_val = int(total_entities * val_ratio)
    val_indices = shuffled_indices[:n_val]
    train_indices = shuffled_indices[n_val:]

    val_id_set = set(unique_ids[val_indices])
    train_id_set = set(unique_ids[train_indices])

    # Strict leakage validation
    intersection = train_id_set.intersection(val_id_set)
    if intersection:
        raise ValueError(f"CRITICAL LEAKAGE DETECTED: {len(intersection)} IDs appear in both train and val!")

    logger.info(
        "Source 1 IDs partitioned: %s train (%0.1f%%), %s val (%0.1f%%)",
        f"{len(train_id_set):,}",
        (len(train_id_set) / total_entities) * 100,
        f"{len(val_id_set):,}",
        (len(val_id_set) / total_entities) * 100,
    )
    return train_id_set, val_id_set


def split_source1_and_ground_truth(
    source1_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    val_ratio: float = 0.20,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split Source 1 and corresponding Ground Truth without data leakage.

    Args:
        source1_df: Normalized Source 1 DataFrame.
        ground_truth_df: Ground truth DataFrame.
        val_ratio: Validation split ratio.
        seed: Random seed.

    Returns:
        (train_source1, val_source1, train_ground_truth, val_ground_truth)
    """
    train_ids, val_ids = partition_entity_ids(source1_df["entity_id"], val_ratio=val_ratio, seed=seed)

    # Split Source 1
    train_s1 = source1_df[source1_df["entity_id"].isin(train_ids)].copy()
    val_s1 = source1_df[source1_df["entity_id"].isin(val_ids)].copy()

    # Split Ground Truth based on source1_entity_id
    train_gt = ground_truth_df[ground_truth_df["source1_entity_id"].isin(train_ids)].copy()
    val_gt = ground_truth_df[ground_truth_df["source1_entity_id"].isin(val_ids)].copy()

    # Integrity assertions
    assert len(train_s1) + len(val_s1) == len(source1_df), "Mismatch in Source 1 split row counts!"
    assert len(train_gt) + len(val_gt) == len(ground_truth_df), "Mismatch in Ground Truth split row counts!"
    assert len(set(train_s1["entity_id"]).intersection(set(val_s1["entity_id"]))) == 0, "Train/Val S1 ID leak!"
    assert len(set(train_gt["source1_entity_id"]).intersection(set(val_gt["source1_entity_id"]))) == 0, "GT ID leak!"

    return train_s1, val_s1, train_gt, val_gt
