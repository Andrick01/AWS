"""Data loading module for Business Entity Resolution.

Handles chunked TSV reading and writing with safe type conversions and quoting.
"""

import csv
import logging
from pathlib import Path
from typing import Generator, List, Optional, Union

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_tsv(
    filepath: Union[str, Path],
    chunksize: Optional[int] = None,
    nrows: Optional[int] = None,
    usecols: Optional[List[str]] = None,
    dtype: Optional[dict] = None,
) -> Union[pd.DataFrame, Generator[pd.DataFrame, None, None]]:
    """Load a TSV file safely with tab delimiter.

    Args:
        filepath: Path to the TSV file.
        chunksize: Number of rows per chunk if chunked loading is desired.
        nrows: Maximum rows to read.
        usecols: Specific columns to load.
        dtype: Dictionary mapping column names to data types (defaults to str).

    Returns:
        pd.DataFrame or Generator of DataFrame chunks.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # Set default string dtype for consistency if not specified
    if dtype is None:
        dtype = str

    return pd.read_csv(
        path,
        sep="\t",
        chunksize=chunksize,
        nrows=nrows,
        usecols=usecols,
        dtype=dtype,
        quoting=csv.QUOTE_NONE,
        keep_default_na=False,  # Treat empty fields as empty strings instead of NaN
        encoding="utf-8",
    )


def stream_tsv_chunks(
    filepath: Union[str, Path],
    chunksize: int = 250000,
    usecols: Optional[List[str]] = None,
) -> Generator[pd.DataFrame, None, None]:
    """Yield chunks of a TSV file for memory-efficient streaming.

    Args:
        filepath: Path to the TSV file.
        chunksize: Number of records per chunk.
        usecols: Specific columns to select.

    Yields:
        pd.DataFrame chunks.
    """
    path = Path(filepath)
    logger.info("Streaming TSV: %s in chunks of %s rows", path.name, f"{chunksize:,}")
    reader = load_tsv(path, chunksize=chunksize, usecols=usecols)
    for chunk in reader:
        yield chunk


def save_tsv(
    df: pd.DataFrame,
    filepath: Union[str, Path],
    mode: str = "w",
    header: bool = True,
) -> None:
    """Save DataFrame to TSV format with tab separator and no quoting issues.

    Args:
        df: DataFrame to save.
        filepath: Destination path.
        mode: File write mode ('w' for new/overwrite, 'a' for append).
        header: Whether to write column header.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        path,
        sep="\t",
        index=False,
        mode=mode,
        header=header,
        quoting=csv.QUOTE_NONE,
        escapechar="\\",
        encoding="utf-8",
    )
