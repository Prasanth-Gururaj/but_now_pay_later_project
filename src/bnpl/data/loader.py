"""Raw data loading from compressed CSV files.

Provides DataLoader for chunked CSV reading of the LendingClub dataset.
TemporalSplitter is re-exported from splitter.py for backward compatibility.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from bnpl.data.splitter import TemporalSplitter
from bnpl.logger import LoggerMixin, log_execution

__all__ = ["DataLoader", "TemporalSplitter"]


class DataLoader(LoggerMixin):
    """Load raw LendingClub loan data from compressed CSV using chunked reading.

    Reads the raw CSV in chunks of 200,000 rows to handle the full
    2.2M-row dataset without loading all 151 columns into memory at
    once. Only the columns needed for modeling are retained.

    For a full pipeline, use DataLoader.load() then pass the result
    to DataCleaner.clean() and TemporalSplitter.split().

    Usage::

        loader = DataLoader()
        df_raw = loader.load("data/raw/accepted_2007_to_2018Q4.csv.gz")

    Depends on:
        - Raw CSV file in LendingClub format
        - LoggerMixin: structured logging
    """

    FEATURE_COLS: list[str] = [
        "dti", "fico_range_low", "revol_util", "annual_inc", "loan_amnt",
        "int_rate", "sub_grade", "term", "emp_length", "home_ownership",
        "verification_status", "purpose", "delinq_2yrs", "inq_last_6mths",
        "open_acc", "pub_rec", "revol_bal",
    ]

    EXTRA_COLS: list[str] = ["loan_status", "issue_d"]

    CHUNK_SIZE: int = 200_000

    @log_execution(operation="DataLoader.load")
    def load(self, raw_path: str | Path) -> pd.DataFrame:
        """Load raw CSV with chunked reading.

        Reads the CSV in chunks, keeping only feature columns plus
        loan_status and issue_d. Does NOT filter or create target;
        that is DataCleaner's responsibility.

        Args:
            raw_path: Path to the compressed CSV file
                      (e.g. ``data/raw/accepted_2007_to_2018Q4.csv.gz``).

        Returns:
            pd.DataFrame: Raw data with feature columns, loan_status,
            and issue_d. No filtering or target creation applied.

        Raises:
            FileNotFoundError: If the raw CSV file does not exist.
        """
        raw_path = Path(raw_path)
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw data not found: {raw_path}")

        usecols = self.FEATURE_COLS + self.EXTRA_COLS
        chunks = self._read_chunks(raw_path, usecols)
        df = pd.concat(chunks, ignore_index=True)
        self.logger.info("Loaded %d rows from %s", len(df), raw_path.name)
        return df

    def _read_chunks(
        self, raw_path: Path, usecols: list[str],
    ) -> list[pd.DataFrame]:
        """Read CSV in chunks, keeping only needed columns.

        Args:
            raw_path: Path to the compressed CSV file.
            usecols: Column names to retain.

        Returns:
            list[pd.DataFrame]: Chunks ready for concatenation.
        """
        chunks: list[pd.DataFrame] = []
        chunk_iter = pd.read_csv(
            raw_path,
            usecols=usecols,
            chunksize=self.CHUNK_SIZE,
            low_memory=False,
        )

        for i, chunk in enumerate(chunk_iter):
            chunks.append(chunk)
            self.logger.debug("Chunk %d: %d rows read", i, len(chunk))

        return chunks
