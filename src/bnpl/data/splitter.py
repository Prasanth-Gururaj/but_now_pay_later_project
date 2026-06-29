"""Temporal train/validation/test splitting strategies.

Provides TemporalSplitter that splits cleaned data by issue year
into train, validation, test, and drift holdout segments.
"""

from __future__ import annotations

import pandas as pd

from bnpl.logger import LoggerMixin, log_execution


class TemporalSplitter(LoggerMixin):
    """Split data into train/val/test/holdout by issue year.

    Uses temporal splitting to prevent future data leaking into
    training. The split years match the notebook ground truth:
    train 2013-2015, val 2016, test 2017, drift holdout 2018.

    Usage::

        splitter = TemporalSplitter()
        splits = splitter.split(df)
        train_df = splits["train"]

    Depends on:
        - DataFrame with ``issue_year`` column from DataCleaner
        - LoggerMixin: structured logging
    """

    TRAIN_YEARS: list[int] = [2013, 2014, 2015]
    VAL_YEAR: int = 2016
    TEST_YEAR: int = 2017
    DRIFT_YEAR: int = 2018

    @log_execution(operation="TemporalSplitter.split")
    def split(self, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Split DataFrame into temporal segments by issue_year.

        Args:
            df: DataFrame with ``issue_year`` integer column.

        Returns:
            dict with keys ``train``, ``val``, ``test``, ``drift_2018``
            each mapping to the corresponding DataFrame subset.

        Raises:
            ValueError: If the DataFrame has no ``issue_year`` column.
        """
        if "issue_year" not in df.columns:
            raise ValueError("DataFrame must have 'issue_year' column")

        splits = {
            "train": df[df["issue_year"].isin(self.TRAIN_YEARS)].copy(),
            "val": df[df["issue_year"] == self.VAL_YEAR].copy(),
            "test": df[df["issue_year"] == self.TEST_YEAR].copy(),
            "drift_2018": df[df["issue_year"] == self.DRIFT_YEAR].copy(),
        }

        for name, split_df in splits.items():
            rate = split_df["default"].mean() * 100 if len(split_df) > 0 else 0
            self.logger.info(
                "Split '%s': %d rows | default_rate=%.1f%%",
                name, len(split_df), rate,
            )

        return splits
