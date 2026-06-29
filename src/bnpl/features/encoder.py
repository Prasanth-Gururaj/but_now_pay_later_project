"""Categorical encoding: ordinal and one-hot with fixed training categories.

Provides CategoricalEncoder that handles sub_grade ordinal encoding and
one-hot encoding for nominal features using categories fixed from training.
"""

from __future__ import annotations

import pandas as pd

from bnpl.logger import LoggerMixin


class CategoricalEncoder(LoggerMixin):
    """Encode categorical features using training-fixed parameters.

    Handles two types of encoding:
    - Ordinal: sub_grade A1-G5 mapped to 0-34
    - One-hot: nominal features using only categories seen in training

    Unseen categories produce all-zero one-hot rows rather than errors,
    preventing crashes in production when new category values appear.

    Usage::

        encoder = CategoricalEncoder(sub_grade_map, train_categories)
        df = encoder.encode_sub_grade(df)
        df = encoder.encode_one_hot(df)

    Depends on:
        - Training-fitted category lists from data_prep_config.json
        - LoggerMixin: structured logging
    """

    ORDINAL_FALLBACK: int = 17

    def __init__(
        self,
        sub_grade_map: dict[str, int],
        train_categories: dict[str, list[str]],
    ) -> None:
        """Initialize with training-fitted encoding parameters.

        Args:
            sub_grade_map: Mapping of sub_grade strings to ordinal
                           integers (e.g. ``{"A1": 0, ..., "G5": 34}``).
            train_categories: Mapping of nominal column names to their
                              category lists from training data.
        """
        self._sub_grade_map = sub_grade_map
        self._train_categories = train_categories

    def encode_sub_grade(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode sub_grade as ordinal integer (A1=0 through G5=34).

        Unknown sub_grades map to the fallback value (17 = D2 midpoint).

        Args:
            df: DataFrame with optional ``sub_grade`` string column.

        Returns:
            DataFrame with ``sub_grade_encoded`` column added.
        """
        if "sub_grade" in df.columns:
            df["sub_grade_encoded"] = (
                df["sub_grade"]
                .map(self._sub_grade_map)
                .fillna(self.ORDINAL_FALLBACK)
            )
        else:
            df["sub_grade_encoded"] = self.ORDINAL_FALLBACK
        return df

    def encode_one_hot(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode nominal columns using training-fixed categories.

        Categories not seen in training produce all-zero rows. This
        prevents both crashes and leakage from new category values.

        Args:
            df: DataFrame with optional nominal string columns
                (home_ownership, verification_status, purpose).

        Returns:
            DataFrame with one-hot encoded binary columns added.
        """
        for col, cats in self._train_categories.items():
            for cat in cats:
                if col in df.columns:
                    df[f"{col}_{cat}"] = (df[col] == cat).astype(int)
                else:
                    df[f"{col}_{cat}"] = 0
        return df
