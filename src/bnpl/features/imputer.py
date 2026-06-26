"""Missing value imputation with training medians and was_missing flags.

Provides MedianImputer that fills nulls with training-derived median
values and creates binary flags indicating which values were imputed.
"""

from __future__ import annotations

import pandas as pd

from bnpl.logger import LoggerMixin


class MedianImputer(LoggerMixin):
    """Impute missing numeric values with training medians.

    For each configured column, creates a binary ``{col}_was_missing``
    flag (1 if the value was originally null, 0 otherwise), then fills
    the null with the training median. The was_missing flags carry real
    predictive signal (emp_length_num_was_missing appears in top 15
    SHAP features).

    Usage::

        imputer = MedianImputer(impute_values)
        df = imputer.impute(df)

    Depends on:
        - Training-fitted median values from data_prep_config.json
        - LoggerMixin: structured logging
    """

    def __init__(self, impute_values: dict[str, float]) -> None:
        """Initialize with training-fitted median values.

        Args:
            impute_values: Mapping of column names to their training
                           median values (e.g. ``{"dti": 17.81, ...}``).
        """
        self._impute_values = impute_values

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values and add was_missing flags.

        For each column in impute_values:
        - If the column exists: check for nulls, create flag, fill
        - If the column is missing: set flag=1, value=median

        Args:
            df: DataFrame with potentially missing numeric values.

        Returns:
            DataFrame with imputed values and was_missing flag columns.
        """
        for col, fill_val in self._impute_values.items():
            if col in df.columns:
                df[f"{col}_was_missing"] = df[col].isnull().astype(int)
                df[col] = df[col].fillna(fill_val)
            else:
                df[f"{col}_was_missing"] = 1
                df[col] = fill_val
        return df

    @property
    def columns(self) -> list[str]:
        """Return the list of columns this imputer handles.

        Returns:
            list[str]: Column names with configured median values.
        """
        return list(self._impute_values.keys())
