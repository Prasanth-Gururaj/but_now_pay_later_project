"""Data cleaning: status filtering, target creation, leakage/redundant column removal.

Provides DataCleaner that takes raw loaded data and produces a clean
DataFrame ready for temporal splitting and feature engineering.
"""

from __future__ import annotations

import pandas as pd

from bnpl.logger import LoggerMixin, log_execution


class DataCleaner(LoggerMixin):
    """Clean raw LendingClub data for the BNPL modeling pipeline.

    Filters to resolved loans only (Fully Paid / Charged Off), creates
    the binary default target, parses the issue date, and drops leakage
    columns and the 6 features eliminated during feature selection
    (Notebook 02).

    Usage::

        cleaner = DataCleaner()
        df_clean = cleaner.clean(df_raw)

    Depends on:
        - Raw DataFrame from DataLoader with loan_status and issue_d
        - LoggerMixin: structured logging
    """

    RESOLVED_STATUSES: list[str] = ["Fully Paid", "Charged Off"]

    LEAKAGE_COLS: list[str] = [
        "total_pymnt", "recoveries", "total_rec_prncp",
        "out_prncp", "total_rec_int", "last_pymnt_amnt",
    ]

    LEAKAGE_PREFIXES: list[str] = ["hardship_", "settlement_"]

    DROPPED_FEATURES: dict[str, str] = {
        "fico_range_high": "Perfect duplicate of fico_range_low (corr=1.0)",
        "funded_amnt": "Perfect duplicate of loan_amnt (corr=1.0)",
        "installment": "Derived from loan_amnt+term+int_rate (corr=0.95)",
        "grade": "Redundant with sub_grade (sub_grade is more granular)",
        "addr_state": "Weakest signal across all statistical tests",
        "total_acc": "Weak across all methods, correlated with open_acc",
    }

    @log_execution(operation="DataCleaner.clean")
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run all cleaning steps on a raw DataFrame.

        Args:
            df: Raw DataFrame from DataLoader containing loan_status,
                issue_d, and all feature columns.

        Returns:
            pd.DataFrame: Cleaned data with only resolved loans, binary
            default target, parsed dates, and leakage/redundant columns
            removed.
        """
        df = self._filter_resolved(df)
        df = self._create_target(df)
        df = self._parse_dates(df)
        df = self._drop_leakage(df)
        df = self._drop_redundant_features(df)
        return df

    def _filter_resolved(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep only Fully Paid and Charged Off loans.

        Args:
            df: DataFrame with ``loan_status`` column.

        Returns:
            DataFrame filtered to resolved loans only.
        """
        before = len(df)
        df = df[df["loan_status"].isin(self.RESOLVED_STATUSES)].copy()
        self.logger.info(
            "Filtered to resolved loans: %d -> %d rows", before, len(df),
        )
        return df

    def _create_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create binary default target: 1 = Charged Off, 0 = Fully Paid.

        Args:
            df: DataFrame with ``loan_status`` column.

        Returns:
            DataFrame with ``default`` binary column added.
        """
        df["default"] = (df["loan_status"] == "Charged Off").astype(int)
        self.logger.info(
            "Target created | default_rate=%.1f%%",
            df["default"].mean() * 100,
        )
        return df

    def _parse_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse issue_d string to datetime and extract year.

        Args:
            df: DataFrame with ``issue_d`` string column (format ``%b-%Y``).

        Returns:
            DataFrame with parsed ``issue_d`` and ``issue_year`` columns.
        """
        if "issue_d" in df.columns:
            df["issue_d"] = pd.to_datetime(
                df["issue_d"], format="%b-%Y", errors="coerce",
            )
            df["issue_year"] = df["issue_d"].dt.year
        return df

    def _drop_leakage(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop post-origination leakage columns.

        These columns contain information that only exists AFTER the
        loan outcome is known (payment totals, recovery amounts) and
        would cause data leakage if used as features.

        Args:
            df: DataFrame potentially containing leakage columns.

        Returns:
            DataFrame with leakage columns removed.
        """
        to_drop = [c for c in self.LEAKAGE_COLS if c in df.columns]
        prefix_drops = [
            c for c in df.columns
            if any(c.startswith(p) for p in self.LEAKAGE_PREFIXES)
        ]
        all_drops = list(set(to_drop + prefix_drops))

        if all_drops:
            df = df.drop(columns=all_drops)
            self.logger.info("Dropped %d leakage columns", len(all_drops))
        return df

    def _drop_redundant_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop the 6 features eliminated during feature selection.

        These features were removed in Notebook 02 because they are
        either perfect duplicates, highly redundant, or have weak signal.

        Args:
            df: DataFrame potentially containing redundant features.

        Returns:
            DataFrame with redundant features removed.
        """
        to_drop = [c for c in self.DROPPED_FEATURES if c in df.columns]
        if to_drop:
            df = df.drop(columns=to_drop)
            self.logger.info("Dropped %d redundant features: %s", len(to_drop), to_drop)
        return df
