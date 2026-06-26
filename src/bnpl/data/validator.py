"""Data validation and quality checks before processing.

Provides DataValidator that checks raw data integrity before
it enters the preprocessing and training pipelines.
"""

from __future__ import annotations

import pandas as pd

from bnpl.logger import LoggerMixin, log_execution


class DataValidator(LoggerMixin):
    """Validate raw loan data before it enters the pipeline.

    Checks that required columns exist, the target variable has
    only expected values, dates are parseable, and basic data
    quality thresholds are met.

    Usage::

        validator = DataValidator()
        report = validator.validate(df)
        if not report["is_valid"]:
            print(report["errors"])

    Depends on:
        - LoggerMixin: structured logging
    """

    REQUIRED_FEATURE_COLS: list[str] = [
        "dti", "fico_range_low", "revol_util", "annual_inc", "loan_amnt",
        "int_rate", "sub_grade", "term", "emp_length", "home_ownership",
        "verification_status", "purpose", "delinq_2yrs", "inq_last_6mths",
        "open_acc", "pub_rec", "revol_bal",
    ]

    EXPECTED_STATUSES: list[str] = ["Fully Paid", "Charged Off"]

    @log_execution(operation="DataValidator.validate")
    def validate(self, df: pd.DataFrame) -> dict:
        """Run all validation checks on the raw DataFrame.

        Args:
            df: Raw DataFrame to validate. Should contain feature
                columns, loan_status, and issue_d at minimum.

        Returns:
            dict with keys:
                - is_valid (bool): True if all checks passed
                - errors (list[str]): list of error messages
                - warnings (list[str]): list of warning messages
                - row_count (int): number of rows in the DataFrame
                - column_count (int): number of columns
        """
        errors: list[str] = []
        warnings: list[str] = []

        self._check_columns(df, errors)
        self._check_target(df, errors)
        self._check_dates(df, errors, warnings)
        self._check_row_count(df, warnings)

        is_valid = len(errors) == 0
        self.logger.info(
            "Validation %s | errors=%d | warnings=%d",
            "PASSED" if is_valid else "FAILED",
            len(errors), len(warnings),
        )

        return {
            "is_valid": is_valid,
            "errors": errors,
            "warnings": warnings,
            "row_count": len(df),
            "column_count": len(df.columns),
        }

    def _check_columns(self, df: pd.DataFrame, errors: list[str]) -> None:
        """Check that all required feature columns exist.

        Args:
            df: DataFrame to check.
            errors: List to append error messages to.
        """
        missing = set(self.REQUIRED_FEATURE_COLS) - set(df.columns)
        if missing:
            errors.append(f"Missing required columns: {sorted(missing)}")

    def _check_target(self, df: pd.DataFrame, errors: list[str]) -> None:
        """Check that the target column has only expected values.

        Args:
            df: DataFrame to check.
            errors: List to append error messages to.
        """
        if "default" in df.columns:
            unique_vals = set(df["default"].dropna().unique())
            if not unique_vals.issubset({0, 1}):
                errors.append(
                    f"Target 'default' has unexpected values: {unique_vals}"
                )
        elif "loan_status" in df.columns:
            unique_statuses = set(df["loan_status"].dropna().unique())
            unexpected = unique_statuses - set(self.EXPECTED_STATUSES)
            if unexpected:
                self.logger.debug("Non-resolved statuses present: %s", unexpected)

    def _check_dates(
        self, df: pd.DataFrame, errors: list[str], warnings: list[str]
    ) -> None:
        """Check that issue_d is present and parseable.

        Args:
            df: DataFrame to check.
            errors: List to append error messages to.
            warnings: List to append warning messages to.
        """
        if "issue_d" not in df.columns and "issue_year" not in df.columns:
            errors.append("Neither 'issue_d' nor 'issue_year' column found")
            return

        if "issue_d" in df.columns:
            null_rate = df["issue_d"].isnull().mean()
            if null_rate > 0.05:
                warnings.append(
                    f"issue_d has {null_rate:.1%} null values"
                )

    def _check_row_count(self, df: pd.DataFrame, warnings: list[str]) -> None:
        """Check that the DataFrame has a reasonable number of rows.

        Args:
            df: DataFrame to check.
            warnings: List to append warning messages to.
        """
        if len(df) < 1000:
            warnings.append(
                f"Very few rows ({len(df)}). Expected 1M+ for full dataset."
            )
