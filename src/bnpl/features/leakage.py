"""Leakage column detection and removal.

Provides LeakageChecker that identifies columns with suspiciously
high correlation with the target or matching known post-issuance
column name patterns.
"""

from __future__ import annotations

import pandas as pd

from bnpl.logger import LoggerMixin, log_execution


class LeakageChecker(LoggerMixin):
    """Detect potential data leakage columns by correlation and name pattern.

    Calculates Pearson correlation of each numeric column with the
    target and flags any with absolute correlation above a threshold.
    Also checks for known post-issuance column name patterns that
    should never be used as features.

    Usage::

        checker = LeakageChecker()
        report = checker.check(df, target_col="default")
        if report["leakage_found"]:
            print(report["flagged_columns"])

    Depends on:
        - LoggerMixin: structured logging
    """

    KNOWN_LEAKAGE_PATTERNS: list[str] = [
        "total_pymnt", "recoveries", "total_rec_",
        "out_prncp", "last_pymnt", "hardship_", "settlement_",
    ]

    CORRELATION_THRESHOLD: float = 0.3

    @log_execution(operation="LeakageChecker.check")
    def check(
        self,
        df: pd.DataFrame,
        target_col: str = "default",
        threshold: float | None = None,
    ) -> dict:
        """Run leakage detection on a DataFrame.

        Args:
            df: DataFrame with feature columns and a target column.
            target_col: Name of the binary target column.
            threshold: Absolute correlation threshold for flagging.
                       Defaults to 0.3.

        Returns:
            dict with keys:
                - leakage_found (bool): True if any leakage detected
                - flagged_columns (list[dict]): name, reason, correlation
                - safe_columns (list[str]): columns that passed
                - total_checked (int): columns checked

        Raises:
            ValueError: If target_col is not in the DataFrame.
        """
        if target_col not in df.columns:
            raise ValueError(f"Target '{target_col}' not in DataFrame")

        thresh = threshold or self.CORRELATION_THRESHOLD
        flagged = self._check_name_patterns(df)
        flagged_names = {f["name"] for f in flagged}
        flagged.extend(self._check_correlations(df, target_col, thresh, flagged_names))

        all_flagged = {f["name"] for f in flagged}
        safe = [c for c in df.columns if c not in all_flagged and c != target_col]

        self.logger.info(
            "Leakage check: %d flagged, %d safe", len(flagged), len(safe),
        )
        return {
            "leakage_found": len(flagged) > 0,
            "flagged_columns": flagged,
            "safe_columns": safe,
            "total_checked": len(df.columns) - 1,
        }

    def _check_name_patterns(self, df: pd.DataFrame) -> list[dict]:
        """Flag columns matching known post-issuance name patterns.

        Args:
            df: DataFrame to check.

        Returns:
            list[dict]: Flagged columns with name and reason.
        """
        flagged: list[dict] = []
        for col in df.columns:
            for pattern in self.KNOWN_LEAKAGE_PATTERNS:
                if col.startswith(pattern) or col == pattern:
                    flagged.append({
                        "name": col,
                        "reason": f"Matches leakage pattern '{pattern}'",
                        "correlation": None,
                    })
                    break
        return flagged

    def _check_correlations(
        self, df: pd.DataFrame, target_col: str,
        threshold: float, exclude: set[str],
    ) -> list[dict]:
        """Flag numeric columns with high absolute correlation to target.

        Args:
            df: DataFrame with numeric columns and target.
            target_col: Target column name.
            threshold: Absolute correlation threshold.
            exclude: Column names to skip (already flagged).

        Returns:
            list[dict]: Flagged columns with name, reason, correlation.
        """
        flagged: list[dict] = []
        for col in df.select_dtypes(include="number").columns:
            if col == target_col or col in exclude:
                continue
            try:
                corr = float(df[col].corr(df[target_col]))
                if abs(corr) > threshold:
                    flagged.append({
                        "name": col,
                        "reason": f"|corr|={abs(corr):.3f} > {threshold}",
                        "correlation": round(corr, 4),
                    })
            except Exception:
                continue
        return flagged
