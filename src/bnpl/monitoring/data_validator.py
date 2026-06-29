"""Production data validation and schema conformance checks.

Provides ProductionDataValidator for checking incoming production
data batches against the training data schema and statistics.
Different from data.validator which validates raw training data.
"""

from __future__ import annotations

import pandas as pd

from bnpl.logger import LoggerMixin, log_execution


class ProductionDataValidator(LoggerMixin):
    """Validate incoming production data against training schema.

    Checks schema consistency, null rates, value ranges, scale
    anomalies, and duplicate rates. This validator runs BEFORE
    drift detection to catch pipeline bugs that would otherwise
    be misinterpreted as real drift.

    Usage::

        validator = ProductionDataValidator(
            reference_df=train_features,
            feature_cols=feature_cols,
            thresholds=monitoring_config,
        )
        result = validator.validate(current_df)

    Depends on:
        - Reference training data for schema and range comparison
        - config/thresholds.yaml monitoring section for thresholds
        - LoggerMixin: structured logging
    """

    def __init__(
        self,
        reference_df: pd.DataFrame,
        feature_cols: list[str],
        thresholds: dict[str, float],
    ) -> None:
        """Initialize with reference data and monitoring thresholds.

        Args:
            reference_df: Training features as reference distribution.
            feature_cols: List of expected feature column names.
            thresholds: Monitoring thresholds dict with max_null_rate,
                        max_duplicate_rate, scale_anomaly_multiplier.
        """
        self._reference = reference_df
        self._feature_cols = feature_cols
        self._thresholds = thresholds

    @log_execution(operation="ProductionDataValidator.validate")
    def validate(self, current_df: pd.DataFrame) -> dict:
        """Run all quality checks on production data.

        Args:
            current_df: Current production data to validate.

        Returns:
            dict with keys:
                - failure_type (str): "pipeline_bug" or "clean"
                - affected_features (list[str]): failed features
                - severity (str): "critical", "warning", or "none"
                - details (dict): per-check result dicts
        """
        affected: list[str] = []
        details: dict = {}
        severity = "none"

        schema = self._check_schema(current_df)
        details["schema"] = schema
        if schema["missing"]:
            affected.extend(schema["missing"])
            severity = "critical"

        scale = self._check_scale_anomalies(current_df)
        details["scale"] = scale
        if scale["anomalous"]:
            affected.extend(scale["anomalous"])
            severity = "critical"

        nulls = self._check_null_rates(current_df)
        details["nulls"] = nulls
        if nulls["high_null_features"]:
            affected.extend(nulls["high_null_features"])
            if severity != "critical":
                severity = "warning"

        dups = self._check_duplicate_rate(current_df)
        details["duplicates"] = dups
        if dups["is_excessive"] and severity != "critical":
            severity = "warning"

        failure_type = "pipeline_bug" if affected else "clean"
        return {
            "failure_type": failure_type,
            "affected_features": list(set(affected)),
            "severity": severity,
            "details": details,
        }

    def _check_schema(self, current_df: pd.DataFrame) -> dict:
        """Check for missing or extra columns.

        Args:
            current_df: Current data.

        Returns:
            dict: missing and extra column lists.
        """
        ref_cols = set(self._feature_cols)
        cur_cols = set(current_df.columns)
        return {
            "missing": sorted(ref_cols - cur_cols),
            "extra": sorted(cur_cols - ref_cols),
        }

    def _check_scale_anomalies(self, current_df: pd.DataFrame) -> dict:
        """Detect values outside plausible range.

        Args:
            current_df: Current data.

        Returns:
            dict: anomalous feature list.
        """
        multiplier = self._thresholds.get("scale_anomaly_multiplier", 10)
        anomalous: list[str] = []
        shared = (
            set(current_df.select_dtypes(include="number").columns)
            & set(self._reference.select_dtypes(include="number").columns)
            & set(self._feature_cols)
        )
        for col in shared:
            ref_range = self._reference[col].max() - self._reference[col].min()
            if ref_range == 0:
                continue
            cur_range = current_df[col].max() - current_df[col].min()
            if cur_range > ref_range * multiplier:
                anomalous.append(col)
        return {"anomalous": anomalous, "multiplier": multiplier}

    def _check_null_rates(self, current_df: pd.DataFrame) -> dict:
        """Check null rates against threshold.

        Args:
            current_df: Current data.

        Returns:
            dict: features exceeding null rate threshold.
        """
        max_rate = self._thresholds.get("max_null_rate", 0.40)
        high: list[str] = []
        for col in set(current_df.columns) & set(self._feature_cols):
            if current_df[col].isnull().mean() > max_rate:
                high.append(col)
        return {"high_null_features": high, "threshold": max_rate}

    def _check_duplicate_rate(self, current_df: pd.DataFrame) -> dict:
        """Check duplicate row rate.

        Args:
            current_df: Current data.

        Returns:
            dict: duplicate rate and whether excessive.
        """
        max_rate = self._thresholds.get("max_duplicate_rate", 0.20)
        rate = current_df.duplicated().mean()
        return {
            "duplicate_rate": round(float(rate), 4),
            "threshold": max_rate,
            "is_excessive": rate > max_rate,
        }
