"""Data and model drift detection using Evidently AI.

Provides the DriftDetector class that implements a two-stage approach:
data quality validation first, then statistical drift detection. This
ordering matters because pipeline bugs (schema changes, unit conversions,
ETL duplication) mimic drift signals and must be caught before running
Evidently to avoid false drift alerts.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from bnpl.logger import LoggerMixin, log_execution


class DriftDetector(LoggerMixin):
    """Two-stage drift detection: data quality validation then statistical tests.

    Stage 1 (validate_data_quality) catches pipeline bugs that would
    otherwise trigger false drift alerts: schema mismatches from upstream
    vendors renaming columns, scale anomalies from unit changes (dollars
    to thousands), high null rates from broken ETL joins, and duplicate
    rows from ETL replays.

    Stage 2 (run_drift) uses Evidently's DataDriftPreset to perform
    statistical hypothesis tests on each feature. Results are classified
    as clean (no drift), investigate (single feature), or real_drift
    (multiple correlated features shifting together).

    Usage::

        detector = DriftDetector(reference_df=train_features, config_path="reports/data_prep_config.json")
        result = detector.run_drift(train_features, current_features, "2018_jan")

    Depends on:
        - Evidently AI: statistical drift detection
        - data_prep_config.json: feature lists and outlier caps
        - config/thresholds.yaml: monitoring thresholds (via Settings)
        - LoggerMixin: structured logging
    """

    def __init__(
        self,
        reference_df: pd.DataFrame,
        config_path: str | Path,
    ) -> None:
        """Initialize with reference data and monitoring thresholds.

        Args:
            reference_df: Training data features used as the reference
                          distribution for drift comparison.
            config_path: Path to data_prep_config.json for feature lists
                         and outlier cap values.

        Raises:
            FileNotFoundError: If config_path does not exist.
        """
        import json

        self._reference = reference_df

        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
        self._feature_cols: list[str] = config["final_model_columns"]
        self._outlier_caps: dict[str, float] = config["outlier_handling"]["caps"]

        self._monitoring = self._load_monitoring_thresholds()

        self.logger.info(
            "DriftDetector initialized | ref_rows=%d | features=%d",
            len(reference_df),
            len(self._feature_cols),
        )

    def _load_monitoring_thresholds(self) -> dict[str, float]:
        """Load monitoring thresholds from the Settings config system.

        Returns:
            dict: Monitoring thresholds including psi_threshold,
                  max_null_rate, max_duplicate_rate, etc.
        """
        try:
            from config.settings import get_settings

            settings = get_settings()
            return dict(settings.thresholds.monitoring)
        except Exception:
            self.logger.warning("Could not load settings, using defaults")
            return {
                "psi_threshold": 0.25,
                "min_drifted_features_for_retrain": 2,
                "approval_rate_deviation_pct": 0.10,
                "max_null_rate": 0.40,
                "max_duplicate_rate": 0.20,
                "scale_anomaly_multiplier": 10,
            }

    def validate_data_quality(self, current_df: pd.DataFrame) -> dict:
        """Run data quality checks to detect pipeline bugs before drift analysis.

        Performs four sequential checks to catch common production data
        pipeline failures that would otherwise be misinterpreted as
        real distribution drift.

        Args:
            current_df: The current/production data to validate against
                        the reference schema and value ranges.

        Returns:
            dict with keys:
                - failure_type (str): ``"pipeline_bug"`` or ``"clean"``
                - affected_features (list[str]): features that failed checks
                - severity (str): ``"critical"`` or ``"warning"`` or ``"none"``
                - details (dict): per-check results with specifics
        """
        affected: list[str] = []
        details: dict[str, dict] = {}
        severity = "none"

        schema_result = self._check_schema(current_df)
        details["schema"] = schema_result
        if schema_result["missing"]:
            affected.extend(schema_result["missing"])
            severity = "critical"

        scale_result = self._check_scale_anomalies(current_df)
        details["scale"] = scale_result
        if scale_result["anomalous"]:
            affected.extend(scale_result["anomalous"])
            severity = "critical"

        null_result = self._check_null_rates(current_df)
        details["nulls"] = null_result
        if null_result["high_null_features"]:
            affected.extend(null_result["high_null_features"])
            if severity != "critical":
                severity = "warning"

        dup_result = self._check_duplicate_rate(current_df)
        details["duplicates"] = dup_result
        if dup_result["is_excessive"]:
            if severity != "critical":
                severity = "warning"

        failure_type = "pipeline_bug" if affected else "clean"
        return {
            "failure_type": failure_type,
            "affected_features": list(set(affected)),
            "severity": severity,
            "details": details,
        }

    def _check_schema(self, current_df: pd.DataFrame) -> dict:
        """Check for missing or extra columns vs reference.

        Column rename or drop by an upstream data vendor is the most
        common production data pipeline failure mode.

        Args:
            current_df: Current data to check.

        Returns:
            dict with missing and extra column lists.
        """
        ref_cols = set(self._feature_cols)
        cur_cols = set(current_df.columns)
        return {
            "missing": sorted(ref_cols - cur_cols),
            "extra": sorted(cur_cols - ref_cols),
        }

    def _check_scale_anomalies(self, current_df: pd.DataFrame) -> dict:
        """Detect values outside plausible range suggesting unit changes.

        A vendor changing units (dollars to thousands, or percentage
        to decimal) looks like drift but is actually a pipeline bug
        that must be fixed at the data source.

        Args:
            current_df: Current data to check.

        Returns:
            dict with list of anomalous features.
        """
        multiplier = self._monitoring.get("scale_anomaly_multiplier", 10)
        anomalous: list[str] = []
        numeric_cols = current_df.select_dtypes(include="number").columns
        ref_numeric = self._reference.select_dtypes(include="number").columns
        shared = set(numeric_cols) & set(ref_numeric) & set(self._feature_cols)

        for col in shared:
            ref_range = self._reference[col].max() - self._reference[col].min()
            if ref_range == 0:
                continue
            cur_range = current_df[col].max() - current_df[col].min()
            if cur_range > ref_range * multiplier:
                anomalous.append(col)

        return {"anomalous": anomalous, "multiplier": multiplier}

    def _check_null_rates(self, current_df: pd.DataFrame) -> dict:
        """Check if any feature exceeds the maximum allowed null rate.

        Args:
            current_df: Current data to check.

        Returns:
            dict with features exceeding the null rate threshold.
        """
        max_null_rate = self._monitoring.get("max_null_rate", 0.40)
        high_null: list[str] = []
        shared = set(current_df.columns) & set(self._feature_cols)

        for col in shared:
            null_rate = current_df[col].isnull().mean()
            if null_rate > max_null_rate:
                high_null.append(col)

        return {
            "high_null_features": high_null,
            "threshold": max_null_rate,
        }

    def _check_duplicate_rate(self, current_df: pd.DataFrame) -> dict:
        """Check if duplicate row rate suggests ETL replay.

        ETL running twice is a common failure mode that inflates row
        counts and distorts distributions without representing real drift.

        Args:
            current_df: Current data to check.

        Returns:
            dict with duplicate rate and whether it is excessive.
        """
        max_dup_rate = self._monitoring.get("max_duplicate_rate", 0.20)
        dup_rate = current_df.duplicated().mean()
        return {
            "duplicate_rate": round(float(dup_rate), 4),
            "threshold": max_dup_rate,
            "is_excessive": dup_rate > max_dup_rate,
        }

    @log_execution(operation="DriftDetector.run_drift")
    def run_drift(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        window_label: str,
        approval_rate: float | None = None,
    ) -> dict:
        """Run the full two-stage drift detection pipeline.

        Stage 1 validates data quality. If a pipeline bug is detected,
        drift analysis is skipped entirely because the statistical tests
        would produce meaningless results on corrupted data.

        Stage 2 runs Evidently DataDriftPreset and classifies the result.

        Args:
            reference_df: Reference (training) feature DataFrame.
            current_df: Current (production) feature DataFrame.
            window_label: Identifier for this monitoring window (e.g.
                          ``"2018_jan"``). Used for report file naming.
            approval_rate: Optional current approval rate for retraining
                           decision. Pass None to skip approval check.

        Returns:
            dict with keys:
                - window (str): the window label
                - drift_detected (bool): whether any drift was found
                - failure_type (str): pipeline_bug/real_drift/investigate/clean
                - drifted_features (list[str]): names of drifted features
                - drift_score (float): proportion of features drifted
                - recommended_action (str): human-readable recommendation
                - should_retrain (bool): whether retraining is recommended
        """
        quality_result = self.validate_data_quality(current_df)
        if quality_result["failure_type"] == "pipeline_bug":
            return self._build_pipeline_bug_result(window_label, quality_result)

        drift_result = self._run_evidently(reference_df, current_df)
        classified = self._classify_drift(drift_result, window_label)

        should_retrain = self._should_retrain(classified, approval_rate, quality_result)
        classified["should_retrain"] = should_retrain

        self._save_report(drift_result["report"], window_label)
        return classified

    def _build_pipeline_bug_result(self, window_label: str, quality: dict) -> dict:
        """Build result dict for pipeline bug detection.

        Args:
            window_label: Monitoring window identifier.
            quality: Data quality validation result dict.

        Returns:
            dict: Drift result indicating pipeline bug, not real drift.
        """
        return {
            "window": window_label,
            "drift_detected": False,
            "failure_type": "pipeline_bug",
            "drifted_features": [],
            "drift_score": 0.0,
            "recommended_action": (
                f"Fix data pipeline. Affected: {quality['affected_features']}"
            ),
            "should_retrain": quality["severity"] == "critical",
        }

    def _run_evidently(
        self, reference_df: pd.DataFrame, current_df: pd.DataFrame
    ) -> dict:
        """Execute Evidently DataDriftPreset on shared numeric columns.

        Args:
            reference_df: Reference distribution DataFrame.
            current_df: Current distribution DataFrame.

        Returns:
            dict with ``report`` (Evidently Report object) and
            ``result_dict`` (the report as a Python dict).
        """
        import warnings

        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report

        shared = sorted(set(reference_df.columns) & set(current_df.columns))
        ref = reference_df[shared].copy()
        cur = current_df[shared].copy()

        report = Report(metrics=[DataDriftPreset()])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            report.run(reference_data=ref, current_data=cur)
        return {"report": report, "result_dict": report.as_dict()}

    def _classify_drift(self, drift_result: dict, window_label: str) -> dict:
        """Classify drift severity based on how many features drifted.

        Args:
            drift_result: Output from _run_evidently.
            window_label: Monitoring window identifier.

        Returns:
            dict: Classified drift result with failure_type and actions.
        """
        result_dict = drift_result["result_dict"]
        metrics = result_dict.get("metrics", [])
        drifted: list[str] = []
        drift_score = 0.0

        for metric in metrics:
            name = metric.get("metric", "")
            result = metric.get("result", {})
            if name == "DataDriftTable":
                drift_score = result.get("share_of_drifted_columns", 0.0)
                for col_name, col_data in result.get("drift_by_columns", {}).items():
                    if col_data.get("drift_detected", False):
                        drifted.append(col_name)
                break

        min_features = int(
            self._monitoring.get("min_drifted_features_for_retrain", 2)
        )
        if len(drifted) == 0:
            failure_type = "clean"
            action = "No action needed."
        elif len(drifted) < min_features:
            failure_type = "investigate"
            action = f"Investigate single feature drift: {drifted}"
        else:
            failure_type = "real_drift"
            action = f"Multiple features drifted ({len(drifted)}). Consider retraining."

        return {
            "window": window_label,
            "drift_detected": len(drifted) > 0,
            "failure_type": failure_type,
            "drifted_features": drifted,
            "drift_score": round(drift_score, 4),
            "recommended_action": action,
        }

    def _should_retrain(
        self,
        drift_result: dict,
        approval_rate: float | None,
        quality_result: dict,
    ) -> bool:
        """Determine whether model retraining is recommended.

        Returns True if ANY of three conditions are met:
        1. Real drift with PSI above threshold on 2+ features.
        2. Approval rate deviates more than the configured percentage
           from the training baseline.
        3. Data quality check failed with critical severity.

        Args:
            drift_result: Classified drift result from _classify_drift.
            approval_rate: Current window approval rate, or None.
            quality_result: Data quality validation result.

        Returns:
            bool: True if retraining is recommended.
        """
        psi_threshold = self._monitoring.get("psi_threshold", 0.25)
        min_features = int(
            self._monitoring.get("min_drifted_features_for_retrain", 2)
        )
        deviation_pct = self._monitoring.get("approval_rate_deviation_pct", 0.10)

        if (
            drift_result["failure_type"] == "real_drift"
            and drift_result["drift_score"] > psi_threshold
            and len(drift_result["drifted_features"]) >= min_features
        ):
            return True

        if approval_rate is not None:
            baseline = 0.55
            if abs(approval_rate - baseline) > deviation_pct:
                return True

        if quality_result.get("severity") == "critical":
            return True

        return False

    def _save_report(self, report: object, window_label: str) -> Path:
        """Save the Evidently HTML report to the drift reports directory.

        Creates the reports/drift_reports/ directory if it does not exist.

        Args:
            report: Evidently Report object with a save_html method.
            window_label: Used to name the output file.

        Returns:
            Path: Absolute path where the HTML report was saved.
        """
        try:
            from config.settings import get_settings

            settings = get_settings()
            reports_dir = Path(settings.paths.drift_reports_dir)
        except Exception:
            reports_dir = Path("reports/drift_reports")

        reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = reports_dir / f"{window_label}_drift.html"
        report.save_html(str(report_path))
        self.logger.info("Drift report saved to %s", report_path)
        return report_path
