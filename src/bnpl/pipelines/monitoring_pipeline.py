"""Scheduled monitoring pipeline: drift checks, business metrics, alerts.

Provides the MonitoringPipeline class and a module-level
run_monitoring_pipeline function that serves as the entry point
for GitHub Actions or any scheduled monitoring job.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd

from bnpl.features.pipeline import PreprocessingPipeline
from bnpl.logger import LoggerMixin, log_execution
from bnpl.monitoring.drift_detector import DriftDetector


class MonitoringPipeline(LoggerMixin):
    """End-to-end monitoring pipeline for drift detection and alerting.

    Loads the training reference data, preprocesses a specified monitoring
    window, scores it with the champion model, calculates approval rates,
    runs drift detection, and appends results to the monitoring log.

    This pipeline is designed to be called daily by GitHub Actions or
    any scheduler. Each run produces a timestamped entry in the
    monitoring log and an Evidently HTML drift report.

    Usage::

        pipeline = MonitoringPipeline("reports/data_prep_config.json")
        result = pipeline.run("2018")

    Depends on:
        - PreprocessingPipeline: feature transformation
        - DriftDetector: two-stage drift detection
        - champion_xgboost.pkl: model for scoring
        - config/settings.py: paths and thresholds
        - LoggerMixin: structured logging
    """

    def __init__(self, config_path: str | Path | None = None) -> None:
        """Initialize the monitoring pipeline with paths from config.

        Args:
            config_path: Optional override for data_prep_config.json path.
                         If None, loads from Settings.

        Raises:
            FileNotFoundError: If config or model files do not exist.
        """
        paths = self._load_paths()

        self._config_path = str(config_path or paths["config_path"])
        self._model_path = paths["model_path"]
        self._train_data_path = paths["train_data_path"]
        self._processed_data_dir = paths["processed_data_dir"]
        self._monitoring_log_path = paths["monitoring_log_path"]

        self._pipeline = PreprocessingPipeline(self._config_path)
        self._model = joblib.load(self._model_path)
        self._threshold = self._load_threshold()

        self.logger.info(
            "MonitoringPipeline initialized | model=%s | threshold=%.2f",
            self._model_path,
            self._threshold,
        )

    def _load_paths(self) -> dict[str, str]:
        """Load file paths from Settings, with fallback defaults.

        Returns:
            dict: Mapping of path names to their string values.
        """
        try:
            from config.settings import get_settings

            settings = get_settings()
            return {
                "config_path": settings.paths.config_path,
                "model_path": settings.paths.model_path,
                "train_data_path": settings.paths.train_data_path,
                "processed_data_dir": settings.paths.processed_data_dir,
                "monitoring_log_path": settings.paths.monitoring_log_path,
            }
        except Exception:
            return {
                "config_path": "reports/data_prep_config.json",
                "model_path": "models/champion_xgboost.pkl",
                "train_data_path": "data/processed/train.parquet",
                "processed_data_dir": "data/processed/",
                "monitoring_log_path": "reports/monitoring_log.json",
            }

    def _load_threshold(self) -> float:
        """Load decision threshold from Settings.

        Returns:
            float: Decision threshold value.
        """
        try:
            from config.settings import get_settings

            return get_settings().thresholds.default_threshold
        except Exception:
            return 0.45

    @log_execution(operation="MonitoringPipeline.run")
    def run(self, window_label: str) -> dict:
        """Run the full monitoring pipeline for a data window.

        Loads reference and window data, preprocesses both, scores
        the window with the champion model, calculates the approval
        rate, runs drift detection, and logs results.

        Args:
            window_label: Identifier for the monitoring window, e.g.
                          ``"2018"`` or ``"2018_jan"``. Used to locate
                          data files and name drift reports.

        Returns:
            dict with keys:
                - should_retrain (bool): whether retraining is recommended
                - drift_report (dict): full drift detection results
                - approval_rate (float): approval rate for this window
                - baseline_approval_rate (float): training approval rate
                - window (str): the window label
                - timestamp (str): UTC ISO timestamp
        """
        train_features = self._load_reference_data()
        window_df = self._load_window_data(window_label)
        window_features = self._preprocess_batch(window_df)
        approval_rate = self._calculate_approval_rate(window_features)
        baseline_rate = self._calculate_approval_rate(train_features)

        detector = DriftDetector(train_features, self._config_path)
        drift_report = detector.run_drift(
            train_features, window_features, window_label,
            approval_rate=approval_rate,
        )

        result = {
            "should_retrain": drift_report.get("should_retrain", False),
            "drift_report": drift_report,
            "approval_rate": round(approval_rate, 4),
            "baseline_approval_rate": round(baseline_rate, 4),
            "window": window_label,
            "timestamp": datetime.now(UTC).isoformat(),
        }

        self._append_to_log(result)
        self.logger.info(
            "Monitoring complete | window=%s | retrain=%s | approval=%.2f%%",
            window_label,
            result["should_retrain"],
            approval_rate * 100,
        )
        return result

    def _load_reference_data(self) -> pd.DataFrame:
        """Load preprocessed training data as the reference distribution.

        Returns:
            pd.DataFrame: Training features from train.parquet,
                          filtered to only the model's feature columns.

        Raises:
            FileNotFoundError: If train.parquet does not exist.
        """
        df = pd.read_parquet(self._train_data_path)
        feature_cols = self._pipeline._feature_cols
        available = [c for c in feature_cols if c in df.columns]
        return df[available]

    def _load_window_data(self, window_label: str) -> pd.DataFrame:
        """Load data for a monitoring window from the processed directory.

        Args:
            window_label: Window identifier used to locate the data file.
                          Known labels: "2018" → drift_2018.parquet,
                          "2017" → test.parquet, "2016" → val.parquet,
                          "train" → train.parquet. Unknown labels try
                          {label}.parquet then drift_{label}.parquet.

        Returns:
            pd.DataFrame: Raw window data.

        Raises:
            FileNotFoundError: If no matching data file is found.
        """
        data_dir = Path(self._processed_data_dir)

        known_windows = {
            "2018": "drift_2018.parquet",
            "2017": "test.parquet",
            "2016": "val.parquet",
            "train": "train.parquet",
        }

        if window_label in known_windows:
            candidates = [data_dir / known_windows[window_label]]
        else:
            candidates = [
                data_dir / f"{window_label}.parquet",
                data_dir / f"drift_{window_label}.parquet",
            ]

        for path in candidates:
            if path.exists():
                self.logger.info("Loading window data from %s", path)
                return pd.read_parquet(path)

        raise FileNotFoundError(
            f"No data found for window '{window_label}' in {data_dir}. "
            f"Tried: {[str(p) for p in candidates]}"
        )

    def _preprocess_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess a batch of rows through the pipeline.

        Applies the pipeline's transform to each row individually
        and concatenates the results, matching how individual API
        requests are processed.

        Args:
            df: Raw data DataFrame with the 17 input fields.

        Returns:
            pd.DataFrame: Model-ready features with 47 columns.
        """
        feature_cols = self._pipeline._feature_cols
        available = [c for c in feature_cols if c in df.columns]
        if len(available) == len(feature_cols):
            return df[feature_cols]

        rows = []
        for _, row in df.head(1000).iterrows():
            transformed = self._pipeline.transform(row.to_dict())
            rows.append(transformed)

        if not rows:
            return pd.DataFrame(columns=feature_cols)
        return pd.concat(rows, ignore_index=True)

    def _calculate_approval_rate(self, features_df: pd.DataFrame) -> float:
        """Calculate the approval rate by scoring features with the model.

        Args:
            features_df: Model-ready features DataFrame.

        Returns:
            float: Fraction of predictions below the decision threshold
                   (i.e., approved applications).
        """
        if features_df.empty:
            return 0.0

        probas = self._model.predict_proba(features_df)[:, 1]
        approved = (probas < self._threshold).sum()
        return float(approved / len(probas))

    def _append_to_log(self, result: dict) -> None:
        """Append monitoring result to the JSON log file.

        Creates the file if it does not exist. Handles file corruption
        gracefully by starting a fresh log if the existing JSON is invalid.

        Args:
            result: Monitoring result dict to append.
        """
        log_path = Path(self._monitoring_log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        entries: list[dict] = []
        if log_path.exists():
            try:
                with open(log_path, encoding="utf-8") as f:
                    entries = json.load(f)
            except (json.JSONDecodeError, ValueError):
                self.logger.warning(
                    "Corrupted monitoring log at %s, starting fresh", log_path
                )
                entries = []

        entries.append(result)
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2, default=str)


def run_monitoring_pipeline(window_label: str, config_path: str | None = None) -> dict:
    """Module-level entry point for the monitoring pipeline.

    This function is the primary interface called by GitHub Actions
    and the CLI. It instantiates MonitoringPipeline and runs it for
    the specified data window.

    Args:
        window_label: Identifier for the monitoring window (e.g.
                      ``"2018_jan"``, ``"2017"``).
        config_path: Optional override for data_prep_config.json path.

    Returns:
        dict: Monitoring results including should_retrain boolean
              and full drift report.
    """
    pipeline = MonitoringPipeline(config_path=config_path)
    return pipeline.run(window_label)
