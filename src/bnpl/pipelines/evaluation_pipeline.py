"""Evaluation pipeline: score test set, generate plots, log metrics.

Owns only model evaluation on the held-out test set. Assumes the
champion model and test.parquet already exist.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from bnpl.logger import LoggerMixin, log_execution


class EvaluationPipeline(LoggerMixin):
    """Evaluate the champion model on the test set.

    Loads test.parquet (touched only once for final unbiased evaluation),
    scores with the champion model, generates calibration and SHAP plots,
    and logs test metrics to MLflow.

    Steps:
    1. Load test.parquet
    2. Load champion model
    3. Score test set at final threshold
    4. Evaluate via ModelEvaluator
    5. Generate SHAP plots via ModelExplainer
    6. Log test metrics to MLflow
    7. Print classification report

    Usage::

        pipeline = EvaluationPipeline()
        metrics = pipeline.run()

    Depends on:
        - ModelEvaluator for metrics and calibration
        - ModelExplainer for SHAP plots
        - config/settings.py for paths and threshold
        - LoggerMixin: structured logging
    """

    def __init__(self) -> None:
        """Initialize with paths from Settings."""
        self._paths = self._load_paths()
        self._threshold = self._load_threshold()

    def _load_paths(self) -> dict[str, str]:
        """Load file paths from Settings.

        Returns:
            dict: Path name to string mapping.
        """
        try:
            from config.settings import get_settings
            s = get_settings()
            return {
                "processed_dir": s.paths.processed_data_dir,
                "model_path": s.paths.model_path,
            }
        except Exception:
            return {
                "processed_dir": "data/processed/",
                "model_path": "models/champion_xgboost.pkl",
            }

    def _load_threshold(self) -> float:
        """Load decision threshold from Settings.

        Returns:
            float: Decision threshold.
        """
        try:
            from config.settings import get_settings
            return get_settings().thresholds.default_threshold
        except Exception:
            return 0.45

    @log_execution(operation="EvaluationPipeline.run")
    def run(self) -> dict:
        """Run the full evaluation pipeline.

        Returns:
            dict: Test set metrics including AUC, precision, recall, F1.
        """
        from bnpl.models.evaluate import ModelEvaluator
        from bnpl.models.explain import ModelExplainer

        test = pd.read_parquet(
            Path(self._paths["processed_dir"]) / "test.parquet"
        )
        model = joblib.load(self._paths["model_path"])

        non_feature = ["default", "issue_d", "issue_year"]
        feature_cols = [c for c in test.columns if c not in non_feature]
        X_test, y_test = test[feature_cols], test["default"]

        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(
            model, X_test, y_test, self._threshold,
        )

        try:
            explainer = ModelExplainer(model)
            explainer.explain_global(X_test)
        except Exception as exc:
            self.logger.warning("SHAP explanation failed: %s", exc)

        approval_rate = float((model.predict_proba(X_test)[:, 1] < self._threshold).mean())
        self._save_test_metrics(metrics, approval_rate)

        self.logger.info(
            "Evaluation complete | AUC=%.4f | threshold=%.2f | approval=%.1f%%",
            metrics["auc"], self._threshold, approval_rate * 100,
        )
        return metrics

    def _save_test_metrics(self, metrics: dict, approval_rate: float) -> None:
        """Save test set metrics back to champion_metadata.json.

        Args:
            metrics: Test set metrics from ModelEvaluator.
            approval_rate: Fraction of test predictions below threshold.
        """
        meta_path = Path("models") / "champion_metadata.json"
        if not meta_path.exists():
            self.logger.warning("champion_metadata.json not found, skipping")
            return

        with open(meta_path, encoding="utf-8") as f:
            metadata = json.load(f)

        metadata["test_set_metrics"] = {
            "auc": metrics.get("auc"),
            "precision_default": metrics.get("precision_default"),
            "recall_default": metrics.get("recall_default"),
            "f1_default": metrics.get("f1_default"),
            "brier_score": metrics.get("brier_score"),
            "approval_rate_at_threshold": round(approval_rate, 4),
        }

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        self.logger.info("Test metrics saved to %s", meta_path)


def run_evaluation_pipeline() -> dict:
    """Module-level entry point for the evaluation pipeline.

    Returns:
        dict: Test set metrics.
    """
    pipeline = EvaluationPipeline()
    return pipeline.run()
