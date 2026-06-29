"""Model evaluation and metrics computation.

Provides ModelEvaluator that scores the champion model on the test
set, generates calibration and SHAP plots, and logs metrics to MLflow.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    brier_score_loss,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from bnpl.logger import LoggerMixin, log_execution


class ModelEvaluator(LoggerMixin):
    """Evaluate the champion model on a held-out test set.

    Computes classification metrics, generates calibration and SHAP
    summary plots, saves them to reports/, and logs everything to MLflow.

    The test set should only be touched ONCE for the final unbiased
    evaluation after all model selection and threshold tuning is done.

    Usage::

        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(model, X_test, y_test, threshold=0.31)

    Depends on:
        - Champion model with predict_proba method
        - SHAP for feature importance visualization
        - MLflow for metrics logging
        - LoggerMixin: structured logging
    """

    @log_execution(operation="ModelEvaluator.evaluate")
    def evaluate(
        self,
        model: object,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        threshold: float,
        log_to_mlflow: bool = True,
    ) -> dict:
        """Run full evaluation: metrics, plots, and MLflow logging.

        Args:
            model: Trained model with ``predict_proba`` method.
            X_test: Test features DataFrame.
            y_test: Test target Series.
            threshold: Decision threshold for classification.
            log_to_mlflow: Whether to log metrics to MLflow.

        Returns:
            dict with keys: auc, precision_default, recall_default,
            f1_default, brier_score, threshold, classification_report.
        """
        test_proba = model.predict_proba(X_test)[:, 1]
        metrics = self._compute_metrics(y_test, test_proba, threshold)

        self._print_report(metrics, y_test, test_proba, threshold)

        try:
            cal_path = self._generate_calibration_plot(model, X_test, y_test)
            self.logger.info("Calibration plot saved to %s", cal_path)
        except Exception as exc:
            self.logger.warning("Calibration plot failed: %s", exc)

        try:
            shap_path = self._generate_shap_plot(model, X_test)
            self.logger.info("SHAP plot saved to %s", shap_path)
        except Exception as exc:
            self.logger.warning("SHAP plot failed: %s", exc)

        if log_to_mlflow:
            self._log_to_mlflow(metrics)

        return metrics

    def _compute_metrics(
        self, y_true: pd.Series, y_proba: np.ndarray, threshold: float
    ) -> dict:
        """Compute all classification metrics.

        Args:
            y_true: True binary labels.
            y_proba: Predicted probabilities.
            threshold: Decision threshold.

        Returns:
            dict: All metrics including AUC, precision, recall, F1, Brier.
        """
        y_pred = (y_proba >= threshold).astype(int)
        return {
            "threshold": threshold,
            "auc": roc_auc_score(y_true, y_proba),
            "precision_default": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
            "recall_default": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
            "f1_default": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
            "brier_score": brier_score_loss(y_true, y_proba),
        }

    def _print_report(
        self, metrics: dict, y_true: pd.Series,
        y_proba: np.ndarray, threshold: float,
    ) -> None:
        """Print formatted classification report.

        Args:
            metrics: Computed metrics dict.
            y_true: True binary labels.
            y_proba: Predicted probabilities.
            threshold: Decision threshold.
        """
        y_pred = (y_proba >= threshold).astype(int)
        print(f"\n=== Test Set Evaluation (threshold={threshold}) ===")
        print(f"  AUC:                 {metrics['auc']:.4f}")
        print(f"  Precision (default): {metrics['precision_default']:.4f}")
        print(f"  Recall (default):    {metrics['recall_default']:.4f}")
        print(f"  F1 (default):        {metrics['f1_default']:.4f}")
        print(f"  Brier score:         {metrics['brier_score']:.4f}")
        print()
        print(classification_report(
            y_true, y_pred, target_names=["Fully Paid", "Charged Off"]
        ))

    def _generate_calibration_plot(
        self, model: object, X_test: pd.DataFrame, y_test: pd.Series,
    ) -> Path:
        """Generate and save a calibration curve plot.

        Args:
            model: Trained model with predict_proba.
            X_test: Test features.
            y_test: Test labels.

        Returns:
            Path: Where the plot was saved.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.calibration import calibration_curve

        y_proba = model.predict_proba(X_test)[:, 1]
        prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(prob_pred, prob_true, marker="o", label="Champion model")
        ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfectly calibrated")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title("Calibration Curve")
        ax.legend()
        ax.grid(True, alpha=0.3)

        out_path = Path("reports") / "calibration_plot.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def _generate_shap_plot(self, model: object, X_test: pd.DataFrame) -> Path:
        """Generate SHAP summary plot on a 5000-row sample.

        Args:
            model: Trained tree model (XGBoost or LightGBM).
            X_test: Test features DataFrame.

        Returns:
            Path: Where the plot was saved.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import shap

        sample_size = min(5000, len(X_test))
        sample_X = X_test.sample(sample_size, random_state=42)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample_X)

        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, sample_X, show=False, max_display=15)

        out_path = Path("reports") / "shap_summary.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close("all")
        return out_path

    def _log_to_mlflow(self, metrics: dict) -> None:
        """Log test metrics to MLflow as a separate evaluation run.

        Args:
            metrics: Dict of metric name-value pairs.
        """
        try:
            import mlflow

            with mlflow.start_run(run_name="champion_test_evaluation"):
                numeric = {
                    f"test_{k}": v
                    for k, v in metrics.items()
                    if isinstance(v, (int, float))
                }
                mlflow.log_metrics(numeric)
                self.logger.info("Test metrics logged to MLflow")
        except Exception as exc:
            self.logger.warning("MLflow logging failed: %s", exc)
