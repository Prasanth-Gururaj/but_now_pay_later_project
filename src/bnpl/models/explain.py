"""Model explainability with SHAP.

Provides ModelExplainer for generating global SHAP summary plots,
local waterfall plots for individual predictions, and drift
comparison between time periods.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from bnpl.logger import LoggerMixin, log_execution


class ModelExplainer(LoggerMixin):
    """Generate SHAP explanations for the champion model.

    Supports three explanation types:
    - Global: summary plot showing top feature importances
    - Local: waterfall plot for a single prediction
    - Drift: comparison of SHAP values between two time periods

    Usage::

        explainer = ModelExplainer(model)
        explainer.explain_global(X_test)
        explainer.explain_local(single_input_df)

    Depends on:
        - SHAP library for TreeExplainer
        - matplotlib for plot generation
        - LoggerMixin: structured logging
    """

    SAMPLE_SIZE: int = 5000
    MAX_DISPLAY: int = 15

    def __init__(self, model: object) -> None:
        """Initialize with a trained model.

        Args:
            model: Trained XGBoost or LightGBM model with
                   predict_proba method.
        """
        self._model = model

    @log_execution(operation="ModelExplainer.explain_global")
    def explain_global(
        self,
        X: pd.DataFrame,
        output_path: str | Path | None = None,
    ) -> Path:
        """Generate SHAP summary plot for top features.

        Args:
            X: Feature DataFrame to explain (sampled to 5000 rows).
            output_path: Where to save the plot. Defaults to
                         ``reports/shap_summary.png``.

        Returns:
            Path: Where the plot was saved.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import shap

        sample = X.sample(min(self.SAMPLE_SIZE, len(X)), random_state=42)
        explainer = shap.TreeExplainer(self._model)
        shap_values = explainer.shap_values(sample)

        out = Path(output_path or "reports/shap_summary.png")
        out.parent.mkdir(parents=True, exist_ok=True)

        shap.summary_plot(
            shap_values, sample, show=False, max_display=self.MAX_DISPLAY,
        )
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close("all")

        self.logger.info("SHAP summary plot saved to %s", out)
        return out

    @log_execution(operation="ModelExplainer.explain_local")
    def explain_local(
        self,
        X_single: pd.DataFrame,
        output_path: str | Path | None = None,
    ) -> Path:
        """Generate SHAP waterfall plot for a single prediction.

        Args:
            X_single: Single-row DataFrame with model features.
            output_path: Where to save the plot. Defaults to
                         ``reports/shap_waterfall.png``.

        Returns:
            Path: Where the plot was saved.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import shap

        explainer = shap.TreeExplainer(self._model)
        shap_values = explainer(X_single)

        out = Path(output_path or "reports/shap_waterfall.png")
        out.parent.mkdir(parents=True, exist_ok=True)

        shap.plots.waterfall(shap_values[0], show=False, max_display=self.MAX_DISPLAY)
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close("all")

        self.logger.info("SHAP waterfall plot saved to %s", out)
        return out

    @log_execution(operation="ModelExplainer.explain_drift")
    def explain_drift(
        self,
        X_period1: pd.DataFrame,
        X_period2: pd.DataFrame,
        output_path: str | Path | None = None,
    ) -> dict:
        """Compare SHAP values between two time periods.

        Shows which features changed most in their SHAP importance
        between period1 and period2, indicating drift drivers.

        Args:
            X_period1: Features from the first time period.
            X_period2: Features from the second time period.
            output_path: Where to save the comparison plot. Defaults to
                         ``reports/shap_drift_comparison.png``.

        Returns:
            dict: Feature importance changes between periods.
        """
        import shap

        explainer = shap.TreeExplainer(self._model)

        s1 = X_period1.sample(min(self.SAMPLE_SIZE, len(X_period1)), random_state=42)
        s2 = X_period2.sample(min(self.SAMPLE_SIZE, len(X_period2)), random_state=42)

        sv1 = explainer.shap_values(s1)
        sv2 = explainer.shap_values(s2)

        import numpy as np
        mean1 = np.abs(sv1).mean(axis=0)
        mean2 = np.abs(sv2).mean(axis=0)

        changes = {}
        for i, col in enumerate(s1.columns):
            changes[col] = {
                "period1_importance": round(float(mean1[i]), 6),
                "period2_importance": round(float(mean2[i]), 6),
                "change": round(float(mean2[i] - mean1[i]), 6),
            }

        sorted_changes = dict(
            sorted(changes.items(), key=lambda x: abs(x[1]["change"]), reverse=True)
        )

        self.logger.info(
            "Drift SHAP comparison: top changed feature = %s",
            next(iter(sorted_changes), "none"),
        )
        return sorted_changes
