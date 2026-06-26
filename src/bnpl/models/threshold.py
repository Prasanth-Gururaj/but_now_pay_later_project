"""Decision threshold optimization using cost-sensitive analysis.

Provides ThresholdSelector that sweeps thresholds to find the one
minimizing total business cost, with an optional approval rate constraint.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from bnpl.logger import LoggerMixin, log_execution


class ThresholdSelector(LoggerMixin):
    """Select the optimal decision threshold based on business costs.

    Sweeps thresholds from 0.05 to 0.95 and calculates the total
    business cost at each point. Also finds a constrained threshold
    that meets a minimum approval rate requirement.

    All cost values are loaded from config/thresholds.yaml via Settings,
    never hardcoded.

    Usage::

        selector = ThresholdSelector()
        result = selector.select_optimal(y_val, val_proba)
        threshold = result["constrained_threshold"]

    Depends on:
        - config/thresholds.yaml: cost_false_negative, cost_false_positive
        - LoggerMixin: structured logging
    """

    def __init__(self) -> None:
        """Load cost values from Settings config."""
        costs = self._load_costs()
        self._cost_fn: float = costs["cost_false_negative"]
        self._cost_fp: float = costs["cost_false_positive"]

    def _load_costs(self) -> dict[str, float]:
        """Load business cost values from thresholds.yaml.

        Returns:
            dict: cost_false_negative and cost_false_positive values.
        """
        try:
            from config.settings import get_settings

            cm = get_settings().thresholds.cost_matrix
            return {
                "cost_false_negative": cm.get("cost_false_negative", 300),
                "cost_false_positive": cm.get("cost_false_positive", 45),
            }
        except Exception:
            return {"cost_false_negative": 300, "cost_false_positive": 45}

    @log_execution(operation="ThresholdSelector.select_optimal")
    def select_optimal(
        self, y_true: pd.Series | np.ndarray, y_proba: np.ndarray,
    ) -> dict:
        """Sweep thresholds and find cost-optimal and constrained options.

        Args:
            y_true: True binary labels.
            y_proba: Predicted probabilities for the positive class.

        Returns:
            dict with keys:
                - unconstrained_threshold (float): minimum cost threshold
                - unconstrained_cost (float): total cost at that threshold
                - unconstrained_approval_rate (float): approval rate
                - constrained_threshold (float): lowest cost at >= 55% approval
                - constrained_cost (float): total cost at constrained threshold
                - constrained_approval_rate (float): approval rate
                - cost_savings_vs_naive (float): savings vs 0.5 threshold
                - sweep_results (pd.DataFrame): full sweep data
        """
        sweep = self._sweep_thresholds(y_true, y_proba)

        optimal_idx = sweep["total_cost"].idxmin()
        optimal = sweep.loc[optimal_idx]

        constrained = self.select_constrained(y_true, y_proba, min_approval_rate=0.55)
        constrained_row = sweep[
            (sweep["threshold"] - constrained).abs() < 0.005
        ]
        constrained_cost = (
            constrained_row["total_cost"].iloc[0]
            if len(constrained_row) > 0 else optimal["total_cost"]
        )
        constrained_approval = (
            constrained_row["approval_rate"].iloc[0]
            if len(constrained_row) > 0 else optimal["approval_rate"]
        )

        naive_row = sweep[(sweep["threshold"] - 0.5).abs() < 0.005]
        naive_cost = naive_row["total_cost"].iloc[0] if len(naive_row) > 0 else 0

        self.logger.info(
            "Optimal threshold=%.2f (cost=$%.0f) | Constrained=%.2f (cost=$%.0f)",
            optimal["threshold"], optimal["total_cost"],
            constrained, constrained_cost,
        )

        return {
            "unconstrained_threshold": float(optimal["threshold"]),
            "unconstrained_cost": float(optimal["total_cost"]),
            "unconstrained_approval_rate": float(optimal["approval_rate"]),
            "constrained_threshold": constrained,
            "constrained_cost": float(constrained_cost),
            "constrained_approval_rate": float(constrained_approval),
            "cost_savings_vs_naive": float(naive_cost - optimal["total_cost"]),
            "sweep_results": sweep,
        }

    def select_constrained(
        self,
        y_true: pd.Series | np.ndarray,
        y_proba: np.ndarray,
        min_approval_rate: float = 0.55,
    ) -> float:
        """Find lowest cost threshold that meets minimum approval rate.

        Args:
            y_true: True binary labels.
            y_proba: Predicted probabilities.
            min_approval_rate: Minimum fraction of applications approved.

        Returns:
            float: The constrained threshold value.
        """
        sweep = self._sweep_thresholds(y_true, y_proba)
        feasible = sweep[sweep["approval_rate"] >= min_approval_rate]

        if feasible.empty:
            self.logger.warning(
                "No threshold achieves %.0f%% approval. Using unconstrained.",
                min_approval_rate * 100,
            )
            return float(sweep.loc[sweep["total_cost"].idxmin(), "threshold"])

        best_idx = feasible["total_cost"].idxmin()
        return float(feasible.loc[best_idx, "threshold"])

    def _sweep_thresholds(
        self, y_true: pd.Series | np.ndarray, y_proba: np.ndarray,
    ) -> pd.DataFrame:
        """Compute cost and approval rate for each threshold.

        Args:
            y_true: True binary labels.
            y_proba: Predicted probabilities.

        Returns:
            pd.DataFrame: One row per threshold with cost and rate columns.
        """
        thresholds = np.arange(0.05, 0.95, 0.01)
        rows: list[dict] = []

        for t in thresholds:
            y_pred = (y_proba >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            total_cost = (fn * self._cost_fn) + (fp * self._cost_fp)
            approval_rate = float((y_pred == 0).mean())
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0

            rows.append({
                "threshold": round(float(t), 2),
                "false_negatives": int(fn),
                "false_positives": int(fp),
                "total_cost": float(total_cost),
                "approval_rate": approval_rate,
                "recall_default": recall,
            })

        return pd.DataFrame(rows)
