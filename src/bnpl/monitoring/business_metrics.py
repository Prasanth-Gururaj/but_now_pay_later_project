"""Business-level metrics: approval rates, expected loss, default rates.

Provides BusinessMetricsCalculator for computing business KPIs from
model predictions on a data window, used by the monitoring pipeline
to detect business-level degradation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bnpl.logger import LoggerMixin, log_execution


class BusinessMetricsCalculator(LoggerMixin):
    """Calculate business-level metrics from model predictions.

    Computes approval rate, expected loss, and default rate on
    approved loans for a given data window. These metrics trigger
    retraining when they deviate from the training baseline.

    Usage::

        calc = BusinessMetricsCalculator(threshold=0.45)
        metrics = calc.calculate(probabilities, loan_amounts, actuals)

    Depends on:
        - Decision threshold from config/thresholds.yaml
        - Cost values from config/thresholds.yaml
        - LoggerMixin: structured logging
    """

    def __init__(
        self,
        threshold: float,
        cost_false_negative: float | None = None,
    ) -> None:
        """Initialize with decision threshold and cost assumptions.

        Args:
            threshold: Decision threshold for APPROVE/DENY.
            cost_false_negative: Cost of approving a defaulter.
                                 If None, loads from config.
        """
        self._threshold = threshold
        self._cost_fn = cost_false_negative or self._load_cost_fn()

    def _load_cost_fn(self) -> float:
        """Load false negative cost from config.

        Returns:
            float: Cost of approving a defaulter.
        """
        try:
            from config.settings import get_settings
            return get_settings().thresholds.cost_matrix.get("cost_false_negative", 300)
        except Exception:
            return 300

    @log_execution(operation="BusinessMetricsCalculator.calculate")
    def calculate(
        self,
        probabilities: np.ndarray,
        loan_amounts: np.ndarray | None = None,
        actuals: np.ndarray | None = None,
    ) -> dict:
        """Calculate all business metrics for a prediction window.

        Args:
            probabilities: Model-predicted default probabilities.
            loan_amounts: Loan amounts per application (for loss calc).
                          If None, expected_loss is not calculated.
            actuals: Actual default outcomes (0/1). If None,
                     default_rate_on_approved is not calculated.

        Returns:
            dict with keys:
                - approval_rate (float): fraction approved
                - denial_rate (float): fraction denied
                - expected_loss (float or None): estimated loss
                - default_rate_on_approved (float or None): actual default
                  rate among approved applications
                - mean_probability (float): average predicted probability
                - threshold_used (float): the threshold applied
        """
        decisions = (probabilities < self._threshold).astype(int)
        n_total = len(probabilities)
        n_approved = int(decisions.sum())
        approval_rate = n_approved / n_total if n_total > 0 else 0.0

        expected_loss = self._calc_expected_loss(
            probabilities, loan_amounts, decisions,
        )

        default_on_approved = self._calc_default_on_approved(
            actuals, decisions,
        )

        self.logger.info(
            "Business metrics: approval=%.1f%% | mean_prob=%.4f",
            approval_rate * 100, float(probabilities.mean()),
        )

        return {
            "approval_rate": round(float(approval_rate), 4),
            "denial_rate": round(1.0 - float(approval_rate), 4),
            "expected_loss": expected_loss,
            "default_rate_on_approved": default_on_approved,
            "mean_probability": round(float(probabilities.mean()), 4),
            "threshold_used": self._threshold,
        }

    def _calc_expected_loss(
        self,
        probabilities: np.ndarray,
        loan_amounts: np.ndarray | None,
        decisions: np.ndarray,
    ) -> float | None:
        """Calculate expected loss on approved loans.

        Args:
            probabilities: Default probabilities.
            loan_amounts: Loan amounts per application.
            decisions: Binary approval decisions (1=approved).

        Returns:
            float or None: Expected loss, or None if no loan amounts.
        """
        if loan_amounts is None:
            return None
        approved_mask = decisions == 1
        if not approved_mask.any():
            return 0.0
        loss = (probabilities[approved_mask] * loan_amounts[approved_mask]).sum()
        return round(float(loss), 2)

    def _calc_default_on_approved(
        self,
        actuals: np.ndarray | None,
        decisions: np.ndarray,
    ) -> float | None:
        """Calculate actual default rate among approved applications.

        Args:
            actuals: Actual default outcomes (0/1).
            decisions: Binary approval decisions (1=approved).

        Returns:
            float or None: Default rate on approved, or None if no actuals.
        """
        if actuals is None:
            return None
        approved_mask = decisions == 1
        if not approved_mask.any():
            return 0.0
        rate = actuals[approved_mask].mean()
        return round(float(rate), 4)
