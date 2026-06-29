"""Model inference for BNPL default prediction.

Provides the Predictor class that combines the preprocessing pipeline
with the champion XGBoost model to produce credit decisions from raw
loan application data.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import joblib

from bnpl.features.pipeline import PreprocessingPipeline
from bnpl.logger import LoggerMixin, log_execution


class Predictor(LoggerMixin):
    """Single point of contact between raw application data and credit decisions.

    Loads the champion XGBoost model once at initialization and reuses it
    for all subsequent predictions. Each prediction transforms the raw
    17-field input through the PreprocessingPipeline, then runs inference
    to produce a default probability and an APPROVE/DENY decision.

    The model and preprocessing parameters are loaded from disk at init
    time, making prediction calls fast (no I/O per request). The decision
    threshold is configurable and defaults to the business-constrained
    value from config.

    Usage::

        predictor = Predictor(
            model_path="models/champion_xgboost.pkl",
            config_path="reports/data_prep_config.json",
            threshold=0.45,
        )
        result = predictor.predict({"dti": 18.5, "fico_range_low": 690, ...})

    Depends on:
        - PreprocessingPipeline: transforms raw input to model-ready features
        - champion_xgboost.pkl: serialized XGBoost model trained in Notebook 04
        - LoggerMixin: structured logging
    """

    MODEL_VERSION: str = "champion_xgboost_v1"

    def __init__(
        self,
        model_path: str | Path,
        config_path: str | Path,
        threshold: float,
    ) -> None:
        """Load model and preprocessing pipeline.

        Args:
            model_path: Path to the serialized XGBoost model (.pkl file).
            config_path: Path to data_prep_config.json with transformation
                         parameters fitted on training data.
            threshold: Decision threshold for the APPROVE/DENY boundary.
                       Probabilities >= threshold produce DENY.

        Raises:
            FileNotFoundError: If model or config file does not exist.
        """
        self._model = joblib.load(model_path)
        self._pipeline = PreprocessingPipeline(config_path)
        self._threshold = threshold

        self.logger.info(
            "Predictor ready | model=%s | threshold=%.2f | version=%s",
            model_path,
            threshold,
            self.MODEL_VERSION,
        )

    @property
    def model_version(self) -> str:
        """Return the model version identifier string.

        Returns:
            str: Version string such as ``champion_xgboost_v1``.
        """
        return self.MODEL_VERSION

    @property
    def threshold(self) -> float:
        """Return the current decision threshold.

        Returns:
            float: Threshold value where probabilities >= threshold are DENY.
        """
        return self._threshold

    @property
    def pipeline(self) -> PreprocessingPipeline:
        """Return the preprocessing pipeline instance.

        Returns:
            PreprocessingPipeline: The pipeline used for feature transforms.
        """
        return self._pipeline

    @log_execution(operation="Predictor.predict")
    def predict(self, raw_input: dict) -> dict:
        """Run a single prediction on raw loan application data.

        Transforms the 17 raw fields through the preprocessing pipeline,
        runs XGBoost inference to get a default probability, and applies
        the decision threshold to produce an APPROVE or DENY decision.

        Args:
            raw_input: Dictionary containing the 17 raw loan application
                       fields (dti, fico_range_low, revol_util, etc.).

        Returns:
            dict with keys:
                - decision (str): ``"APPROVE"`` or ``"DENY"``
                - default_probability (float): probability rounded to 4 dp
                - threshold_used (float): the threshold applied
                - model_version (str): model identifier
                - timestamp (str): ISO 8601 UTC timestamp

        Raises:
            ValueError: If preprocessing produces invalid features.
        """
        features = self._pipeline.transform(raw_input)
        probability = float(self._model.predict_proba(features)[0][1])
        decision = "DENY" if probability >= self._threshold else "APPROVE"

        return {
            "decision": decision,
            "default_probability": round(probability, 4),
            "threshold_used": self._threshold,
            "model_version": self.MODEL_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
