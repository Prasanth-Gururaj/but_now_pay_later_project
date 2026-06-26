"""Model inference service with validation and prediction logging.

Provides InferenceEngine that wraps the Predictor with additional
input validation, prediction logging, and batch support.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from bnpl.logger import LoggerMixin, log_execution
from bnpl.models.predictor import Predictor


class InferenceEngine(LoggerMixin):
    """Production inference engine wrapping the Predictor class.

    Adds input validation beyond Pydantic (business rule checks),
    prediction logging to a local JSONL file for monitoring, and
    batch prediction support.

    Usage::

        engine = InferenceEngine(predictor, log_path="logs/predictions.jsonl")
        result = engine.predict(raw_input)
        batch_results = engine.predict_batch([input1, input2])

    Depends on:
        - Predictor: model loading and inference
        - LoggerMixin: structured logging
    """

    def __init__(
        self,
        predictor: Predictor,
        log_path: str | Path | None = None,
    ) -> None:
        """Initialize with a Predictor instance.

        Args:
            predictor: Configured Predictor with model and pipeline loaded.
            log_path: Path to JSONL file for prediction logging.
                      If None, prediction logging is disabled.
        """
        self._predictor = predictor
        self._log_path = Path(log_path) if log_path else None

    @log_execution(operation="InferenceEngine.predict")
    def predict(self, raw_input: dict) -> dict:
        """Run a single prediction with validation and logging.

        Args:
            raw_input: Dictionary with 17 raw loan application fields.

        Returns:
            dict: Prediction result with decision, probability, etc.

        Raises:
            ValueError: If business rule validation fails.
        """
        self._validate_business_rules(raw_input)
        result = self._predictor.predict(raw_input)
        self._log_prediction(raw_input, result)
        return result

    def predict_batch(self, inputs: list[dict]) -> list[dict]:
        """Run predictions on a batch of inputs.

        Args:
            inputs: List of raw input dicts, each with 17 fields.

        Returns:
            list[dict]: One prediction result per input.
        """
        return [self.predict(raw) for raw in inputs]

    def _validate_business_rules(self, raw_input: dict) -> None:
        """Check business rules beyond Pydantic type validation.

        Args:
            raw_input: Raw input dict to validate.

        Raises:
            ValueError: If a business rule is violated.
        """
        term = raw_input.get("term", "")
        if isinstance(term, str) and term not in ("36 months", "60 months", ""):
            self.logger.warning("Unusual term value: %s", term)

        fico = raw_input.get("fico_range_low", 0)
        if isinstance(fico, (int, float)) and fico < 300:
            raise ValueError(f"FICO score {fico} is below minimum 300")

    def _log_prediction(self, raw_input: dict, result: dict) -> None:
        """Append prediction to JSONL log file for monitoring.

        Args:
            raw_input: The input that was scored.
            result: The prediction result dict.
        """
        if self._log_path is None:
            return

        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "input": raw_input,
                "result": result,
            }
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception as exc:
            self.logger.warning("Prediction logging failed: %s", exc)

    @property
    def predictor(self) -> Predictor:
        """Return the underlying Predictor instance.

        Returns:
            Predictor: The wrapped predictor.
        """
        return self._predictor
