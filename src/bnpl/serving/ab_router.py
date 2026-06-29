"""A/B testing router for champion vs challenger model comparison.

Routes prediction requests between champion and challenger models
with deterministic assignment and counterfactual logging.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from bnpl.logger import LoggerMixin, log_execution
from bnpl.models.predictor import Predictor


class ABRouter(LoggerMixin):
    """Route prediction requests between champion and challenger models.

    Implements a 90/10 traffic split with counterfactual logging.
    Both models score every request but only the assigned model's
    decision is returned to the caller. This allows safe evaluation
    of a challenger model without exposing users to full risk.

    The same applicant always hits the same model because routing
    uses a deterministic hash of the request_id. This prevents
    the same person getting inconsistent decisions on retries.

    Usage::

        router = ABRouter(champion_predictor)
        router.load_challenger("models/challenger.pkl", "reports/data_prep_config.json", 0.45)
        result = router.route(raw_input, request_id="req-123")

    Depends on:
        - Predictor class from src/bnpl/models/predictor.py
        - LoggerMixin: structured logging
    """

    def __init__(
        self,
        champion_predictor: Predictor,
        challenger_predictor: Predictor | None = None,
        challenger_pct: float = 0.10,
        log_path: str | Path = "reports/ab_log.jsonl",
    ) -> None:
        """Initialize ABRouter with champion and optional challenger.

        Args:
            champion_predictor: Loaded Predictor instance for champion.
            challenger_predictor: Loaded Predictor for challenger.
                                 If None, all traffic goes to champion.
            challenger_pct: Fraction of traffic routed to challenger.
                           Default 0.10 means 10 percent.
            log_path: Path to JSONL file for counterfactual logging.

        Raises:
            ValueError: If challenger_pct is not between 0 and 1.
        """
        if not 0 <= challenger_pct <= 1:
            raise ValueError(f"challenger_pct must be 0-1, got {challenger_pct}")

        self._champion = champion_predictor
        self._challenger = challenger_predictor
        self._challenger_pct = challenger_pct
        self._log_path = Path(log_path)

    @log_execution(operation="ABRouter.route")
    def route(self, raw_input: dict, request_id: str) -> dict:
        """Route request to champion or challenger based on request_id hash.

        Both models score the input regardless of assignment.
        Only the assigned model's decision is returned.

        Args:
            raw_input: Dictionary with 17 raw loan application fields.
            request_id: Unique ID for this request. Used for routing.

        Returns:
            dict with decision, default_probability, threshold_used,
            model_version, assigned_model, request_id, timestamp.

        Raises:
            RuntimeError: If champion predictor fails to score.
        """
        assigned = self._assign_model(request_id)
        champion_result = self._champion.predict(raw_input)

        challenger_result = None
        if self._challenger is not None:
            try:
                challenger_result = self._challenger.predict(raw_input)
            except Exception as exc:
                self.logger.warning("Challenger scoring failed: %s", exc)

        returned = challenger_result if (assigned == "challenger" and challenger_result) else champion_result
        returned["assigned_model"] = assigned
        returned["request_id"] = request_id

        self._log_counterfactual(
            request_id, assigned, champion_result, challenger_result, returned,
        )

        return returned

    def load_challenger(
        self, model_path: str, config_path: str, threshold: float,
    ) -> None:
        """Load a challenger model for A/B testing.

        Args:
            model_path: Path to challenger model pkl file.
            config_path: Path to data_prep_config.json for challenger.
            threshold: Decision threshold for challenger.

        Raises:
            FileNotFoundError: If model_path does not exist.
        """
        self._challenger = Predictor(model_path, config_path, threshold)
        self.logger.info("Challenger loaded from %s (threshold=%.2f)", model_path, threshold)

    def get_summary(self) -> dict:
        """Return summary statistics from the A/B log file.

        Returns:
            dict with total_requests, champion/challenger counts,
            approval rates, average probabilities, agreement rate.
        """
        if not self._log_path.exists():
            return self._empty_summary()

        champion_probs, challenger_probs = [], []
        champion_decisions, challenger_decisions = [], []
        champion_count, challenger_count = 0, 0

        try:
            with open(self._log_path, encoding="utf-8") as f:
                for line in f:
                    entry = json.loads(line.strip())
                    if entry.get("assigned_model") == "champion":
                        champion_count += 1
                    else:
                        challenger_count += 1
                    if entry.get("champion_probability") is not None:
                        champion_probs.append(entry["champion_probability"])
                        champion_decisions.append(entry["champion_decision"])
                    if entry.get("challenger_probability") is not None:
                        challenger_probs.append(entry["challenger_probability"])
                        challenger_decisions.append(entry["challenger_decision"])
        except Exception:
            return self._empty_summary()

        total = champion_count + challenger_count
        agree = sum(
            1 for c, ch in zip(champion_decisions, challenger_decisions) if c == ch
        ) if challenger_decisions else 0

        return {
            "total_requests": total,
            "champion_requests": champion_count,
            "challenger_requests": challenger_count,
            "champion_approval_rate": self._approval_rate(champion_decisions),
            "challenger_approval_rate": self._approval_rate(challenger_decisions),
            "champion_avg_probability": self._avg(champion_probs),
            "challenger_avg_probability": self._avg(challenger_probs),
            "agreement_rate": round(agree / len(challenger_decisions), 4) if challenger_decisions else 0.0,
            "has_challenger": self._challenger is not None,
        }

    def _assign_model(self, request_id: str) -> str:
        """Deterministically assign a model based on request_id hash.

        Args:
            request_id: Unique request identifier.

        Returns:
            str: "champion" or "challenger".
        """
        hash_val = int(hashlib.md5(request_id.encode()).hexdigest(), 16) % 100
        if hash_val < self._challenger_pct * 100 and self._challenger is not None:
            return "challenger"
        return "champion"

    def _log_counterfactual(
        self, request_id: str, assigned: str,
        champion: dict, challenger: dict | None, returned: dict,
    ) -> None:
        """Log both model scores for offline comparison.

        Args:
            request_id: Request identifier.
            assigned: Which model was assigned.
            champion: Champion prediction result.
            challenger: Challenger prediction result or None.
            returned: The result actually returned to caller.
        """
        entry = {
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "assigned_model": assigned,
            "champion_probability": champion.get("default_probability"),
            "champion_decision": champion.get("decision"),
            "challenger_probability": challenger.get("default_probability") if challenger else None,
            "challenger_decision": challenger.get("decision") if challenger else None,
            "returned_decision": returned.get("decision"),
        }

        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as exc:
            self.logger.warning("A/B log write failed: %s", exc)

    def _empty_summary(self) -> dict:
        """Return empty summary when no log data exists.

        Returns:
            dict: All counts zero, rates zero.
        """
        return {
            "total_requests": 0, "champion_requests": 0,
            "challenger_requests": 0, "champion_approval_rate": 0.0,
            "challenger_approval_rate": 0.0, "champion_avg_probability": 0.0,
            "challenger_avg_probability": 0.0, "agreement_rate": 0.0,
            "has_challenger": self._challenger is not None,
        }

    @staticmethod
    def _approval_rate(decisions: list[str]) -> float:
        """Calculate approval rate from a list of decisions.

        Args:
            decisions: List of "APPROVE" or "DENY" strings.

        Returns:
            float: Fraction of APPROVE decisions.
        """
        if not decisions:
            return 0.0
        return round(sum(1 for x in decisions if x == "APPROVE") / len(decisions), 4)

    @staticmethod
    def _avg(values: list[float]) -> float:
        """Calculate average of a list of floats.

        Args:
            values: List of numeric values.

        Returns:
            float: Mean value rounded to 4 decimal places.
        """
        if not values:
            return 0.0
        return round(sum(values) / len(values), 4)
