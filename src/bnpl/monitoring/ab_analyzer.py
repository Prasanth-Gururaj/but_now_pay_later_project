"""A/B test analysis for champion vs challenger model comparison.

Analyzes counterfactual log data to recommend whether the challenger
should be promoted to champion.
"""

from __future__ import annotations

import json
from pathlib import Path

from bnpl.logger import LoggerMixin, log_execution


class ABAnalyzer(LoggerMixin):
    """Analyze A/B test results to recommend champion or challenger.

    Since full loan outcome labels take 6 weeks to materialize
    in BNPL, this analyzer uses proxy metrics observable immediately:
    approval rate, probability distributions, and agreement rate.

    A challenger is recommended for promotion if it achieves a
    meaningfully higher approval rate without a corresponding
    increase in average predicted default probability.

    Usage::

        analyzer = ABAnalyzer()
        result = analyzer.analyze(min_requests=1000)

    Depends on:
        - reports/ab_log.jsonl for A/B test data
        - LoggerMixin: structured logging
    """

    def __init__(
        self,
        log_path: str | Path = "reports/ab_log.jsonl",
    ) -> None:
        """Initialize ABAnalyzer.

        Args:
            log_path: Path to the A/B test log JSONL file.
        """
        self._log_path = Path(log_path)

    @log_execution(operation="ABAnalyzer.analyze")
    def analyze(self, min_requests: int = 1000) -> dict:
        """Analyze A/B log and return promotion recommendation.

        Args:
            min_requests: Minimum total requests needed before
                          analysis is meaningful. Default 1000.

        Returns:
            dict with sufficient_data, counts, rates, diff,
            recommendation (promote/keep_champion/insufficient_data),
            and reason string.

        Raises:
            FileNotFoundError: If log file does not exist.
        """
        if not self._log_path.exists():
            return self._insufficient("Log file not found")

        entries = self._read_log()
        total = len(entries)

        if total < min_requests:
            return self._insufficient(
                f"Only {total} requests, need {min_requests}"
            )

        champ_entries = [e for e in entries if e.get("assigned_model") == "champion"]
        chall_entries = [e for e in entries if e.get("assigned_model") == "challenger"]

        champ_approval = self._calc_approval(champ_entries, "champion_decision")
        chall_approval = self._calc_approval(chall_entries, "challenger_decision")
        champ_avg_prob = self._calc_avg_prob(entries, "champion_probability")
        chall_avg_prob = self._calc_avg_prob(entries, "challenger_probability")

        diff = chall_approval - champ_approval
        prob_diff = chall_avg_prob - champ_avg_prob

        recommendation, reason = self._decide(diff, prob_diff)

        self.logger.info(
            "A/B analysis: %s | approval_diff=%.3f | prob_diff=%.3f",
            recommendation, diff, prob_diff,
        )

        return {
            "sufficient_data": True,
            "total_requests": total,
            "champion_requests": len(champ_entries),
            "challenger_requests": len(chall_entries),
            "champion_approval_rate": round(champ_approval, 4),
            "challenger_approval_rate": round(chall_approval, 4),
            "approval_rate_diff": round(diff, 4),
            "recommendation": recommendation,
            "reason": reason,
        }

    def _read_log(self) -> list[dict]:
        """Read all entries from the JSONL log file.

        Returns:
            list[dict]: Parsed log entries.
        """
        entries = []
        with open(self._log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries

    def _calc_approval(self, entries: list[dict], key: str) -> float:
        """Calculate approval rate from log entries.

        Args:
            entries: Log entries to analyze.
            key: Dict key for the decision field.

        Returns:
            float: Fraction of APPROVE decisions.
        """
        if not entries:
            return 0.0
        approved = sum(1 for e in entries if e.get(key) == "APPROVE")
        return approved / len(entries)

    def _calc_avg_prob(self, entries: list[dict], key: str) -> float:
        """Calculate average probability from log entries.

        Args:
            entries: Log entries to analyze.
            key: Dict key for the probability field.

        Returns:
            float: Mean probability value.
        """
        probs = [e[key] for e in entries if e.get(key) is not None]
        if not probs:
            return 0.0
        return sum(probs) / len(probs)

    def _decide(self, approval_diff: float, prob_diff: float) -> tuple[str, str]:
        """Make promotion recommendation based on rate differences.

        Args:
            approval_diff: Challenger approval rate minus champion.
            prob_diff: Challenger avg probability minus champion.

        Returns:
            tuple: (recommendation string, reason string).
        """
        if approval_diff > 0.02 and prob_diff <= 0.02:
            return (
                "promote",
                f"Challenger has {approval_diff:.1%} higher approval rate "
                f"without increased default probability ({prob_diff:+.1%}). "
                f"Safe to promote.",
            )
        return (
            "keep_champion",
            f"Challenger approval diff={approval_diff:+.1%}, "
            f"prob diff={prob_diff:+.1%}. "
            f"Not enough improvement to justify promotion.",
        )

    def _insufficient(self, reason: str) -> dict:
        """Return insufficient data response.

        Args:
            reason: Why analysis cannot proceed.

        Returns:
            dict: Response with recommendation="insufficient_data".
        """
        return {
            "sufficient_data": False,
            "total_requests": 0,
            "champion_requests": 0,
            "challenger_requests": 0,
            "champion_approval_rate": 0.0,
            "challenger_approval_rate": 0.0,
            "approval_rate_diff": 0.0,
            "recommendation": "insufficient_data",
            "reason": reason,
        }
