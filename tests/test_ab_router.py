"""Tests for the A/B testing router."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from bnpl.serving.ab_router import ABRouter


def _make_mock_predictor(decision: str = "APPROVE", prob: float = 0.3) -> MagicMock:
    """Create a mock Predictor that returns fixed results."""
    predictor = MagicMock()
    predictor.predict.return_value = {
        "decision": decision,
        "default_probability": prob,
        "threshold_used": 0.45,
        "model_version": "mock_v1",
        "timestamp": "2026-01-01T00:00:00+00:00",
    }
    predictor.model_version = "mock_v1"
    predictor.threshold = 0.45
    return predictor


class TestABRouterChampionOnly:
    """Verify champion is always used when no challenger is loaded."""

    def test_champion_always_used_when_no_challenger(self, tmp_path: Path) -> None:
        """All requests must be routed to champion when no challenger exists."""
        champion = _make_mock_predictor("APPROVE", 0.25)
        router = ABRouter(champion, log_path=tmp_path / "ab.jsonl")

        for i in range(10):
            result = router.route({"dti": 18.5}, request_id=f"req-{i}")
            assert result["assigned_model"] == "champion"


class TestDeterministicRouting:
    """Verify same request_id always routes to same model."""

    def test_deterministic_routing_same_id_same_model(self, tmp_path: Path) -> None:
        """The same request_id must always hit the same model across calls."""
        champion = _make_mock_predictor("APPROVE", 0.25)
        challenger = _make_mock_predictor("DENY", 0.55)
        router = ABRouter(
            champion, challenger, challenger_pct=0.50,
            log_path=tmp_path / "ab.jsonl",
        )

        results = [
            router.route({"dti": 18.5}, request_id="fixed-id-123")
            for _ in range(5)
        ]
        models = [r["assigned_model"] for r in results]
        assert len(set(models)) == 1, f"Got inconsistent routing: {models}"


class TestABLogWritten:
    """Verify the log file is created after a route call."""

    def test_ab_log_written_after_route_call(self, tmp_path: Path) -> None:
        """Log file must exist and contain valid JSON after one route call."""
        import json

        champion = _make_mock_predictor("APPROVE", 0.25)
        log_path = tmp_path / "ab.jsonl"
        router = ABRouter(champion, log_path=log_path)

        router.route({"dti": 18.5}, request_id="test-req-1")

        assert log_path.exists(), "Log file should exist after route()"
        with open(log_path, encoding="utf-8") as f:
            line = f.readline().strip()
        entry = json.loads(line)
        assert entry["request_id"] == "test-req-1"
        assert entry["assigned_model"] == "champion"
        assert entry["champion_probability"] == 0.25
