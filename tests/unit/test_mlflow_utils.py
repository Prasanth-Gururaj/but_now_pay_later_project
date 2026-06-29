"""Tests for MLflow tracking wrapper."""

from __future__ import annotations

import os

import mlflow
import pytest

from bnpl.tracking.mlflow_utils import configure_tracking, get_or_create_experiment, mlflow_run

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"


@pytest.fixture()
def mlflow_tmp(tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> str:
    """Point MLflow at a temp directory to avoid polluting the repo."""
    tracking_uri = tmp_path.as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    return tracking_uri


class TestConfigureTracking:
    """Verify tracking URI setup."""

    def test_local_backend(self, mlflow_tmp: str) -> None:
        configure_tracking(mlflow_tmp, "local")
        assert mlflow.get_tracking_uri() == mlflow_tmp


class TestGetOrCreateExperiment:
    """Verify experiment creation and retrieval."""

    def test_creates_new(self, mlflow_tmp: str) -> None:
        exp_id = get_or_create_experiment("test-experiment")
        assert exp_id is not None

        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment(exp_id)
        assert exp.name == "test-experiment"

    def test_returns_existing(self, mlflow_tmp: str) -> None:
        id1 = get_or_create_experiment("same-experiment")
        id2 = get_or_create_experiment("same-experiment")
        assert id1 == id2


class TestMlflowRun:
    """Verify the mlflow_run context manager."""

    def test_completes_with_tags(self, mlflow_tmp: str) -> None:
        with mlflow_run("test-run", experiment_name="test-exp") as run:
            mlflow.log_param("key", "value")
            mlflow.log_metric("score", 0.95)
            run_id = run.info.run_id

        client = mlflow.tracking.MlflowClient()
        finished_run = client.get_run(run_id)
        assert finished_run.data.tags["run_status"] == "COMPLETED"
        assert "git_commit" in finished_run.data.tags
        assert "config_env" in finished_run.data.tags

    def test_failure_sets_failed_tag(self, mlflow_tmp: str) -> None:
        run_id = None
        with pytest.raises(RuntimeError, match="deliberate"), \
             mlflow_run("fail-run", experiment_name="test-exp") as run:
                run_id = run.info.run_id
                raise RuntimeError("deliberate failure")

        assert run_id is not None
        client = mlflow.tracking.MlflowClient()
        finished_run = client.get_run(run_id)
        assert finished_run.data.tags["run_status"] == "FAILED"
