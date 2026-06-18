"""MLflow tracking wrapper with DagsHub support.

Provides a context manager and helpers that integrate with
the project's config system and logger.
"""

from __future__ import annotations

import os
import subprocess
from contextlib import contextmanager
from typing import Any, Generator, Optional

import mlflow
from mlflow.tracking import MlflowClient

from bnpl.logger import get_logger

logger = get_logger(__name__)


def _get_git_commit_hash() -> str:
    """Return the short git commit hash, or 'unknown' on failure."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        if result.returncode == 0:
            return result.stdout.strip()[:8]
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return "unknown"


def configure_tracking(tracking_uri: str, backend: str = "local") -> None:
    """Configure MLflow tracking URI and DagsHub credentials if needed.

    Args:
        tracking_uri: MLflow tracking server URI or local path.
        backend: ``"local"`` or ``"dagshub"``.
    """
    if backend == "dagshub":
        from config.settings import get_settings

        settings = get_settings()
        if settings.dagshub_username and settings.dagshub_token:
            os.environ["MLFLOW_TRACKING_USERNAME"] = settings.dagshub_username
            os.environ["MLFLOW_TRACKING_PASSWORD"] = settings.dagshub_token
            logger.info(
                f"Configured DagsHub MLflow tracking for user={settings.dagshub_username}"
            )
        else:
            logger.warning(
                "DagsHub backend selected but DAGSHUB_USERNAME or DAGSHUB_TOKEN not set. "
                "MLflow operations may fail."
            )

    mlflow.set_tracking_uri(tracking_uri)
    logger.info(f"MLflow tracking URI set to: {tracking_uri}")


def get_or_create_experiment(name: str) -> str:
    """Get an existing experiment by name or create a new one.

    Returns:
        The experiment ID as a string.
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name(name)
    if experiment is not None:
        logger.debug(f"Found existing experiment: {name} (id={experiment.experiment_id})")
        return experiment.experiment_id

    experiment_id = client.create_experiment(name)
    logger.info(f"Created new experiment: {name} (id={experiment_id})")
    return experiment_id


@contextmanager
def mlflow_run(
    run_name: str,
    experiment_name: Optional[str] = None,
    tags: Optional[dict[str, str]] = None,
    nested: bool = False,
) -> Generator[mlflow.ActiveRun, None, None]:
    """Context manager for MLflow runs with automatic tagging.

    Automatically tags each run with the current git commit hash
    and config environment. Logs run lifecycle through the project logger.

    Args:
        run_name: Display name for the run.
        experiment_name: Experiment to log under. Uses config default if None.
        tags: Additional tags to attach to the run.
        nested: Whether this is a nested run.

    Yields:
        The active MLflow run object.
    """
    from config.settings import get_settings

    settings = get_settings()

    configure_tracking(
        tracking_uri=settings.tracking.tracking_uri,
        backend=settings.tracking.backend,
    )

    exp_name = experiment_name or settings.model.experiment_name
    experiment_id = get_or_create_experiment(exp_name)

    auto_tags = {
        "git_commit": _get_git_commit_hash(),
        "config_env": settings.app_env,
    }
    if tags:
        auto_tags.update(tags)

    logger.info(
        f"Starting MLflow run: name={run_name}, experiment={exp_name}, tags={auto_tags}"
    )

    with mlflow.start_run(
        run_name=run_name,
        experiment_id=experiment_id,
        tags=auto_tags,
        nested=nested,
    ) as active_run:
        logger.info(f"MLflow run started: id={active_run.info.run_id}")
        try:
            yield active_run
        except Exception:
            logger.error(
                f"MLflow run failed: id={active_run.info.run_id}",
                exc_info=True,
            )
            mlflow.set_tag("run_status", "FAILED")
            raise
        else:
            mlflow.set_tag("run_status", "COMPLETED")
            logger.info(f"MLflow run completed: id={active_run.info.run_id}")
