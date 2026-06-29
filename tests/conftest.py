"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _set_dev_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure tests always run in dev environment."""
    monkeypatch.setenv("APP_ENV", "dev")


@pytest.fixture(autouse=True)
def _reset_settings() -> None:
    """Reset the settings singleton before and after each test."""
    from config.settings import reset_settings

    reset_settings()
    yield  # type: ignore[misc]
    reset_settings()


@pytest.fixture()
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parent.parent
