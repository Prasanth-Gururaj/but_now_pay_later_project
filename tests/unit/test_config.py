"""Tests for the configuration system."""

from __future__ import annotations

import pytest
from config.settings import Settings, get_settings


class TestDevSettings:
    """Verify dev environment loads correctly."""

    def test_app_env_is_dev(self) -> None:
        settings = get_settings()
        assert settings.app_env == "dev"

    def test_sample_fraction_overridden(self) -> None:
        settings = get_settings()
        assert settings.data.sample_fraction == 0.1

    def test_tracking_backend_is_local(self) -> None:
        settings = get_settings()
        assert settings.tracking.backend == "local"
        assert settings.tracking.tracking_uri == "mlruns"

    def test_logging_level_is_debug(self) -> None:
        settings = get_settings()
        assert settings.logging.level == "DEBUG"

    def test_json_output_off_in_dev(self) -> None:
        settings = get_settings()
        assert settings.logging.json_output is False


class TestDeepMerge:
    """Verify base.yaml keys survive environment merge."""

    def test_base_keys_preserved(self) -> None:
        settings = get_settings()
        assert settings.project.name == "bnpl-default-prediction"
        assert settings.serving.host == "0.0.0.0"
        assert settings.serving.port == 8000
        assert settings.monitoring.psi_threshold == 0.2

    def test_data_paths_from_base(self) -> None:
        settings = get_settings()
        assert settings.data.raw_dir == "data/raw"
        assert settings.data.processed_dir == "data/processed"


class TestFeatures:
    """Verify features.yaml loading."""

    def test_feature_count(self) -> None:
        settings = get_settings()
        assert len(settings.features) == 20

    def test_feature_fields(self) -> None:
        settings = get_settings()
        for f in settings.features:
            assert f.name
            assert f.dtype
            assert f.category


class TestThresholds:
    """Verify threshold validation."""

    def test_default_threshold_in_range(self) -> None:
        settings = get_settings()
        assert 0.1 <= settings.thresholds.default_threshold <= 0.9

    def test_cost_matrix_loaded(self) -> None:
        settings = get_settings()
        cm = settings.thresholds.cost_matrix
        assert cm["false_negative"] == -500
        assert cm["false_positive"] == -100


class TestValidation:
    """Verify fail-fast validation."""

    def test_staging_missing_secrets_fails(self) -> None:
        with pytest.raises(ValueError, match="DAGSHUB_TOKEN"):
            Settings(app_env="staging", dagshub_token=None)

    def test_threshold_out_of_range_fails(self) -> None:
        with pytest.raises(ValueError, match="outside allowed range"):
            Settings(
                app_env="dev",
                thresholds={
                    "default_threshold": 0.05,
                    "business_rules": {"min_threshold": 0.1, "max_threshold": 0.9},
                },
            )


class TestSingleton:
    """Verify singleton behavior."""

    def test_returns_same_instance(self) -> None:
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_override_returns_fresh(self) -> None:
        s1 = get_settings()
        s2 = get_settings(app_env="dev")
        assert s1 is not s2

    def test_project_root_is_absolute(self) -> None:
        settings = get_settings()
        assert settings.project_root.is_absolute()
