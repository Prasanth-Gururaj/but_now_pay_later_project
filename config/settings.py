"""Pydantic-based configuration system with environment-aware YAML loading.

Loading order:
1. config/base.yaml (shared defaults)
2. config/{APP_ENV}.yaml (deep-merged over base)
3. Environment variables and .env file (for secrets only)
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base. Override wins on leaf conflicts."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file, returning empty dict if missing or empty."""
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        content = yaml.safe_load(f)
    return content if content is not None else {}


def load_merged_config(env: str) -> dict[str, Any]:
    """Load base.yaml then deep-merge with {env}.yaml."""
    base = _load_yaml(CONFIG_DIR / "base.yaml")
    env_specific = _load_yaml(CONFIG_DIR / f"{env}.yaml")
    return _deep_merge(base, env_specific)


# ---------------------------------------------------------------------------
# Nested sub-models
# ---------------------------------------------------------------------------
class ProjectConfig(BaseModel):
    """Project identity."""

    name: str = "bnpl-default-prediction"
    version: str = "0.1.0"


class PathsConfig(BaseModel):
    """Data directory paths."""

    raw_dir: str = "data/raw"
    interim_dir: str = "data/interim"
    processed_dir: str = "data/processed"
    reference_dir: str = "data/reference"


class DataConfig(BaseModel):
    """Data processing parameters."""

    raw_dir: str = "data/raw"
    interim_dir: str = "data/interim"
    processed_dir: str = "data/processed"
    reference_dir: str = "data/reference"
    test_size: float = 0.2
    random_state: int = 42
    sample_fraction: float = 1.0


class TrackingConfig(BaseModel):
    """MLflow tracking configuration."""

    backend: str = "local"
    tracking_uri: str = "mlruns"


class ModelConfig(BaseModel):
    """Model training configuration."""

    target_column: str = "default"
    primary_metric: str = "f1"
    experiment_name: str = "bnpl-default-prediction"


class ServingConfig(BaseModel):
    """API serving configuration."""

    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1


class MonitoringConfig(BaseModel):
    """Monitoring thresholds and intervals."""

    drift_check_interval_hours: int = 24
    psi_threshold: float = 0.2


class LoggingConfig(BaseModel):
    """Logging behaviour toggles."""

    level: str = "INFO"
    json_output: bool = False


class FeatureDefinition(BaseModel):
    """Single feature specification from features.yaml."""

    name: str
    dtype: str
    category: str


class ThresholdsConfig(BaseModel):
    """Decision threshold and cost matrix."""

    default_threshold: float = 0.5
    optimized_threshold: Optional[float] = None
    cost_matrix: dict[str, float] = Field(default_factory=dict)
    business_rules: dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main Settings
# ---------------------------------------------------------------------------
class Settings(BaseSettings):
    """Central configuration merging YAML files and environment variables.

    Secrets come only from env vars / .env:
    DAGSHUB_TOKEN, SUPABASE_URL, SUPABASE_KEY, RENDER_DEPLOY_HOOK.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Environment identifier
    app_env: str = Field(default="dev")

    # Secrets (from .env / env vars only)
    dagshub_token: Optional[str] = None
    dagshub_username: Optional[str] = None
    dagshub_repo: Optional[str] = None
    supabase_url: Optional[str] = None
    supabase_key: Optional[str] = None
    render_deploy_hook: Optional[str] = None

    # Nested configs populated from YAML
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    serving: ServingConfig = Field(default_factory=ServingConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    # Feature definitions from features.yaml
    features: list[FeatureDefinition] = Field(default_factory=list)

    # Thresholds from thresholds.yaml
    thresholds: ThresholdsConfig = Field(default_factory=ThresholdsConfig)

    # Computed
    project_root: Path = Field(default=PROJECT_ROOT)

    @model_validator(mode="before")
    @classmethod
    def _load_from_yaml(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Inject YAML-loaded values before Pydantic constructs sub-models."""
        env = data.get("app_env", os.getenv("APP_ENV", "dev"))
        data["app_env"] = env

        merged = load_merged_config(env)

        for section in (
            "project",
            "data",
            "tracking",
            "model",
            "serving",
            "monitoring",
            "logging",
        ):
            if section in merged and section not in data:
                data[section] = merged[section]
            elif section in merged and section in data:
                data[section] = _deep_merge(merged[section], data[section])

        if "features" not in data:
            features_raw = _load_yaml(CONFIG_DIR / "features.yaml")
            data["features"] = features_raw.get("features", [])

        if "thresholds" not in data:
            thresholds_raw = _load_yaml(CONFIG_DIR / "thresholds.yaml")
            if thresholds_raw:
                decision = thresholds_raw.get("decision", {})
                data["thresholds"] = {
                    "default_threshold": decision.get("default_threshold", 0.5),
                    "optimized_threshold": decision.get("optimized_threshold"),
                    "cost_matrix": thresholds_raw.get("cost_matrix", {}),
                    "business_rules": thresholds_raw.get("business_rules", {}),
                }

        return data

    @model_validator(mode="after")
    def _validate_settings(self) -> "Settings":
        """Fail-fast validation for environment-specific constraints."""
        if self.app_env in ("staging", "prod"):
            missing = []
            if not self.dagshub_token:
                missing.append("DAGSHUB_TOKEN")
            if not self.supabase_url:
                missing.append("SUPABASE_URL")
            if not self.supabase_key:
                missing.append("SUPABASE_KEY")
            if missing:
                raise ValueError(
                    f"Missing required secrets for {self.app_env}: {', '.join(missing)}"
                )

        if len(self.features) == 0:
            raise ValueError("No features defined in features.yaml")

        t = self.thresholds
        rules = t.business_rules
        if rules:
            min_t = rules.get("min_threshold", 0.1)
            max_t = rules.get("max_threshold", 0.9)
            if t.default_threshold < min_t or t.default_threshold > max_t:
                raise ValueError(
                    f"default_threshold {t.default_threshold} outside "
                    f"allowed range [{min_t}, {max_t}]"
                )

        return self


# ---------------------------------------------------------------------------
# Thread-safe singleton
# ---------------------------------------------------------------------------
_settings_lock = threading.Lock()
_settings_instance: Optional[Settings] = None


def get_settings(**overrides: Any) -> Settings:
    """Return the cached Settings singleton. Thread-safe.

    Pass overrides only in tests — they force a fresh, uncached instance.
    """
    global _settings_instance

    if overrides:
        return Settings(**overrides)

    if _settings_instance is not None:
        return _settings_instance

    with _settings_lock:
        if _settings_instance is not None:
            return _settings_instance
        _settings_instance = Settings()
        return _settings_instance


def reset_settings() -> None:
    """Clear the singleton. For use in tests only."""
    global _settings_instance
    with _settings_lock:
        _settings_instance = None
