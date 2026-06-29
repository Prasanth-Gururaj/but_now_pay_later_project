"""Model registry and versioning via MLflow on DagsHub.

Provides ModelRegistry for registering champion models, transitioning
them to Production stage, and retrieving production model URIs.
"""

from __future__ import annotations

from bnpl.logger import LoggerMixin, log_execution


class ModelRegistry(LoggerMixin):
    """Register and manage models in the MLflow model registry on DagsHub.

    After training and evaluation, the champion model is registered
    in the MLflow registry with metadata about its performance and
    known limitations. The model is then transitioned to the
    Production stage for serving.

    Usage::

        registry = ModelRegistry()
        version = registry.register_champion(
            run_id="abc123",
            model_name="bnpl-default-prediction-champion",
            description="AUC 0.7045 on test set",
        )

    Depends on:
        - DagsHub MLflow for model registry
        - Settings for DagsHub credentials
        - LoggerMixin: structured logging
    """

    DEFAULT_MODEL_NAME: str = "bnpl-default-prediction-champion"

    def __init__(self) -> None:
        """Initialize MLflow connection via dagshub.init."""
        self._init_mlflow()

    def _init_mlflow(self) -> None:
        """Initialize DagsHub MLflow connection.

        Uses dagshub.init() pattern with credentials from Settings.
        """
        try:
            import dagshub
            from config.settings import get_settings

            settings = get_settings()
            if settings.dagshub_username and settings.dagshub_repo:
                dagshub.init(
                    repo_owner=settings.dagshub_username,
                    repo_name=settings.dagshub_repo,
                    mlflow=True,
                )
        except Exception as exc:
            self.logger.warning("DagsHub init failed: %s", exc)

    @log_execution(operation="ModelRegistry.register_champion")
    def register_champion(
        self,
        run_id: str,
        model_name: str | None = None,
        description: str | None = None,
    ) -> str:
        """Register the champion model from a completed MLflow run.

        Registers the model artifact, adds a description, and
        transitions the latest version to the Production stage.

        Args:
            run_id: MLflow run ID containing the logged model artifact.
            model_name: Registry model name. Defaults to
                        ``bnpl-default-prediction-champion``.
            description: Human-readable description of the model version.

        Returns:
            str: The model version number string (e.g. ``"1"``).

        Raises:
            RuntimeError: If registration fails.
        """
        import mlflow
        from mlflow.tracking import MlflowClient

        name = model_name or self.DEFAULT_MODEL_NAME
        model_uri = f"runs:/{run_id}/model"

        result = mlflow.register_model(model_uri, name)
        version = result.version
        self.logger.info(
            "Registered model '%s' version %s from run %s",
            name, version, run_id,
        )

        client = MlflowClient()
        if description:
            client.update_model_version(
                name=name, version=version, description=description,
            )

        self._transition_to_production(client, name, version)
        return str(version)

    def _transition_to_production(
        self, client: object, name: str, version: str,
    ) -> None:
        """Transition a model version to the Production stage.

        Args:
            client: MlflowClient instance.
            name: Registry model name.
            version: Model version to transition.
        """
        try:
            client.transition_model_version_stage(
                name=name, version=version, stage="Production",
            )
            self.logger.info(
                "Model '%s' v%s transitioned to Production", name, version,
            )
        except Exception as exc:
            self.logger.warning(
                "Stage transition failed (may require newer MLflow): %s", exc,
            )
            try:
                client.set_registered_model_alias(name, "champion", version)
                self.logger.info("Set alias 'champion' on v%s instead", version)
            except Exception:
                pass

    def get_champion_uri(self, model_name: str | None = None) -> str:
        """Get the URI of the current Production model.

        Args:
            model_name: Registry model name. Defaults to the standard name.

        Returns:
            str: Model URI for loading (e.g. ``models:/name/Production``).
        """
        name = model_name or self.DEFAULT_MODEL_NAME
        return f"models:/{name}/Production"
