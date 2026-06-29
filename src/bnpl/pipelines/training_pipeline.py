"""Training pipeline: train models, select champion, optimize threshold, register.

Owns only model training. Assumes processed parquet files already
exist from DataPipeline. Produces a champion model, metadata, and
MLflow registry entry.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd

from bnpl.logger import LoggerMixin, log_execution


class TrainingPipeline(LoggerMixin):
    """Train models, select champion, and register in MLflow.

    Assumes processed parquet files exist in data/processed/ (produced
    by DataPipeline). Steps:
    1. Load train.parquet and val.parquet
    2. Train all three models via ModelTrainer
    3. Select champion by highest validation AUC
    4. Optimize decision threshold via ThresholdSelector
    5. Save champion model and metadata
    6. Register champion in MLflow model registry

    Usage::

        pipeline = TrainingPipeline()
        result = pipeline.run()

    Depends on:
        - ModelTrainer, ThresholdSelector, ModelRegistry
        - Processed parquet files from DataPipeline
        - config/settings.py for paths and thresholds
        - LoggerMixin: structured logging
    """

    def __init__(self) -> None:
        """Initialize with paths from Settings."""
        self._paths = self._load_paths()

    def _load_paths(self) -> dict[str, str]:
        """Load file paths from Settings with fallback defaults.

        Returns:
            dict: Path name to string mapping.
        """
        try:
            from config.settings import get_settings
            s = get_settings()
            return {
                "processed_dir": s.paths.processed_data_dir,
                "model_path": s.paths.model_path,
            }
        except Exception:
            return {
                "processed_dir": "data/processed/",
                "model_path": "models/champion_xgboost.pkl",
            }

    @log_execution(operation="TrainingPipeline.run")
    def run(self) -> dict:
        """Run the full training pipeline.

        Returns:
            dict with champion_name, val_auc, threshold,
            registry_version, all_model_aucs.
        """
        from bnpl.models.threshold import ThresholdSelector
        from bnpl.models.train import ModelTrainer

        processed_dir = Path(self._paths["processed_dir"])
        train = pd.read_parquet(processed_dir / "train.parquet")
        val = pd.read_parquet(processed_dir / "val.parquet")

        non_feature = ["default", "issue_d", "issue_year"]
        feature_cols = [c for c in train.columns if c not in non_feature]

        X_train, y_train = train[feature_cols], train["default"]
        X_val, y_val = val[feature_cols], val["default"]

        trainer = ModelTrainer()
        results = trainer.train_all(X_train, y_train, X_val, y_val)
        champion_name = trainer.select_champion(results)
        champion_data = results[champion_name]

        selector = ThresholdSelector()
        threshold_result = selector.select_optimal(y_val, champion_data["val_proba"])
        final_threshold = threshold_result["constrained_threshold"]

        model = champion_data["model"]
        self._save_model(model, champion_name)
        self._save_metadata(
            champion_name, feature_cols, final_threshold,
            champion_data["metrics"], trainer.scale_pos_weight,
        )

        version = self._register_model(
            champion_name, champion_data.get("run_id"),
            champion_data["metrics"], final_threshold,
        )

        if version:
            self._update_metadata_registry(version)

        return {
            "champion_name": champion_name,
            "val_auc": champion_data["metrics"]["auc"],
            "threshold": final_threshold,
            "registry_version": version,
            "all_model_aucs": {
                name: data["metrics"]["auc"] for name, data in results.items()
            },
        }

    def _save_model(self, model: object, champion_name: str) -> None:
        """Save champion model to disk.

        Args:
            model: Trained model object.
            champion_name: Name for logging.
        """
        model_path = Path(self._paths["model_path"])
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        self.logger.info("Champion '%s' saved to %s", champion_name, model_path)

    def _save_metadata(
        self, champion_name: str, feature_cols: list[str],
        threshold: float, metrics: dict, scale_pos_weight: float | None,
    ) -> None:
        """Save champion_metadata.json.

        Args:
            champion_name: Name of the champion model.
            feature_cols: List of 47 feature column names.
            threshold: Final decision threshold.
            metrics: Validation metrics dict.
            scale_pos_weight: Class imbalance weight.
        """
        metadata = {
            "model_type": champion_name,
            "trained_on": "2013 to 2015, train split",
            "validated_on": "2016, val split",
            "tested_on": "2017, test split, touched once",
            "feature_columns": feature_cols,
            "decision_threshold": threshold,
            "threshold_selection_method": (
                "minimum total business cost with 55% minimum approval rate"
            ),
            "validation_metrics": {
                k: v for k, v in metrics.items() if isinstance(v, (int, float))
            },
            "scale_pos_weight": scale_pos_weight,
            "timestamp": datetime.now(UTC).isoformat(),
        }

        out = Path("models") / "champion_metadata.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        self.logger.info("Metadata saved to %s", out)

    def _register_model(
        self, champion_name: str, run_id: str | None,
        metrics: dict, threshold: float,
    ) -> str | None:
        """Register champion in MLflow model registry.

        Args:
            champion_name: Champion model name.
            run_id: MLflow run ID.
            metrics: Validation metrics.
            threshold: Decision threshold.

        Returns:
            str or None: Model version if succeeded.
        """
        if not run_id:
            self.logger.warning("No run_id, skipping registry")
            return None
        try:
            from bnpl.models.registry import ModelRegistry
            registry = ModelRegistry()
            desc = (
                f"XGBoost BNPL default predictor. "
                f"Validation AUC {metrics.get('auc', 0):.4f}. "
                f"Decision threshold {threshold}. "
                f"Trained on 2013-2015 LendingClub data. "
                f"Known limitations: calibration overestimates probability, "
                f"sub_grade accounts for 70 percent of feature importance, "
                f"selection bias from accepted-loans-only training data."
            )
            return registry.register_champion(run_id=run_id, description=desc)
        except Exception as exc:
            self.logger.warning("Registration failed: %s", exc)
            return None


    def _update_metadata_registry(self, version: str) -> None:
        """Save registry name and version to champion_metadata.json.

        Args:
            version: MLflow model registry version string.
        """
        meta_path = Path("models") / "champion_metadata.json"
        if not meta_path.exists():
            return
        with open(meta_path, encoding="utf-8") as f:
            metadata = json.load(f)
        metadata["registry_name"] = "bnpl-default-prediction-champion"
        metadata["registry_version"] = version
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        self.logger.info("Registry info saved: version=%s", version)


def run_training_pipeline() -> dict:
    """Module-level entry point for the training pipeline.

    Returns:
        dict: Training results including champion name and AUC.
    """
    pipeline = TrainingPipeline()
    return pipeline.run()
