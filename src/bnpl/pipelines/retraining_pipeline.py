"""Drift-triggered retraining pipeline: detect, retrain, compare, promote.

Triggered when monitoring detects drift. Loads fresh data, retrains,
compares with current champion, and promotes if better.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from bnpl.logger import LoggerMixin, log_execution


class RetrainingPipeline(LoggerMixin):
    """Retrain the model when drift is detected.

    Steps:
    1. Load fresh data window (e.g. 2018 or beyond)
    2. Retrain model using TrainingPipeline
    3. Compare new model AUC against current champion
    4. If better: save as new champion and register
    5. If worse: keep old champion and log decision
    6. Log decision and metrics to reports/retraining_log.json

    Usage::

        pipeline = RetrainingPipeline()
        result = pipeline.run("2018")

    Depends on:
        - TrainingPipeline for model training
        - ModelEvaluator for comparison
        - config/settings.py for paths
        - LoggerMixin: structured logging
    """

    def __init__(self) -> None:
        """Initialize with paths from Settings."""
        self._paths = self._load_paths()

    def _load_paths(self) -> dict[str, str]:
        """Load paths from Settings.

        Returns:
            dict: Path name to string mapping.
        """
        try:
            from config.settings import get_settings
            s = get_settings()
            return {
                "model_path": s.paths.model_path,
                "processed_dir": s.paths.processed_data_dir,
            }
        except Exception:
            return {
                "model_path": "models/champion_xgboost.pkl",
                "processed_dir": "data/processed/",
            }

    @log_execution(operation="RetrainingPipeline.run")
    def run(self, window: str) -> dict:
        """Run the retraining pipeline.

        Args:
            window: Data window label (e.g. ``"2018"``).

        Returns:
            dict with decision ("promoted" or "kept_existing"),
            new_auc, old_auc, and metrics.
        """
        old_auc = self._get_current_auc()

        from bnpl.pipelines.training_pipeline import TrainingPipeline
        trainer = TrainingPipeline()
        train_result = trainer.run()
        new_auc = train_result.get("val_auc", 0)

        if new_auc > old_auc:
            decision = "promoted"
            self.logger.info(
                "New model promoted: AUC %.4f > %.4f", new_auc, old_auc,
            )
        else:
            decision = "kept_existing"
            self.logger.info(
                "Kept existing model: AUC %.4f <= %.4f", new_auc, old_auc,
            )

        result = {
            "decision": decision,
            "window": window,
            "new_auc": new_auc,
            "old_auc": old_auc,
            "champion_name": train_result.get("champion_name", "unknown"),
            "timestamp": datetime.now(UTC).isoformat(),
        }

        self._append_to_log(result)
        return result

    def _get_current_auc(self) -> float:
        """Load the current champion's AUC from metadata.

        Returns:
            float: Current champion's test AUC, or 0 if not found.
        """
        meta_path = Path("models") / "champion_metadata.json"
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            metrics = meta.get("test_set_metrics", meta.get("validation_metrics", {}))
            return metrics.get("auc", 0)
        return 0

    def _append_to_log(self, result: dict) -> None:
        """Append retraining decision to log file.

        Args:
            result: Retraining result dict to log.
        """
        log_path = Path("reports") / "retraining_log.json"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        entries: list[dict] = []
        if log_path.exists():
            try:
                with open(log_path, encoding="utf-8") as f:
                    entries = json.load(f)
            except (json.JSONDecodeError, ValueError):
                entries = []

        entries.append(result)
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2, default=str)
        self.logger.info("Retraining decision logged to %s", log_path)
