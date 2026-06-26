"""Hyperparameter tuning with Optuna for XGBoost.

Provides HyperparameterTuner that uses Optuna to search for optimal
XGBoost parameters, with each trial logged to MLflow.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from bnpl.logger import LoggerMixin, log_execution


class HyperparameterTuner(LoggerMixin):
    """Tune XGBoost hyperparameters using Optuna with MLflow logging.

    Search space is loaded from config/model_params.yaml. Each trial
    trains an XGBoost model and evaluates on the validation set.
    Best parameters are returned for use in the training pipeline.

    Usage::

        tuner = HyperparameterTuner()
        best_params = tuner.tune(X_train, y_train, X_val, y_val, n_trials=50)

    Depends on:
        - Optuna for Bayesian hyperparameter search
        - config/model_params.yaml for search space bounds
        - LoggerMixin: structured logging
    """

    def __init__(self) -> None:
        """Initialize and load search space from config."""
        self._search_space = self._load_search_space()

    def _load_search_space(self) -> dict:
        """Load search space bounds from config/model_params.yaml.

        Returns:
            dict: XGBoost search space with min/max ranges.
        """
        try:
            import yaml
            from config.settings import CONFIG_DIR

            path = CONFIG_DIR / "model_params.yaml"
            if path.exists():
                with open(path, encoding="utf-8") as f:
                    config = yaml.safe_load(f)
                return config.get("xgboost", {}).get("search_space", {})
        except Exception:
            pass
        return {
            "max_depth": [3, 10],
            "learning_rate": [0.01, 0.3],
            "subsample": [0.6, 1.0],
            "colsample_bytree": [0.6, 1.0],
            "min_child_weight": [1, 10],
            "gamma": [0.0, 5.0],
            "reg_alpha": [0.0, 10.0],
            "reg_lambda": [0.0, 10.0],
        }

    @log_execution(operation="HyperparameterTuner.tune")
    def tune(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        n_trials: int = 50,
        timeout: int | None = None,
    ) -> dict:
        """Run Optuna hyperparameter search.

        Args:
            X_train: Training features.
            y_train: Training target.
            X_val: Validation features.
            y_val: Validation target.
            n_trials: Number of Optuna trials.
            timeout: Timeout in seconds (None for no limit).

        Returns:
            dict: Best hyperparameters found.
        """
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())

        def objective(trial: optuna.Trial) -> float:
            return self._objective(
                trial, X_train, y_train, X_val, y_val, scale_pos_weight,
            )

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, timeout=timeout)

        best = study.best_params
        best["scale_pos_weight"] = scale_pos_weight
        self.logger.info(
            "Best trial: AUC=%.4f | params=%s",
            study.best_value, best,
        )
        return best

    def _objective(
        self,
        trial: object,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        scale_pos_weight: float,
    ) -> float:
        """Optuna objective function for a single trial.

        Args:
            trial: Optuna Trial object.
            X_train: Training features.
            y_train: Training target.
            X_val: Validation features.
            y_val: Validation target.
            scale_pos_weight: Class imbalance weight.

        Returns:
            float: Validation AUC score for this trial.
        """
        import xgboost as xgb

        ss = self._search_space
        params = {
            "max_depth": trial.suggest_int("max_depth", *ss.get("max_depth", [3, 10])),
            "learning_rate": trial.suggest_float(
                "learning_rate", *ss.get("learning_rate", [0.01, 0.3]), log=True,
            ),
            "subsample": trial.suggest_float("subsample", *ss.get("subsample", [0.6, 1.0])),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", *ss.get("colsample_bytree", [0.6, 1.0]),
            ),
            "min_child_weight": trial.suggest_int(
                "min_child_weight", *ss.get("min_child_weight", [1, 10]),
            ),
            "gamma": trial.suggest_float("gamma", *ss.get("gamma", [0.0, 5.0])),
            "reg_alpha": trial.suggest_float("reg_alpha", *ss.get("reg_alpha", [0.0, 10.0])),
            "reg_lambda": trial.suggest_float("reg_lambda", *ss.get("reg_lambda", [0.0, 10.0])),
        }

        model = xgb.XGBClassifier(
            n_estimators=500,
            scale_pos_weight=scale_pos_weight,
            eval_metric="auc",
            early_stopping_rounds=30,
            random_state=42,
            **params,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        val_proba = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, val_proba)
