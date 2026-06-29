"""Model training orchestration with MLflow logging.

Provides ModelTrainer that trains Logistic Regression, XGBoost, and
LightGBM. All model parameters are loaded from config/model_params.yaml
so you can tune by editing the YAML and re-running the pipeline.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from bnpl.logger import LoggerMixin, log_execution


class ModelTrainer(LoggerMixin):
    """Train and compare multiple models with MLflow experiment tracking.

    All model hyperparameters are loaded from config/model_params.yaml.
    To try different parameters, edit the YAML file and re-run::

        python -m bnpl.main training-pipeline

    Each run is logged to DagsHub MLflow so you can compare runs with
    different parameters in the experiment dashboard.

    Usage::

        trainer = ModelTrainer()
        results = trainer.train_all(X_train, y_train, X_val, y_val)
        champion = trainer.select_champion(results)

    Depends on:
        - config/model_params.yaml for all hyperparameters
        - DagsHub MLflow for experiment tracking
        - LoggerMixin: structured logging
    """

    def __init__(self, experiment_name: str | None = None) -> None:
        """Initialize with model params from config and MLflow experiment.

        Args:
            experiment_name: MLflow experiment name. If None, uses
                             the value from Settings.
        """
        self._experiment_name = experiment_name or self._get_experiment_name()
        self._params = self._load_model_params()
        self._scale_pos_weight: float | None = None
        self._run_ids: dict[str, str] = {}
        self._run_tag = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

        self.logger.info(
            "ModelTrainer loaded params for: %s (run_tag=%s)",
            list(self._params.keys()),
            self._run_tag,
        )

    def _load_model_params(self) -> dict:
        """Load all model parameters from config/model_params.yaml.

        Returns:
            dict: Full config with keys for each model type.
        """
        try:
            from config.settings import CONFIG_DIR
            config_path = CONFIG_DIR / "model_params.yaml"
        except Exception:
            config_path = Path("config") / "model_params.yaml"

        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                return yaml.safe_load(f)
        self.logger.warning("model_params.yaml not found, using defaults")
        return {}

    def _get_experiment_name(self) -> str:
        """Load experiment name from Settings.

        Returns:
            str: MLflow experiment name.
        """
        try:
            from config.settings import get_settings
            return get_settings().model.experiment_name
        except Exception:
            return "bnpl_default_prediction"

    def _init_mlflow(self) -> None:
        """Initialize DagsHub MLflow connection.

        Uses dagshub.init() pattern with credentials from Settings/.env.
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
                self.logger.info("DagsHub MLflow initialized")
            else:
                self.logger.warning("DagsHub credentials not found, using local MLflow")
        except ImportError:
            self.logger.warning("dagshub not installed, using local MLflow")
        except Exception as exc:
            self.logger.warning("DagsHub init failed: %s", exc)

    @log_execution(operation="ModelTrainer.train_all")
    def train_all(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> dict:
        """Train all three models and return results.

        Args:
            X_train: Training features (47 columns).
            y_train: Training target (binary).
            X_val: Validation features.
            y_val: Validation target.

        Returns:
            dict mapping model names to dicts containing:
                - model: trained model object
                - val_proba: predicted probabilities on validation set
                - metrics: dict of evaluation metrics
                - run_id: MLflow run ID
                - scaler: StandardScaler (only for logistic_regression)
        """
        import mlflow

        self._init_mlflow()
        mlflow.set_experiment(self._experiment_name)

        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        self._scale_pos_weight = float(n_neg / n_pos)
        self.logger.info(
            "Class balance: neg=%d pos=%d scale_pos_weight=%.3f",
            n_neg, n_pos, self._scale_pos_weight,
        )

        results = {}
        results["logistic_regression"] = self._train_logistic_regression(
            X_train, y_train, X_val, y_val
        )
        results["xgboost"] = self._train_xgboost(
            X_train, y_train, X_val, y_val
        )
        results["lightgbm"] = self._train_lightgbm(
            X_train, y_train, X_val, y_val
        )

        return results

    def _train_logistic_regression(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> dict:
        """Train Logistic Regression with scaling and balanced weights.

        Parameters loaded from config/model_params.yaml ``logistic_regression`` section.

        Args:
            X_train: Training features.
            y_train: Training target.
            X_val: Validation features.
            y_val: Validation target.

        Returns:
            dict with model, scaler, val_proba, metrics, and run_id.
        """
        import mlflow
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        cfg = self._params.get("logistic_regression", {})

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        with mlflow.start_run(run_name=f"logistic_regression_{self._run_tag}") as run:
            model = LogisticRegression(
                class_weight=cfg.get("class_weight", "balanced"),
                max_iter=cfg.get("max_iter", 1000),
                C=cfg.get("C", 1.0),
                solver=cfg.get("solver", "lbfgs"),
                random_state=cfg.get("random_state", 42),
            )
            model.fit(X_train_scaled, y_train)
            val_proba = model.predict_proba(X_val_scaled)[:, 1]
            metrics = self._evaluate("Logistic Regression", y_val, val_proba)

            mlflow.log_params({"model_type": "logistic_regression", **cfg})
            self._log_metrics(mlflow, metrics)
            mlflow.sklearn.log_model(model, "model")
            self._run_ids["logistic_regression"] = run.info.run_id

        return {
            "model": model, "scaler": scaler,
            "val_proba": val_proba, "metrics": metrics,
            "run_id": run.info.run_id,
        }

    def _train_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> dict:
        """Train XGBoost with early stopping and class weighting.

        Parameters loaded from config/model_params.yaml ``xgboost`` section.
        Applies the ``_estimator_type = "classifier"`` fix before MLflow logging.

        Args:
            X_train: Training features.
            y_train: Training target.
            X_val: Validation features.
            y_val: Validation target.

        Returns:
            dict with model, val_proba, metrics, and run_id.
        """
        import mlflow
        import xgboost as xgb

        cfg = self._params.get("xgboost", {})
        train_params = {k: v for k, v in cfg.items() if k != "search_space"}

        with mlflow.start_run(run_name=f"xgboost_{self._run_tag}") as run:
            model = xgb.XGBClassifier(
                n_estimators=train_params.get("n_estimators", 500),
                max_depth=train_params.get("max_depth", 6),
                learning_rate=train_params.get("learning_rate", 0.05),
                subsample=train_params.get("subsample", 0.8),
                colsample_bytree=train_params.get("colsample_bytree", 0.8),
                min_child_weight=train_params.get("min_child_weight", 1),
                gamma=train_params.get("gamma", 0),
                reg_alpha=train_params.get("reg_alpha", 0),
                reg_lambda=train_params.get("reg_lambda", 1),
                scale_pos_weight=self._scale_pos_weight,
                eval_metric=train_params.get("eval_metric", "auc"),
                early_stopping_rounds=train_params.get("early_stopping_rounds", 30),
                random_state=train_params.get("random_state", 42),
            )
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            val_proba = model.predict_proba(X_val)[:, 1]
            metrics = self._evaluate("XGBoost", y_val, val_proba)

            log_params = {
                "model_type": "xgboost",
                "n_estimators_actual": model.best_iteration,
                "scale_pos_weight": round(self._scale_pos_weight, 3),
                **{k: v for k, v in train_params.items() if k != "eval_metric"},
            }
            mlflow.log_params(log_params)
            self._log_metrics(mlflow, metrics)

            model._estimator_type = "classifier"
            mlflow.xgboost.log_model(model, "model")
            self._run_ids["xgboost"] = run.info.run_id

        return {
            "model": model, "val_proba": val_proba,
            "metrics": metrics, "run_id": run.info.run_id,
        }

    def _train_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> dict:
        """Train LightGBM with parameters from config.

        Parameters loaded from config/model_params.yaml ``lightgbm`` section.
        Applies the ``_estimator_type = "classifier"`` fix before MLflow logging.

        Args:
            X_train: Training features.
            y_train: Training target.
            X_val: Validation features.
            y_val: Validation target.

        Returns:
            dict with model, val_proba, metrics, and run_id.
        """
        import lightgbm as lgb
        import mlflow

        cfg = self._params.get("lightgbm", {})
        train_params = {k: v for k, v in cfg.items() if k != "search_space"}
        es_rounds = train_params.pop("early_stopping_rounds", 15)

        with mlflow.start_run(run_name=f"lightgbm_{self._run_tag}") as run:
            model = lgb.LGBMClassifier(
                n_estimators=train_params.get("n_estimators", 300),
                max_depth=train_params.get("max_depth", 6),
                learning_rate=train_params.get("learning_rate", 0.05),
                subsample=train_params.get("subsample", 0.8),
                colsample_bytree=train_params.get("colsample_bytree", 0.8),
                scale_pos_weight=self._scale_pos_weight,
                n_jobs=train_params.get("n_jobs", -1),
                num_leaves=train_params.get("num_leaves", 63),
                min_child_samples=train_params.get("min_child_samples", 50),
                reg_alpha=train_params.get("reg_alpha", 0),
                reg_lambda=train_params.get("reg_lambda", 1),
                random_state=train_params.get("random_state", 42),
                verbose=train_params.get("verbose", -1),
            )
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="auc",
                callbacks=[
                    lgb.early_stopping(es_rounds, verbose=False),
                    lgb.log_evaluation(period=-1),
                ],
            )
            val_proba = model.predict_proba(X_val)[:, 1]
            metrics = self._evaluate("LightGBM", y_val, val_proba)

            log_params = {
                "model_type": "lightgbm",
                "n_estimators_actual": model.best_iteration_,
                "scale_pos_weight": round(self._scale_pos_weight, 3),
                "early_stopping_rounds": es_rounds,
                **{k: v for k, v in train_params.items()},
            }
            mlflow.log_params(log_params)
            self._log_metrics(mlflow, metrics)

            model._estimator_type = "classifier"
            mlflow.lightgbm.log_model(model, "model")
            self._run_ids["lightgbm"] = run.info.run_id

        return {
            "model": model, "val_proba": val_proba,
            "metrics": metrics, "run_id": run.info.run_id,
        }

    def _evaluate(
        self, name: str, y_true: pd.Series, y_proba: np.ndarray, threshold: float = 0.5,
    ) -> dict:
        """Compute evaluation metrics at both 0.5 and 0.3 thresholds.

        Models trained with scale_pos_weight output shifted probabilities.
        Evaluating only at 0.5 can show 0 precision/recall even when the
        model has learned well (AUC > 0.7). We evaluate at 0.3 too to
        see the model's true discriminative power.

        Args:
            name: Model display name for logging.
            y_true: True binary labels.
            y_proba: Predicted probabilities for the positive class.
            threshold: Primary classification threshold (default 0.5).

        Returns:
            dict with auc, brier_score, and precision/recall/f1 at
            both threshold 0.5 and 0.3.
        """
        auc = roc_auc_score(y_true, y_proba)
        brier = brier_score_loss(y_true, y_proba)

        metrics = {"model": name, "auc": auc, "brier_score": brier}

        for t in [threshold, 0.3]:
            y_pred = (y_proba >= t).astype(int)
            suffix = "" if t == threshold else "_at_0.3"
            metrics[f"threshold{suffix}"] = t
            metrics[f"precision_default{suffix}"] = precision_score(
                y_true, y_pred, pos_label=1, zero_division=0,
            )
            metrics[f"recall_default{suffix}"] = recall_score(
                y_true, y_pred, pos_label=1, zero_division=0,
            )
            metrics[f"f1_default{suffix}"] = f1_score(
                y_true, y_pred, pos_label=1, zero_division=0,
            )

        self.logger.info(
            "%s | AUC=%.4f | F1@0.5=%.4f | F1@0.3=%.4f",
            name, auc, metrics["f1_default"], metrics["f1_default_at_0.3"],
        )
        return metrics

    def _log_metrics(self, mlflow_module: object, metrics: dict) -> None:
        """Log numeric metrics to MLflow.

        Args:
            mlflow_module: The mlflow module.
            metrics: Dict of metric name-value pairs.
        """
        numeric = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        mlflow_module.log_metrics(numeric)

    def select_champion(self, results: dict) -> str:
        """Select the champion model by highest AUC on validation set.

        Args:
            results: Dict mapping model names to their result dicts,
                     each containing a ``metrics`` dict with an ``auc`` key.

        Returns:
            str: Name of the champion model (e.g. ``"xgboost"``).
        """
        best_name = max(results, key=lambda k: results[k]["metrics"]["auc"])
        best_auc = results[best_name]["metrics"]["auc"]
        self.logger.info("Champion: %s (AUC=%.4f)", best_name, best_auc)
        return best_name

    @property
    def run_ids(self) -> dict[str, str]:
        """Return mapping of model names to MLflow run IDs.

        Returns:
            dict[str, str]: Model name to run ID mapping.
        """
        return self._run_ids

    @property
    def scale_pos_weight(self) -> float | None:
        """Return the calculated class imbalance weight.

        Returns:
            float or None if not yet calculated.
        """
        return self._scale_pos_weight
