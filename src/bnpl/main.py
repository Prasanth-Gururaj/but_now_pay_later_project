"""Central CLI entry point for all BNPL pipeline tasks.

Usage::

    python -m bnpl.main <command> [args]
    bnpl <command> [args]           # via installed script

Available commands::

    preprocess      Test the preprocessing pipeline
    predict         Run a single prediction
    serve           Start the FastAPI server
    monitor         Run the monitoring pipeline
    drift           Run drift detection only
    validate-config Validate the config system
    health-check    Check if the API is running
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bnpl.logger import LoggerMixin, get_logger, setup_logging

logger = get_logger(__name__)

SAMPLE_INPUT: dict = {
    "dti": 18.5,
    "fico_range_low": 690,
    "revol_util": 45.2,
    "annual_inc": 68000.0,
    "loan_amnt": 10000.0,
    "int_rate": 12.5,
    "sub_grade": "B3",
    "term": "36 months",
    "emp_length": "5 years",
    "home_ownership": "RENT",
    "verification_status": "Verified",
    "purpose": "debt_consolidation",
    "delinq_2yrs": 0.0,
    "inq_last_6mths": 1.0,
    "open_acc": 8.0,
    "pub_rec": 0.0,
    "revol_bal": 12000.0,
}


class BNPLRunner(LoggerMixin):
    """Central runner class for all BNPL pipeline tasks.

    Provides a unified interface for running any individual component
    or full pipeline from the command line. It is the production entry
    point for all operational tasks including serving, monitoring,
    drift detection, and prediction.

    Usage::

        python -m bnpl.main <command> [args]

    Available commands::

        preprocess       Test the preprocessing pipeline
        predict          Run a single prediction
        serve            Start the FastAPI server
        monitor          Run the monitoring pipeline
        drift            Run drift detection only
        validate-config  Validate the config system
        health-check     Check if the API is running

    Depends on:
        - PreprocessingPipeline: feature transformation
        - Predictor: model inference
        - TrainingPipeline: end-to-end training
        - MonitoringPipeline: drift detection and alerting
        - Settings: configuration system
        - LoggerMixin: structured logging
    """

    def __init__(self) -> None:
        """Initialize the runner with logging."""
        setup_logging()

    def run_preprocess(self, input_json: str | None) -> None:
        """Run preprocessing pipeline on a sample or provided input.

        Args:
            input_json: Path to JSON file with 17 raw fields.
                        If None, uses the built-in sample input.

        Returns:
            None. Prints output shape and column names to stdout.
        """
        from bnpl.features.pipeline import PreprocessingPipeline

        raw = self._load_input(input_json)
        config_path = self._get_config_path()
        pipeline = PreprocessingPipeline(config_path)
        result = pipeline.transform(raw)

        print("\nPreprocessing result:")
        print(f"  Shape: {result.shape}")
        print(f"  Columns ({len(result.columns)}):")
        for i, col in enumerate(result.columns, 1):
            print(f"    {i:2d}. {col} = {result[col].iloc[0]}")

    def run_predict(
        self,
        input_json: str,
        model_path: str | None,
        threshold: float | None,
    ) -> None:
        """Run a single prediction and print the result.

        Args:
            input_json: Path to JSON file with 17 raw input fields.
            model_path: Optional override for model file path.
            threshold: Optional override for decision threshold.

        Returns:
            None. Prints PredictionResponse as formatted JSON.
        """
        from bnpl.models.predictor import Predictor

        raw = self._load_input(input_json)
        resolved_model = model_path or self._get_model_path()
        resolved_threshold = threshold or self._get_threshold()
        config_path = self._get_config_path()

        predictor = Predictor(resolved_model, config_path, resolved_threshold)
        result = predictor.predict(raw)

        print(json.dumps(result, indent=2))

    def run_serve(self, host: str, port: int, reload: bool) -> None:
        """Start the FastAPI prediction server using uvicorn.

        Args:
            host: Host address to bind to.
            port: Port number to listen on.
            reload: Whether to enable hot reload for development.

        Returns:
            None. Blocks until server is stopped.
        """
        import uvicorn

        print(f"Starting BNPL API server on {host}:{port}")
        uvicorn.run(
            "bnpl.serving.api:app",
            host=host,
            port=port,
            reload=reload,
        )

    def run_monitor(self, window: str, config_path: str | None) -> None:
        """Run the full monitoring pipeline for a data window.

        Args:
            window: Window label identifying which data to monitor.
            config_path: Optional override for config file path.

        Returns:
            None. Prints monitoring results and retraining recommendation.
        """
        from bnpl.pipelines.monitoring_pipeline import run_monitoring_pipeline

        result = run_monitoring_pipeline(window, config_path=config_path)
        print(json.dumps(result, indent=2, default=str))

        if result.get("should_retrain"):
            print("\n*** RETRAINING RECOMMENDED ***")
        else:
            print("\nNo retraining needed.")

    def run_drift(self, window: str) -> None:
        """Run drift detection only without model scoring.

        Loads reference and window data, runs preprocessing on both,
        and performs drift detection without calculating approval rates
        or scoring predictions.

        Args:
            window: Window label identifying which data to check.

        Returns:
            None. Prints drift detection results.
        """
        import pandas as pd

        from bnpl.features.pipeline import PreprocessingPipeline
        from bnpl.monitoring.drift_detector import DriftDetector

        config_path = self._get_config_path()
        train_path = self._get_train_data_path()

        pipeline = PreprocessingPipeline(config_path)
        train_df = pd.read_parquet(train_path)
        feature_cols = pipeline._feature_cols
        available = [c for c in feature_cols if c in train_df.columns]
        train_features = train_df[available]

        detector = DriftDetector(train_features, config_path)
        result = detector.run_drift(train_features, train_features, window)

        print(json.dumps(result, indent=2, default=str))

    def run_validate_config(self) -> None:
        """Validate and print the full configuration.

        Loads the Settings singleton, validates all sections, and
        prints a summary of every loaded value.

        Returns:
            None. Prints config values. Exits with code 1 if invalid.
        """
        try:
            from config.settings import get_settings

            settings = get_settings()
            print("Configuration valid!\n")
            print(f"  Environment:  {settings.app_env}")
            print(f"  Project:      {settings.project.name} v{settings.project.version}")
            print(f"  Model path:   {settings.paths.model_path}")
            print(f"  Config path:  {settings.paths.config_path}")
            print(f"  Threshold:    {settings.thresholds.default_threshold}")
            print(f"  Serving:      {settings.serving.host}:{settings.serving.port}")
            print(f"  Log level:    {settings.logging.level}")
            print(f"  Features:     {len(settings.features)} defined")
            print(f"  PSI thresh:   {settings.thresholds.monitoring.get('psi_threshold', 'N/A')}")
            print(f"  Project root: {settings.project_root}")
        except Exception as exc:
            print(f"Configuration error: {exc}", file=sys.stderr)
            sys.exit(1)

    def run_pipeline(
        self,
        raw_data_path: str | None,
        skip_data_prep: bool,
        skip_training: bool,
    ) -> None:
        """Run the full training pipeline from raw data to model registry.

        Orchestrates all steps: data loading, validation, preprocessing,
        training, evaluation, threshold selection, and model registration.

        Args:
            raw_data_path: Path to raw CSV file. If None, uses config.
            skip_data_prep: If True, skip data loading and preprocessing.
            skip_training: If True, skip training, use existing champion.

        Returns:
            None. Prints pipeline summary and registry details.
        """
        from bnpl.pipelines.training_pipeline import run_training_pipeline

        result = run_training_pipeline(
            raw_data_path=raw_data_path,
            skip_data_prep=skip_data_prep,
            skip_training=skip_training,
        )
        print(json.dumps(result, indent=2, default=str))

    def run_train(self, experiment_name: str | None) -> None:
        """Run only the model training step on existing processed data.

        Assumes processed parquet files already exist in data/processed/.

        Args:
            experiment_name: Optional MLflow experiment name override.

        Returns:
            None. Prints training results.
        """
        from bnpl.pipelines.training_pipeline import run_training_pipeline

        result = run_training_pipeline(skip_data_prep=True)
        print(json.dumps(result, indent=2, default=str))

    def run_evaluate(self) -> None:
        """Run model evaluation on the test set.

        Assumes the champion model already exists at the configured path.
        Generates calibration and SHAP plots, logs to MLflow.

        Returns:
            None. Prints test set metrics.
        """
        import joblib
        import pandas as pd

        from bnpl.models.evaluate import ModelEvaluator

        model_path = self._get_model_path()
        model = joblib.load(model_path)
        threshold = self._get_threshold()

        train_path = self._get_train_data_path()
        processed_dir = Path(train_path).parent
        test = pd.read_parquet(processed_dir / "test.parquet")

        non_feature = ["default", "issue_d", "issue_year"]
        feature_cols = [c for c in test.columns if c not in non_feature]
        X_test, y_test = test[feature_cols], test["default"]

        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(model, X_test, y_test, threshold)
        print(json.dumps(
            {k: v for k, v in metrics.items() if isinstance(v, (int, float, str))},
            indent=2,
        ))

    def run_data_prep(self, raw_data_path: str | None) -> None:
        """Run only the data preparation step.

        Args:
            raw_data_path: Path to raw CSV. If None, uses config default.

        Returns:
            None. Prints summary of saved parquet files.
        """
        from bnpl.pipelines.data_pipeline import DataPipeline

        pipeline = DataPipeline()
        result = pipeline.run(raw_data_path)
        print(json.dumps(result, indent=2, default=str))

    def run_data_pipeline(self, raw_data_path: str | None) -> None:
        """Run DataPipeline: load, clean, split, preprocess, save.

        Args:
            raw_data_path: Path to raw CSV. If None, uses config default.

        Returns:
            None. Prints pipeline summary.
        """
        from bnpl.pipelines.data_pipeline import DataPipeline

        pipeline = DataPipeline()
        result = pipeline.run(raw_data_path)
        print(json.dumps(result, indent=2, default=str))

    def run_training_pipeline(self) -> None:
        """Run TrainingPipeline: train models, select champion, register.

        Assumes data-pipeline already ran and parquets exist.

        Returns:
            None. Prints training results.
        """
        from bnpl.pipelines.training_pipeline import run_training_pipeline

        result = run_training_pipeline()
        print(json.dumps(result, indent=2, default=str))

    def run_evaluation_pipeline(self) -> None:
        """Run EvaluationPipeline: score test set, generate plots.

        Assumes training-pipeline already ran and champion exists.

        Returns:
            None. Prints test metrics.
        """
        from bnpl.pipelines.evaluation_pipeline import run_evaluation_pipeline

        metrics = run_evaluation_pipeline()
        print(json.dumps(
            {k: v for k, v in metrics.items() if isinstance(v, (int, float, str))},
            indent=2,
        ))

    def run_full_pipeline(
        self, raw_data_path: str | None, skip_data: bool, skip_training: bool,
    ) -> None:
        """Run all three pipelines in sequence.

        Args:
            raw_data_path: Path to raw CSV. If None, uses config default.
            skip_data: If True, skip DataPipeline.
            skip_training: If True, skip TrainingPipeline.

        Returns:
            None. Prints final summary.
        """
        if not skip_data:
            print("=== Step 1: Data Pipeline ===")
            self.run_data_pipeline(raw_data_path)

        if not skip_training:
            print("\n=== Step 2: Training Pipeline ===")
            self.run_training_pipeline()

        print("\n=== Step 3: Evaluation Pipeline ===")
        self.run_evaluation_pipeline()

    def run_retrain(self, window: str) -> None:
        """Run drift-triggered retraining pipeline.

        Args:
            window: Data window label (e.g. ``"2018"``).

        Returns:
            None. Prints retraining decision.
        """
        from bnpl.pipelines.retraining_pipeline import RetrainingPipeline

        pipeline = RetrainingPipeline()
        result = pipeline.run(window)
        print(json.dumps(result, indent=2, default=str))

    def run_tune(self, n_trials: int) -> None:
        """Run Optuna hyperparameter tuning.

        Args:
            n_trials: Number of Optuna trials to run.

        Returns:
            None. Prints best parameters.
        """
        import pandas as pd

        from bnpl.models.tune import HyperparameterTuner

        train_path = self._get_train_data_path()
        processed_dir = Path(train_path).parent
        train = pd.read_parquet(train_path)
        val = pd.read_parquet(processed_dir / "val.parquet")

        non_feature = ["default", "issue_d", "issue_year"]
        feature_cols = [c for c in train.columns if c not in non_feature]

        tuner = HyperparameterTuner()
        best = tuner.tune(
            train[feature_cols], train["default"],
            val[feature_cols], val["default"],
            n_trials=n_trials,
        )
        print(json.dumps(best, indent=2, default=str))

    def run_explain(self, input_json: str | None) -> None:
        """Run SHAP explanation for a single prediction or global summary.

        Args:
            input_json: Path to JSON file for local explanation.
                        If None, generates global summary on test set.

        Returns:
            None. Prints path to saved SHAP plot.
        """
        import joblib
        import pandas as pd

        from bnpl.models.explain import ModelExplainer

        model = joblib.load(self._get_model_path())
        explainer = ModelExplainer(model)

        if input_json:
            from bnpl.features.pipeline import PreprocessingPipeline
            raw = self._load_input(input_json)
            pipeline = PreprocessingPipeline(self._get_config_path())
            features = pipeline.transform(raw)
            path = explainer.explain_local(features)
            print(f"Local explanation saved to: {path}")
        else:
            train_path = self._get_train_data_path()
            test = pd.read_parquet(Path(train_path).parent / "test.parquet")
            non_feature = ["default", "issue_d", "issue_year"]
            feature_cols = [c for c in test.columns if c not in non_feature]
            path = explainer.explain_global(test[feature_cols])
            print(f"Global explanation saved to: {path}")

    def run_ab_summary(self) -> None:
        """Print current A/B test summary from the log file.

        Returns:
            None. Prints formatted summary to stdout.
        """
        from bnpl.models.predictor import Predictor
        from bnpl.serving.ab_router import ABRouter

        predictor = Predictor(
            self._get_model_path(), self._get_config_path(), self._get_threshold(),
        )
        router = ABRouter(predictor)
        summary = router.get_summary()
        print(json.dumps(summary, indent=2))

    def run_ab_analyze(self, min_requests: int) -> None:
        """Analyze A/B test results and print promotion recommendation.

        Args:
            min_requests: Minimum requests needed for meaningful analysis.

        Returns:
            None. Prints recommendation and reasoning to stdout.
        """
        from bnpl.monitoring.ab_analyzer import ABAnalyzer

        analyzer = ABAnalyzer()
        result = analyzer.analyze(min_requests=min_requests)
        print(json.dumps(result, indent=2))

    def run_ab_load_challenger(self, model_path: str) -> None:
        """Register a challenger model path for A/B testing.

        Args:
            model_path: Path to the challenger model pkl file.

        Returns:
            None. Prints confirmation.

        Raises:
            FileNotFoundError: If model_path does not exist.
        """
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Challenger model not found: {path}")
        print(f"Challenger model verified at: {path}")
        print("To activate A/B testing, set environment variable:")
        print(f"  CHALLENGER_MODEL_PATH={path}")
        print("Then restart the API server.")

    def run_health_check(self, url: str) -> None:
        """Check if the API is running and healthy.

        Args:
            url: Base URL of the running API.

        Returns:
            None. Prints health response or error message.
        """
        import httpx

        health_url = f"{url.rstrip('/')}/health"
        try:
            response = httpx.get(health_url, timeout=5.0)
            response.raise_for_status()
            print(json.dumps(response.json(), indent=2))
        except httpx.ConnectError:
            print(f"Cannot connect to {health_url}. Is the server running?",
                  file=sys.stderr)
            sys.exit(1)
        except Exception as exc:
            print(f"Health check failed: {exc}", file=sys.stderr)
            sys.exit(1)

    def _load_input(self, input_json: str | None) -> dict:
        """Load raw input from a JSON file or use the built-in sample.

        Args:
            input_json: Path to JSON file, or None for sample input.

        Returns:
            dict: The 17 raw loan application fields.

        Raises:
            FileNotFoundError: If the specified JSON file does not exist.
        """
        if input_json is None:
            return SAMPLE_INPUT.copy()

        path = Path(input_json)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def _get_config_path(self) -> str:
        """Resolve the data_prep_config.json path from env or Settings.

        Returns:
            str: Path to data_prep_config.json.
        """
        import os

        env_path = os.getenv("CONFIG_PATH")
        if env_path:
            return env_path
        try:
            from config.settings import get_settings
            return get_settings().paths.config_path
        except Exception:
            return "reports/data_prep_config.json"

    def _get_model_path(self) -> str:
        """Resolve the model path from env or Settings.

        Returns:
            str: Path to the champion model file.
        """
        import os

        env_path = os.getenv("MODEL_PATH")
        if env_path:
            return env_path
        try:
            from config.settings import get_settings
            return get_settings().paths.model_path
        except Exception:
            return "models/champion_xgboost.pkl"

    def _get_threshold(self) -> float:
        """Resolve the decision threshold from env or Settings.

        Returns:
            float: Decision threshold value.
        """
        import os

        env_val = os.getenv("DECISION_THRESHOLD")
        if env_val:
            return float(env_val)
        try:
            from config.settings import get_settings
            return get_settings().thresholds.default_threshold
        except Exception:
            return 0.45

    def _get_train_data_path(self) -> str:
        """Resolve the training data parquet path from Settings.

        Returns:
            str: Path to train.parquet.
        """
        try:
            from config.settings import get_settings
            return get_settings().paths.train_data_path
        except Exception:
            return "data/processed/train.parquet"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for the BNPL CLI.

    Args:
        argv: Optional argument list. Defaults to sys.argv[1:].

    Returns:
        argparse.Namespace: Parsed arguments with the command and
        its specific options.
    """
    parser = argparse.ArgumentParser(
        prog="bnpl",
        description="BNPL Default Prediction CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # preprocess
    p_pre = subparsers.add_parser("preprocess", help="Test the preprocessing pipeline")
    p_pre.add_argument("--input-json", default=None, help="Path to input JSON file")

    # predict
    p_pred = subparsers.add_parser("predict", help="Run a single prediction")
    p_pred.add_argument("--input-json", required=True, help="Path to input JSON file")
    p_pred.add_argument("--model-path", default=None, help="Override model file path")
    p_pred.add_argument("--threshold", type=float, default=None, help="Override threshold")

    # serve
    p_serve = subparsers.add_parser("serve", help="Start the FastAPI server")
    p_serve.add_argument("--host", default="0.0.0.0", help="Host address")
    p_serve.add_argument("--port", type=int, default=8000, help="Port number")
    p_serve.add_argument("--reload", action="store_true", help="Enable hot reload")

    # monitor
    p_mon = subparsers.add_parser("monitor", help="Run the monitoring pipeline")
    p_mon.add_argument("--window", required=True, help="Data window label")
    p_mon.add_argument("--config-path", default=None, help="Override config path")

    # drift
    p_drift = subparsers.add_parser("drift", help="Run drift detection only")
    p_drift.add_argument("--window", required=True, help="Data window label")

    # validate-config
    subparsers.add_parser("validate-config", help="Validate the config system")

    # pipeline
    p_pipe = subparsers.add_parser("pipeline", help="Run full training pipeline")
    p_pipe.add_argument("--raw-data-path", default=None, help="Path to raw CSV")
    p_pipe.add_argument("--skip-data-prep", action="store_true", help="Use existing parquets")
    p_pipe.add_argument("--skip-training", action="store_true", help="Use existing model")

    # train
    p_train = subparsers.add_parser("train", help="Train models on processed data")
    p_train.add_argument("--experiment-name", default=None, help="MLflow experiment name")

    # evaluate
    subparsers.add_parser("evaluate", help="Evaluate champion on test set")

    # data-prep
    p_dp = subparsers.add_parser("data-prep", help="Run data preparation only")
    p_dp.add_argument("--raw-data-path", default=None, help="Path to raw CSV")

    # data-pipeline
    p_dpipe = subparsers.add_parser("data-pipeline", help="Run data preparation pipeline")
    p_dpipe.add_argument("--raw-data-path", default=None, help="Path to raw CSV")

    # training-pipeline
    subparsers.add_parser("training-pipeline", help="Run training pipeline only")

    # evaluation-pipeline
    subparsers.add_parser("evaluation-pipeline", help="Run evaluation pipeline only")

    # full-pipeline
    p_full = subparsers.add_parser("full-pipeline", help="Run all three pipelines")
    p_full.add_argument("--raw-data-path", default=None, help="Path to raw CSV")
    p_full.add_argument("--skip-data", action="store_true", help="Skip data pipeline")
    p_full.add_argument("--skip-training", action="store_true", help="Skip training")

    # retrain
    p_retrain = subparsers.add_parser("retrain", help="Drift-triggered retraining")
    p_retrain.add_argument("--window", required=True, help="Data window label")

    # tune
    p_tune = subparsers.add_parser("tune", help="Optuna hyperparameter tuning")
    p_tune.add_argument("--n-trials", type=int, default=50, help="Number of trials")

    # explain
    p_explain = subparsers.add_parser("explain", help="SHAP explanation")
    p_explain.add_argument("--input-json", default=None, help="Input for local explanation")

    # ab-summary
    subparsers.add_parser("ab-summary", help="Print A/B test summary")

    # ab-analyze
    p_ab_analyze = subparsers.add_parser("ab-analyze", help="Analyze A/B test results")
    p_ab_analyze.add_argument(
        "--min-requests", type=int, default=1000, help="Min requests for analysis",
    )

    # ab-load-challenger
    p_ab_load = subparsers.add_parser("ab-load-challenger", help="Register challenger model")
    p_ab_load.add_argument("--model-path", required=True, help="Path to challenger pkl")

    # health-check
    p_health = subparsers.add_parser("health-check", help="Check API health")
    p_health.add_argument("--url", default="http://localhost:8000", help="API base URL")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Main entry point for the BNPL CLI.

    Parses arguments and delegates to BNPLRunner methods.
    Handles top-level exceptions with clean error messages.
    Exits with code 0 on success, 1 on error.

    Args:
        argv: Optional argument list for testing. Defaults to sys.argv.
    """
    args = parse_args(argv)

    if args.command is None:
        parse_args(["--help"])
        return

    runner = BNPLRunner()

    try:
        if args.command == "preprocess":
            runner.run_preprocess(args.input_json)
        elif args.command == "predict":
            runner.run_predict(args.input_json, args.model_path, args.threshold)
        elif args.command == "serve":
            runner.run_serve(args.host, args.port, args.reload)
        elif args.command == "monitor":
            runner.run_monitor(args.window, args.config_path)
        elif args.command == "drift":
            runner.run_drift(args.window)
        elif args.command == "validate-config":
            runner.run_validate_config()
        elif args.command == "pipeline":
            runner.run_pipeline(args.raw_data_path, args.skip_data_prep, args.skip_training)
        elif args.command == "train":
            runner.run_train(args.experiment_name)
        elif args.command == "evaluate":
            runner.run_evaluate()
        elif args.command == "data-prep":
            runner.run_data_prep(args.raw_data_path)
        elif args.command == "data-pipeline":
            runner.run_data_pipeline(args.raw_data_path)
        elif args.command == "training-pipeline":
            runner.run_training_pipeline()
        elif args.command == "evaluation-pipeline":
            runner.run_evaluation_pipeline()
        elif args.command == "full-pipeline":
            runner.run_full_pipeline(args.raw_data_path, args.skip_data, args.skip_training)
        elif args.command == "retrain":
            runner.run_retrain(args.window)
        elif args.command == "tune":
            runner.run_tune(args.n_trials)
        elif args.command == "explain":
            runner.run_explain(args.input_json)
        elif args.command == "ab-summary":
            runner.run_ab_summary()
        elif args.command == "ab-analyze":
            runner.run_ab_analyze(args.min_requests)
        elif args.command == "ab-load-challenger":
            runner.run_ab_load_challenger(args.model_path)
        elif args.command == "health-check":
            runner.run_health_check(args.url)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
    except Exception as exc:
        logger.error("Command '%s' failed: %s", args.command, exc, exc_info=True)
        print(f"\nError: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
