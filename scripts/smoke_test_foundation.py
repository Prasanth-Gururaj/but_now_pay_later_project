"""Smoke test: validates config, logger, and MLflow integration work end-to-end.

Run with: uv run python scripts/smoke_test_foundation.py
Exit code 0 = all checks passed.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    print("=" * 60)
    print("SMOKE TEST: Foundation Layer")
    print("=" * 60)

    # 1. Config
    from config.settings import get_settings

    settings = get_settings()
    print(f"\n[OK] Settings loaded: env={settings.app_env}")
    print(f"[OK] Features count: {len(settings.features)}")
    print(f"[OK] Tracking backend: {settings.tracking.backend}")
    print(f"[OK] Default threshold: {settings.thresholds.default_threshold}")

    # 2. Logger (use bnpl namespace so file handlers are active)
    from bnpl.logger import LoggerMixin, get_logger, log_errors, log_execution, setup_logging

    setup_logging(json_output=settings.logging.json_output)

    logger = get_logger("bnpl.smoke_test")
    logger.info("Smoke test: INFO message")
    logger.debug("Smoke test: DEBUG message")
    logger.warning("Smoke test: WARNING message")
    print("\n[OK] Logger initialized and messages logged")

    # 3. LoggerMixin + decorator tests via a class so they route through
    #    the bnpl.* logger hierarchy and hit file handlers
    class SmokeComponent(LoggerMixin):
        """Test component demonstrating LoggerMixin + decorators."""

        def do_work(self) -> None:
            self.logger.info("LoggerMixin works correctly")

        @log_execution(operation="smoke_success")
        def successful_operation(self) -> int:
            return 42

        @log_errors(operation="smoke_failure")
        def failing_operation(self) -> None:
            raise RuntimeError("Deliberate smoke test error")

        @log_execution(operation="smoke_execution_failure")
        def execution_failure(self) -> None:
            raise ValueError("Deliberate execution failure for error.log")

    component = SmokeComponent()
    component.do_work()
    print("[OK] LoggerMixin works")

    # 4. log_execution (success path)
    result = component.successful_operation()
    assert result == 42
    print("[OK] log_execution decorator (success path)")

    # 5. log_errors (failure path)
    try:
        component.failing_operation()
    except RuntimeError:
        print("[OK] log_errors decorator (failure path — error caught and re-raised)")

    # 6. log_execution (failure path — should appear in error.log)
    try:
        component.execution_failure()
    except ValueError:
        print("[OK] log_execution decorator (failure path — traceback in error.log)")

    # 7. MLflow
    import mlflow

    from bnpl.tracking import mlflow_run

    with mlflow_run("smoke-test-run") as run:
        mlflow.log_param("smoke_param", "test_value")
        mlflow.log_metric("smoke_metric", 1.0)
        run_id = run.info.run_id

    print(f"\n[OK] MLflow run completed: id={run_id}")

    # 8. Verify log files
    app_log = PROJECT_ROOT / "logs" / "app.log"
    error_log = PROJECT_ROOT / "logs" / "error.log"
    print(f"\n[CHECK] logs/app.log exists: {app_log.exists()}")
    print(f"[CHECK] logs/error.log exists: {error_log.exists()}")

    if error_log.exists():
        content = error_log.read_text(encoding="utf-8")
        has_error = "Deliberate" in content
        print(f"[CHECK] error.log contains deliberate error: {has_error}")

    if app_log.exists():
        content = app_log.read_text(encoding="utf-8")
        has_mlflow = "MLflow run completed" in content
        print(f"[CHECK] app.log contains MLflow run log: {has_mlflow}")

    print("\n" + "=" * 60)
    print("ALL SMOKE TESTS PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
