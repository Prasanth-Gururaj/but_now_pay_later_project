"""Tests for the logging system."""

from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock

import pytest

from bnpl.logger import LoggerMixin, get_logger, log_errors, log_execution, setup_logging
from bnpl.logger.logger_config import JsonFormatter


class TestSetupLogging:
    """Verify logging initialization."""

    def test_creates_handlers(self, tmp_path: object) -> None:
        setup_logging()
        bnpl_logger = logging.getLogger("bnpl")
        handler_types = [type(h).__name__ for h in bnpl_logger.handlers]
        assert "StreamHandler" in handler_types
        assert "RotatingFileHandler" in handler_types
        assert len(bnpl_logger.handlers) >= 3


class TestGetLogger:
    """Verify logger factory."""

    def test_returns_named_logger(self) -> None:
        log = get_logger("bnpl.test.module")
        assert log.name == "bnpl.test.module"

    def test_returns_logger_instance(self) -> None:
        log = get_logger("bnpl.test")
        assert isinstance(log, logging.Logger)


class TestLoggerMixin:
    """Verify mixin provides correctly named logger."""

    def test_logger_name(self) -> None:
        class MyComponent(LoggerMixin):
            pass

        obj = MyComponent()
        expected = f"{MyComponent.__module__}.MyComponent"
        assert obj.logger.name == expected

    def test_logger_cached(self) -> None:
        class AnotherComponent(LoggerMixin):
            pass

        obj = AnotherComponent()
        assert obj.logger is obj.logger


class TestLogExecution:
    """Verify @log_execution decorator."""

    def test_start_and_success(self, caplog: pytest.LogCaptureFixture) -> None:
        @log_execution(operation="test_op")
        def do_work() -> int:
            return 42

        with caplog.at_level(logging.DEBUG):
            result = do_work()

        assert result == 42
        messages = caplog.text
        assert "op=test_op | START" in messages
        assert "op=test_op | SUCCESS" in messages
        assert "duration=" in messages

    def test_failure_with_traceback(self, caplog: pytest.LogCaptureFixture) -> None:
        @log_execution(operation="fail_op")
        def bad_work() -> None:
            raise ValueError("test error")

        with caplog.at_level(logging.DEBUG), pytest.raises(ValueError, match="test error"):
            bad_work()

        messages = caplog.text
        assert "op=fail_op | FAILURE" in messages
        assert "ValueError: test error" in messages

    def test_works_on_methods(self, caplog: pytest.LogCaptureFixture) -> None:
        class Worker(LoggerMixin):
            @log_execution(operation="method_op")
            def run(self) -> str:
                return "done"

        with caplog.at_level(logging.DEBUG):
            result = Worker().run()

        assert result == "done"
        assert "op=method_op | START" in caplog.text
        assert "op=method_op | SUCCESS" in caplog.text

    def test_default_operation_name(self, caplog: pytest.LogCaptureFixture) -> None:
        @log_execution()
        def my_function() -> None:
            pass

        with caplog.at_level(logging.DEBUG):
            my_function()

        assert "my_function | START" in caplog.text


class TestLogErrors:
    """Verify @log_errors decorator."""

    def test_silent_on_success(self, caplog: pytest.LogCaptureFixture) -> None:
        @log_errors(operation="safe_op")
        def safe_work() -> int:
            return 1

        with caplog.at_level(logging.DEBUG):
            result = safe_work()

        assert result == 1
        assert "safe_op" not in caplog.text

    def test_logs_on_failure(self, caplog: pytest.LogCaptureFixture) -> None:
        @log_errors(operation="risky_op")
        def risky_work() -> None:
            raise RuntimeError("boom")

        with caplog.at_level(logging.DEBUG), pytest.raises(RuntimeError, match="boom"):
            risky_work()

        assert "op=risky_op | ERROR" in caplog.text
        assert "RuntimeError: boom" in caplog.text


class TestJsonFormatter:
    """Verify JSON formatter output."""

    def test_produces_valid_json(self) -> None:
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="hello",
            args=None,
            exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["level"] == "INFO"
        assert parsed["message"] == "hello"
        assert "timestamp" in parsed
