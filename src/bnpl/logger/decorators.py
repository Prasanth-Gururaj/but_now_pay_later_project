"""Logging decorators: @log_execution and @log_errors.

Work on both plain functions and instance methods.
"""

from __future__ import annotations

import functools
import logging
import time
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


def _resolve_logger(func: Callable[..., Any], args: tuple[Any, ...]) -> logging.Logger:
    """Pick the right logger: self.logger for methods, module logger for functions."""
    if args and hasattr(args[0], "logger"):
        return args[0].logger

    module = func.__module__ or "__main__"
    qualname = func.__qualname__
    parts = qualname.split(".")
    if len(parts) > 1:
        class_name = parts[-2]
        return logging.getLogger(f"{module}.{class_name}")

    return logging.getLogger(module)


def _format_operation(operation: str | None, func: Callable[..., Any]) -> str:
    """Build the operation tag for log messages."""
    return operation if operation else func.__qualname__


def log_execution(
    operation: str | None = None,
    *,
    log_args: bool = False,
    level: int = logging.INFO,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator that logs START, SUCCESS (with duration), and FAILURE (with traceback).

    Args:
        operation: Custom operation name. Defaults to function qualname.
        log_args: If True, includes function arguments in the START message.
        level: Log level for START/SUCCESS. FAILURE always uses ERROR.
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            logger = _resolve_logger(func, args)
            op = _format_operation(operation, func)

            msg_parts = [f"op={op} | START"]
            if log_args:
                display_args = args[1:] if args and hasattr(args[0], "logger") else args
                msg_parts.append(f"args={display_args}, kwargs={kwargs}")
            logger.log(level, " | ".join(msg_parts))

            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start_time
                logger.log(level, f"op={op} | SUCCESS | duration={elapsed:.3f}s")
                return result
            except Exception as exc:
                elapsed = time.perf_counter() - start_time
                logger.error(
                    f"op={op} | FAILURE | duration={elapsed:.3f}s | "
                    f"error={type(exc).__name__}: {exc}",
                    exc_info=True,
                )
                raise

        return wrapper

    return decorator


def log_errors(
    operation: str | None = None,
    *,
    level: int = logging.ERROR,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Lightweight decorator that only logs failures — no START/SUCCESS noise.

    Args:
        operation: Custom operation name. Defaults to function qualname.
        level: Log level for the error message.
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                logger = _resolve_logger(func, args)
                op = _format_operation(operation, func)
                logger.log(
                    level,
                    f"op={op} | ERROR | {type(exc).__name__}: {exc}",
                    exc_info=True,
                )
                raise

        return wrapper

    return decorator
