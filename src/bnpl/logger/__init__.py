"""Structured logging with automatic context: module, class, function, operation."""

from bnpl.logger.custom_logger import LoggerMixin, get_logger
from bnpl.logger.decorators import log_errors, log_execution
from bnpl.logger.logger_config import setup_logging

__all__ = [
    "setup_logging",
    "get_logger",
    "LoggerMixin",
    "log_execution",
    "log_errors",
]
