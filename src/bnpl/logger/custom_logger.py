"""Custom logger factory and LoggerMixin for class-based logging."""

from __future__ import annotations

import logging


def get_logger(name: str) -> logging.Logger:
    """Get a named logger.

    Usage at module level::

        logger = get_logger(__name__)

    The returned logger inherits handlers from the ``bnpl`` logger
    configured in logging.yaml.
    """
    return logging.getLogger(name)


class LoggerMixin:
    """Mixin providing ``self.logger`` with automatic module.Class naming.

    Usage::

        class DataLoader(LoggerMixin):
            def load(self):
                self.logger.info("Loading data...")

    The logger name becomes e.g. ``bnpl.data.loader.DataLoader``.
    """

    @property
    def logger(self) -> logging.Logger:
        attr = "_logger_instance"
        if not hasattr(self, attr):
            name = f"{self.__class__.__module__}.{self.__class__.__name__}"
            object.__setattr__(self, attr, logging.getLogger(name))
        return getattr(self, attr)
