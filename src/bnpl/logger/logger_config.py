"""Logging configuration: reads config/logging.yaml, applies dictConfig,
and provides a JSON formatter for production use.
"""

from __future__ import annotations

import json
import logging
import logging.config
from datetime import datetime, timezone
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent.parent.parent / "config"
PROJECT_ROOT = CONFIG_DIR.parent
LOGS_DIR = PROJECT_ROOT / "logs"


class JsonFormatter(logging.Formatter):
    """Structured JSON log formatter for production environments."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "function": record.funcName,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
        }

        if hasattr(record, "operation"):
            log_entry["operation"] = record.operation

        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, default=str)


def _ensure_log_dirs(config_dict: dict[str, object]) -> None:
    """Create log directories and convert relative paths to absolute."""
    handlers = config_dict.get("handlers", {})
    if not isinstance(handlers, dict):
        return
    for handler_cfg in handlers.values():
        if not isinstance(handler_cfg, dict):
            continue
        filename = handler_cfg.get("filename")
        if filename and isinstance(filename, str):
            log_path = Path(filename)
            if not log_path.is_absolute():
                log_path = PROJECT_ROOT / log_path
                handler_cfg["filename"] = str(log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)


def setup_logging(json_output: bool = False) -> None:
    """Initialize logging from config/logging.yaml.

    Call once at process start from each entrypoint
    (api.py, dashboard/app.py, scripts/*.py).

    Args:
        json_output: Switch console formatter to JSON (for prod).
    """
    yaml_path = CONFIG_DIR / "logging.yaml"
    if yaml_path.exists():
        with open(yaml_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
    else:
        logging.basicConfig(level=logging.INFO)
        return

    _ensure_log_dirs(config_dict)

    if json_output:
        handlers = config_dict.get("handlers", {})
        if isinstance(handlers, dict) and "console" in handlers:
            handlers["console"]["formatter"] = "json"

    logging.config.dictConfig(config_dict)
