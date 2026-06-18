"""Entrypoint script for the retraining pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from bnpl.logger import get_logger, setup_logging

logger = get_logger(__name__)

if __name__ == "__main__":
    from config.settings import get_settings

    settings = get_settings()
    setup_logging(json_output=settings.logging.json_output)
    logger.info("Retraining pipeline not yet implemented")
