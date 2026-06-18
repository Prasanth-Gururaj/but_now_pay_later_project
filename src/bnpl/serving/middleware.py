"""FastAPI middleware: request logging, error handling, metrics."""

from bnpl.logger import get_logger

logger = get_logger(__name__)
