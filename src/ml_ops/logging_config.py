"""Logging configuration for the ml_ops package."""

from __future__ import annotations

import logging
from logging.config import dictConfig


def setup_logging(level: str = "INFO") -> None:
    """Configure logging for the project.

    This is safe to call multiple times; it will not add duplicate handlers.

    Args:
        level: Root log level (e.g., "INFO", "DEBUG").
    """
    root = logging.getLogger()
    if root.handlers:
        return

    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {"format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s"},
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "standard",
                    "level": level,
                },
            },
            "root": {"handlers": ["console"], "level": level},
        }
    )
