"""Logging configuration for the ml_ops package."""

from __future__ import annotations

import logging
from logging.config import dictConfig
from pathlib import Path
from typing import Any


def setup_logging(level: str = "INFO", log_file: str | Path | None = None) -> logging.Logger:
    """Configure logging for the project.

    This is safe to call multiple times; it will not add duplicate handlers.

    Args:
        level: Root log level (e.g., "INFO", "DEBUG").
        log_file: Optional log file path.

    Returns:
        The configured root logger.
    """
    root = logging.getLogger()
    if root.handlers:
        return root

    handlers: dict[str, dict[str, Any]] = {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "level": level,
        },
    }
    root_handlers: list[str] = ["console"]

    if log_file is not None:
        log_path = Path(log_file)
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handlers["file"] = {
                "class": "logging.FileHandler",
                "formatter": "standard",
                "level": level,
                "filename": str(log_path),
            }
            root_handlers.append("file")
        except Exception:
            pass

    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {"format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s"},
            },
            "handlers": handlers,
            "root": {"handlers": root_handlers, "level": level},
        }
    )
    return logging.getLogger()
