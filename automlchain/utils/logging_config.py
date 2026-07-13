"""Logging configuration for AutoMLChain."""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog


def setup_logging(
    level: str = "INFO",
    format: str = "text",
    **kwargs: Any,
) -> None:
    """Configure structured logging for AutoMLChain.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR).
        format: Log format ("text" or "json").
        **kwargs: Additional structlog configuration.
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=numeric_level,
    )

    # Configure structlog
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    if format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
        **kwargs,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a structured logger.

    Args:
        name: Logger name (usually __name__).

    Returns:
        Configured logger.
    """
    return structlog.get_logger(name)
