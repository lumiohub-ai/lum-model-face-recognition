"""Logging configuration for the face recognition system.

This module provides structured logging setup using loguru with JSON formatting,
file rotation, and context-aware logging for better observability.
"""

import sys
from typing import Optional
from loguru import logger

from .constants import LOG_LEVEL_INFO


def setup_structured_logging(
    log_level: str = LOG_LEVEL_INFO,
    log_file: Optional[str] = None,
    enable_json: bool = False,
    enable_console: bool = True,
) -> None:
    """Configure structured logging for the application.

    Args:
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file for file-based logging
        enable_json: Whether to use JSON format for structured logs
        enable_console: Whether to enable console logging

    Example:
        >>> setup_structured_logging(
        ...     log_level="INFO",
        ...     log_file="logs/face_recognition.log",
        ...     enable_json=True
        ... )
    """
    # Remove default logger
    logger.remove()

    # Console logging with colors
    if enable_console:
        console_format = (
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<level>{message}</level>"
        )
        logger.add(
            sys.stderr,
            format=console_format,
            level=log_level,
            colorize=True,
            backtrace=True,
            diagnose=True,
        )

    # File logging with rotation
    if log_file:
        if enable_json:
            # JSON format for structured logging (better for log aggregation)
            logger.add(
                log_file,
                format="{message}",  # JSON serialization is done by serialize=True
                level=log_level,
                rotation="100 MB",  # Rotate when file reaches 100MB
                retention="30 days",  # Keep logs for 30 days
                compression="zip",  # Compress rotated logs
                serialize=True,  # Enable JSON serialization
                backtrace=True,
                diagnose=True,
            )
        else:
            # Standard text format for file logging
            file_format = (
                "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
                "{name}:{function}:{line} | {message}"
            )
            logger.add(
                log_file,
                format=file_format,
                level=log_level,
                rotation="100 MB",
                retention="30 days",
                compression="zip",
                backtrace=True,
                diagnose=True,
            )
