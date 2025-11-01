"""Logging configuration for the face recognition system.

This module provides structured logging setup using loguru with JSON formatting,
file rotation, and context-aware logging for better observability.
"""

import os
import sys
from typing import Optional
from loguru import logger

from ..config.constants import (
    LOG_LEVEL_DEBUG,
    LOG_LEVEL_INFO,
    LOG_LEVEL_WARNING,
    LOG_LEVEL_ERROR,
)


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
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
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


def get_structured_logger(name: str, **context):
    """Get a logger with context binding for structured logging.

    Args:
        name: Logger name (typically module name)
        **context: Additional context to bind to the logger

    Returns:
        Logger instance with bound context

    Example:
        >>> log = get_structured_logger("camera.processor", camera_id=1)
        >>> log.info("Frame processed", frame_num=42, fps=30.5)
        # Outputs: {"camera_id": 1, "frame_num": 42, "fps": 30.5, ...}
    """
    return logger.bind(logger_name=name, **context)


def log_function_call(func_name: str, **params):
    """Log a function call with parameters (for debugging).

    Args:
        func_name: Name of the function being called
        **params: Function parameters to log

    Example:
        >>> log_function_call("recognize_face", track_id=123, threshold=0.5)
    """
    logger.debug(
        f"Function call: {func_name}",
        extra={"function": func_name, "parameters": params}
    )


def log_performance_metric(metric_name: str, value: float, unit: str = "", **context):
    """Log a performance metric with structured data.

    Args:
        metric_name: Name of the metric (e.g., "fps", "latency")
        value: Metric value
        unit: Optional unit (e.g., "ms", "fps")
        **context: Additional context

    Example:
        >>> log_performance_metric("frame_processing_time", 33.5, unit="ms", camera_id=1)
    """
    logger.info(
        f"Performance: {metric_name} = {value}{unit}",
        extra={
            "metric_type": "performance",
            "metric_name": metric_name,
            "value": value,
            "unit": unit,
            **context
        }
    )


def log_recognition_event(
    event_type: str,
    person_name: str,
    track_id: int,
    similarity: float,
    camera_id: Optional[int] = None,
    camera_name: Optional[str] = None,
    **extra_context
):
    """Log a face recognition event with structured data.

    Args:
        event_type: Type of event (RECOGNIZED, UNRECOGNIZED, PARTIAL_MATCH)
        person_name: Name of the person (or "Unknown")
        track_id: Tracking ID
        similarity: Similarity score
        camera_id: Optional camera ID
        camera_name: Optional camera name
        **extra_context: Additional context to include

    Example:
        >>> log_recognition_event(
        ...     "RECOGNIZED",
        ...     "John Doe",
        ...     track_id=123,
        ...     similarity=0.85,
        ...     camera_id=1,
        ...     camera_name="Main Entrance"
        ... )
    """
    logger.info(
        f"Recognition: {event_type} | {person_name} | Track: {track_id} | Similarity: {similarity:.2f}",
        extra={
            "event_type": "recognition",
            "recognition_status": event_type,
            "person_name": person_name,
            "track_id": track_id,
            "similarity": similarity,
            "camera_id": camera_id,
            "camera_name": camera_name,
            **extra_context
        }
    )


def log_entry_exit_event(
    person_name: str,
    status: str,
    timestamp: str,
    camera_id: Optional[int] = None,
    camera_name: Optional[str] = None,
    **extra_context
):
    """Log an entry/exit event with structured data.

    Args:
        person_name: Name of the person
        status: Entry status (IN/OUT)
        timestamp: Event timestamp
        camera_id: Optional camera ID
        camera_name: Optional camera name
        **extra_context: Additional context

    Example:
        >>> log_entry_exit_event(
        ...     "Jane Smith",
        ...     "IN",
        ...     "2025-01-30 14:30:00",
        ...     camera_id=2,
        ...     camera_name="Back Door"
        ... )
    """
    logger.info(
        f"Entry/Exit: {person_name} | {status} | {timestamp}",
        extra={
            "event_type": "entry_exit",
            "person_name": person_name,
            "status": status,
            "timestamp": timestamp,
            "camera_id": camera_id,
            "camera_name": camera_name,
            **extra_context
        }
    )


def log_api_request(
    method: str,
    endpoint: str,
    status_code: Optional[int] = None,
    duration_ms: Optional[float] = None,
    **extra_context
):
    """Log an API request with structured data.

    Args:
        method: HTTP method (GET, POST, etc.)
        endpoint: API endpoint
        status_code: Optional HTTP status code
        duration_ms: Optional request duration in milliseconds
        **extra_context: Additional context

    Example:
        >>> log_api_request(
        ...     "POST",
        ...     "/api/users/entry",
        ...     status_code=200,
        ...     duration_ms=45.3
        ... )
    """
    message = f"API {method} {endpoint}"
    if status_code:
        message += f" -> {status_code}"
    if duration_ms:
        message += f" ({duration_ms:.1f}ms)"

    log_level = "info" if status_code and 200 <= status_code < 300 else "warning"

    getattr(logger, log_level)(
        message,
        extra={
            "event_type": "api_request",
            "http_method": method,
            "endpoint": endpoint,
            "status_code": status_code,
            "duration_ms": duration_ms,
            **extra_context
        }
    )


def log_error_with_context(
    error: Exception,
    context_message: str,
    **extra_context
):
    """Log an error with additional context information.

    Args:
        error: The exception that occurred
        context_message: Human-readable context about what was happening
        **extra_context: Additional context data

    Example:
        >>> try:
        ...     risky_operation()
        ... except Exception as e:
        ...     log_error_with_context(
        ...         e,
        ...         "Failed to process frame",
        ...         frame_num=42,
        ...         camera_id=1
        ...     )
    """
    logger.error(
        f"{context_message}: {str(error)}",
        extra={
            "event_type": "error",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context_message,
            **extra_context
        },
        exc_info=True
    )


# Convenience function to initialize logging from environment variables
def init_logging_from_env():
    """Initialize logging using environment variables.

    Environment variables:
        - LOG_LEVEL: Log level (default: INFO)
        - LOG_FILE: Path to log file (optional)
        - LOG_JSON: Enable JSON logging (default: false)
        - LOG_CONSOLE: Enable console logging (default: true)
    """
    log_level = os.getenv("LOG_LEVEL", LOG_LEVEL_INFO).upper()
    log_file = os.getenv("LOG_FILE")
    enable_json = os.getenv("LOG_JSON", "false").lower() == "true"
    enable_console = os.getenv("LOG_CONSOLE", "true").lower() == "true"

    setup_structured_logging(
        log_level=log_level,
        log_file=log_file,
        enable_json=enable_json,
        enable_console=enable_console,
    )

    logger.info(
        "Logging initialized",
        extra={
            "log_level": log_level,
            "log_file": log_file,
            "json_enabled": enable_json,
            "console_enabled": enable_console,
        }
    )
