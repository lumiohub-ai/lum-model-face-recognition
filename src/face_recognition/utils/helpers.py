"""Utility functions for the face recognition system."""

import time
from functools import wraps
from typing import Callable, Optional, Type, Tuple, Any

from .constants import (
    MAX_API_RETRY_ATTEMPTS,
    API_RETRY_BASE_DELAY,
)

def retry_on_exception(
    max_attempts: int = MAX_API_RETRY_ATTEMPTS,
    base_delay: float = API_RETRY_BASE_DELAY,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    logger: Optional[Any] = None
) -> Callable:
    """Decorator to retry a function on exception with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts
        base_delay: Base delay in seconds (will be exponentially increased)
        exceptions: Tuple of exceptions to catch and retry on
        logger: Optional logger instance for logging retry attempts

    Returns:
        Decorated function with retry logic

    Example:
        >>> @retry_on_exception(max_attempts=3, base_delay=1, exceptions=(ConnectionError,))
        ... def fetch_data():
        ...     # code that might fail
        ...     pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts:
                        if logger:
                            logger.error(
                                f"{func.__name__} failed after {max_attempts} attempts: {e}"
                            )
                        raise

                    delay = base_delay * (2 ** (attempt - 1))  # Exponential backoff
                    if logger:
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt}/{max_attempts}), "
                            f"retrying in {delay}s: {e}"
                        )
                    time.sleep(delay)

            # This should never be reached, but just in case
            if last_exception:
                raise last_exception

        return wrapper
    return decorator

def validate_threshold(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Validate that a threshold value is within acceptable range.

    Args:
        value: Threshold value to validate
        min_val: Minimum acceptable value
        max_val: Maximum acceptable value

    Returns:
        The validated value

    Raises:
        ValueError: If value is outside the acceptable range
    """
    if not min_val <= value <= max_val:
        raise ValueError(
            f"Threshold must be between {min_val} and {max_val}, got {value}"
        )
    return value

def ensure_directory_exists(directory_path: str, logger: Optional[Any] = None) -> bool:
    """Ensure a directory exists, creating it if necessary.

    Args:
        directory_path: Path to the directory
        logger: Optional logger for logging creation

    Returns:
        True if directory exists or was created successfully, False otherwise
    """
    import os

    try:
        if not os.path.exists(directory_path):
            os.makedirs(directory_path, exist_ok=True)
            if logger:
                logger.debug(f"Created directory: {directory_path}")
        return True
    except OSError as e:
        if logger:
            logger.error(f"Failed to create directory {directory_path}: {e}")
        return False

def format_timestamp(dt: Any, timezone_str: str = "UTC", format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Format a datetime object with timezone awareness.

    Args:
        dt: Datetime object to format
        timezone_str: Timezone string (e.g., "UTC", "Asia/Tashkent")
        format_str: strftime format string

    Returns:
        Formatted timestamp string
    """
    import pytz
    from datetime import datetime

    if not isinstance(dt, datetime):
        raise TypeError(f"Expected datetime object, got {type(dt)}")

    tz = pytz.timezone(timezone_str)
    if dt.tzinfo is None:
        # Naive datetime, localize it
        dt = tz.localize(dt)
    else:
        # Already aware, convert to target timezone
        dt = dt.astimezone(tz)

    return dt.strftime(format_str)

def calculate_iou(box1: Tuple[float, float, float, float],
                  box2: Tuple[float, float, float, float]) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1: First bounding box (x1, y1, x2, y2)
        box2: Second bounding box (x1, y1, x2, y2)

    Returns:
        IoU score between 0 and 1
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Calculate intersection area
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate union area
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - intersection_area

    if union_area == 0:
        return 0.0

    return intersection_area / union_area

def sanitize_filename(filename: str, replacement: str = "_") -> str:
    """Sanitize a filename by removing or replacing invalid characters.

    Args:
        filename: Original filename
        replacement: Character to use as replacement for invalid chars

    Returns:
        Sanitized filename
    """
    import re

    # Remove or replace characters that are invalid in filenames
    invalid_chars = r'[<>:"/\\|?*\x00-\x1f]'
    sanitized = re.sub(invalid_chars, replacement, filename)

    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip('. ')

    # Ensure filename is not empty
    if not sanitized:
        sanitized = "unnamed"

    return sanitized
