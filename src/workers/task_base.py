"""
Base Task Classes and Exception Types for Celery Tasks.

This module is separate from celery_app.py to avoid circular imports.
Import these classes in task modules instead of from celery_app.

Exception Hierarchy:
- RetryableError: Errors that should trigger a retry
  - ImageFetchError: Failed to fetch image (network issue)
  - DatabaseError: Database operation failed
  - ModelInferenceError: ML model inference failed
- NonRetryableError: Errors that should NOT be retried
  - ValidationError: Input validation failed
"""

import json
import traceback
from datetime import datetime
from typing import Any
from celery import Task
from loguru import logger


# ============================================================
# Custom Exception Classes for Error Differentiation
# ============================================================

class RetryableError(Exception):
    """Errors that should trigger a retry (network issues, temporary failures)."""
    pass


class NonRetryableError(Exception):
    """Errors that should NOT be retried (validation errors, permanent failures)."""
    pass


class ImageFetchError(RetryableError):
    """Failed to fetch image - may be temporary network issue."""
    pass


class ValidationError(NonRetryableError):
    """Input validation failed - retrying won't help."""
    pass


class DatabaseError(RetryableError):
    """Database operation failed - may be temporary."""
    pass


class ModelInferenceError(RetryableError):
    """ML model inference failed - may be due to resource constraints."""
    pass


# ============================================================
# Dead-Letter Queue Handler
# ============================================================

def send_to_dlq(task_name: str, task_id: str, args: tuple, kwargs: dict,
                exc: Exception, traceback_str: str) -> None:
    """Send failed task to Dead-Letter Queue for manual inspection.

    Args:
        task_name: Name of the failed task
        task_id: Celery task ID
        args: Task positional arguments
        kwargs: Task keyword arguments
        exc: Exception that caused the failure
        traceback_str: Traceback string
    """

    dlq_message = {
        'task_name': task_name,
        'task_id': task_id,
        'args': list(args) if args else [],
        'kwargs': kwargs or {},
        'error': str(exc),
        'error_type': type(exc).__name__,
        'traceback': traceback_str,
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'retries_exhausted': True,
    }

    try:
        # Determine DLQ based on task queue
        if 'embedding' in task_name.lower():
            dlq_key = 'dlq:embeddings'
        elif 'detection' in task_name.lower():
            dlq_key = 'dlq:detections'
        else:
            dlq_key = 'dlq:unknown'

        # Store in Redis list (LPUSH for FIFO when consuming with RPOP)
        from messaging.redis_client import RedisClient
        r = RedisClient.get_instance().client
        r.lpush(dlq_key, json.dumps(dlq_message))

        # Also store in a hash for quick lookup by task_id
        r.hset(f'{dlq_key}:index', task_id, json.dumps(dlq_message))

        # Expire old DLQ messages after 7 days
        r.expire(dlq_key, 7 * 24 * 3600)
        r.expire(f'{dlq_key}:index', 7 * 24 * 3600)

        logger.warning(f"[DLQ] Task {task_id} moved to {dlq_key}: {type(exc).__name__}")

    except Exception as e:
        logger.exception(f"[DLQ] Failed to send task to DLQ: {e}")


# ============================================================
# Task Base Class with Enhanced Error Handling
# ============================================================

class BaseTaskWithRetry(Task):
    """Base task class with enhanced retry logic and DLQ support.

    Features:
    - Exponential backoff
    - Error type differentiation
    - Automatic DLQ on max retries
    - Soft timeout handling
    """

    # Override these in subclasses if needed
    autoretry_for = (RetryableError, ConnectionError, TimeoutError)
    dont_autoretry_for = (NonRetryableError, ValidationError)
    retry_backoff = True
    retry_backoff_max = 300  # Max 5 minutes between retries
    retry_jitter = True       # Add randomness to prevent thundering herd
    max_retries = 5

    def on_failure(self, exc: Exception, task_id: str, args: tuple, kwargs: dict, einfo: Any) -> None:
        """Called when task fails after all retries.

        Sends failed task to Dead-Letter Queue for manual inspection.
        """
        # Get traceback string
        tb_str = ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))

        # Send to DLQ
        send_to_dlq(
            task_name=self.name,
            task_id=task_id,
            args=args,
            kwargs=kwargs,
            exc=exc,
            traceback_str=tb_str
        )

        logger.error(f"[Celery] Task {self.name}[{task_id}] failed permanently: {exc}")

    def on_retry(self, exc: Exception, task_id: str, args: tuple, kwargs: dict, einfo: Any) -> None:
        """Called when task is being retried."""
        retry_count = self.request.retries
        max_retries = self.max_retries

        logger.warning(
            f"[Celery] Retrying {self.name}[{task_id}] "
            f"(attempt {retry_count + 1}/{max_retries}): {exc}"
        )

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Override in subclasses."""
        raise NotImplementedError("Subclasses must implement run()")
