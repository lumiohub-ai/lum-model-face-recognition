"""
Celery Application Configuration

Celery is used for async task processing in True MDA architecture.
Tasks are queued from MDA handlers and processed by workers.

IMPORTANT: This configuration includes:
- Task timeouts to prevent hanging workers
- Exponential backoff retry strategy
- Dead-letter queue for failed tasks
- Error differentiation (retryable vs non-retryable errors)

NOTE: Exception classes and BaseTaskWithRetry are in task_base.py
to avoid circular imports. Import from there in task modules.
"""

import os
from celery import Celery
from kombu import Queue, Exchange
from loguru import logger

# Get Redis config directly from env (avoids circular import with messaging module)
REDIS_HOST = os.getenv('SO_REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('SO_REDIS_PORT', 6379))

CELERY_BROKER_URL = os.getenv('SO_CELERY_BROKER_URL', f'redis://{REDIS_HOST}:{REDIS_PORT}/0')
CELERY_RESULT_BACKEND = os.getenv('SO_CELERY_RESULT_BACKEND', f'redis://{REDIS_HOST}:{REDIS_PORT}/1')

# Create Celery app
celery = Celery(
    'smart_office',
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=[
        'workers.embedding_tasks',
        'workers.detection_tasks',
    ]
)

# Define exchanges and queues
default_exchange = Exchange('default', type='direct')
dlq_exchange = Exchange('dlq', type='direct')

# Celery configuration
celery.conf.update(
    # Task settings
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,

    # Task routing
    task_routes={
        'workers.embedding_tasks.*': {'queue': 'embeddings'},
        'workers.detection_tasks.*': {'queue': 'detections'},
        'detection.*': {'queue': 'detections'},
        'embedding.*': {'queue': 'embeddings'},
    },

    # Queue definitions with DLQ support
    task_queues=(
        Queue('embeddings', exchange=default_exchange, routing_key='embeddings'),
        Queue('detections', exchange=default_exchange, routing_key='detections'),
        Queue('dlq.embeddings', exchange=dlq_exchange, routing_key='dlq.embeddings'),
        Queue('dlq.detections', exchange=dlq_exchange, routing_key='dlq.detections'),
    ),

    # Task execution settings
    task_acks_late=True,
    task_reject_on_worker_lost=True,

    # CRITICAL: Task timeouts to prevent hanging workers
    task_time_limit=600,           # Hard limit: 10 minutes (kills task)
    task_soft_time_limit=540,      # Soft limit: 9 minutes (raises exception)

    # Worker settings
    worker_prefetch_multiplier=1,
    worker_concurrency=int(os.getenv('SO_CELERY_CONCURRENCY', 2)),

    # Result settings
    result_expires=3600,  # 1 hour

    # Retry settings with exponential backoff
    task_default_retry_delay=5,
    task_max_retries=5,

    # Exponential backoff: 5s, 10s, 20s, 40s, 80s
    # Configured per-task using retry_backoff=True

    # Track task state for monitoring
    task_track_started=True,

    # Send task-sent event for monitoring
    task_send_sent_event=True,

    # Worker will send task-related events
    worker_send_task_events=True,

    # Enable task events for monitoring
    task_events=True,
)

# Log configuration (use logger, not print)
logger.info(f"[Celery] Configured with broker: {CELERY_BROKER_URL}")
logger.info(f"[Celery] Task timeouts: soft={celery.conf.task_soft_time_limit}s, hard={celery.conf.task_time_limit}s")

# Re-export from task_base for backward compatibility
# NOTE: Import these from workers.task_base in new code to avoid circular imports
from .task_base import (
    BaseTaskWithRetry,
    RetryableError,
    NonRetryableError,
    ImageFetchError,
    ValidationError,
    DatabaseError,
    ModelInferenceError,
    send_to_dlq,
)

__all__ = [
    'celery',
    'BaseTaskWithRetry',
    'RetryableError',
    'NonRetryableError',
    'ImageFetchError',
    'ValidationError',
    'DatabaseError',
    'ModelInferenceError',
    'send_to_dlq',
    'CELERY_BROKER_URL',
    'CELERY_RESULT_BACKEND',
]
