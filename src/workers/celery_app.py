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

from celery import Celery
from kombu import Queue, Exchange
from loguru import logger

from config.settings import settings

# Create Celery app
celery = Celery(
    'smart_office',
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=[
        'workers.embedding_tasks',
        'workers.detection_tasks',
        'workers.camera_tasks',
    ]
)

# Define exchanges and queues
default_exchange = Exchange('default', type='direct')
dlq_exchange = Exchange('dlq', type='direct')

# Celery configuration
celery.conf.update(
    # Task settings
    #
    # Results are pickled, not JSON (LSO-67 Stage 2). The face inference
    # result dicts carry numpy throughout — `embedding` (512-float vector),
    # `face_image` (a raw pixel crop), and even `face_bbox`, whose elements
    # are numpy.int64 from `face.bbox.astype(int)` rather than Python ints.
    # json.dumps rejects all of these, so a JSON result serializer fails the
    # first face task outright. Converting field-by-field was considered and
    # rejected: the numpy leaks are not all obvious (face_bbox looks like a
    # plain list), so it would be a standing trap for anyone adding a field.
    #
    # Task *payloads* stay JSON: they only ever carry small handles/scalars,
    # and keeping them JSON preserves the readable-in-Redis property that
    # makes broker inspection useful during incidents.
    #
    # Pickle is safe here only because the broker is internal and every
    # producer and consumer is one of our own processes. If either stops
    # being true, this needs revisiting.
    task_serializer='json',
    accept_content=['json', 'pickle'],
    result_serializer='pickle',
    result_accept_content=['json', 'pickle'],
    timezone='UTC',
    enable_utc=True,

    # Task routing
    task_routes={
        'workers.embedding_tasks.*': {'queue': 'embeddings'},
        'workers.detection_tasks.*': {'queue': 'detections'},
        'workers.camera_tasks.*': {'queue': 'camera_frames'},
        'detection.*': {'queue': 'detections'},
        'embedding.*': {'queue': 'embeddings'},
        'camera.*': {'queue': 'camera_frames'},
    },

    # Queue definitions with DLQ support
    task_queues=(
        Queue('embeddings', exchange=default_exchange, routing_key='embeddings'),
        Queue('detections', exchange=default_exchange, routing_key='detections'),
        Queue('camera_frames', exchange=default_exchange, routing_key='camera_frames'),
        Queue('dlq.embeddings', exchange=dlq_exchange, routing_key='dlq.embeddings'),
        Queue('dlq.detections', exchange=dlq_exchange, routing_key='dlq.detections'),
        Queue('dlq.camera_frames', exchange=dlq_exchange, routing_key='dlq.camera_frames'),
    ),

    # Task execution settings
    task_acks_late=True,
    task_reject_on_worker_lost=True,

    # CRITICAL: Task timeouts to prevent hanging workers
    task_time_limit=600,           # Hard limit: 10 minutes (kills task)
    task_soft_time_limit=540,      # Soft limit: 9 minutes (raises exception)

    # Worker settings
    worker_prefetch_multiplier=1,
    worker_concurrency=settings.celery_concurrency,

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
logger.info(f"[Celery] Configured with broker: {settings.celery_broker_url}")
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
]
