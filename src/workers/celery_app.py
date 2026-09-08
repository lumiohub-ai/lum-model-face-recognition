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

import zlib

from celery import Celery
from kombu import Queue, Exchange
from loguru import logger

from config.settings import settings


def camera_slot(camera_id: int, n_slots: int) -> int:
    """Stable slot number for a camera id, 0 <= slot < n_slots (LSO-186).

    crc32, not the builtin `hash()`: `hash()` on strings is randomized per
    process (PYTHONHASHSEED), so the decode-worker and the yolo-worker — two
    separate processes that BOTH compute this to agree on a queue — could hash
    the same camera to different slots and split its frames across two
    camera-workers, fragmenting tracker state. crc32 is a fixed checksum: same
    input -> same number in every process, on every host, across restarts.
    `str(camera_id)` so ids that differ only in type (5 vs "5") still collide
    to one slot, and so the spread doesn't degenerate for small sequential ids.
    """
    return zlib.crc32(str(camera_id).encode()) % n_slots

# Create Celery app
celery = Celery(
    'smart_office',
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=[
        'workers.embedding_tasks',
        'workers.detection_tasks',
        'workers.camera_tasks',
        'workers.yolo_tasks',
        'workers.face_tasks',
    ]
)

# Define exchanges and queues
default_exchange = Exchange('default', type='direct')
dlq_exchange = Exchange('dlq', type='direct')


def camera_queue_name(camera_id: int) -> str:
    """The queue `yolo.detect` forwards a camera's detections to, and which a
    camera-worker consumes.

    Hash-to-slot (LSO-186): a camera routes to one of `camera_slot_count`
    fixed slot queues (`cam-slot-0`, `cam-slot-1`, …) by a stable hash of its
    id, NOT a per-camera `cam.<id>` queue. This decouples the queue set from
    the live camera set: a camera added in the app hashes to an existing slot
    that a worker is already consuming — no `-Q` edit, no restart. Sticky by
    construction: same id -> same slot -> same worker, so per-camera tracker
    state stays put (as long as N is unchanged and that slot's worker lives).

    Not declared in `task_queues` below: `apply_async(queue="cam-slot-0")`
    needs no prior declaration (task_create_missing_queues defaults to True).
    """
    return f"cam-slot-{camera_slot(camera_id, settings.camera_slot_count)}"


def slot_queue_name(slot: int) -> str:
    """The queue a camera-worker consumes once it has leased `slot`
    (pipeline/slot_lease.py). The consumer side of `camera_queue_name`."""
    return f"cam-slot-{slot}"

# Celery configuration
celery.conf.update(
    # Task settings
    #
    # Results are pickled, not JSON. The face inference result dicts carry
    # numpy throughout — `embedding` (512-float vector),
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
    #
    # 'camera.*' is deliberately absent: camera.track has no static queue at
    # all (see its @celery.task decorator) — it's dispatched exclusively via
    # yolo.detect's send_task(queue=camera_queue_name(camera_id)), one queue
    # per camera, statically assigned to a camera-worker via compose.yml's
    # -Q list. A route entry here could only name one fixed queue, which is
    # exactly the thing per-camera queues need to NOT be.
    task_routes={
        'workers.embedding_tasks.*': {'queue': 'embeddings'},
        'workers.detection_tasks.*': {'queue': 'detections'},
        'workers.yolo_tasks.*': {'queue': 'yolo'},
        'workers.face_tasks.*': {'queue': 'face'},
        'detection.*': {'queue': 'detections'},
        'embedding.*': {'queue': 'embeddings'},
        'yolo.*': {'queue': 'yolo'},
        'face.*': {'queue': 'face'},
    },

    # Queue definitions with DLQ support
    #
    # Three kinds of queue in this app:
    #   - shared, stateless: embeddings/detections/yolo/face. Any worker
    #     consuming that queue may process any task on it.
    #   - pinned, one per camera: cam.<id> (see camera_queue_name above).
    #     Not declared here — task_create_missing_queues handles them, and a
    #     fixed list here would need editing on every camera add/remove.
    #
    # yolo and face MUST stay separate queues with their own worker
    # processes: camera-worker's process_frame blocks on face.embed_batch's
    # result, so if face inference shared a queue with yolo's batch work, a
    # backlog on one could starve the worker the other is waiting on.
    # Keeping them separate is what makes the blocking call safe — do not
    # consolidate these to "simplify".
    task_queues=(
        Queue('embeddings', exchange=default_exchange, routing_key='embeddings'),
        Queue('detections', exchange=default_exchange, routing_key='detections'),
        Queue('yolo', exchange=default_exchange, routing_key='yolo'),
        Queue('face', exchange=default_exchange, routing_key='face'),
        Queue('dlq.embeddings', exchange=dlq_exchange, routing_key='dlq.embeddings'),
        Queue('dlq.detections', exchange=dlq_exchange, routing_key='dlq.detections'),
        Queue('dlq.camera_frames', exchange=dlq_exchange, routing_key='dlq.camera_frames'),
        Queue('dlq.yolo', exchange=dlq_exchange, routing_key='dlq.yolo'),
        Queue('dlq.face', exchange=dlq_exchange, routing_key='dlq.face'),
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
