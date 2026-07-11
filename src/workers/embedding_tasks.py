"""
Celery Tasks for Embedding Processing

These tasks are dispatched by StreamConsumer and processed by Celery workers.
Results are published as events to Backend via MDAPublisher.

Flow:
1. Backend sends CreateEmbedding command (Redis Streams)
2. StreamConsumer receives and queues Celery task
3. Celery Worker processes embeddings
4. Worker publishes EmbeddingCreated/EmbeddingFailed event (Redis Pub/Sub)
5. Backend receives event and notifies frontend via Socket.IO

IMPORTANT: Tasks use BaseTaskWithRetry for:
- Exponential backoff on retries
- Error type differentiation (retryable vs non-retryable)
- Automatic DLQ on permanent failures
"""

import time
import uuid
from contextlib import contextmanager
from typing import Any, Dict, Optional

from celery.exceptions import SoftTimeLimitExceeded
from loguru import logger

from workers.celery_app import celery

# Import from task_base to avoid circular imports
from .task_base import (
    BaseTaskWithRetry,
    RetryableError,
    NonRetryableError,
    ImageFetchError,
    ValidationError,
    DatabaseError,
    ModelInferenceError,
)

# Cache the (multi-second-to-load, GPU-resident) FaceDetector per worker
# process instead of constructing a fresh one on every single task
# invocation — Celery worker processes are long-lived, so this is safe to
# reuse across many task calls within the same process.
_detector_cache: Dict[tuple, Any] = {}


def _get_cached_detector(gpu_id: int = 0, padding_percent: float = 20.0):
    key = (gpu_id, padding_percent)
    if key not in _detector_cache:
        from domain.face_detection import FaceDetector
        _detector_cache[key] = FaceDetector(gpu_id=gpu_id, padding_percent=padding_percent)
    return _detector_cache[key]


def get_embedding_sync_service(client_slug: str):
    """Lazy import to avoid circular imports and ensure proper initialization.

    Reuses the process-cached FaceDetector (see _get_cached_detector) rather
    than letting EmbeddingSyncService.__init__ build a brand new one.
    """
    from infrastructure.storage import EmbeddingSyncService
    return EmbeddingSyncService(client_slug, detector=_get_cached_detector())


@contextmanager
def _user_embedding_lock(client_slug: str, user_id: Any, timeout: float = 30.0):
    """Distributed lock (Redis SET NX EX) serializing add/update/delete
    embedding tasks for the same user.

    Celery's default (prefork) pool runs tasks in separate OS processes, so
    an in-process threading.Lock would NOT prevent two different worker
    processes from handling, say, an UpdateEmbedding and a DeleteEmbedding
    for the same user concurrently — which can resurrect a just-deleted
    user's embeddings if the update's write lands after the delete.

    Yields True if the lock was acquired, False otherwise. Fails open (logs
    and yields True) on a Redis error, since blocking all embedding writes
    on Redis availability would be worse than the rare race this guards
    against.
    """
    from messaging.redis_client import RedisClient

    key = f"embedding_lock:{client_slug}:{user_id}"
    token = str(uuid.uuid4())
    acquired = False
    client = None
    try:
        client = RedisClient.get_instance().client
        # Brief wait-and-retry rather than failing immediately, since the
        # holder is expected to release within a normal task's runtime.
        for _ in range(int(timeout)):
            if client.set(key, token, nx=True, ex=int(timeout)):
                acquired = True
                break
            time.sleep(1)
    except Exception as e:
        logger.warning(f"[embedding lock] Redis error acquiring lock for {key}, proceeding without it: {e}")
        yield True
        return

    try:
        yield acquired
    finally:
        if acquired and client is not None:
            try:
                if client.get(key) == token:
                    client.delete(key)
            except Exception as e:
                logger.warning(f"[embedding lock] Failed to release lock {key}: {e}")


def validate_embedding_request(client_slug: str, user_data: Dict[str, Any]) -> None:
    """Validate embedding request parameters.

    Args:
        client_slug: Organization slug
        user_data: User data dict

    Raises:
        ValidationError: If validation fails (non-retryable)
    """
    if not client_slug or not isinstance(client_slug, str):
        raise ValidationError(f"Invalid client_slug: {client_slug}")

    if not user_data:
        raise ValidationError("user_data is required")

    user_id = user_data.get('id')
    if user_id is None:
        raise ValidationError("user_data.id is required")

    # Validate image_urls if present
    image_urls = user_data.get('image_urls', [])
    if not isinstance(image_urls, list):
        raise ValidationError(f"image_urls must be a list, got {type(image_urls)}")


def get_event_publisher(client_slug: str):
    """Get event publisher for sending results to Backend."""
    from messaging.publisher import MDAPublisher
    return MDAPublisher(client_slug)


def notify_embedding_reload(client_slug: str, user_id: int, action: str):
    """Notify camera engine to reload embeddings via Redis Pub/Sub.

    Best-effort: the embedding write itself already committed successfully
    by the time this runs, so a failure here doesn't undo that — it just
    means the live camera engine's in-memory cache won't refresh until the
    next unrelated reload event. One retry after a short delay covers the
    common transient-Redis-blip case; a persistent failure is still just
    logged rather than failing the whole (already-successful) task.
    """
    from messaging.redis_client import RedisClient
    from messaging.channels import INTERNAL_CHANNELS
    import json

    message = json.dumps({
        'client_slug': client_slug,
        'user_id': user_id,
        'action': action,
    })

    for attempt in range(2):
        try:
            RedisClient.get_instance().client.publish(
                INTERNAL_CHANNELS['EMBEDDING_RELOAD'], message
            )
            logger.info(f"[Celery] Notified camera engine to reload embeddings for {client_slug}")
            return
        except Exception as e:
            if attempt == 0:
                logger.warning(f"[Celery] Embedding reload notify failed, retrying once: {e}")
                time.sleep(1)
            else:
                logger.warning(
                    f"[Celery] Failed to notify embedding reload after retry: {e}"
                )


@celery.task(
    bind=True,
    base=BaseTaskWithRetry,
    name='embedding.add_user',
    queue='embeddings',
    autoretry_for=(RetryableError, ImageFetchError, DatabaseError, ConnectionError),
    dont_autoretry_for=(ValidationError, NonRetryableError),
    retry_backoff=True,
    retry_backoff_max=300,
    retry_jitter=True,
    max_retries=5,
)
def process_add_user(self, command_id: str, client_slug: str, user_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process CreateEmbedding command.

    Args:
        command_id: Original command ID from Backend
        client_slug: Organization slug
        user_data: User data with id, full_name, image_urls

    Returns:
        Result dict with status and embeddings_created count

    Raises:
        ValidationError: If input validation fails (non-retryable)
        ImageFetchError: If image download fails (retryable)
        DatabaseError: If database operation fails (retryable)
    """
    task_id = self.request.id
    user_id = user_data.get('id')

    logger.info(f"[Celery] Processing CreateEmbedding: id={user_id} task_id={task_id}")

    try:
        # STEP 1: Validate inputs (non-retryable errors)
        validate_embedding_request(client_slug, user_data)

        # Serialize against a concurrent DeleteEmbedding for the same user —
        # see _user_embedding_lock.
        with _user_embedding_lock(client_slug, user_id) as locked:
            if not locked:
                raise RetryableError(
                    f"Could not acquire embedding lock for user {user_id}, retrying"
                )

            # STEP 2: Get embedding sync service
            service = get_embedding_sync_service(client_slug)

            # STEP 3: Process user embeddings
            result = service.handle_user_created(user_data)
            embeddings_created = result.get('embeddings_added', 0)

        logger.info(f"[Celery] CreateEmbedding completed: {embeddings_created} embeddings task_id={task_id}")

        # STEP 4: Publish EmbeddingCreated event to Backend
        publisher = get_event_publisher(client_slug)
        publisher.publish_embedding_created(
            command_id=command_id,
            user_id=user_id,
            embeddings_created=embeddings_created
        )

        # STEP 5: Notify camera engine to reload embeddings
        notify_embedding_reload(client_slug, user_id, 'add')

        return {
            'status': 'success',
            'action': 'CreateEmbedding',
            'user_id': user_id,
            'embeddings_created': embeddings_created,
            'task_id': task_id,
        }

    except ValidationError:
        # Non-retryable: Don't retry validation errors
        logger.exception(f"[Celery] CreateEmbedding validation failed for {user_id}: validation error")
        _publish_failure_event(client_slug, command_id, user_id, "Validation failed")
        raise  # Let BaseTaskWithRetry handle DLQ

    except SoftTimeLimitExceeded:
        # Task timed out - log and fail
        logger.exception(f"[Celery] CreateEmbedding timed out for {user_id} task_id={task_id}")
        _publish_failure_event(client_slug, command_id, user_id, "Task timed out")
        raise RetryableError(f"Task timed out for user {user_id}")

    except (ConnectionError, TimeoutError) as e:
        # Network errors - retryable
        logger.warning(f"[Celery] CreateEmbedding network error for {user_id}: {e}")
        raise ImageFetchError(str(e))

    except Exception as e:
        # Unknown errors - wrap as retryable and let BaseTaskWithRetry decide
        logger.exception(f"[Celery] CreateEmbedding failed for {user_id}: {e}")
        _publish_failure_event(client_slug, command_id, user_id, str(e))
        raise RetryableError(str(e))


def _publish_failure_event(client_slug: str, command_id: str, user_id: Any, error: str) -> None:
    """Publish EmbeddingFailed event to Backend."""
    try:
        publisher = get_event_publisher(client_slug)
        publisher.publish_embedding_failed(
            command_id=command_id,
            user_id=user_id,
            error=error
        )
    except Exception as pub_error:
        logger.exception(f"[Celery] Failed to publish error event: {pub_error}")


@celery.task(
    bind=True,
    base=BaseTaskWithRetry,
    name='embedding.update_user',
    queue='embeddings',
    autoretry_for=(RetryableError, ImageFetchError, DatabaseError, ConnectionError),
    dont_autoretry_for=(ValidationError, NonRetryableError),
    retry_backoff=True,
    retry_backoff_max=300,
    retry_jitter=True,
    max_retries=5,
)
def process_update_user(self, command_id: str, client_slug: str, user_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process UpdateEmbedding command.

    Args:
        command_id: Original command ID from Backend
        client_slug: Organization slug
        user_data: User data with id, full_name, image_urls

    Returns:
        Result dict with status and embeddings_created count
    """
    task_id = self.request.id
    user_id = user_data.get('id')

    logger.info(f"[Celery] Processing UpdateEmbedding: id={user_id} task_id={task_id}")

    try:
        # Validate inputs
        validate_embedding_request(client_slug, user_data)

        # Serialize against a concurrent DeleteEmbedding for the same user
        # (see _user_embedding_lock) — otherwise a delete landing between
        # this delete-old/add-new pair can be silently undone.
        with _user_embedding_lock(client_slug, user_id) as locked:
            if not locked:
                raise RetryableError(
                    f"Could not acquire embedding lock for user {user_id}, retrying"
                )

            # Get embedding sync service
            service = get_embedding_sync_service(client_slug)

            # Delete old embeddings first
            if user_id:
                service.store.delete_all_for_user(str(user_id))

            # Process new embeddings
            result = service.handle_user_created(user_data)
            embeddings_created = result.get('embeddings_added', 0)

        logger.info(f"[Celery] UpdateEmbedding completed: {embeddings_created} embeddings task_id={task_id}")

        # Publish EmbeddingCreated event to Backend
        publisher = get_event_publisher(client_slug)
        publisher.publish_embedding_created(
            command_id=command_id,
            user_id=user_id,
            embeddings_created=embeddings_created
        )

        # Notify camera engine to reload embeddings
        notify_embedding_reload(client_slug, user_id, 'update')

        return {
            'status': 'success',
            'action': 'UpdateEmbedding',
            'user_id': user_id,
            'embeddings_created': embeddings_created,
            'task_id': task_id,
        }

    except ValidationError:
        logger.exception(f"[Celery] UpdateEmbedding validation failed for {user_id}")
        _publish_failure_event(client_slug, command_id, user_id, "Validation failed")
        raise

    except SoftTimeLimitExceeded:
        logger.exception(f"[Celery] UpdateEmbedding timed out for {user_id} task_id={task_id}")
        _publish_failure_event(client_slug, command_id, user_id, "Task timed out")
        raise RetryableError(f"Task timed out for user {user_id}")

    except Exception as e:
        logger.exception(f"[Celery] UpdateEmbedding failed for {user_id}: {e}")
        _publish_failure_event(client_slug, command_id, user_id, str(e))
        raise RetryableError(str(e))


@celery.task(
    bind=True,
    base=BaseTaskWithRetry,
    name='embedding.delete_user',
    queue='embeddings',
    autoretry_for=(RetryableError, DatabaseError, ConnectionError),
    dont_autoretry_for=(ValidationError, NonRetryableError),
    retry_backoff=True,
    retry_backoff_max=300,
    retry_jitter=True,
    max_retries=5,
)
def process_delete_user(self, command_id: str, client_slug: str, user_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process DeleteEmbedding command.

    Args:
        command_id: Original command ID from Backend
        client_slug: Organization slug
        user_data: User data with id

    Returns:
        Result dict with status
    """
    task_id = self.request.id
    user_id = user_data.get('id')

    logger.info(f"[Celery] Processing DeleteEmbedding: id={user_id} task_id={task_id}")

    try:
        # Validate inputs
        if not client_slug:
            raise ValidationError("client_slug is required")
        if not user_id:
            raise ValidationError("user_id is required")

        # Serialize against a concurrent UpdateEmbedding for the same user —
        # see _user_embedding_lock.
        with _user_embedding_lock(client_slug, user_id) as locked:
            if not locked:
                raise RetryableError(
                    f"Could not acquire embedding lock for user {user_id}, retrying"
                )

            # Delete only ever needs the pgvector store — go straight to it
            # instead of get_embedding_sync_service(), which would load the
            # (multi-second, GPU-resident) FaceDetector this task never uses.
            from infrastructure.storage import PgVectorStore
            store = PgVectorStore(client_slug)
            deleted_count = store.delete_all_for_user(str(user_id))

        logger.info(f"[Celery] DeleteEmbedding completed: {deleted_count} embeddings deleted task_id={task_id}")

        # Publish EmbeddingDeleted event to Backend
        publisher = get_event_publisher(client_slug)
        publisher.publish_embedding_created(
            command_id=command_id,
            user_id=user_id,
            embeddings_created=0
        )

        # Notify camera engine to reload embeddings
        notify_embedding_reload(client_slug, user_id, 'delete')

        return {
            'status': 'success',
            'action': 'DeleteEmbedding',
            'user_id': user_id,
            'deleted_count': deleted_count,
            'task_id': task_id,
        }

    except ValidationError:
        logger.exception(f"[Celery] DeleteEmbedding validation failed for {user_id}")
        _publish_failure_event(client_slug, command_id, user_id, "Validation failed")
        raise

    except SoftTimeLimitExceeded:
        logger.exception(f"[Celery] DeleteEmbedding timed out for {user_id} task_id={task_id}")
        _publish_failure_event(client_slug, command_id, user_id, "Task timed out")
        raise RetryableError(f"Task timed out for user {user_id}")

    except Exception as e:
        logger.exception(f"[Celery] DeleteEmbedding failed for {user_id}: {e}")
        _publish_failure_event(client_slug, command_id, user_id, str(e))
        raise RetryableError(str(e))
