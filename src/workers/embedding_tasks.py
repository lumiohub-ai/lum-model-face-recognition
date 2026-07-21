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

from typing import Dict, Any, Optional
from workers.celery_app import celery
from celery.exceptions import SoftTimeLimitExceeded
from loguru import logger

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


def get_embedding_sync_service(client_slug: str):
    """Lazy import to avoid circular imports and ensure proper initialization."""
    from infrastructure.storage import EmbeddingSyncService
    return EmbeddingSyncService(client_slug)


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
    """Notify camera engine to reload embeddings via Redis Pub/Sub."""
    try:
        from messaging.redis_client import RedisClient
        from messaging.channels import INTERNAL_CHANNELS
        import json

        message = {
            'client_slug': client_slug,
            'user_id': user_id,
            'action': action,
        }
        RedisClient.get_instance().client.publish(
            INTERNAL_CHANNELS['EMBEDDING_RELOAD'],
            json.dumps(message)
        )
        logger.info(f"[Celery] Notified camera engine to reload embeddings for {client_slug}")
    except Exception as e:
        logger.warning(f"[Celery] Failed to notify embedding reload: {e}")


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

        # Get embedding sync service
        service = get_embedding_sync_service(client_slug)

        # Delete all embeddings for user
        deleted_count = service.store.delete_all_for_user(str(user_id))

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
