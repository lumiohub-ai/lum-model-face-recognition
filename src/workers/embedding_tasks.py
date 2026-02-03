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
"""

import os
from typing import Dict, Any, Optional
from celery import shared_task
from loguru import logger


def get_embedding_sync_service(client_slug: str):
    """Lazy import to avoid circular imports and ensure proper initialization."""
    from infrastructure.storage import EmbeddingSyncService
    return EmbeddingSyncService(client_slug)


def get_event_publisher(client_slug: str):
    """Get event publisher for sending results to Backend."""
    from messaging.publisher import MDAPublisher
    return MDAPublisher(client_slug)


def notify_embedding_reload(client_slug: str, user_id: int, action: str):
    """Notify camera engine to reload embeddings via Redis Pub/Sub."""
    try:
        from messaging.redis_client import get_redis_client
        from messaging.channels import INTERNAL_CHANNELS
        import json

        redis_client = get_redis_client()
        message = {
            'client_slug': client_slug,
            'user_id': user_id,
            'action': action,
        }
        redis_client.client.publish(
            INTERNAL_CHANNELS['EMBEDDING_RELOAD'],
            json.dumps(message)
        )
        logger.info(f"[Celery] Notified camera engine to reload embeddings for {client_slug}")
    except Exception as e:
        logger.warning(f"[Celery] Failed to notify embedding reload: {e}")


@shared_task(bind=True, name='embedding.add_user', queue='embeddings', max_retries=3)
def process_add_user(self, command_id: str, client_slug: str, user_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process CreateEmbedding command.

    Args:
        command_id: Original command ID from Backend
        client_slug: Organization slug
        user_data: User data with id, full_name, image_urls

    Returns:
        Result dict with status and embeddings_created count
    """
    user_id = user_data.get('id')
    full_name = user_data.get('full_name', 'unknown')

    logger.info(f"[Celery] Processing CreateEmbedding: {full_name} (id={user_id})")

    try:
        # Get embedding sync service
        service = get_embedding_sync_service(client_slug)

        # Process user embeddings
        result = service.handle_user_created(user_data)
        embeddings_created = result.get('embeddings_added', 0)

        logger.info(f"[Celery] CreateEmbedding completed: {full_name} - {embeddings_created} embeddings")

        # Publish EmbeddingCreated event to Backend
        publisher = get_event_publisher(client_slug)
        publisher.publish_embedding_created(
            command_id=command_id,
            user_id=user_id,
            embeddings_created=embeddings_created
        )

        # Notify camera engine to reload embeddings
        notify_embedding_reload(client_slug, user_id, 'add')

        return {
            'status': 'success',
            'action': 'CreateEmbedding',
            'user_id': user_id,
            'embeddings_created': embeddings_created
        }

    except Exception as e:
        logger.error(f"[Celery] CreateEmbedding failed for {full_name}: {e}")

        # Publish EmbeddingFailed event to Backend
        try:
            publisher = get_event_publisher(client_slug)
            publisher.publish_embedding_failed(
                command_id=command_id,
                user_id=user_id,
                error=str(e)
            )
        except Exception as pub_error:
            logger.error(f"[Celery] Failed to publish error event: {pub_error}")

        # Re-raise for Celery retry logic
        raise self.retry(exc=e, countdown=5)


@shared_task(bind=True, name='embedding.update_user', queue='embeddings', max_retries=3)
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
    user_id = user_data.get('id')
    full_name = user_data.get('full_name', 'unknown')

    logger.info(f"[Celery] Processing UpdateEmbedding: {full_name} (id={user_id})")

    try:
        # Get embedding sync service
        service = get_embedding_sync_service(client_slug)

        # Delete old embeddings first
        if user_id:
            service.store.delete_all_for_user(str(user_id))

        # Process new embeddings
        result = service.handle_user_created(user_data)
        embeddings_created = result.get('embeddings_added', 0)

        logger.info(f"[Celery] UpdateEmbedding completed: {full_name} - {embeddings_created} embeddings")

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
            'embeddings_created': embeddings_created
        }

    except Exception as e:
        logger.error(f"[Celery] UpdateEmbedding failed for {full_name}: {e}")

        # Publish EmbeddingFailed event to Backend
        try:
            publisher = get_event_publisher(client_slug)
            publisher.publish_embedding_failed(
                command_id=command_id,
                user_id=user_id,
                error=str(e)
            )
        except Exception as pub_error:
            logger.error(f"[Celery] Failed to publish error event: {pub_error}")

        raise self.retry(exc=e, countdown=5)


@shared_task(bind=True, name='embedding.delete_user', queue='embeddings', max_retries=3)
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
    user_id = user_data.get('id')
    full_name = user_data.get('full_name', 'unknown')

    logger.info(f"[Celery] Processing DeleteEmbedding: {full_name} (id={user_id})")

    try:
        # Get embedding sync service
        service = get_embedding_sync_service(client_slug)

        # Delete all embeddings for user
        deleted_count = 0
        if user_id:
            deleted_count = service.store.delete_all_for_user(str(user_id))

        logger.info(f"[Celery] DeleteEmbedding completed: {full_name} - {deleted_count} embeddings deleted")

        # Publish EmbeddingCreated event (with 0 embeddings) to Backend
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
            'deleted_count': deleted_count
        }

    except Exception as e:
        logger.error(f"[Celery] DeleteEmbedding failed for {full_name}: {e}")

        # Publish EmbeddingFailed event to Backend
        try:
            publisher = get_event_publisher(client_slug)
            publisher.publish_embedding_failed(
                command_id=command_id,
                user_id=user_id,
                error=str(e)
            )
        except Exception as pub_error:
            logger.error(f"[Celery] Failed to publish error event: {pub_error}")

        raise self.retry(exc=e, countdown=5)
