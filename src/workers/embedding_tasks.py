"""
Celery Tasks for Embedding Processing

These tasks are queued by MDA handlers and processed by Celery workers.
Results are published back to Redis via MDA publisher.
"""

import os
from typing import Dict, Any, Optional
from celery import shared_task
from loguru import logger


def get_embedding_sync_service(client_slug: str):
    """Lazy import to avoid circular imports and ensure proper initialization."""
    from infrastructure.storage import EmbeddingSyncService
    return EmbeddingSyncService(client_slug)


def get_mda_publisher(client_slug: str):
    """Get MDA publisher for sending results."""
    from messaging.publisher import MDAPublisher
    return MDAPublisher(client_slug)


@shared_task(bind=True, name='embedding.add_user', queue='embeddings')
def process_add_user(self, message_id: str, client_slug: str, user_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process add_user embedding request.

    Args:
        message_id: Original request message ID
        client_slug: Organization slug
        user_data: User data with id, full_name, image_urls

    Returns:
        Result dict with status and embeddings_created count
    """
    user_id = user_data.get('id')
    full_name = user_data.get('full_name', 'unknown')

    logger.info(f"[Celery] Processing add_user: {full_name} (id={user_id})")

    try:
        # Get embedding sync service
        service = get_embedding_sync_service(client_slug)

        # Process user embeddings
        result = service.handle_user_created(user_data)
        embeddings_created = result.get('embeddings_added', 0)

        logger.info(f"[Celery] add_user completed: {full_name} - {embeddings_created} embeddings")

        # Publish success result via MDA
        publisher = get_mda_publisher(client_slug)
        publisher.publish_embedding_result(
            request_id=message_id,
            status='success',
            action='add_user',
            user_id=user_id,
            embeddings_created=embeddings_created
        )

        return {
            'status': 'success',
            'action': 'add_user',
            'user_id': user_id,
            'embeddings_created': embeddings_created
        }

    except Exception as e:
        logger.error(f"[Celery] add_user failed for {full_name}: {e}")

        # Publish failure result via MDA
        try:
            publisher = get_mda_publisher(client_slug)
            publisher.publish_embedding_result(
                request_id=message_id,
                status='failure',
                action='add_user',
                user_id=user_id,
                error=str(e)
            )
        except Exception as pub_error:
            logger.error(f"[Celery] Failed to publish error: {pub_error}")

        # Re-raise for Celery retry logic
        raise


@shared_task(bind=True, name='embedding.update_user', queue='embeddings')
def process_update_user(self, message_id: str, client_slug: str, user_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process update_user embedding request.

    Args:
        message_id: Original request message ID
        client_slug: Organization slug
        user_data: User data with id, full_name, image_urls

    Returns:
        Result dict with status and embeddings_created count
    """
    user_id = user_data.get('id')
    full_name = user_data.get('full_name', 'unknown')

    logger.info(f"[Celery] Processing update_user: {full_name} (id={user_id})")

    try:
        # Get embedding sync service
        service = get_embedding_sync_service(client_slug)

        # Delete old embeddings first
        if user_id:
            service.store.delete_all_for_user(str(user_id))

        # Process new embeddings
        result = service.handle_user_created(user_data)
        embeddings_created = result.get('embeddings_added', 0)

        logger.info(f"[Celery] update_user completed: {full_name} - {embeddings_created} embeddings")

        # Publish success result via MDA
        publisher = get_mda_publisher(client_slug)
        publisher.publish_embedding_result(
            request_id=message_id,
            status='success',
            action='update_user',
            user_id=user_id,
            embeddings_created=embeddings_created
        )

        return {
            'status': 'success',
            'action': 'update_user',
            'user_id': user_id,
            'embeddings_created': embeddings_created
        }

    except Exception as e:
        logger.error(f"[Celery] update_user failed for {full_name}: {e}")

        # Publish failure result via MDA
        try:
            publisher = get_mda_publisher(client_slug)
            publisher.publish_embedding_result(
                request_id=message_id,
                status='failure',
                action='update_user',
                user_id=user_id,
                error=str(e)
            )
        except Exception as pub_error:
            logger.error(f"[Celery] Failed to publish error: {pub_error}")

        raise


@shared_task(bind=True, name='embedding.delete_user', queue='embeddings')
def process_delete_user(self, message_id: str, client_slug: str, user_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process delete_user embedding request.

    Args:
        message_id: Original request message ID
        client_slug: Organization slug
        user_data: User data with id

    Returns:
        Result dict with status
    """
    user_id = user_data.get('id')
    full_name = user_data.get('full_name', 'unknown')

    logger.info(f"[Celery] Processing delete_user: {full_name} (id={user_id})")

    try:
        # Get embedding sync service
        service = get_embedding_sync_service(client_slug)

        # Delete all embeddings for user
        deleted_count = 0
        if user_id:
            deleted_count = service.store.delete_all_for_user(str(user_id))

        logger.info(f"[Celery] delete_user completed: {full_name} - {deleted_count} embeddings deleted")

        # Publish success result via MDA
        publisher = get_mda_publisher(client_slug)
        publisher.publish_embedding_result(
            request_id=message_id,
            status='success',
            action='delete_user',
            user_id=user_id,
            embeddings_created=0
        )

        return {
            'status': 'success',
            'action': 'delete_user',
            'user_id': user_id,
            'deleted_count': deleted_count
        }

    except Exception as e:
        logger.error(f"[Celery] delete_user failed for {full_name}: {e}")

        # Publish failure result via MDA
        try:
            publisher = get_mda_publisher(client_slug)
            publisher.publish_embedding_result(
                request_id=message_id,
                status='failure',
                action='delete_user',
                user_id=user_id,
                error=str(e)
            )
        except Exception as pub_error:
            logger.error(f"[Celery] Failed to publish error: {pub_error}")

        raise
