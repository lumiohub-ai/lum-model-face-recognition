"""
MDA Message Handlers

Handlers for processing messages from Backend.
Uses Celery for async task processing (True MDA architecture).
"""

import logging
from typing import Dict, Any

from face_recognition.celery_app import celery
from face_recognition.tasks.embedding_tasks import (
    process_add_user,
    process_update_user,
    process_delete_user,
)

logger = logging.getLogger(__name__)

# Print Celery broker info on import
print(f"[Handler] Celery broker: {celery.conf.broker_url}", flush=True)


class EmbeddingRequestHandler:
    """
    Handler for embedding requests from Backend.

    Queues Celery tasks for add_user, update_user, and delete_user actions.
    Results are published back via MDA by the Celery tasks.
    """

    def __init__(self, embedding_sync_service=None, publisher=None):
        """
        Initialize the handler.

        Args:
            embedding_sync_service: Unused (kept for API compatibility)
            publisher: Unused (Celery tasks handle publishing)
        """
        # These are no longer used - Celery tasks handle everything
        pass

    def handle(self, message: Dict[str, Any]):
        """
        Handle an embedding request message by queueing a Celery task.

        Args:
            message: Dict containing action, client_slug, user_data
        """
        message_id = message.get('message_id', 'unknown')
        action = message.get('action')
        client_slug = message.get('client_slug')
        user_data = message.get('user_data', {})

        full_name = user_data.get('full_name', 'unknown')
        logger.info(f"[MDA] Received embedding request: {action} for {full_name}")

        try:
            if action == 'add_user':
                self._queue_add_user(message_id, client_slug, user_data)
            elif action == 'update_user':
                self._queue_update_user(message_id, client_slug, user_data)
            elif action == 'delete_user':
                self._queue_delete_user(message_id, client_slug, user_data)
            else:
                logger.error(f"[MDA] Unknown action: {action}")
                return

            logger.info(f"[MDA] Queued {action} task for {full_name}")

        except Exception as e:
            logger.error(f"[MDA] Failed to queue {action} task: {e}")

    def _queue_add_user(self, message_id: str, client_slug: str, user_data: Dict[str, Any]):
        """Queue add_user task to Celery."""
        task_id = f"add-{user_data.get('id', 'unknown')}-{message_id[:8]}"

        print(f"[Handler] Submitting add_user task {task_id} to Celery (queue=embeddings)...", flush=True)

        result = process_add_user.apply_async(
            args=[message_id, client_slug, user_data],
            task_id=task_id,
            queue='embeddings'
        )

        print(f"[Handler] Celery returned: task_id={result.id}, state={result.state}", flush=True)

    def _queue_update_user(self, message_id: str, client_slug: str, user_data: Dict[str, Any]):
        """Queue update_user task to Celery."""
        task_id = f"update-{user_data.get('id', 'unknown')}-{message_id[:8]}"

        print(f"[Handler] Submitting update_user task {task_id} to Celery (queue=embeddings)...", flush=True)

        result = process_update_user.apply_async(
            args=[message_id, client_slug, user_data],
            task_id=task_id,
            queue='embeddings'
        )

        print(f"[Handler] Celery returned: task_id={result.id}, state={result.state}", flush=True)

    def _queue_delete_user(self, message_id: str, client_slug: str, user_data: Dict[str, Any]):
        """Queue delete_user task to Celery."""
        task_id = f"delete-{user_data.get('id', 'unknown')}-{message_id[:8]}"

        print(f"[Handler] Submitting delete_user task {task_id} to Celery (queue=embeddings)...", flush=True)

        result = process_delete_user.apply_async(
            args=[message_id, client_slug, user_data],
            task_id=task_id,
            queue='embeddings'
        )

        print(f"[Handler] Celery returned: task_id={result.id}, state={result.state}", flush=True)
