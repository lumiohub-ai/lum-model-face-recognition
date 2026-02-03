"""
Celery Workers

Async task processing for:
- Embedding generation (Backend → AI commands)
- Detection recording (Camera engine → persistent storage)

Architecture:
- Celery = Tasks (reliable, retryable, trackable, durable)
- Pub/Sub = Signals (ephemeral UI notifications)
"""

from .celery_app import celery
from .embedding_tasks import (
    process_add_user,
    process_update_user,
    process_delete_user,
)
from .detection_tasks import (
    task_record_attendance,
    task_save_unrecognized_face,
    task_record_activity,
    task_update_user_location,
)

__all__ = [
    "celery",
    # Embedding tasks
    "process_add_user",
    "process_update_user",
    "process_delete_user",
    # Detection tasks
    "task_record_attendance",
    "task_save_unrecognized_face",
    "task_record_activity",
    "task_update_user_location",
]
