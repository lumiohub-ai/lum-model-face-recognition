"""
Celery Workers

Async task processing for heavy operations like embedding generation.
"""

from .celery_app import celery
from .embedding_tasks import (
    process_add_user,
    process_update_user,
    process_delete_user,
)

__all__ = [
    "celery",
    "process_add_user",
    "process_update_user",
    "process_delete_user",
]
