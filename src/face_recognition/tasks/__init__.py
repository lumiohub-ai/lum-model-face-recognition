"""Celery tasks for async processing."""

from .embedding_tasks import (
    process_add_user,
    process_update_user,
    process_delete_user,
)

__all__ = [
    'process_add_user',
    'process_update_user',
    'process_delete_user',
]
