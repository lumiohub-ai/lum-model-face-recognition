"""Data storage components for face embeddings and cloud storage."""

from .database import Database
from .cloud_storage import CloudStorageManager

__all__ = [
    "Database",
    "CloudStorageManager",
]
