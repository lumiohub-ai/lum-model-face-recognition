"""
Infrastructure Layer

External services and I/O operations.
- Storage: Database (Repository, PgVector) and cloud storage (GCS)
- Video: Camera streams and annotation
"""

from .storage import PgVectorStore, ImageFetcher, EmbeddingSyncService, Repository
from .video import StreamHandler, StreamManager, FrameAnnotator
from .entry_logger import EntryLogger
from .async_logger import AsyncLogger

__all__ = [
    # Storage
    "PgVectorStore",
    "ImageFetcher",
    "EmbeddingSyncService",
    "Repository",
    # Video
    "StreamHandler",
    "StreamManager",
    "FrameAnnotator",
    # Services
    "EntryLogger",
    "AsyncLogger",
]
