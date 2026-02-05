"""
Infrastructure Layer

External services and I/O operations.
- Storage: Database (Repository, PgVector) and cloud storage (GCS)
- Video: Camera streams and annotation
"""

from .storage import PgVectorStore, GCSClient, EmbeddingSyncService, Repository
from .video import StreamHandler, StreamManager, FrameAnnotator
from .lifecycle import EngineLifecycle
from .entry_logger import EntryLogger
from .csv_logger import CSVLogger

__all__ = [
    # Storage
    "PgVectorStore",
    "GCSClient",
    "EmbeddingSyncService",
    "Repository",
    # Video
    "StreamHandler",
    "StreamManager",
    "FrameAnnotator",
    # Services
    "EngineLifecycle",
    "EntryLogger",
    "CSVLogger",
]
