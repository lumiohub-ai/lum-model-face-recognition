"""
Infrastructure Layer

External services and I/O operations.
- API: Backend communication
- Storage: Database and cloud storage
- Video: Camera streams and annotation
"""

from .api import APIClient, AuthenticationService
from .storage import PgVectorStore, GCSClient, EmbeddingSyncService
from .video import StreamHandler, StreamManager, FrameAnnotator
from .lifecycle import EngineLifecycle
from .entry_logger import EntryLogger
from .csv_logger import CSVLogger

__all__ = [
    # API
    "APIClient",
    "AuthenticationService",
    # Storage
    "PgVectorStore",
    "GCSClient",
    "EmbeddingSyncService",
    # Video
    "StreamHandler",
    "StreamManager",
    "FrameAnnotator",
    # Services
    "EngineLifecycle",
    "EntryLogger",
    "CSVLogger",
]
