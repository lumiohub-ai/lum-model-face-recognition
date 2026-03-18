"""
Storage Infrastructure

Database and cloud storage operations.
- PgVector: Face embeddings storage
- GCS: Image upload/download
- Repository: Direct database queries (replaces HTTP calls)
"""

from .pgvector import PgVectorStore
from .db_config import DatabaseConfig
from .gcs import ImageFetcher
from .embedding_sync import EmbeddingSyncService
from .url_utils import normalize_image_url
from .validators import validate_client_slug, validate_schema_name
from .repository import Repository
from .detection_repository import DetectionRepository

__all__ = [
    "PgVectorStore",
    "DatabaseConfig",
    "Repository",
    "DetectionRepository",
    "ImageFetcher",
    "EmbeddingSyncService",
    "normalize_image_url",
    "validate_client_slug",
    "validate_schema_name",
]
