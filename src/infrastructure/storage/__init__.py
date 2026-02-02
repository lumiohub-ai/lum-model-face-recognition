"""
Storage Infrastructure

Database and cloud storage operations.
- PgVector: Face embeddings storage
- GCS: Image upload/download
"""

from .pgvector import PgVectorStore
from .db_config import DatabaseConfig
from .gcs import GCSClient, upload_proof_image, get_gcs_client
from .embedding_sync import EmbeddingSyncService
from .url_utils import normalize_image_url
from .validators import validate_client_slug, validate_schema_name

__all__ = [
    "PgVectorStore",
    "DatabaseConfig",
    "GCSClient",
    "upload_proof_image",
    "get_gcs_client",
    "EmbeddingSyncService",
    "normalize_image_url",
    "validate_client_slug",
    "validate_schema_name",
]
