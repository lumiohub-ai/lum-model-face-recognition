"""Centralized application settings.

Single source of truth for all environment variables.
Loads .env file automatically on import.

Usage:
    from config.settings import settings

"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    # Try project root (.env sits next to src/)
    for _candidate in [Path(__file__).parents[2] / ".env", Path(".env")]:
        if _candidate.exists():
            load_dotenv(_candidate, override=False)
            break

except ImportError:
    pass


class Settings:
    # PostgreSQL
    postgres_host = os.getenv("SO_POSTGRES_HOST", "localhost")
    postgres_port = int(os.getenv("SO_POSTGRES_PORT", 5433))
    postgres_user = os.getenv("SO_POSTGRES_USER", "face_recognition")
    postgres_password = os.getenv("SO_POSTGRES_PASSWORD")
    postgres_db = os.getenv("SO_POSTGRES_DB", "face_embeddings")

    # Redis
    redis_host = os.getenv("SO_REDIS_HOST", "localhost")
    redis_port = int(os.getenv("SO_REDIS_PORT", 6379))
    redis_db = int(os.getenv("SO_REDIS_DB", 0))
    redis_url = os.getenv("SO_REDIS_URL", f"redis://{redis_host}:{redis_port}")

    # Celery
    celery_broker_url = os.getenv("SO_CELERY_BROKER_URL", f"redis://{redis_host}:{redis_port}/0")
    celery_result_backend = os.getenv("SO_CELERY_RESULT_BACKEND", f"redis://{redis_host}:{redis_port}/1")
    celery_concurrency = int(os.getenv("SO_CELERY_CONCURRENCY", 2))

    # Google Cloud Storage
    gcs_credentials_path = os.getenv("SO_GCS_CREDENTIALS_PATH", "")
    gcs_bucket = os.getenv("SO_GCS_BUCKET", "hbai-general-data")

    # Application
    client_slug = os.getenv("SO_CLIENT_SLUG", "")
    log_level = os.getenv("SO_LOG_LEVEL", "INFO")
    config_path = os.getenv("SMART_OFFICE_CONFIG", "")
    hostname = os.getenv("HOSTNAME", "unknown")

    # Edge MediaMTX (LSO-27): when set, the AI reads every camera via the edge
    # (rtsp://<base>/<slug(camera_name)>) instead of the DB stream_url — single
    # pull per camera, no credentials in the AI. Empty = use the DB stream_url.
    edge_rtsp_base = os.getenv("SO_EDGE_RTSP_BASE", "").strip().rstrip("/")

    # Ollama (action recognition)
    ollama_api_url = os.getenv("SO_OLLAMA_API_URL", "http://localhost:11434")
    ollama_model = os.getenv("SO_OLLAMA_MODEL", "gemma3:4b")

    # Metrics monitoring
    metrics_enabled = os.getenv("SO_METRICS_ENABLED", "true").lower() not in ("false", "0", "no")
    metrics_port = int(os.getenv("SO_METRICS_PORT", 8765))


# Module-level singleton — created once at import time after .env is loaded
settings = Settings()
