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

    # Camera-worker slot routing (LSO-186). yolo routes each camera's
    # `camera.track` to `cam-slot-{crc32(id) % N}` instead of a per-camera
    # `cam.<id>` queue, and each camera-worker replica leases exactly one slot
    # (see pipeline/slot_lease.py). Hardcoded, NOT an env var: decode, yolo and
    # camera-worker all run this one image, so a single constant keeps them in
    # lockstep with zero chance of a per-service env drifting out of agreement.
    # N MUST equal the camera-worker `replicas` in compose — a replica that
    # can't lease a slot exits loudly. Changing N remaps every camera to a
    # different slot (a deliberate rebalance + image rebuild; tracker state
    # re-inits), so it lives here in code, not in a hot config knob.
    camera_slot_count = 2

    # Google Cloud Storage
    gcs_credentials_path = os.getenv("SO_GCS_CREDENTIALS_PATH", "")
    gcs_bucket = os.getenv("SO_GCS_BUCKET", "hbai-general-data")

    # Application
    client_slug = os.getenv("SO_CLIENT_SLUG", "")
    log_level = os.getenv("SO_LOG_LEVEL", "INFO")
    config_path = os.getenv("SMART_OFFICE_CONFIG", "")
    hostname = os.getenv("HOSTNAME", "unknown")

    # Edge MediaMTX (LSO-27): the AI reads EVERY camera via the edge
    # (rtsp://<base>/<slug(camera_name)>, the high-res main path) — single pull
    # per camera, no credentials in the AI. REQUIRED: the AI never pulls cameras
    # directly; if this is empty the camera loader raises. Same-host deploys use
    # rtsp://host.docker.internal:8554; cross-host use the edge's Tailnet IP.
    edge_rtsp_base = os.getenv("SO_EDGE_RTSP_BASE", "").strip().rstrip("/")

    # Branch scoping (LSO-133): restrict this AI to one branch's cameras.
    # The AI resolves EVERY camera against its LOCAL edge (see edge_rtsp_base),
    # so an unscoped AI in a multi-site org loads other branches' cameras and
    # then tries to read them from an edge that has never heard of them.
    # Mirrors the edge sidecar's EDGE_BRANCH_CODE. Empty = every camera, which
    # is correct only for a single-site org.
    edge_branch_code = os.getenv("SO_EDGE_BRANCH_CODE", "").strip().lower()

    # Ollama (action recognition)
    ollama_api_url = os.getenv("SO_OLLAMA_API_URL", "http://localhost:11434")
    ollama_model = os.getenv("SO_OLLAMA_MODEL", "gemma3:4b")

    # Metrics monitoring
    metrics_enabled = os.getenv("SO_METRICS_ENABLED", "true").lower() not in ("false", "0", "no")
    metrics_port = int(os.getenv("SO_METRICS_PORT", 8765))


# Module-level singleton — created once at import time after .env is loaded
settings = Settings()
