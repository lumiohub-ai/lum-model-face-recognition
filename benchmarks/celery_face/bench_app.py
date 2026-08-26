"""Isolated Celery app for the face-detection benchmark.

Talks to a throwaway Redis on port 6402 (NOT 6379 — another developer's
stack; NOT 6400 — this repo's own dev stack; NOT 6401 — the YOLO benchmark's).
Nothing here may import ``workers.celery_app``, ``config.settings``, or
anything else under ``src/``: that would drag in the stale ``.env`` plus
sqlalchemy/psycopg2, none of which are installed in ``.venv``. See
benchmarks/celery_worker/README.md for the reasoning this mirrors.
"""

import os

from celery import Celery

BROKER = os.environ.get("FACEBENCH_BROKER", "redis://127.0.0.1:6402/0")
BACKEND = os.environ.get("FACEBENCH_BACKEND", "redis://127.0.0.1:6402/1")
QUEUE = os.environ.get("FACEBENCH_QUEUE", "facebench")
PREFETCH = int(os.environ.get("FACEBENCH_PREFETCH", "1"))
ACKS_LATE = os.environ.get("FACEBENCH_ACKS_LATE", "1") == "1"

# App name is 'facebench', never 'smart_office' — a distinct name keeps the
# pidbox control exchange separate from production's (and from yolobench's).
app = Celery(
    "facebench",
    broker=BROKER,
    backend=BACKEND,
    include=["celery_face.bench_tasks"],
)

app.conf.update(
    task_default_queue=QUEUE,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    worker_prefetch_multiplier=PREFETCH,
    task_acks_late=ACKS_LATE,
    broker_transport_options={"fanout_prefix": True, "fanout_patterns": True},
    result_expires=600,
    broker_connection_retry_on_startup=True,
)
