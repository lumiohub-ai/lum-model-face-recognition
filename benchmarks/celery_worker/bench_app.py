"""Isolated Celery app for the YOLO worker spike.

Talks to a throwaway Redis on port 6401 (NOT 6379 — that belongs to another
developer's stack, and NOT 6400 — that is this repo's own dev stack). Nothing
here may import ``workers.celery_app``, ``config.settings``, or anything else
under ``src/``: that would drag in the stale ``.env`` plus sqlalchemy/psycopg2,
none of which are installed in ``.venv``.
"""

import os

from celery import Celery

BROKER = os.environ.get("YOLOBENCH_BROKER", "redis://127.0.0.1:6401/0")
BACKEND = os.environ.get("YOLOBENCH_BACKEND", "redis://127.0.0.1:6401/1")
QUEUE = os.environ.get("YOLOBENCH_QUEUE", "yolobench")
# Delivery semantics are knobs, not constants: the threads pool interacts badly
# with the production defaults (acks_late + prefetch 1). See README.
PREFETCH = int(os.environ.get("YOLOBENCH_PREFETCH", "1"))
ACKS_LATE = os.environ.get("YOLOBENCH_ACKS_LATE", "1") == "1"

# App name is 'yolobench', never 'smart_office' — a distinct name keeps the
# pidbox control exchange separate from production's.
app = Celery(
    "yolobench",
    broker=BROKER,
    backend=BACKEND,
    include=["celery_worker.bench_tasks"],
)

app.conf.update(
    task_default_queue=QUEUE,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    # Match production's delivery semantics so the numbers transfer.
    worker_prefetch_multiplier=PREFETCH,
    task_acks_late=ACKS_LATE,
    # Redis pub/sub is NOT database-scoped; prefixing keeps fanout channels ours.
    broker_transport_options={"fanout_prefix": True, "fanout_patterns": True},
    result_expires=600,
    broker_connection_retry_on_startup=True,
    # No task_soft_time_limit: the threads pool silently ignores it, which would
    # make one pool behave differently from the others.
)
