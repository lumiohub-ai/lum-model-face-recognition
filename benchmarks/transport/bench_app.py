"""Isolated Celery app for the frame-transport comparison.

Talks to a throwaway Redis on port 6401 (NOT 6379, NOT 6380 — 6380 is this
repo's live dev stack). Nothing here may import ``workers.celery_app``,
``config.settings``, or anything else under ``src/``: that would drag in the
stale ``.env`` plus sqlalchemy/psycopg2, none of which are installed in
``.venv``. Same reasoning as celery_worker/bench_app.py.

Differs from that app in one way that matters: ``accept_content`` includes
``pickle``. JSON cannot carry raw bytes, so the pickle/jpeg cells would be
impossible otherwise. The *default* serializer stays ``json`` — production's
setting (src/workers/celery_app.py:75) — and each cell passes an explicit
``serializer=`` to send_task, so no cell silently changes another's transport.
"""

import os

from celery import Celery

BROKER = os.environ.get("XPORTBENCH_BROKER", "redis://127.0.0.1:6401/0")
BACKEND = os.environ.get("XPORTBENCH_BACKEND", "redis://127.0.0.1:6401/1")
QUEUE = os.environ.get("XPORTBENCH_QUEUE", "xportbench")

# App name is 'xportbench', never 'smart_office' — a distinct name keeps the
# pidbox control exchange separate from production's.
app = Celery(
    "xportbench",
    broker=BROKER,
    backend=BACKEND,
    include=["transport.bench_tasks"],
)

app.conf.update(
    task_default_queue=QUEUE,
    task_serializer="json",
    result_serializer="json",
    # pickle is accepted so cells 4/5 can put raw bytes on the wire. Results
    # stay json: a task returns a small dict, never pixels.
    accept_content=["json", "pickle"],
    timezone="UTC",
    enable_utc=True,
    # prefetch=1 with in-flight depth 1 (the driver submits serially), so a
    # task is never sitting in a worker-side buffer while its clock runs.
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    # Redis pub/sub is NOT database-scoped; prefixing keeps fanout channels ours.
    broker_transport_options={"fanout_prefix": True, "fanout_patterns": True},
    result_expires=600,
    broker_connection_retry_on_startup=True,
)
