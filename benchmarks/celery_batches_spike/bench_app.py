"""Isolated Celery app for the celery-batches go/no-go spike (LSO-67 follow-up).

Talks to a throwaway Redis on port 6402 (NOT 6379 — another developer's stack;
NOT 6400 — this repo's own dev stack; NOT 6401 — benchmarks/celery_worker's
own throwaway Redis, kept distinct so both spikes can run side by side).
Nothing here imports ``workers.celery_app``, ``config.settings``, or anything
under ``src/`` — see benchmarks/celery_worker/README.md for why that drags in
a stale ``.env`` plus sqlalchemy/psycopg2, neither installed in ``.venv``.
"""

import os

from celery import Celery

BROKER = os.environ.get("BATCHBENCH_BROKER", "redis://127.0.0.1:6402/0")
BACKEND = os.environ.get("BATCHBENCH_BACKEND", "redis://127.0.0.1:6402/1")
QUEUE = os.environ.get("BATCHBENCH_QUEUE", "batchbench")

# App name is 'batchbench', never 'smart_office' — keeps the pidbox control
# exchange separate from production's and from yolobench's.
app = Celery(
    "batchbench",
    broker=BROKER,
    backend=BACKEND,
    include=["celery_batches_spike.bench_tasks"],
)

app.conf.update(
    task_default_queue=QUEUE,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    # The plan's real design runs task_acks_late=True everywhere (celery_app.py)
    # and worker_prefetch_multiplier=1 globally, overridden to 32 on the
    # yolo-worker CLI specifically for the Batches queue. Match that shape here.
    task_acks_late=True,
    worker_prefetch_multiplier=32,
    broker_transport_options={"fanout_prefix": True, "fanout_patterns": True},
    result_expires=600,
    broker_connection_retry_on_startup=True,
)
