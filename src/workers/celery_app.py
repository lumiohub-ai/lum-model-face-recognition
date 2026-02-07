"""
Celery Application Configuration

Celery is used for async task processing in True MDA architecture.
Tasks are queued from MDA handlers and processed by workers.
"""

import os
from celery import Celery

from messaging.redis_config import REDIS_HOST, REDIS_PORT

CELERY_BROKER_URL = os.getenv('CELERY_BROKER_URL', f'redis://{REDIS_HOST}:{REDIS_PORT}/0')
CELERY_RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND', f'redis://{REDIS_HOST}:{REDIS_PORT}/1')

# Create Celery app
celery = Celery(
    'smart_office',
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=[
        'workers.embedding_tasks',
        'workers.detection_tasks',
    ]
)

# Celery configuration
celery.conf.update(
    # Task settings
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,

    # Task routing
    task_routes={
        'workers.embedding_tasks.*': {'queue': 'embeddings'},
        'workers.detection_tasks.*': {'queue': 'detections'},
        'detection.*': {'queue': 'detections'},
    },

    # Task execution settings
    task_acks_late=True,
    task_reject_on_worker_lost=True,

    # Worker settings
    worker_prefetch_multiplier=1,
    worker_concurrency=2,

    # Result settings
    result_expires=3600,  # 1 hour

    # Retry settings
    task_default_retry_delay=5,
    task_max_retries=3,
)

print(f"[Celery] Configured with broker: {CELERY_BROKER_URL}")
