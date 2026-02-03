"""
Messaging Layer (MDA Core)

Message-Driven Architecture implementation:
- Commands: Backend → AI (Redis Streams) - handled by StreamConsumer
- Events: AI → Backend (Redis Pub/Sub) - handled by MDAPublisher
"""

from .redis_client import RedisClient, get_redis_client
from .publisher import MDAPublisher
from .subscriber import MDASubscriber  # Legacy - being replaced by StreamConsumer
from .stream_consumer import (
    StreamConsumer,
    get_stream_consumer,
    start_stream_consumer,
    stop_stream_consumer,
)

__all__ = [
    # Redis client
    "RedisClient",
    "get_redis_client",

    # Event publisher (AI → Backend)
    "MDAPublisher",

    # Stream consumer (Backend → AI commands)
    "StreamConsumer",
    "get_stream_consumer",
    "start_stream_consumer",
    "stop_stream_consumer",

    # Legacy (deprecated)
    "MDASubscriber",
]
