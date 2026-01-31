"""
MDA (Message-Driven Architecture) Package

Pure MDA implementation for AI Service communication with Backend.
Replaces HTTP calls with Redis Pub/Sub messaging.
"""

from face_recognition.mda.redis_client import RedisClient, get_redis_client
from face_recognition.mda.publisher import MDAPublisher
from face_recognition.mda.subscriber import MDASubscriber
from face_recognition.mda.handlers import EmbeddingRequestHandler

__all__ = [
    'RedisClient',
    'get_redis_client',
    'MDAPublisher',
    'MDASubscriber',
    'EmbeddingRequestHandler',
]
