"""
Message Handlers

Process incoming messages from Backend via Redis Pub/Sub.
"""

from .embedding import EmbeddingRequestHandler

__all__ = [
    "EmbeddingRequestHandler",
]
