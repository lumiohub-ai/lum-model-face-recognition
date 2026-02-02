"""
API Infrastructure

Backend API communication using HTTP for read operations.
Write operations use MDA (messaging layer).
"""

from .client import APIClient
from .auth import AuthenticationService

__all__ = [
    "APIClient",
    "AuthenticationService",
]
