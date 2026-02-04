"""API communication package for SmartOffice backend integration."""

from .client import APIClient
from .auth import AuthenticationService

__all__ = [
    "APIClient",
    "AuthenticationService",
]
