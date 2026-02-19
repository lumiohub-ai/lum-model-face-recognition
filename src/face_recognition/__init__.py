# -*- coding: utf-8 -*-
"""Face Recognition System for SmartOffice.

This package provides a comprehensive face recognition system with:
- Real-time face detection and tracking
- Face recognition with embedding-based matching
- Multi-camera support
- API integration for attendance tracking
"""

from .__version__ import __version__
from .hbface import HBFace
from .smart_office_engine import SmartOfficeEngine

# Core components
from .core import FaceEngine, FaceDetector, FaceTracker, FaceRecognizer, TrackManager

# Configuration
from .config import ConfigurationManager, SystemConfig
from .config import constants

# Logging
from .logging import setup_structured_logging, EntryLogger, CSVLogger

# Storage
from .storage import Database, CloudStorageManager

# Video processing
from .video import StreamHandler, FrameProcessor

# API
from .api import APIClient, AuthenticationService

__all__ = [
    # Main entry points
    "HBFace",
    "SmartOfficeEngine",

    # Core
    "FaceEngine",
    "FaceDetector",
    "FaceTracker",
    "FaceRecognizer",
    "TrackManager",

    # Configuration
    "ConfigurationManager",
    "SystemConfig",
    "constants",

    # Logging
    "setup_structured_logging",
    "EntryLogger",
    "CSVLogger",

    # Storage
    "Database",
    "CloudStorageManager",

    # Video
    "StreamHandler",
    "FrameProcessor",

    # API
    "APIClient",
    "AuthenticationService",

    # Version
    "__version__",
]