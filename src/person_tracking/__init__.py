"""
Person Tracking & Phone Usage Detection System

This package provides person detection, tracking, face recognition integration,
and phone usage detection capabilities for the Smart Office project.
"""

__version__ = "0.1.0"
__author__ = "Smart Office Team"

# Main engine
from .engine import PersonTrackingEngine

# Configuration
from .config.manager import ConfigurationManager
from .config.models import PersonTrackingAppConfig, CameraConfig

# Core components
from .core.person_detector import PersonDetector
from .core.person_tracker import PersonTracker
from .core.track_manager import PersonTrackManager
from .core.identity_manager import IdentityManager
from .core.state_manager import PersonStateManager

# Video processing
from .video.frame_annotator import FrameAnnotator

# Logging
from .logging.csv_logger import CSVLogger

__all__ = [
    # Main
    "PersonTrackingEngine",

    # Config
    "ConfigurationManager",
    "PersonTrackingAppConfig",
    "CameraConfig",

    # Core
    "PersonDetector",
    "PersonTracker",
    "PersonTrackManager",
    "IdentityManager",
    "PersonStateManager",

    # Video
    "FrameAnnotator",

    # Logging
    "CSVLogger",
]
