"""
Person Tracking Module

This package provides person detection, tracking, and face recognition integration
for the Smart Office project.
"""

__version__ = "0.1.0"

# Core components
from .core.person_detector import PersonDetector
from .core.person_tracker import PersonTracker
from .core.track_manager import PersonTrackManager
from .core.identity_manager import IdentityManager
from .core.state_manager import PersonStateManager
from .core.global_track_manager import GlobalTrackManager

# Video processing
from .video.frame_annotator import FrameAnnotator

__all__ = [
    # Core
    "PersonDetector",
    "PersonTracker",
    "PersonTrackManager",
    "IdentityManager",
    "PersonStateManager",
    "GlobalTrackManager",

    # Video
    "FrameAnnotator",
]
