"""
Person Tracking Domain

Multi-person tracking with stable ID assignment and re-identification.
"""

from .tracker import PersonTracker
from .detector import PersonDetector
from .identity import IdentityManager
from .state import PersonStateManager
from .track import PersonTrackManager
from .global_track import GlobalTrackManager
from .global_track_model import GlobalTrack
from .id_corrector import IDSwitchCorrector
from .face_adapter import crop_person_roi

__all__ = [
    "PersonTracker",
    "PersonDetector",
    "IdentityManager",
    "PersonStateManager",
    "PersonTrackManager",
    "GlobalTrackManager",
    "GlobalTrack",
    "IDSwitchCorrector",
    "crop_person_roi",
]
