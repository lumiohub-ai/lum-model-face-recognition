"""Person detection, tracking, and cross-camera re-identification."""

from .detector import PersonDetector
from .face_adapter import crop_person_roi
from .global_track import GlobalTrackManager
from .global_track_model import GlobalTrack
from .id_corrector import IDSwitchCorrector
from .identity import IdentityManager
from .ids import GlobalTrackIDGenerator
from .state import PersonStateManager
from .track import PersonTrackManager
from .tracker import PersonTracker

__all__ = [
    "PersonDetector",
    "PersonTracker",
    "PersonTrackManager",
    "PersonStateManager",
    "IdentityManager",
    "IDSwitchCorrector",
    "GlobalTrackManager",
    "GlobalTrack",
    "GlobalTrackIDGenerator",
    "crop_person_roi",
]
