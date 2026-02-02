"""
Domain Layer

Core business logic for Smart Office AI features.
Contains pure business logic with no external I/O dependencies.
"""

from .face_detection import FaceDetector, FaceRecognizer, ModelFactory, FrameProcessor
from .person_tracking import (
    PersonTracker,
    PersonDetector,
    IdentityManager,
    PersonStateManager,
    PersonTrackManager,
    GlobalTrackManager,
    GlobalTrack,
    IDSwitchCorrector,
    crop_person_roi,
)
from .action_recognition import ActionRecognizer

__all__ = [
    # Face Detection
    "FaceDetector",
    "FaceRecognizer",
    "ModelFactory",
    "FrameProcessor",
    # Person Tracking
    "PersonTracker",
    "PersonDetector",
    "IdentityManager",
    "PersonStateManager",
    "PersonTrackManager",
    "GlobalTrackManager",
    "GlobalTrack",
    "IDSwitchCorrector",
    "crop_person_roi",
    # Action Recognition
    "ActionRecognizer",
]
