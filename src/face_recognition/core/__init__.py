"""Core face processing components: detection, tracking, and recognition."""

from .detector import FaceDetector
from .tracker import FaceTracker
from .recognizer import FaceRecognition as FaceRecognizer  # Alias for consistency
from .track_manager import TrackManager
from .engine import FaceEngine

__all__ = [
    "FaceDetector",
    "FaceTracker",
    "FaceRecognizer",
    "TrackManager",
    "FaceEngine",
]
