"""Core face processing components: detection and recognition."""

from .detector import FaceDetector
from .recognizer import FaceRecognition as FaceRecognizer  # Alias for consistency
from .model_factory import ModelFactory
from .processing import FrameProcessor

__all__ = [
    "FaceDetector",
    "FaceRecognizer",
    "ModelFactory",
    "FrameProcessor",
]
