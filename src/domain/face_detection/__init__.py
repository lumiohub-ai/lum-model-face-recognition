"""
Face Detection Domain

Core face detection and recognition logic.
"""

from .detector import FaceDetector
from .recognizer import FaceRecognition as FaceRecognizer
from .model_factory import ModelFactory

__all__ = [
    "FaceDetector",
    "FaceRecognizer",
    "ModelFactory",
]
