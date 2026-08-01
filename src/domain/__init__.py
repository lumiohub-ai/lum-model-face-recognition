"""
Domain Layer

Core business logic for Smart Office AI features.

The detection and recognition models now live in their own package,
lum_vision (github.com/lumiohub-ai/lum-model-vision), installed as a normal
dependency — see requirements.txt. What remains here is camera calibration,
which is database-backed and therefore application-specific.
"""

from .calibration import CameraCalibrator

__all__ = [
    "CameraCalibrator",
]
