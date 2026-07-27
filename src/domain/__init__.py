"""
Domain Layer

Core business logic for Smart Office AI features.

The detection and recognition models now live in the `lum_vision` package
(packages/lum-model-vision); what remains here is camera calibration, which is
database-backed and therefore application-specific.
"""

from .calibration import CameraCalibrator

__all__ = [
    "CameraCalibrator",
]
