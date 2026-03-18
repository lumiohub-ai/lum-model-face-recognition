"""
Video Infrastructure

Camera stream handling and frame annotation.
"""

from .stream_handler import StreamHandler
from .stream_manager import StreamManager
from .annotator import FrameAnnotator

__all__ = [
    "StreamHandler",
    "StreamManager",
    "FrameAnnotator",
]
