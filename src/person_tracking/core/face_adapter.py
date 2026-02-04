"""
Face Recognition Integration for Person Tracking.

Simple helper functions to use the existing face recognition system
with person ROIs from person detection.
"""

from typing import Optional, Tuple
import numpy as np
from numpy.typing import NDArray


def crop_person_roi(
    frame: NDArray,
    bbox: NDArray,
    expand: float = 0.1
) -> Tuple[Optional[NDArray], Tuple[int, int]]:
    """
    Crop person ROI from frame with optional expansion.

    Args:
        frame: Full frame (BGR format)
        bbox: Person bounding box [x1, y1, x2, y2]
        expand: Expansion factor (0.1 = 10% larger)

    Returns:
        Tuple of (cropped_roi, (x_offset, y_offset))
    """
    h, w = frame.shape[:2]

    # Expand bbox
    x1, y1, x2, y2 = bbox
    bbox_w = x2 - x1
    bbox_h = y2 - y1

    expand_w = bbox_w * expand
    expand_h = bbox_h * expand

    x1 = max(0, int(x1 - expand_w))
    y1 = max(0, int(y1 - expand_h))
    x2 = min(w, int(x2 + expand_w))
    y2 = min(h, int(y2 + expand_h))

    # Crop ROI
    roi = frame[y1:y2, x1:x2]

    return roi, (x1, y1)
