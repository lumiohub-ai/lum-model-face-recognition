"""
Face Recognition Integration for Person Tracking.

Simple helper functions to use the existing face recognition system
with person ROIs from person detection.
"""

from typing import Optional, Dict, Tuple, Any
import numpy as np
from numpy.typing import NDArray
from loguru import logger

# Import existing face recognition components
from face_recognition.core.detector import FaceDetector
from face_recognition.core.recognizer import FaceRecognition


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


def recognize_person_in_roi(
    face_detector: FaceDetector,
    face_recognizer: FaceRecognition,
    frame: NDArray,
    person_bbox: NDArray,
    match_threshold: float = 0.3
) -> Dict[str, Any]:
    """
    Recognize person using existing face recognition system.

    Args:
        face_detector: Existing FaceDetector instance
        face_recognizer: Existing FaceRecognition instance
        frame: Full video frame
        person_bbox: Person bounding box [x1, y1, x2, y2]
        match_threshold: Similarity threshold for matching

    Returns:
        Recognition result:
        {
            'recognized': bool,
            'name': str or None,
            'similarity': float,
            'face_detected': bool,
            'embedding': np.ndarray or None
        }
    """
    # Crop person ROI
    person_roi, roi_offset = crop_person_roi(frame, person_bbox)

    if person_roi is None or person_roi.size == 0:
        return {
            'recognized': False,
            'name': None,
            'similarity': 0.0,
            'face_detected': False,
            'embedding': None
        }

    # Detect faces in ROI using existing detector
    try:
        faces = face_detector.detect(person_roi)
    except Exception as e:
        logger.error(f"Face detection failed: {e}")
        return {
            'recognized': False,
            'name': None,
            'similarity': 0.0,
            'face_detected': False,
            'embedding': None
        }

    if not faces or len(faces) == 0:
        return {
            'recognized': False,
            'name': None,
            'similarity': 0.0,
            'face_detected': False,
            'embedding': None
        }

    # Use first (largest) face
    face = faces[0]
    embedding = face.embedding

    if embedding is None:
        return {
            'recognized': False,
            'name': None,
            'similarity': 0.0,
            'face_detected': True,
            'embedding': None
        }

    # Normalize embedding
    embedding = embedding / np.linalg.norm(embedding)

    # Match using existing recognizer
    from sklearn.metrics.pairwise import cosine_similarity

    if face_recognizer.db_embs is None or len(face_recognizer.db_embs) == 0:
        return {
            'recognized': False,
            'name': None,
            'similarity': 0.0,
            'face_detected': True,
            'embedding': embedding
        }

    # Compute similarity
    similarities = cosine_similarity([embedding], face_recognizer.db_embs)[0]
    best_idx = np.argmax(similarities)
    best_similarity = float(similarities[best_idx])

    # Check threshold
    if best_similarity >= match_threshold:
        best_name = face_recognizer.db_names[best_idx]
        return {
            'recognized': True,
            'name': best_name,
            'similarity': best_similarity,
            'face_detected': True,
            'embedding': embedding
        }
    else:
        return {
            'recognized': False,
            'name': None,
            'similarity': best_similarity,
            'face_detected': True,
            'embedding': embedding
        }
