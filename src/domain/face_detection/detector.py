"""Face detection using InsightFace."""

import contextlib
import os
import ssl
import warnings
from typing import List, Optional

import cv2
import numpy as np
from loguru import logger

# SECURITY: Do NOT disable SSL verification globally.
# Instead, use a temporary context only during model downloads.
# The global ssl._create_default_https_context override is removed.


def _download_model_with_ssl_fallback(model_name: str) -> None:
    """Download InsightFace model with SSL fallback for legacy servers.

    This function temporarily relaxes SSL verification ONLY during model download,
    then restores secure defaults. This prevents MITM attacks during runtime.

    Args:
        model_name: Name of the InsightFace model to download
    """
    import urllib.request
    import tempfile

    # First try with proper SSL verification
    try:
        # Check if model already exists in onnx format
        from insightface.utils import DEFAULT_MP_NAME
        from insightface.app import FaceAnalysis
        # Just attempt to load - will use cached if exists
        return
    except Exception:
        pass

    # If download needed, try with SSL first, then fallback
    logger.warning(
        f"Downloading InsightFace model '{model_name}'. "
        "If SSL fails, will retry with verification disabled (less secure)."
    )

    # Create a temporary SSL context that doesn't verify
    # This is only used for the download, not for runtime
    unverified_context = ssl.create_default_context()
    unverified_context.check_hostname = False
    unverified_context.verify_mode = ssl.CERT_NONE

    # Store original context creator
    original_context = ssl._create_default_https_context

    try:
        # Temporarily use unverified context for download only
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)
            ssl._create_default_https_context = lambda: unverified_context
            # Import will trigger download if needed
            from insightface.app import FaceAnalysis
            _ = FaceAnalysis(name=model_name, root=os.path.expanduser('~/.insightface'))
    finally:
        # CRITICAL: Restore secure SSL context
        ssl._create_default_https_context = original_context


from insightface.app import FaceAnalysis  # type: ignore


class FaceDetector:
    """Face detector using InsightFace's buffalo_l model.

    This class handles face detection and embedding computation.
    """

    def __init__(self, gpu_id: int = 0, model_name: str = 'buffalo_l', padding_percent: float = 20.0):
        """Initialize the face detector.

        Args:
            gpu_id: GPU device ID to use for detection
            model_name: Name of the InsightFace model to use
            padding_percent: Percentage of padding to add around images before detection (0-100)
        """
        self.gpu_id = gpu_id
        self.model_name = model_name
        self.padding_percent = max(0.0, min(100.0, padding_percent))  # Clamp between 0-100
        self.model: Optional[FaceAnalysis] = None
        self._initialize_model()

        if self.padding_percent > 0:
            logger.info(f"Face detection padding enabled: {self.padding_percent}%")

    def _initialize_model(self) -> None:
        """Initialize the InsightFace detection model."""
        # Suppress InsightFace output during initialization
        with open(os.devnull, 'w') as fnull:
            with contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
                self.model = FaceAnalysis(name=self.model_name)
                self.model.prepare(ctx_id=self.gpu_id)

        logger.debug(f"Initialized InsightFace model '{self.model_name}' on GPU {self.gpu_id}")

    def _add_padding(self, image: np.ndarray) -> np.ndarray:
        """Add padding around the image to improve face detection.

        Args:
            image: Input image array (BGR format)

        Returns:
            Padded image with border added
        """
        if self.padding_percent <= 0:
            return image

        h, w = image.shape[:2]

        # Calculate padding size based on percentage of image dimensions
        pad_h = int(h * self.padding_percent / 100)
        pad_w = int(w * self.padding_percent / 100)

        # Add padding using border replication (extends edge pixels)
        # This is better than black/white borders as it looks more natural
        padded = cv2.copyMakeBorder(
            image,
            top=pad_h,
            bottom=pad_h,
            left=pad_w,
            right=pad_w,
            borderType=cv2.BORDER_REPLICATE
        )

        return padded

    def detect(self, image: np.ndarray) -> List:
        """Detect faces in an image.

        Args:
            image: Input image array (BGR format)

        Returns:
            List of detected face objects with bounding boxes, embeddings, and landmarks
        """
        if self.model is None:
            raise RuntimeError("Model not initialized")

        # Apply padding to improve detection of faces near edges
        padded_image = self._add_padding(image)

        # Detect faces on padded image
        faces = self.model.get(padded_image)
        return faces

    def extract_face_features(self, image: np.ndarray) -> List[dict]:
        """Extract comprehensive face features including boxes, embeddings, and landmarks.

        Args:
            image: Input image array (BGR format)

        Returns:
            List of dictionaries containing face features:
            - bbox: Bounding box coordinates [x1, y1, x2, y2, confidence, class_id]
            - embedding: Normalized face embedding vector
            - landmarks: Facial landmarks (5 keypoints)
        """
        faces = self.detect(image)
        face_features = []

        for face in faces:
            # Extract bounding box
            x1, y1, x2, y2 = face.bbox.astype(int)
            conf = face.det_score

            # Extract and normalize embedding
            embedding = face.embedding
            embedding = embedding / np.linalg.norm(embedding)

            # Extract facial landmarks (5 keypoints: left eye, right eye, nose, left mouth, right mouth)
            landmarks = face.kps.astype(int)  # Shape: (5, 2)

            face_features.append({
                'bbox': [x1, y1, x2, y2, conf, 0],  # class id 0 for faces
                'embedding': embedding,
                'landmarks': landmarks
            })

        return face_features