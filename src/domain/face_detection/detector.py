"""Face detection using InsightFace."""

import contextlib
import os
from typing import List, Optional

import cv2
import numpy as np
from loguru import logger

from insightface.app import FaceAnalysis 


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

        # detect() ran the model on a padded image (see _add_padding), so
        # face.bbox/face.kps come back in padded-image coordinate space.
        # Shift them back to the original image's coordinates — the same
        # correction pipeline/gpu_worker.py already applies for its own
        # face.bbox/face.kps usage, which this method was missing, so any
        # bbox/landmarks stored from here (e.g. embedding_sync.py's DB
        # metadata) were off by (pad_w, pad_h) whenever padding is enabled.
        h, w = image.shape[:2]
        pad_h = int(h * self.padding_percent / 100)
        pad_w = int(w * self.padding_percent / 100)

        for face in faces:
            # Extract bounding box
            x1, y1, x2, y2 = face.bbox.astype(int)
            x1 -= pad_w
            y1 -= pad_h
            x2 -= pad_w
            y2 -= pad_h
            conf = face.det_score

            # Extract and normalize embedding
            embedding = face.embedding
            norm = np.linalg.norm(embedding)
            embedding = embedding / norm if norm > 0 else embedding

            # Extract facial landmarks (5 keypoints: left eye, right eye, nose, left mouth, right mouth)
            landmarks = face.kps.astype(int) - [pad_w, pad_h]  # Shape: (5, 2)

            face_features.append({
                'bbox': [x1, y1, x2, y2, conf, 0],  # class id 0 for faces
                'embedding': embedding,
                'landmarks': landmarks
            })

        return face_features