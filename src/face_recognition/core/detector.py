"""Face detection using InsightFace."""

import contextlib
import os
import ssl
from typing import List, Optional

import cv2
import numpy as np

# Disable SSL verification for InsightFace model downloads
ssl._create_default_https_context = ssl._create_unverified_context

from insightface.app import FaceAnalysis  # type: ignore
from loguru import logger


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

        logger.info(f"Initialized InsightFace model '{self.model_name}' on GPU {self.gpu_id}")

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
            - bbox: Bounding box coordinates [x1, y1, x2, y2, confidence, class_id] in original image space
            - embedding: Normalized face embedding vector
            - landmarks: Facial landmarks (5 keypoints) in original image space
        """
        h, w = image.shape[:2]

        # Calculate padding offset (same as _add_padding)
        pad_h = int(h * self.padding_percent / 100) if self.padding_percent > 0 else 0
        pad_w = int(w * self.padding_percent / 100) if self.padding_percent > 0 else 0

        faces = self.detect(image)
        face_features = []

        for face in faces:
            # Extract bounding box (in padded image coordinates)
            x1, y1, x2, y2 = face.bbox.astype(int)
            conf = face.det_score

            # Adjust coordinates back to original image space
            x1 = max(0, x1 - pad_w)
            y1 = max(0, y1 - pad_h)
            x2 = min(w, x2 - pad_w)
            y2 = min(h, y2 - pad_h)

            # Validate bbox is still valid after adjustment
            if x2 <= x1 or y2 <= y1:
                continue  # Skip invalid boxes

            # Extract and normalize embedding
            embedding = face.embedding
            embedding = embedding / np.linalg.norm(embedding)

            # Extract facial landmarks (5 keypoints: left eye, right eye, nose, left mouth, right mouth)
            landmarks = face.kps.astype(int)  # Shape: (5, 2)

            # Adjust landmarks back to original image space
            landmarks[:, 0] = np.clip(landmarks[:, 0] - pad_w, 0, w)
            landmarks[:, 1] = np.clip(landmarks[:, 1] - pad_h, 0, h)

            face_features.append({
                'bbox': [x1, y1, x2, y2, conf, 0],  # class id 0 for faces
                'embedding': embedding,
                'landmarks': landmarks
            })

        return face_features