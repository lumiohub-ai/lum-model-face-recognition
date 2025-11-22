"""Face detection using InsightFace."""

import contextlib
import os
from typing import List, Optional

import cv2
import numpy as np
from insightface.app import FaceAnalysis  # type: ignore
from loguru import logger


class FaceDetector:
    """Face detector using InsightFace's buffalo_l model.

    This class handles face detection and embedding computation.
    """

    def __init__(self, gpu_id: int = 0, model_name: str = 'buffalo_l'):
        """Initialize the face detector.

        Args:
            gpu_id: GPU device ID to use for detection
            model_name: Name of the InsightFace model to use
        """
        self.gpu_id = gpu_id
        self.model_name = model_name
        self.model: Optional[FaceAnalysis] = None
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize the InsightFace detection model."""
        # Suppress InsightFace output during initialization
        with open(os.devnull, 'w') as fnull:
            with contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
                self.model = FaceAnalysis(name=self.model_name)
                self.model.prepare(ctx_id=self.gpu_id)

        logger.info(f"Initialized InsightFace model '{self.model_name}' on GPU {self.gpu_id}")

    def detect(self, image: np.ndarray) -> List:
        """Detect faces in an image.

        Args:
            image: Input image array (BGR format)

        Returns:
            List of detected face objects with bounding boxes, embeddings, and landmarks
        """
        if self.model is None:
            raise RuntimeError("Model not initialized")

        faces = self.model.get(image)
        return faces

    def compute_embedding(
        self,
        image: np.ndarray,
        alpha: float = 0.9
    ) -> Optional[np.ndarray]:
        """Compute face embedding for a given image.

        Args:
            image: Input image array (BGR format)
            alpha: Normalization factor for the embedding (not used in current implementation)

        Returns:
            Face embedding vector or None if no face is detected
        """
        faces = self.detect(image)

        if not faces:
            return None

        # Use the first detected face
        face = faces[0]
        emb = face.embedding

        # Apply normalization (alpha is currently not used, kept for compatibility)
        emb = emb / np.linalg.norm(emb)

        return emb

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
'''


'''