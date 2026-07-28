"""Face detection using InsightFace."""

import contextlib
import os
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np
from loguru import logger

from insightface.app import FaceAnalysis


class FaceDetector:
    """Face detector using InsightFace's buffalo_l model.

    This class handles face detection and embedding computation.
    """

    def __init__(
        self,
        gpu_id: int = 0,
        model_name: str = 'buffalo_l',
        padding_percent: float = 20.0,
        model_root: Optional[Union[str, Path]] = None,
    ):
        """Initialize the face detector.

        Args:
            gpu_id: GPU device ID to use for detection
            model_name: Name of the InsightFace model to use
            padding_percent: Percentage of padding to add around images before detection (0-100)
            model_root: Directory InsightFace downloads its model zoo into.
                        Defaults to InsightFace's own ``~/.insightface``.
        """
        self.gpu_id = gpu_id
        self.model_name = model_name
        self.padding_percent = max(0.0, min(100.0, padding_percent))  # Clamp between 0-100
        self.model_root = Path(model_root).expanduser() if model_root else None
        self.model: Optional[FaceAnalysis] = None
        self._initialize_model()

        if self.padding_percent > 0:
            logger.info(f"Face detection padding enabled: {self.padding_percent}%")

    def _initialize_model(self) -> None:
        """Initialize the InsightFace detection model."""
        # FaceAnalysis defaults root to '~/.insightface'; only override when the
        # caller asked for a specific location.
        kwargs = {}
        if self.model_root is not None:
            self.model_root.mkdir(parents=True, exist_ok=True)
            kwargs['root'] = str(self.model_root)

        # Suppress InsightFace output during initialization
        with open(os.devnull, 'w') as fnull:
            with contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
                self.model = FaceAnalysis(name=self.model_name, **kwargs)
                self.model.prepare(ctx_id=self.gpu_id)

        logger.debug(
            f"Initialized InsightFace model '{self.model_name}' on GPU {self.gpu_id}"
            + (f" (root={self.model_root})" if self.model_root else "")
        )

    #: Attributes InsightFace fills with image-space point coordinates. Each is an
    #: (N, 2) or (N, 3) array whose first two columns are x, y.
    _POINT_ATTRS = ('kps', 'landmark_2d_106', 'landmark_3d_68')

    def _padding_offset(self, image: np.ndarray) -> tuple:
        """Return the (pad_w, pad_h) border added to ``image`` by :meth:`_add_padding`."""
        if self.padding_percent <= 0:
            return 0, 0

        h, w = image.shape[:2]
        return int(w * self.padding_percent / 100), int(h * self.padding_percent / 100)

    def _add_padding(self, image: np.ndarray) -> np.ndarray:
        """Add padding around the image to improve face detection.

        Args:
            image: Input image array (BGR format)

        Returns:
            Padded image with border added
        """
        pad_w, pad_h = self._padding_offset(image)
        if not (pad_w or pad_h):
            return image

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

    def _remove_padding_offset(self, faces: List, pad_w: int, pad_h: int) -> List:
        """Shift face coordinates from padded-image space back to input-image space.

        Mutates and returns ``faces``. Angular attributes (``pose``) and the
        embedding are translation-invariant and left alone.
        """
        if not (pad_w or pad_h):
            return faces

        offset = np.array([pad_w, pad_h], dtype=np.float32)

        for face in faces:
            if face.bbox is not None:
                face.bbox = face.bbox - np.tile(offset, 2)

            for attr in self._POINT_ATTRS:
                points = face.get(attr)
                if points is None:
                    continue
                points = points.astype(np.float32, copy=True)
                points[:, 0:2] -= offset
                face[attr] = points

        return faces

    def detect(self, image: np.ndarray) -> List:
        """Detect faces in an image.

        Padding is an internal detail: coordinates come back in the coordinate
        space of ``image``, so callers must not compensate for it themselves.
        Boxes may still fall partly outside the frame for faces detected against
        the replicated border, so clamp before cropping.

        Args:
            image: Input image array (BGR format)

        Returns:
            List of detected face objects with bounding boxes, embeddings, and landmarks
        """
        if self.model is None:
            raise RuntimeError("Model not initialized")

        # Apply padding to improve detection of faces near edges
        pad_w, pad_h = self._padding_offset(image)
        padded_image = self._add_padding(image)

        # Here we infer the faces using the InsightFace model
        faces = self.model.get(padded_image)
        return self._remove_padding_offset(faces, pad_w, pad_h)

    def extract_face_features(self, image: np.ndarray) -> List[dict]:
        """Extract comprehensive face features including boxes, embeddings, and landmarks.

        Args:
            image: Input image array (BGR format)

        Returns:
            List of dictionaries containing face features, in ``image`` coordinates:
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