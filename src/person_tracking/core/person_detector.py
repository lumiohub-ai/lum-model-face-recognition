"""
Person Detector using YOLOv8-Pose.

Detects persons in video frames and extracts pose keypoints (17 points in COCO format).
Uses Ultralytics YOLOv8-Pose model for real-time person detection.
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
from numpy.typing import NDArray
from ultralytics import YOLO
from loguru import logger
import torch
import ssl
import urllib3

# Disable SSL verification for model downloads (development only)
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class PersonDetector:
    """
    Person detection using YOLOv8-Pose.

    This detector identifies persons in video frames and extracts:
    - Bounding boxes (x1, y1, x2, y2)
    - Confidence scores
    - 17 pose keypoints (COCO format)

    Keypoints (COCO format):
        0: Nose, 1: Left Eye, 2: Right Eye, 3: Left Ear, 4: Right Ear,
        5: Left Shoulder, 6: Right Shoulder, 7: Left Elbow, 8: Right Elbow,
        9: Left Wrist, 10: Right Wrist, 11: Left Hip, 12: Right Hip,
        13: Left Knee, 14: Right Knee, 15: Left Ankle, 16: Right Ankle
    """

    def __init__(
        self,
        model_size: str = "s",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: Optional[str] = None
    ):
        """
        Initialize Person Detector.

        Args:
            model_size: YOLOv8-Pose model size (n/s/m/l/x)
                       n=nano, s=small, m=medium, l=large, x=xlarge
            confidence_threshold: Minimum confidence for detection (0.0-1.0)
            iou_threshold: IoU threshold for NMS
            device: Device to run model on ('cuda', 'cpu', or None for auto)
        """
        self.model_size = model_size
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

        # Auto-detect device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        logger.info(
            f"Initializing PersonDetector with YOLOv8{model_size}-pose "
            f"on device: {self.device}"
        )

        # Load YOLOv8-Pose model
        model_name = f"yolov8{model_size}-pose.pt"
        try:
            self.model = YOLO(model_name)
            logger.info(f"YOLOv8-Pose model loaded: {model_name}")

            # Move model to device
            self.model.to(self.device)

        except Exception as e:
            logger.error(f"Failed to load YOLOv8-Pose model: {e}")
            raise RuntimeError(f"Failed to load model {model_name}: {e}")

        # Model info
        self.input_size = 640  # YOLOv8 default
        self.num_keypoints = 17  # COCO format

        logger.info(
            f"PersonDetector initialized: conf={confidence_threshold}, "
            f"iou={iou_threshold}, device={self.device}"
        )

    def detect_persons(
        self,
        frame: NDArray,
        min_keypoint_confidence: float = 0.3
    ) -> List[Dict]:
        """
        Detect persons in a frame and extract pose keypoints.

        Args:
            frame: Input frame (BGR format, shape: [H, W, 3])
            min_keypoint_confidence: Minimum confidence for keypoint visibility

        Returns:
            List of detected persons, each as a dict:
            {
                'bbox': [x1, y1, x2, y2],  # Bounding box coordinates
                'confidence': float,        # Detection confidence
                'keypoints': np.array,      # Shape: (17, 3) [x, y, confidence]
                'person_id': int            # Temporary ID (will be assigned by tracker)
            }
        """
        if frame is None or frame.size == 0:
            logger.warning("Empty frame received")
            return []

        try:
            # Run YOLOv8-Pose inference
            results = self.model(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False,
                device=self.device
            )

            # Extract detections
            detections = []
            for result in results:
                # Get bounding boxes
                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    continue

                # Get keypoints
                keypoints = result.keypoints
                if keypoints is None:
                    continue

                # Process each detection
                for idx in range(len(boxes)):
                    # Bounding box
                    bbox = boxes.xyxy[idx].cpu().numpy()  # [x1, y1, x2, y2]
                    conf = float(boxes.conf[idx].cpu().numpy())

                    # Keypoints: shape (17, 3) where each row is [x, y, confidence]
                    kpts = keypoints.xy[idx].cpu().numpy()  # (17, 2)
                    kpts_conf = keypoints.conf[idx].cpu().numpy()  # (17,)

                    # Combine keypoints with confidence
                    kpts_full = np.zeros((self.num_keypoints, 3))
                    kpts_full[:, :2] = kpts  # x, y
                    kpts_full[:, 2] = kpts_conf  # confidence

                    detection = {
                        'bbox': bbox.tolist(),
                        'confidence': conf,
                        'keypoints': kpts_full,
                        'person_id': idx  # Temporary ID (tracker will assign real ID)
                    }

                    detections.append(detection)

            # Verbose logging disabled to reduce log noise
            # logger.debug(f"Detected {len(detections)} person(s) in frame")
            return detections

        except Exception as e:
            logger.error(f"Error during person detection: {e}")
            return []

    def get_visible_keypoints(
        self,
        keypoints: NDArray,
        min_confidence: float = 0.3
    ) -> Dict[str, NDArray]:
        """
        Get visible keypoints above confidence threshold.

        Args:
            keypoints: Keypoints array (17, 3) [x, y, confidence]
            min_confidence: Minimum confidence for visibility

        Returns:
            Dictionary mapping keypoint names to coordinates
        """
        keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]

        visible_keypoints = {}
        for idx, name in enumerate(keypoint_names):
            if keypoints[idx, 2] >= min_confidence:
                visible_keypoints[name] = keypoints[idx, :2]

        return visible_keypoints

    def get_keypoint_by_name(
        self,
        keypoints: NDArray,
        name: str,
        min_confidence: float = 0.3
    ) -> Optional[NDArray]:
        """
        Get a specific keypoint by name if visible.

        Args:
            keypoints: Keypoints array (17, 3)
            name: Keypoint name (e.g., 'nose', 'left_wrist')
            min_confidence: Minimum confidence

        Returns:
            Keypoint coordinates [x, y] or None if not visible
        """
        keypoint_map = {
            'nose': 0, 'left_eye': 1, 'right_eye': 2,
            'left_ear': 3, 'right_ear': 4,
            'left_shoulder': 5, 'right_shoulder': 6,
            'left_elbow': 7, 'right_elbow': 8,
            'left_wrist': 9, 'right_wrist': 10,
            'left_hip': 11, 'right_hip': 12,
            'left_knee': 13, 'right_knee': 14,
            'left_ankle': 15, 'right_ankle': 16
        }

        if name not in keypoint_map:
            logger.warning(f"Unknown keypoint name: {name}")
            return None

        idx = keypoint_map[name]
        if keypoints[idx, 2] >= min_confidence:
            return keypoints[idx, :2]
        return None

    def calculate_person_height(self, keypoints: NDArray) -> Optional[float]:
        """
        Estimate person height from keypoints.

        Args:
            keypoints: Keypoints array (17, 3)

        Returns:
            Estimated height in pixels, or None if not enough keypoints
        """
        # Try nose to ankle distance
        nose = self.get_keypoint_by_name(keypoints, 'nose')
        left_ankle = self.get_keypoint_by_name(keypoints, 'left_ankle')
        right_ankle = self.get_keypoint_by_name(keypoints, 'right_ankle')

        if nose is not None and (left_ankle is not None or right_ankle is not None):
            ankle = left_ankle if left_ankle is not None else right_ankle
            height = np.linalg.norm(nose - ankle)
            return float(height)

        # Fallback: try shoulder to ankle
        left_shoulder = self.get_keypoint_by_name(keypoints, 'left_shoulder')
        right_shoulder = self.get_keypoint_by_name(keypoints, 'right_shoulder')

        if (left_shoulder is not None or right_shoulder is not None) and \
           (left_ankle is not None or right_ankle is not None):
            shoulder = left_shoulder if left_shoulder is not None else right_shoulder
            ankle = left_ankle if left_ankle is not None else right_ankle
            # Multiply by ~1.3 to account for head
            height = np.linalg.norm(shoulder - ankle) * 1.3
            return float(height)

        return None

    def visualize_detections(
        self,
        frame: NDArray,
        detections: List[Dict],
        draw_bbox: bool = True,
        draw_keypoints: bool = True,
        draw_skeleton: bool = True
    ) -> NDArray:
        """
        Visualize detections on frame.

        Args:
            frame: Input frame
            detections: List of detection dicts
            draw_bbox: Draw bounding boxes
            draw_keypoints: Draw keypoint circles
            draw_skeleton: Draw skeleton connections

        Returns:
            Annotated frame
        """
        import cv2

        annotated = frame.copy()

        # COCO skeleton connections
        skeleton = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 12),  # Torso
            (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
        ]

        for det in detections:
            bbox = det['bbox']
            keypoints = det['keypoints']
            conf = det['confidence']

            # Draw bounding box
            if draw_bbox:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated,
                    f"{conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

            # Draw skeleton
            if draw_skeleton:
                for start_idx, end_idx in skeleton:
                    if keypoints[start_idx, 2] > 0.3 and keypoints[end_idx, 2] > 0.3:
                        start_point = tuple(map(int, keypoints[start_idx, :2]))
                        end_point = tuple(map(int, keypoints[end_idx, :2]))
                        cv2.line(annotated, start_point, end_point, (255, 0, 0), 2)

            # Draw keypoints
            if draw_keypoints:
                for idx in range(len(keypoints)):
                    if keypoints[idx, 2] > 0.3:
                        x, y = map(int, keypoints[idx, :2])
                        cv2.circle(annotated, (x, y), 3, (0, 0, 255), -1)

        return annotated

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"PersonDetector(model=yolov8{self.model_size}-pose, "
            f"conf={self.confidence_threshold}, iou={self.iou_threshold}, "
            f"device={self.device})"
        )
