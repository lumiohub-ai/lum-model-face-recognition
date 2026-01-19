"""
Person Detector using YOLOv8 or YOLOv8-Pose.

Detects persons in video frames using Ultralytics YOLO models.
- YOLOv8: Fast person detection (2-3x faster, recommended)
- YOLOv8-Pose: Person detection + 17 pose keypoints (COCO format, slower)
"""

from typing import List, Dict, Optional
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
    Person detection using YOLOv8 or YOLOv8-Pose.

    This detector identifies persons in video frames and extracts:
    - Bounding boxes (x1, y1, x2, y2)
    - Confidence scores
    - [Optional] 17 pose keypoints (COCO format) if use_pose=True

    Keypoints (COCO format, only when use_pose=True):
        0: Nose, 1: Left Eye, 2: Right Eye, 3: Left Ear, 4: Right Ear,
        5: Left Shoulder, 6: Right Shoulder, 7: Left Elbow, 8: Right Elbow,
        9: Left Wrist, 10: Right Wrist, 11: Left Hip, 12: Right Hip,
        13: Left Knee, 14: Right Knee, 15: Left Ankle, 16: Right Ankle

    Performance:
        - YOLOv8 (use_pose=False): ~30-40 FPS on GPU, 2-3x faster, recommended
        - YOLOv8-Pose (use_pose=True): ~12-18 FPS on GPU, for skeleton visualization
    """

    def __init__(
        self,
        model_size: str = "s",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: Optional[str] = None,
        use_pose: bool = False  # NEW: Toggle between YOLOv8 and YOLOv8-Pose
    ):
        """
        Initialize Person Detector.

        Args:
            model_size: YOLOv8 model size (n/s/m/l/x)
                       n=nano, s=small, m=medium, l=large, x=xlarge
            confidence_threshold: Minimum confidence for detection (0.0-1.0)
            iou_threshold: IoU threshold for NMS
            device: Device to run model on ('cuda', 'cpu', or None for auto)
            use_pose: Use YOLOv8-Pose for keypoints (slower) or regular YOLOv8 (faster)
        """
        self.model_size = model_size
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.use_pose = use_pose

        # Auto-detect device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        model_type = "YOLOv8-Pose" if use_pose else "YOLOv8"
        logger.info(
            f"Initializing PersonDetector with {model_type}{model_size} "
            f"on device: {self.device}"
        )

        # Load YOLOv8 or YOLOv8-Pose model
        model_name = f"yolov8{model_size}-pose.pt" if use_pose else f"yolov8{model_size}.pt"
        try:
            self.model = YOLO(model_name)
            logger.info(f"{model_type} model loaded: {model_name}")

            # Move model to device
            self.model.to(self.device)

        except Exception as e:
            logger.error(f"Failed to load {model_type} model: {e}")
            raise RuntimeError(f"Failed to load model {model_name}: {e}")

        # Model info
        self.input_size = 640  # YOLOv8 default
        self.num_keypoints = 17 if use_pose else 0  # COCO format (only if using pose)

        logger.info(
            f"PersonDetector initialized: conf={confidence_threshold}, "
            f"iou={iou_threshold}, device={self.device}, pose={use_pose}"
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

                # Process each detection
                for idx in range(len(boxes)):
                    # Get class ID
                    cls_id = int(boxes.cls[idx].cpu().numpy())

                    # Filter: Only accept person class (class 0 in COCO)
                    if cls_id != 0:
                        continue

                    # Bounding box
                    bbox = boxes.xyxy[idx].cpu().numpy()  # [x1, y1, x2, y2]
                    conf = float(boxes.conf[idx].cpu().numpy())

                    # Get keypoints only if using pose model
                    kpts_full = None
                    if self.use_pose:
                        keypoints = result.keypoints
                        if keypoints is not None:
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
                        'keypoints': kpts_full,  # None if not using pose
                        'person_id': idx  # Temporary ID (tracker will assign real ID)
                    }

                    detections.append(detection)

            # Verbose logging disabled to reduce log noise
            # logger.debug(f"Detected {len(detections)} person(s) in frame")
            return detections

        except Exception as e:
            logger.error(f"Error during person detection: {e}")
            return []
