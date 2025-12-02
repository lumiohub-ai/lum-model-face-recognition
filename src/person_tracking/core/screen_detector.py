"""
Screen Detector using YOLOv8.

Detects computer screens, laptops, and monitors in video frames using YOLOv8
model trained on COCO dataset. Used for idle detection by identifying when
persons are not looking at their screens.
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


class ScreenDetector:
    """
    Screen detector using YOLOv8.

    Detects computer screens, monitors, and laptops (COCO classes 62, 63)
    in video frames for idle detection purposes.
    """

    # COCO class IDs for screens/monitors
    TV_MONITOR_CLASS_ID = 62  # tv/monitor
    LAPTOP_CLASS_ID = 63      # laptop

    def __init__(
        self,
        model_size: str = "n",
        confidence_threshold: float = 0.4,
        device: Optional[str] = None
    ):
        """
        Initialize Screen Detector.

        Args:
            model_size: YOLOv8 model size (n/s/m/l/x)
            confidence_threshold: Minimum confidence for screen detection
            device: Device to run model on ('cuda', 'cpu', or None for auto)
        """
        self.model_size = model_size
        self.confidence_threshold = confidence_threshold

        # Auto-detect device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        logger.info(
            f"Initializing ScreenDetector with YOLOv8{model_size} "
            f"on device: {self.device}"
        )

        # Load YOLOv8 model
        model_name = f"yolov8{model_size}.pt"
        try:
            self.model = YOLO(model_name)
            logger.info(f"YOLOv8 model loaded: {model_name}")
            self.model.to(self.device)
        except Exception as e:
            logger.error(f"Failed to load YOLOv8 model: {e}")
            raise RuntimeError(f"Failed to load model {model_name}: {e}")

        logger.info(
            f"ScreenDetector initialized: conf={confidence_threshold}, "
            f"device={self.device}"
        )

    def detect_screens(self, frame: NDArray) -> List[Dict]:
        """
        Detect computer screens, monitors, and laptops in a frame.

        Args:
            frame: Input frame (BGR format)

        Returns:
            List of detected screens:
            [{
                'bbox': [x1, y1, x2, y2],
                'confidence': float,
                'class_id': int,
                'class_name': str
            }]
        """
        if frame is None or frame.size == 0:
            logger.warning("Empty frame received")
            return []

        try:
            # Run YOLOv8 inference - detect both monitors and laptops
            results = self.model(
                frame,
                conf=self.confidence_threshold,
                verbose=False,
                device=self.device,
                classes=[self.TV_MONITOR_CLASS_ID, self.LAPTOP_CLASS_ID]
            )

            # Extract screen detections
            screens = []
            for result in results:
                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    continue

                for idx in range(len(boxes)):
                    # Get bbox and confidence
                    bbox = boxes.xyxy[idx].cpu().numpy()  # [x1, y1, x2, y2]
                    conf = float(boxes.conf[idx].cpu().numpy())
                    class_id = int(boxes.cls[idx].cpu().numpy())

                    # Determine class name
                    if class_id == self.TV_MONITOR_CLASS_ID:
                        class_name = 'monitor'
                    elif class_id == self.LAPTOP_CLASS_ID:
                        class_name = 'laptop'
                    else:
                        continue  # Skip unknown classes

                    screen = {
                        'bbox': bbox.tolist(),
                        'confidence': conf,
                        'class_id': class_id,
                        'class_name': class_name
                    }
                    screens.append(screen)
                    logger.debug(
                        f"Screen detected: {class_name}, conf={conf:.2f}, "
                        f"bbox={bbox.tolist()}"
                    )

            return screens

        except Exception as e:
            logger.error(f"Error during screen detection: {e}")
            return []

    def associate_screens_with_persons(
        self,
        screens: List[Dict],
        persons: List[Dict],
        distance_threshold: float = 200.0
    ) -> Dict[int, List[Dict]]:
        """
        Associate detected screens with persons using proximity.

        Finds the nearest screen for each person within distance threshold.

        Args:
            screens: List of screen detections
            persons: List of person detections (must have 'track_id' and 'bbox')
            distance_threshold: Maximum distance (pixels) for association

        Returns:
            Dictionary mapping track_id to list of associated screens
        """
        associations = {}

        for person in persons:
            track_id = person.get('track_id')
            if track_id is None:
                continue

            person_bbox = np.array(person['bbox'])
            person_center = self._get_bbox_center(person_bbox)

            # Find nearest screen
            nearest_screen = None
            min_distance = float('inf')

            for screen in screens:
                screen_bbox = np.array(screen['bbox'])
                screen_center = self._get_bbox_center(screen_bbox)

                # Calculate Euclidean distance between centers
                distance = np.linalg.norm(person_center - screen_center)

                if distance < min_distance:
                    min_distance = distance
                    nearest_screen = screen

            # Associate if within threshold
            if nearest_screen is not None and min_distance <= distance_threshold:
                associations[track_id] = [{
                    **nearest_screen,
                    'distance': float(min_distance)
                }]

        return associations

    @staticmethod
    def _get_bbox_center(bbox: NDArray) -> NDArray:
        """
        Calculate center point of bounding box.

        Args:
            bbox: Bounding box [x1, y1, x2, y2]

        Returns:
            Center point [cx, cy]
        """
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        return np.array([cx, cy])

    def visualize_screens(
        self,
        frame: NDArray,
        screens: List[Dict],
        color: tuple = (0, 255, 255)  # Yellow
    ) -> NDArray:
        """
        Visualize screen detections on frame.

        Args:
            frame: Input frame
            screens: List of screen detections
            color: BGR color for bounding boxes

        Returns:
            Annotated frame
        """
        import cv2

        annotated = frame.copy()

        for screen in screens:
            bbox = screen['bbox']
            conf = screen['confidence']
            class_name = screen['class_name']

            # Draw bbox
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Add label
            label = f"{class_name} {conf:.2f}"
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

        return annotated

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ScreenDetector(model=yolov8{self.model_size}, "
            f"conf={self.confidence_threshold}, device={self.device})"
        )
