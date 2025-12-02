"""
Phone Detector using YOLOv8.

Detects cell phones in video frames using YOLOv8 model trained on COCO dataset.
Associates detected phones with persons using spatial proximity.
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


class PhoneDetector:
    """
    Phone detector using YOLOv8.

    Detects cell phones (COCO class 67) in video frames and performs
    basic spatial association with person bounding boxes.
    """

    # COCO class ID for cell phone
    CELL_PHONE_CLASS_ID = 67

    def __init__(
        self,
        model_size: str = "n",
        confidence_threshold: float = 0.4,
        device: Optional[str] = None
    ):
        """
        Initialize Phone Detector.

        Args:
            model_size: YOLOv8 model size (n/s/m/l/x)
            confidence_threshold: Minimum confidence for phone detection
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
            f"Initializing PhoneDetector with YOLOv8{model_size} "
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
            f"PhoneDetector initialized: conf={confidence_threshold}, "
            f"device={self.device}"
        )

    def detect_phones(self, frame: NDArray) -> List[Dict]:
        """
        Detect cell phones in a frame.

        Args:
            frame: Input frame (BGR format)

        Returns:
            List of detected phones:
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
            # Run YOLOv8 inference
            results = self.model(
                frame,
                conf=self.confidence_threshold,
                verbose=False,
                device=self.device,
                classes=[self.CELL_PHONE_CLASS_ID]  # Only detect cell phones
            )

            # Extract phone detections
            phones = []
            for result in results:
                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    continue

                for idx in range(len(boxes)):
                    # Get bbox and confidence
                    bbox = boxes.xyxy[idx].cpu().numpy()  # [x1, y1, x2, y2]
                    conf = float(boxes.conf[idx].cpu().numpy())
                    class_id = int(boxes.cls[idx].cpu().numpy())

                    # Verify it's a cell phone (should always be true due to classes filter)
                    if class_id == self.CELL_PHONE_CLASS_ID:
                        phone = {
                            'bbox': bbox.tolist(),
                            'confidence': conf,
                            'class_id': class_id,
                            'class_name': 'cell phone'
                        }
                        phones.append(phone)
                        logger.debug(f"Phone detected: conf={conf:.2f}, bbox={bbox.tolist()}")

            return phones

        except Exception as e:
            logger.error(f"Error during phone detection: {e}")
            return []

    def associate_phones_with_persons(
        self,
        phones: List[Dict],
        persons: List[Dict],
        iou_threshold: float = 0.1
    ) -> Dict[int, List[Dict]]:
        """
        Associate detected phones with persons using bbox overlap.

        NOTE: This method works with raw phone detections (no tracking).
        For tracked phones, use associate_tracked_phones_with_persons().

        Args:
            phones: List of phone detections
            persons: List of person detections (must have 'track_id' and 'bbox')
            iou_threshold: Minimum IoU for association

        Returns:
            Dictionary mapping track_id to list of associated phones
        """
        associations = {}

        for person in persons:
            track_id = person.get('track_id')
            if track_id is None:
                continue

            person_bbox = np.array(person['bbox'])
            associated_phones = []

            for phone in phones:
                phone_bbox = np.array(phone['bbox'])

                # Check if phone bbox overlaps with person bbox
                if self._bboxes_overlap(person_bbox, phone_bbox, iou_threshold):
                    associated_phones.append(phone)

            if associated_phones:
                associations[track_id] = associated_phones

        return associations

    def associate_tracked_phones_with_persons(
        self,
        tracked_phones: List[Dict],
        persons: List[Dict],
        iou_threshold: float = 0.1
    ) -> Dict[int, List[Dict]]:
        """
        Associate tracked phones with persons using bbox overlap.

        This method works with tracked phones that have phone_track_id.
        Maintains phone tracking information in associations.

        Args:
            tracked_phones: List of tracked phone detections (with phone_track_id)
            persons: List of person detections (must have 'track_id' and 'bbox')
            iou_threshold: Minimum IoU for association

        Returns:
            Dictionary mapping person track_id to list of associated tracked phones
            {person_track_id: [{'phone_track_id': int, 'bbox': [...], ...}]}
        """
        associations = {}

        for person in persons:
            person_track_id = person.get('track_id')
            if person_track_id is None:
                continue

            person_bbox = np.array(person['bbox'])
            associated_phones = []

            for phone in tracked_phones:
                phone_bbox = np.array(phone['bbox'])

                # Check if phone bbox overlaps with person bbox
                if self._bboxes_overlap(person_bbox, phone_bbox, iou_threshold):
                    # Include phone track info for temporal consistency
                    phone_info = phone.copy()
                    associated_phones.append(phone_info)

                    logger.debug(
                        f"Phone track {phone.get('phone_track_id')} "
                        f"associated with person {person_track_id} "
                        f"(age={phone.get('track_age', 0)}, "
                        f"predicted={phone.get('is_predicted', False)})"
                    )

            if associated_phones:
                associations[person_track_id] = associated_phones

        return associations

    @staticmethod
    def _bboxes_overlap(
        bbox1: NDArray,
        bbox2: NDArray,
        threshold: float = 0.0
    ) -> bool:
        """
        Check if two bounding boxes overlap.

        Args:
            bbox1: First bbox [x1, y1, x2, y2]
            bbox2: Second bbox [x1, y1, x2, y2]
            threshold: Minimum IoU threshold (0.0 = any overlap)

        Returns:
            True if overlap above threshold
        """
        # Intersection coordinates
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        # Intersection area
        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        if intersection == 0:
            return False

        if threshold == 0.0:
            return True  # Any overlap

        # Calculate IoU
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection

        if union == 0:
            return False

        iou = intersection / union
        return iou >= threshold

    def visualize_phones(
        self,
        frame: NDArray,
        phones: List[Dict],
        color: tuple = (255, 0, 0)
    ) -> NDArray:
        """
        Visualize phone detections on frame.

        Args:
            frame: Input frame
            phones: List of phone detections
            color: BGR color for bounding boxes

        Returns:
            Annotated frame
        """
        import cv2

        annotated = frame.copy()

        for phone in phones:
            bbox = phone['bbox']
            conf = phone['confidence']

            # Draw bbox
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Add label
            label = f"Phone {conf:.2f}"
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
            f"PhoneDetector(model=yolov8{self.model_size}, "
            f"conf={self.confidence_threshold}, device={self.device})"
        )
