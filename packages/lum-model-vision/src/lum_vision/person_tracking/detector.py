"""
Person Detector using YOLOv8/YOLO26 or Pose variants.

Detects persons in video frames using Ultralytics YOLO models.
- YOLO26: Latest model with NMS-free end-to-end design (recommended)
- YOLOv8: Fast person detection
- Pose variants: Person detection + 17 pose keypoints (COCO format, slower)
"""

from pathlib import Path
from typing import Optional, Union
from loguru import logger
import torch


# SECURITY: Do NOT disable SSL verification globally.
# SSL verification is critical for preventing MITM attacks.
# Model downloads should use proper SSL or be done offline.
#
# If you need to download models in environments with SSL issues:
# 1. Download models manually and place in the expected location
# 2. Use YOLO_CONFIG_DIR environment variable to specify local model path
# 3. Use a proper certificate authority

# Import YOLO after ensuring SSL is not globally compromised
from ultralytics import YOLO


class PersonDetector:
    """
    Person detection using YOLOv8/YOLO26 or Pose variants.

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
        - YOLO26 (model_version='yolo26'): NMS-free, ~43% faster CPU inference
        - YOLOv8 (model_version='yolov8'): ~30-40 FPS on GPU
        - Pose variants (use_pose=True): ~12-18 FPS on GPU, for skeleton visualization
    """

    def __init__(
        self,
        model_size: str = "s",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: Optional[str] = None,
        use_pose: bool = False,
        model_version: str = "yolo26",  # 'yolo26' or 'yolov8'
        weights_dir: Optional[Union[str, Path]] = None
    ):
        """
        Initialize Person Detector.

        Args:
            model_size: Model size (n/s/m/l/x)
                       n=nano, s=small, m=medium, l=large, x=xlarge
            confidence_threshold: Minimum confidence for detection (0.0-1.0)
            iou_threshold: IoU threshold for NMS (ignored for YOLO26 which is NMS-free)
            device: Device to run model on ('cuda', 'cpu', or None for auto)
            use_pose: Use Pose variant for keypoints (slower) or regular detection (faster)
            model_version: 'yolo26' (recommended, NMS-free) or 'yolov8'
            weights_dir: Directory to load/download weights into. Defaults to the
                         current working directory, which is what ultralytics does
                         on its own — pass this to keep weights somewhere stable.
        """
        self.model_size = model_size
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.use_pose = use_pose
        self.model_version = model_version.lower()

        # Auto-detect device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Build model type string for logging
        if self.model_version == "yolo26":
            model_type = "YOLO26-Pose" if use_pose else "YOLO26"
        else:
            model_type = "YOLOv8-Pose" if use_pose else "YOLOv8"

        logger.debug(
            f"Initializing PersonDetector with {model_type}{model_size} "
            f"on device: {self.device}"
        )

        # Build model filename based on version and pose setting
        if self.model_version == "yolo26":
            model_name = f"yolo26{model_size}-pose.pt" if use_pose else f"yolo26{model_size}.pt"
        else:
            model_name = f"yolov8{model_size}-pose.pt" if use_pose else f"yolov8{model_size}.pt"

        # Resolve to an absolute path so the weights land in a stable location
        # instead of wherever the host process happens to be running from.
        if weights_dir is not None:
            weights_dir = Path(weights_dir).expanduser()
            weights_dir.mkdir(parents=True, exist_ok=True)
            model_path = str(weights_dir / model_name)
        else:
            model_path = model_name

        try:
            self.model = YOLO(model_path)
            logger.debug(f"{model_type} model loaded: {model_path}")

            # Move model to device
            self.model.to(self.device)

        except Exception as e:
            logger.exception(f"Failed to load {model_type}{model_size} model: {e}")
            raise RuntimeError(f"Failed to load model {model_path}: {e}")

        # Model info
        self.input_size = 640  # YOLO default input size
        self.num_keypoints = 17 if use_pose else 0  # COCO format (only if using pose)

        logger.info(
            f"PersonDetector initialized: model={model_path}, conf={confidence_threshold}, "
            f"iou={iou_threshold}, device={self.device}, pose={use_pose}"
        )
