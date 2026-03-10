"""Model factory for initializing detection and recognition models.

This module centralizes the creation and configuration of all ML models
used in the SmartOffice system.
"""

from typing import Dict, Any, Optional

from loguru import logger
from config.settings import settings

from .detector import FaceDetector
from .recognizer import FaceRecognition
from domain.action_recognition import ActionRecognizer
from domain.person_tracking import PersonDetector
from domain.person_tracking.global_track import GlobalTrackManager


class ModelFactory:
    """Factory for creating detection and recognition models.

    Centralizes model initialization with consistent configuration,
    enabling GPU sharing and resource management.
    """

    def __init__(self, config: Dict[str, Any], client_slug: str):
        """Initialize model factory.

        Args:
            config: Configuration dictionary with model settings
            client_slug: Organization slug for embedding storage
        """
        self.config = config
        self.client_slug = client_slug
        self._face_detector: Optional[FaceDetector] = None
        self._face_recognizer: Optional[FaceRecognition] = None
        self._person_detector: Optional[PersonDetector] = None
        self._action_recognizer: Optional[ActionRecognizer] = None
        self._global_track_manager: Optional[GlobalTrackManager] = None
        self._global_id_generator = None  # GlobalTrackIDGenerator (lazy import)

    @property
    def face_detector(self) -> FaceDetector:
        """Get or create face detector (lazy initialization)."""
        if self._face_detector is None:
            logger.debug("Initializing FaceDetector...")
            self._face_detector = FaceDetector(
                gpu_id=0,
                model_name='buffalo_l'
            )
        return self._face_detector

    @property
    def face_recognizer(self) -> FaceRecognition:
        """Get or create face recognizer (lazy initialization)."""
        if self._face_recognizer is None:
            logger.debug("Initializing FaceRecognizer...")
            self._face_recognizer = self._create_face_recognizer()
        return self._face_recognizer

    @property
    def person_detector(self) -> PersonDetector:
        """Get or create person detector (lazy initialization)."""
        if self._person_detector is None:
            person_conf_threshold = self.config.get('person_detection_threshold', 0.5)
            model_version = self.config.get('person_detection_model', 'yolo26')
            logger.debug(f"Initializing PersonDetector ({model_version}, threshold: {person_conf_threshold})...")
            self._person_detector = PersonDetector(
                model_size='s',
                confidence_threshold=person_conf_threshold,
                use_pose=False,
                model_version=model_version  # 'yolo26' (NMS-free, faster) or 'yolov8'
            )
        return self._person_detector

    @property
    def action_recognizer(self) -> ActionRecognizer:
        """Get or create action recognizer (lazy initialization)."""
        if self._action_recognizer is None:
            action_config = self.config.get('action_recognition', {})
            enabled = action_config.get('enabled', False)
            ollama_api_url = action_config.get('ollama_api_url')
            model_name = settings.ollama_model

            actions = action_config.get('actions')
            self._action_recognizer = ActionRecognizer(
                ollama_api_url=ollama_api_url,
                client_slug=self.client_slug,
                enabled=enabled,
                check_interval_seconds=action_config.get('check_interval_seconds', 30),
                max_queue_size=action_config.get('max_queue_size', 50),
                num_workers=action_config.get('async_workers', 1),
                model_name=model_name,
                inference_timeout=action_config.get('inference_timeout', 30),
                actions=actions
            )

            # Start worker threads if enabled
            if enabled:
                self._action_recognizer.start_workers()
                logger.debug("Action recognition workers started")

        return self._action_recognizer

    @property
    def global_track_manager(self) -> GlobalTrackManager:
        """Get or create global track manager."""
        if self._global_track_manager is None:
            self._global_track_manager = GlobalTrackManager(app_config=self.config)
            if self._global_track_manager.enabled:
                logger.debug("GlobalTrackManager enabled - collecting baseline metrics")
        return self._global_track_manager

    @property
    def global_id_generator(self):
        """Get or create global ID generator."""
        if self._global_id_generator is None:
            # Lazy import to avoid circular dependency
            from pipeline.camera_engine import GlobalTrackIDGenerator
            self._global_id_generator = GlobalTrackIDGenerator(start_id=1)
            logger.debug("Global track ID generator enabled - track IDs will be unique across cameras")
        return self._global_id_generator

    def _create_face_recognizer(self) -> FaceRecognition:
        """Create and configure face recognizer."""
        args = type('Args', (), {})()
        args.client_slug = self.client_slug
        args.match_threshold = self.config.get('match_threshold', 0.3)
        args.logger = logger

        return FaceRecognition(args)

    def initialize_all(self) -> None:
        """Initialize all models upfront (optional optimization)."""
        logger.info("Initializing all detection models...")
        _ = self.face_detector
        _ = self.face_recognizer
        _ = self.person_detector
        _ = self.action_recognizer  # Initialize action recognizer
        _ = self.global_track_manager
        _ = self.global_id_generator
        logger.info("All models initialized")

    def reload_embeddings(self) -> None:
        """Reload face embeddings from database."""
        if self._face_recognizer is not None:
            self._face_recognizer.reload_embeddings()
            logger.info("Reloaded face embeddings")

    # ── Batch inference helpers (used by GPUInferenceWorker) ─────────────────

    def batch_detect_persons(self, frames: list) -> list:
        """Run YOLO on a batch of frames.

        Args:
            frames: List of BGR numpy arrays

        Returns:
            List of detection lists, one per frame.
            Each detection list contains dicts with bbox/confidence/keypoints.
        """
        if not frames:
            return []
        try:
            detector = self.person_detector
            results = detector.model(
                frames,
                conf=detector.confidence_threshold,
                iou=detector.iou_threshold,
                verbose=False,
                device=detector.device,
            )
            output = []
            for result in results:
                detections = []
                boxes = result.boxes
                if boxes is not None:
                    for idx in range(len(boxes)):
                        if int(boxes.cls[idx].cpu().numpy()) != 0:
                            continue
                        detections.append(
                            {
                                "bbox": boxes.xyxy[idx].cpu().numpy().tolist(),
                                "confidence": float(boxes.conf[idx].cpu().numpy()),
                                "keypoints": None,
                                "person_id": idx,
                            }
                        )
                output.append(detections)
            return output
        except Exception as e:
            logger.error(f"batch_detect_persons failed: {e}")
            return [[] for _ in frames]

    def batch_get_embeddings(self, face_crops: list) -> list:
        """Extract ArcFace embeddings from a list of face/person crops.

        Each crop is passed through face_detector.detect() to locate the face
        and obtain its embedding.  Returns None for crops where no face is found.

        Args:
            face_crops: List of BGR numpy arrays (person ROIs or face crops)

        Returns:
            List of numpy arrays (embeddings) or None, same length as face_crops.
        """
        embeddings = []
        detector = self.face_detector
        for crop in face_crops:
            if crop is None or crop.size == 0:
                embeddings.append(None)
                continue
            try:
                faces = detector.detect(crop)
                if faces:
                    embeddings.append(faces[0].embedding)
                else:
                    embeddings.append(None)
            except Exception as e:
                logger.debug(f"batch_get_embeddings: face detection error: {e}")
                embeddings.append(None)
        return embeddings

    def cleanup(self) -> None:
        """Clean up resources (stop workers, free GPU memory)."""
        logger.info("Cleaning up model factory resources...")

        # Stop action recognizer workers
        if self._action_recognizer is not None:
            self._action_recognizer.stop_workers()
            logger.info("Action recognizer workers stopped")

        logger.info("Model factory cleanup complete")
