"""Model factory for initializing detection and recognition models.

This module centralizes the creation and configuration of all ML models
used in the SmartOffice system.
"""

from typing import Dict, Any, Optional

from loguru import logger
from config.settings import settings

from .detector import FaceDetector
from .recognizer import FaceRecognition
from domain.action_recognition import ActionRecognizer, PhoneDetector
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
        self._phone_detector: Optional[PhoneDetector] = None
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
    def phone_detector(self) -> PhoneDetector:
        """Get or create phone object detector (lazy initialization)."""
        if self._phone_detector is None:
            logger.debug("Initializing PhoneDetector...")
            action_config = self.config.get('action_recognition', {})
            self._phone_detector = PhoneDetector(
                model_path=None,        # auto-resolve from YOLO_CONFIG_DIR
                confidence_threshold=float(action_config.get('phone_confidence_threshold', 0.15)),
            )
        return self._phone_detector

    @property
    def action_recognizer(self) -> ActionRecognizer:
        """Get or create action recognizer (lazy initialization)."""
        if self._action_recognizer is None:
            action_config = self.config.get('action_recognition', {})
            enabled = action_config.get('enabled', False)
            ollama_api_url = settings.ollama_api_url
            model_name = settings.ollama_model

            actions = self.config.get('actions') or action_config.get('actions')

            # Debug crop directory — resolve to an absolute path so mkdir works
            # regardless of the container's working directory.
            debug_save_dir = None
            if enabled:
                import pathlib
                output_dir = self.config.get('output_dir', 'volumes/storage/videos')
                out_path = pathlib.Path(output_dir)
                if not out_path.is_absolute():
                    # Anchor to /app (container root) when relative path given
                    out_path = pathlib.Path('/app') / out_path
                debug_save_dir = str(out_path.parent / 'debug' / 'action_recognition')

            self._action_recognizer = ActionRecognizer(
                ollama_api_url=ollama_api_url,
                client_slug=self.client_slug,
                enabled=enabled,
                check_interval_seconds=action_config.get('check_interval_seconds', 8),
                max_queue_size=action_config.get('max_queue_size', 20),
                num_workers=action_config.get('async_workers', 2),
                model_name=model_name,
                inference_timeout=action_config.get('inference_timeout', 30),
                max_queue_delay_seconds=float(action_config.get('max_queue_delay_seconds', 12.0)),
                actions=actions,
                phone_detector=self.phone_detector if enabled else None,
                debug_save_dir=debug_save_dir,
                phone_precheck_interval_seconds=float(
                    action_config.get('phone_precheck_interval_seconds', 2.0)
                ),
                general_poll_interval_seconds=float(
                    action_config.get('general_poll_interval_seconds', 30.0)
                ),
                unknown_poll_interval_seconds=float(
                    action_config.get('unknown_poll_interval_seconds', 45.0)
                ),
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
        _ = self.phone_detector
        _ = self.action_recognizer
        _ = self.global_track_manager
        _ = self.global_id_generator
        logger.info("All models initialized")

    def reload_embeddings(self) -> None:
        """Reload face embeddings from database."""
        if self._face_recognizer is not None:
            self._face_recognizer.reload_embeddings()
            logger.info("Reloaded face embeddings")

    def cleanup(self) -> None:
        """Clean up resources (stop workers, free GPU memory)."""
        logger.info("Cleaning up model factory resources...")

        # Stop action recognizer workers
        if self._action_recognizer is not None:
            self._action_recognizer.stop_workers()
            logger.info("Action recognizer workers stopped")

        logger.info("Model factory cleanup complete")
