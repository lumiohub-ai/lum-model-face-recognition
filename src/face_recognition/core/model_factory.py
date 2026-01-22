"""Model factory for initializing detection and recognition models.

This module centralizes the creation and configuration of all ML models
used in the SmartOffice system.
"""

import os
from typing import Dict, Any, Optional

from loguru import logger

from .detector import FaceDetector
from .recognizer import FaceRecognition
from .action_recognizer import ActionRecognizer
from person_tracking.core.person_detector import PersonDetector
from person_tracking.core.global_track_manager import GlobalTrackManager


class ModelFactory:
    """Factory for creating detection and recognition models.

    Centralizes model initialization with consistent configuration,
    enabling GPU sharing and resource management.
    """

    def __init__(self, config: Dict[str, Any], client_slug: str, api_client=None):
        """Initialize model factory.

        Args:
            config: Configuration dictionary with model settings
            client_slug: Organization slug for embedding storage
            api_client: APIClient instance for backend communication
        """
        self.config = config
        self.client_slug = client_slug
        self.api_client = api_client
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
            logger.info("Initializing FaceDetector...")
            self._face_detector = FaceDetector(
                gpu_id=0,
                model_name='buffalo_l'
            )
        return self._face_detector

    @property
    def face_recognizer(self) -> FaceRecognition:
        """Get or create face recognizer (lazy initialization)."""
        if self._face_recognizer is None:
            logger.info("Initializing FaceRecognizer...")
            self._face_recognizer = self._create_face_recognizer()
        return self._face_recognizer

    @property
    def person_detector(self) -> PersonDetector:
        """Get or create person detector (lazy initialization)."""
        if self._person_detector is None:
            person_conf_threshold = self.config.get('person_detection_threshold', 0.5)
            model_version = self.config.get('person_detection_model', 'yolo26')
            logger.info(f"Initializing PersonDetector ({model_version}, threshold: {person_conf_threshold})...")
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
            ollama_api_url = action_config.get('ollama_api_url') or os.getenv('OLLAMA_API_URL')
            model_name = action_config.get('model_name') or os.getenv('OLLAMA_MODEL', 'gemma3:4b')

            logger.info(f"Initializing ActionRecognizer (enabled: {enabled}) | Ollama API: {ollama_api_url} | Model: {model_name}")
            self._action_recognizer = ActionRecognizer(
                ollama_api_url=ollama_api_url,
                api_client=self.api_client,
                enabled=enabled,
                check_interval_seconds=action_config.get('check_interval_seconds', 30),
                max_queue_size=action_config.get('max_queue_size', 50),
                num_workers=action_config.get('async_workers', 1),
                model_name=model_name
            )

            # Start worker threads if enabled
            if enabled:
                self._action_recognizer.start_workers()
                logger.info("Action recognition workers started")

        return self._action_recognizer

    @property
    def global_track_manager(self) -> GlobalTrackManager:
        """Get or create global track manager."""
        if self._global_track_manager is None:
            self._global_track_manager = GlobalTrackManager(app_config=self.config)
            if self._global_track_manager.enabled:
                logger.info("GlobalTrackManager enabled - collecting baseline metrics")
        return self._global_track_manager

    @property
    def global_id_generator(self):
        """Get or create global ID generator."""
        if self._global_id_generator is None:
            # Lazy import to avoid circular dependency
            from ..camera_engine import GlobalTrackIDGenerator
            self._global_id_generator = GlobalTrackIDGenerator(start_id=1)
            logger.info("Global track ID generator enabled - track IDs will be unique across cameras")
        return self._global_id_generator

    def _create_face_recognizer(self) -> FaceRecognition:
        """Create and configure face recognizer."""
        # Create args object for FaceRecognition
        args = type('Args', (), {})()

        # Use config setting with fallback to env var
        args.use_pgvector = self.config.get(
            'use_pgvector',
            os.getenv('USE_PGVECTOR', 'true').lower() == 'true'
        )
        args.client_slug = self.client_slug
        args.match_threshold = self.config.get('match_threshold', 0.3)
        args.logger = logger
        args.db_path = None

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

    def cleanup(self) -> None:
        """Clean up resources (stop workers, free GPU memory)."""
        logger.info("Cleaning up model factory resources...")

        # Stop action recognizer workers
        if self._action_recognizer is not None:
            self._action_recognizer.stop_workers()
            logger.info("Action recognizer workers stopped")

        logger.info("Model factory cleanup complete")
