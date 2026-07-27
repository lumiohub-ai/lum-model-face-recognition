"""Model factory for initializing detection and recognition models.

Centralizes the creation and configuration of every ML model in the package,
with lazy initialization so unused models never touch the GPU.
"""

from typing import Optional

from loguru import logger

from .action_recognition import ActionRecognizer
from .config import VisionConfig
from .face_detection.detector import FaceDetector
from .face_detection.matcher import FaceMatcher
from .person_tracking.detector import PersonDetector
from .person_tracking.global_track import GlobalTrackManager
from .person_tracking.ids import GlobalTrackIDGenerator
from .ports import EmbeddingProvider


class ModelFactory:
    """Factory for creating detection and recognition models.

    Every model is built on first access and cached, so constructing the factory
    is cheap and callers only pay for what they use.
    """

    def __init__(
        self,
        config: VisionConfig,
        embedding_provider: Optional[EmbeddingProvider] = None,
    ):
        """Initialize model factory.

        Args:
            config: Model settings
            embedding_provider: Source of known-face embeddings. Required only
                                if ``face_matcher`` is accessed.
        """
        self.config = config
        self.embedding_provider = embedding_provider

        self._face_detector: Optional[FaceDetector] = None
        self._face_matcher: Optional[FaceMatcher] = None
        self._person_detector: Optional[PersonDetector] = None
        self._action_recognizer: Optional[ActionRecognizer] = None
        self._global_track_manager: Optional[GlobalTrackManager] = None
        self._global_id_generator: Optional[GlobalTrackIDGenerator] = None

    @property
    def face_detector(self) -> FaceDetector:
        """Get or create face detector (lazy initialization)."""
        if self._face_detector is None:
            logger.debug("Initializing FaceDetector...")
            self._face_detector = FaceDetector(
                gpu_id=self.config.gpu_id,
                model_name=self.config.face_model_name,
                padding_percent=self.config.face_detection_padding,
                model_root=self.config.insightface_dir,
            )
        return self._face_detector

    @property
    def face_matcher(self) -> FaceMatcher:
        """Get or create face matcher (lazy initialization).

        Raises:
            ValueError: If no embedding provider was supplied
        """
        if self._face_matcher is None:
            if self.embedding_provider is None:
                raise ValueError(
                    "face_matcher requires an embedding_provider; pass one to ModelFactory()"
                )
            logger.debug("Initializing FaceMatcher...")
            self._face_matcher = FaceMatcher(
                provider=self.embedding_provider,
                match_threshold=self.config.match_threshold,
            )
        return self._face_matcher

    @property
    def person_detector(self) -> PersonDetector:
        """Get or create person detector (lazy initialization)."""
        if self._person_detector is None:
            logger.debug(
                f"Initializing PersonDetector ({self.config.person_detection_model}, "
                f"threshold: {self.config.person_detection_threshold})..."
            )
            self._person_detector = PersonDetector(
                model_size=self.config.person_model_size,
                confidence_threshold=self.config.person_detection_threshold,
                use_pose=False,
                model_version=self.config.person_detection_model,
                weights_dir=self.config.weights_dir,
            )
        return self._person_detector

    @property
    def action_recognizer(self) -> ActionRecognizer:
        """Get or create action recognizer (lazy initialization).

        Returns a synchronous recognizer — no workers are started. Drive it from
        your own thread pool.
        """
        if self._action_recognizer is None:
            self._action_recognizer = ActionRecognizer(self.config.action)
        return self._action_recognizer

    @property
    def global_track_manager(self) -> GlobalTrackManager:
        """Get or create global track manager."""
        if self._global_track_manager is None:
            config_path = self.config.global_tracking_config_path
            self._global_track_manager = GlobalTrackManager(
                config_path=str(config_path) if config_path else None,
                enabled=self.config.enable_global_tracking,
                weights_dir=self.config.weights_dir,
            )
            if self._global_track_manager.enabled:
                logger.debug("GlobalTrackManager enabled - collecting baseline metrics")
        return self._global_track_manager

    @property
    def global_id_generator(self) -> GlobalTrackIDGenerator:
        """Get or create global ID generator."""
        if self._global_id_generator is None:
            self._global_id_generator = GlobalTrackIDGenerator(start_id=1)
            logger.debug(
                "Global track ID generator enabled - track IDs will be unique across cameras"
            )
        return self._global_id_generator

    def initialize_all(self) -> None:
        """Initialize all models upfront (optional optimization).

        Skips the face matcher when no embedding provider was supplied.
        """
        logger.info("Initializing all detection models...")
        _ = self.face_detector
        if self.embedding_provider is not None:
            _ = self.face_matcher
        _ = self.person_detector
        _ = self.action_recognizer
        _ = self.global_track_manager
        _ = self.global_id_generator
        logger.info("All models initialized")

    def reload_embeddings(self) -> None:
        """Reload face embeddings from the provider."""
        if self._face_matcher is not None:
            self._face_matcher.reload_embeddings()
            logger.info("Reloaded face embeddings")

    def cleanup(self) -> None:
        """Release resources.

        Currently a no-op: this package owns no threads or sockets. Kept so
        callers can wire shutdown handling once and not revisit it.
        """
        logger.debug("Model factory cleanup complete")
