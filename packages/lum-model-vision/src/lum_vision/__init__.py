"""lum-model-vision — face detection, person tracking, cross-camera ReID, action recognition.

The package holds inference only. It has no database, queue, or cloud-storage
dependency: known-face embeddings arrive through an
:class:`~lum_vision.ports.EmbeddingProvider`, and results are returned to the
caller rather than dispatched anywhere.

Invariants worth relying on: it spawns no threads, opens no sockets other than
to Ollama, and writes no files outside ``VisionConfig.model_cache_dir``.

    from lum_vision import ModelFactory, VisionConfig

    models = ModelFactory(VisionConfig(), embedding_provider=my_store)
    faces = models.face_detector.extract_face_features(frame)
"""

__version__ = "0.1.0"

from .action_recognition import ActionRecognizer, ActionResult
from .config import ActionConfig, VisionConfig
from .face_detection import FaceDetector, FaceMatcher
from .factory import ModelFactory
from .person_tracking import (
    GlobalTrack,
    GlobalTrackIDGenerator,
    GlobalTrackManager,
    IdentityManager,
    IDSwitchCorrector,
    PersonDetector,
    PersonStateManager,
    PersonTracker,
    PersonTrackManager,
    crop_person_roi,
)
from .ports import EmbeddingProvider, InMemoryEmbeddingProvider

__all__ = [
    "__version__",
    # Config
    "VisionConfig",
    "ActionConfig",
    # Ports
    "EmbeddingProvider",
    "InMemoryEmbeddingProvider",
    # Factory
    "ModelFactory",
    # Face
    "FaceDetector",
    "FaceMatcher",
    # Person
    "PersonDetector",
    "PersonTracker",
    "PersonTrackManager",
    "PersonStateManager",
    "IdentityManager",
    "IDSwitchCorrector",
    "GlobalTrackManager",
    "GlobalTrack",
    "GlobalTrackIDGenerator",
    "crop_person_roi",
    # Action
    "ActionRecognizer",
    "ActionResult",
]
