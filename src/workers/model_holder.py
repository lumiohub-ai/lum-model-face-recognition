"""Per-process GPU model singletons for the inference Celery workers.

Each GPU worker process loads exactly one model family and keeps it for
the process lifetime — the YOLO worker loads `PersonDetector`,
the face worker loads `FaceDetector`, and neither loads the other's. Adapted
from the two proven benchmark holders (`benchmarks/celery_worker/` and
`benchmarks/celery_face/model_holder.py`), which validated this pattern
end-to-end before any of it reached production code.

Three constraints carried over verbatim, all of them subprocess-specific and
all of them observed rather than theoretical:

1. **Nothing here imports torch/ultralytics/insightface/onnxruntime at module
   scope.** The parent imports this module to register tasks; any CUDA touch
   at import time initialises CUDA in the parent and poisons `fork()` for a
   prefork pool.

2. **`device` is passed explicitly.** `PersonDetector`'s default path calls
   `torch.cuda.is_available()`, which initialises CUDA in whichever process
   reaches it first — same poisoning, harder to trace.

3. **Model paths are resolved to absolute.** `src/config/vision.py` sets
   `MODEL_CACHE_DIR = Path("volumes/models")`, a **CWD-relative** path. Today
   the workers happen to share the main process's CWD, so this would appear
   to work — but a worker started from anywhere else would silently look for
   weights in the wrong place, or re-download them. Resolving here removes
   the dependency on that coincidence.

Loading is double-checked-locked because the `solo` and `threads` pools do
not emit `worker_process_init`; every task therefore also calls
`ensure_*_loaded()` rather than relying on the signal alone.
"""

from __future__ import annotations

import threading
from typing import Any, Optional

from loguru import logger

_person_detector: Optional[Any] = None
_face_detector: Optional[Any] = None
_reid_extractor: Optional[Any] = None
_person_lock = threading.Lock()
_face_lock = threading.Lock()
_reid_lock = threading.Lock()


def _vision_config():
    """Build the same VisionConfig the main process uses, with
    `model_cache_dir` forced absolute (see constraint 3 in the module
    docstring)."""
    from pathlib import Path

    from config.startup import load_config
    from config.vision import MODEL_CACHE_DIR, build_vision_config

    config = load_config() or {}
    base = build_vision_config(config)

    absolute_cache_dir = Path(MODEL_CACHE_DIR).resolve()
    if base.model_cache_dir == absolute_cache_dir:
        return base

    # VisionConfig is a frozen dataclass; rebuild rather than mutate.
    import dataclasses

    return dataclasses.replace(base, model_cache_dir=absolute_cache_dir)


def ensure_person_detector_loaded():
    """Load YOLO once per process. Idempotent, thread-safe."""
    global _person_detector
    if _person_detector is not None:
        return _person_detector
    with _person_lock:
        if _person_detector is not None:
            return _person_detector

        from lum_vision.person_tracking.detector import PersonDetector

        cfg = _vision_config()
        logger.info(
            f"model_holder: loading PersonDetector "
            f"({cfg.person_detection_model}, size={cfg.person_model_size}, "
            f"conf={cfg.person_detection_threshold}, weights={cfg.weights_dir})"
        )
        _person_detector = PersonDetector(
            model_size=cfg.person_model_size,
            confidence_threshold=cfg.person_detection_threshold,
            use_pose=False,
            model_version=cfg.person_detection_model,
            weights_dir=cfg.weights_dir,
            # Explicit, never auto-detected — see constraint 2.
            device=_resolve_device(),
        )
        logger.info("model_holder: PersonDetector ready")
    return _person_detector


def ensure_face_detector_loaded():
    """Load SCRFD + ArcFace once per process. Idempotent, thread-safe.

    The benchmark verified one shared `FaceDetector` is safe across threads
    (identical results at 4 concurrent threads, cosine similarity 1.000000,
    2.07x throughput scaling) — unlike ultralytics' YOLO wrapper, there is no
    internal lock forcing serialisation, so no per-thread object is needed.
    """
    global _face_detector
    if _face_detector is not None:
        return _face_detector
    with _face_lock:
        if _face_detector is not None:
            return _face_detector

        from lum_vision.face_detection.detector import FaceDetector

        cfg = _vision_config()
        logger.info(
            f"model_holder: loading FaceDetector "
            f"({cfg.face_model_name}, root={cfg.insightface_dir})"
        )
        _face_detector = FaceDetector(
            gpu_id=cfg.gpu_id,
            model_name=cfg.face_model_name,
            padding_percent=cfg.face_detection_padding,
            model_root=cfg.insightface_dir,
            allowed_modules=cfg.face_modules,
        )
        logger.info("model_holder: FaceDetector ready")
    return _face_detector


def _body_reid_config() -> dict:
    """Read the `body_reid` section straight from configs/global_tracking.yaml,
    the same file and the same merge shape GlobalTrackManager itself reads
    (lum_vision.person_tracking.global_track.GlobalTrackManager._load_config:
    `yaml.safe_load(f)["global_tracking"]`). This worker never constructs a
    GlobalTrackManager - it only needs the one sub-section GlobalTrackManager
    would otherwise use to build its own default BodyReidExtractor - so this
    reads the file directly rather than instantiating anything from lum_vision
    just to reach one dict.

    Falls back to {} (so BodyReidExtractor's own defaults apply) if the
    app-level override file is absent, matching GlobalTrackManager's own
    fallback to its packaged config.
    """
    import yaml

    from config.vision import GLOBAL_TRACKING_CONFIG

    if not GLOBAL_TRACKING_CONFIG.exists():
        return {}
    with open(GLOBAL_TRACKING_CONFIG) as f:
        config = yaml.safe_load(f) or {}
    return config.get("global_tracking", {}).get("body_reid", {})


def ensure_reid_extractor_loaded():
    """Load the body ReID model once per process. Idempotent, thread-safe.

    Builds the exact BodyReidExtractor GlobalTrackManager would build itself
    by default (person_tracking/reid.py) - reid-worker just constructs it
    directly instead of indirectly through a GlobalTrackManager instance,
    since this process runs no matching logic, only extraction.
    """
    global _reid_extractor
    if _reid_extractor is not None:
        return _reid_extractor
    with _reid_lock:
        if _reid_extractor is not None:
            return _reid_extractor

        from lum_vision import BodyReidExtractor

        cfg = _vision_config()
        reid_cfg = _body_reid_config()
        logger.info(
            f"model_holder: loading BodyReidExtractor "
            f"({reid_cfg.get('model', 'osnet_x0_25_msmt17')}, "
            f"weights_dir={cfg.weights_dir})"
        )
        _reid_extractor = BodyReidExtractor(
            model_name=reid_cfg.get("model", "osnet_x0_25_msmt17"),
            weights_path=reid_cfg.get("weights_path"),
            # Explicit, never auto-detected inside BodyReidExtractor if this
            # is None — see constraint 2 above. A worker pinned to a second
            # GPU is told which one via SO_GPU_DEVICE, same as PersonDetector.
            device=reid_cfg.get("device") or _resolve_device(),
            half_precision=reid_cfg.get("half_precision", False),
            weights_dir=cfg.weights_dir,
        )
        logger.info("model_holder: BodyReidExtractor ready")
    return _reid_extractor


def _resolve_device() -> str:
    """Device for the YOLO model, from config/env rather than auto-detection.

    `SO_GPU_DEVICE` exists so a worker pinned to a second GPU (the reason
    this migration exists) can be told which one without a code change.
    """
    import os

    return os.environ.get("SO_GPU_DEVICE", "cuda")


def identity() -> dict:
    """Which process am I, and what is loaded — used to prove in a live run
    that each worker holds exactly one model family, and holds it once."""
    import os

    return {
        "pid": os.getpid(),
        "person_detector_id": id(_person_detector) if _person_detector else None,
        "face_detector_id": id(_face_detector) if _face_detector else None,
        "reid_extractor_id": id(_reid_extractor) if _reid_extractor else None,
    }
