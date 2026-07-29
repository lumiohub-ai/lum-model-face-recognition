"""Typed configuration for the vision models.

Replaces the untyped ``Dict[str, Any]`` the old ModelFactory took, and removes
the package's dependency on the application's global ``settings`` object — the
caller is now responsible for supplying every value.
"""

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def _default_model_cache_dir() -> Path:
    """Root for every model weight this package downloads.

    ``LUM_VISION_MODEL_DIR`` overrides it; deployments normally set the field
    directly instead. Mounting this path is the caller's concern — the package
    only guarantees it writes nothing outside it.
    """
    env = os.getenv("LUM_VISION_MODEL_DIR", "").strip()
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cache" / "lum-vision"


@dataclass(frozen=True)
class ActionConfig:
    """Settings for Ollama-backed action recognition."""

    enabled: bool = False
    ollama_api_url: str = "http://localhost:11434"
    model_name: str = "gemma3:4b"
    check_interval_seconds: int = 30
    # A cold model load alone costs ~28s; see configs/config.yaml.
    inference_timeout: int = 60
    actions: Dict[str, Dict[str, str]] = field(default_factory=dict)

    @classmethod
    def from_dict(
        cls,
        cfg: Optional[Dict[str, Any]],
        *,
        ollama_api_url: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> "ActionConfig":
        """Build from an ``action_recognition:`` config.yaml block.

        ``ollama_api_url`` and ``model_name`` come from the caller's environment
        rather than the YAML, matching how the application sources them.
        """
        cfg = cfg or {}
        base = cls(
            enabled=bool(cfg.get("enabled", False)),
            check_interval_seconds=int(cfg.get("check_interval_seconds", 30)),
            inference_timeout=int(cfg.get("inference_timeout", 60)),
            actions=cfg.get("actions") or {},
        )
        overrides = {}
        if ollama_api_url:
            overrides["ollama_api_url"] = ollama_api_url
        if model_name:
            overrides["model_name"] = model_name
        return replace(base, **overrides) if overrides else base


@dataclass(frozen=True)
class VisionConfig:
    """Everything the models need to be constructed."""

    # Face
    match_threshold: float = 0.3
    face_model_name: str = "buffalo_l"
    face_detection_padding: float = 20.0
    gpu_id: int = 0
    # InsightFace tasks to load. None means FaceDetector.DEFAULT_MODULES
    # (detection + recognition); widen it if you need the landmark models.
    face_modules: Optional[Tuple[str, ...]] = None

    # Person detection
    person_detection_threshold: float = 0.5
    person_detection_model: str = "yolo26"  # 'yolo26' (NMS-free) or 'yolov8'
    person_model_size: str = "s"

    # Cross-camera ReID
    enable_global_tracking: bool = True
    global_tracking_config_path: Optional[Path] = None

    # Where every model weight is downloaded to and loaded from
    model_cache_dir: Path = field(default_factory=_default_model_cache_dir)

    action: ActionConfig = field(default_factory=ActionConfig)

    @property
    def weights_dir(self) -> Path:
        """Directory for YOLO and ReID weight files."""
        return self.model_cache_dir / "weights"

    @property
    def insightface_dir(self) -> Path:
        """Root InsightFace downloads its model zoo into."""
        return self.model_cache_dir / "insightface"

    @classmethod
    def from_dict(
        cls,
        cfg: Optional[Dict[str, Any]],
        *,
        ollama_api_url: Optional[str] = None,
        ollama_model: Optional[str] = None,
        model_cache_dir: Optional[Path] = None,
        global_tracking_config_path: Optional[Path] = None,
    ) -> "VisionConfig":
        """Build from an application config.yaml dict.

        Unknown keys are ignored, so the caller can pass its whole config
        mapping without filtering it first.
        """
        cfg = cfg or {}
        kwargs: Dict[str, Any] = {
            "match_threshold": float(cfg.get("match_threshold", 0.3)),
            "face_detection_padding": float(cfg.get("face_detection_padding", 20.0)),
            "person_detection_threshold": float(
                cfg.get("person_detection_threshold", 0.5)
            ),
            "person_detection_model": str(cfg.get("person_detection_model", "yolo26")),
            "enable_global_tracking": bool(cfg.get("enable_global_tracking", True)),
            "action": ActionConfig.from_dict(
                cfg.get("action_recognition"),
                ollama_api_url=ollama_api_url,
                model_name=ollama_model,
            ),
        }
        if model_cache_dir is not None:
            kwargs["model_cache_dir"] = Path(model_cache_dir)
        if global_tracking_config_path is not None:
            kwargs["global_tracking_config_path"] = Path(global_tracking_config_path)
        return cls(**kwargs)
