"""Bridge between application config and the lum_vision package.

The models package takes a typed VisionConfig and knows nothing about this
application's settings object or its directory layout. This module is the one
place that translation happens.
"""

from pathlib import Path
from typing import Any, Dict, Optional

from lum_vision import VisionConfig

from .settings import settings

# Where model weights live. Mounted into the container by compose.yml — the
# models package only ever writes beneath whatever path it is given.
MODEL_CACHE_DIR = Path("volumes/models")

# Application-tuned ReID config; overrides the copy packaged with lum_vision.
GLOBAL_TRACKING_CONFIG = Path("configs/global_tracking.yaml")


def build_vision_config(config: Optional[Dict[str, Any]]) -> VisionConfig:
    """Build a VisionConfig from config.yaml plus environment settings.

    Args:
        config: Parsed configs/config.yaml (unknown keys are ignored)

    Returns:
        Config ready to hand to lum_vision.ModelFactory
    """
    return VisionConfig.from_dict(
        config,
        ollama_api_url=settings.ollama_api_url,
        ollama_model=settings.ollama_model,
        model_cache_dir=MODEL_CACHE_DIR,
        global_tracking_config_path=(
            GLOBAL_TRACKING_CONFIG if GLOBAL_TRACKING_CONFIG.exists() else None
        ),
    )
