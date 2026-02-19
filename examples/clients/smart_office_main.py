"""Entry point for SmartOfficeEngine - Unified person tracking and face recognition.

This replaces the legacy HBFace with improved architecture:
- Person detection and tracking (YOLOv8-Pose + BoT-SORT)
- Face recognition within person ROIs
- Phone usage detection (optional, per camera)
- Attendance logging with temporal voting

Usage:
    python -m examples.clients.smart_office_main

Environment Variables Required:
    SA_EMAIL: API authentication email
    SA_PASSWORD: API authentication password
    HB_CLIENTSLUG: Organization slug
    API_HOST: API base URL
    USE_PGVECTOR: Enable pgvector for embeddings (default: true)
"""

import os
import sys
from pathlib import Path

import yaml
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from face_recognition import SmartOfficeEngine

# Load .env file only if it exists (for local development)
# Docker deployment uses environment variables from docker-compose
try:
    from dotenv import load_dotenv
    env_file = project_root / '.env'
    if env_file.exists():
        load_dotenv(env_file, override=True)
        logger.debug("Loaded .env file from {}", env_file)
except ImportError:
    pass  # dotenv not installed, use system environment variables


def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = project_root / "configs" / "smart_office.yaml"
    else:
        config_path = Path(config_path)

    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return {}

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO")
logger.remove()
logger.add(
    sys.stderr,
    level=log_level,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)
logger.add(
    "logs/smart_office_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="7 days",
    level=log_level
)


def main():
    """Main entry point for SmartOfficeEngine."""
    # Load configuration from YAML
    config = load_config(os.getenv("SMART_OFFICE_CONFIG"))

    # Validate required environment variables
    required_vars = ["SA_EMAIL", "SA_PASSWORD", "HB_CLIENTSLUG", "API_HOST"]
    missing = [var for var in required_vars if not os.getenv(var)]

    if missing:
        logger.error(f"Missing required environment variables: {missing}")
        sys.exit(1)

    # Log configuration
    logger.info("=" * 60)
    logger.info("SmartOfficeEngine Starting")
    logger.info("=" * 60)
    logger.info(f"Client: {os.getenv('HB_CLIENTSLUG')}")
    logger.info(f"API Host: {os.getenv('API_HOST')}")
    logger.info(f"pgvector: {os.getenv('USE_PGVECTOR', 'true')}")
    logger.info(f"Save Video: {config.get('save_video', False)}")
    logger.info(f"Output Dir: {config.get('output_dir', 'volumes/storage/person-tracking')}")

    # Initialize SmartOfficeEngine
    engine = SmartOfficeEngine(
        email=os.getenv("SA_EMAIL"),
        password=os.getenv("SA_PASSWORD"),
        client_slug=os.getenv("HB_CLIENTSLUG"),
        api_host=os.getenv("API_HOST"),
        # Fetch both FaceRecognision and PhoneUsageDetection cameras
        applications=['FaceRecognision', 'PhoneUsageDetection'],
        # Load settings from config file
        output_dir=config.get('output_dir', 'volumes/storage/person-tracking'),
        save_video=config.get('save_video', False),
        show=config.get('show', False)
    )

    # Run the engine
    engine.run()


if __name__ == "__main__":
    main()
