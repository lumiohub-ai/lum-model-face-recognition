"""
Entry point for SmartOfficeEngine - Unified person tracking and face recognition.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from face_recognition import (
    SmartOfficeEngine,
    init_smart_office_app,
    log_startup_info,
)


def main():
    """Main entry point for SmartOfficeEngine."""
    # Initialize application (loads .env, validates env vars, sets up logging, loads config)
    config = init_smart_office_app(
        env_file=project_root / '.env',
    )

    # Log startup information
    log_startup_info(
        client_slug=os.getenv('HB_CLIENTSLUG'),
        api_host=os.getenv('API_HOST'),
        pgvector=os.getenv('USE_PGVECTOR', 'true'),
    )

    # Initialize SmartOfficeEngine
    engine = SmartOfficeEngine(
        email=os.getenv("SA_EMAIL"),
        password=os.getenv("SA_PASSWORD"),
        client_slug=os.getenv("HB_CLIENTSLUG"),
        api_host=os.getenv("API_HOST"),
        applications=['attendance'],
        output_dir=config.get('output_dir', 'volumes/storage/person-tracking'),
        save_video=config.get('save_video', False),
        show=config.get('show', False),
        person_detection_threshold=config.get('person_detection_threshold', 0.5)
    )

    # Run the engine
    engine.run()


if __name__ == "__main__":
    main()
