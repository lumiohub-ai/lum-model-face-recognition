"""
Logging Setup for Person Tracking.

Configures structured logging using Loguru for:
- Console output
- File logging
- Performance metrics
- Debug information
"""

import sys
from pathlib import Path
from loguru import logger


def setup_logging(
    level: str = "INFO",
    log_file: str = "logs/person_tracking.log",
    console: bool = True,
    file_logging: bool = True,
    rotation: str = "1 day",
    retention: str = "7 days"
) -> None:
    """
    Setup structured logging with loguru.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Path to log file
        console: Enable console logging
        file_logging: Enable file logging
        rotation: Log rotation policy
        retention: Log retention policy
    """
    # Remove default handler
    logger.remove()

    # Console logging
    if console:
        logger.add(
            sys.stderr,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                "<level>{message}</level>"
            ),
            level=level,
            colorize=True
        )

    # File logging
    if file_logging:
        # Create log directory
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            log_file,
            format=(
                "{time:YYYY-MM-DD HH:mm:ss} | "
                "{level: <8} | "
                "{name}:{function}:{line} | "
                "{message}"
            ),
            level=level,
            rotation=rotation,
            retention=retention,
            compression="zip"
        )

    logger.info(f"Logging configured: level={level}, console={console}, file={file_logging}")


def get_logger(name: str = "person_tracking"):
    """
    Get logger instance.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logger.bind(name=name)
