"""CSV logging for person status information."""

import os
from typing import Any, Optional
from loguru import logger


class CSVLogger:
    """Handles CSV logging for person entry/exit status.

    This class manages CSV file creation, rotation, and logging of
    person status information.
    """

    def __init__(
        self,
        log_file_path: str,
        rotation_period: str = '2 weeks',
        logger_instance: Optional[Any] = None
    ):
        """Initialize the CSV logger.

        Args:
            log_file_path: Path to the CSV log file
            rotation_period: Rotation period for log files (e.g., '2 weeks', '1 month')
            logger_instance: Loguru logger instance to use
        """
        self.log_file_path = log_file_path
        self.rotation_period = rotation_period
        self.logger = logger_instance or logger

        # Ensure directory exists
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

        # Configure CSV-specific logger sink
        self._setup_csv_sink()

        # Write header if file is new or empty
        if not os.path.exists(log_file_path) or os.path.getsize(log_file_path) == 0:
            self._write_header()

    def _setup_csv_sink(self) -> None:
        """Set up the CSV-specific logging sink."""
        # Filter to ensure only CSV-intended messages go to this file
        csv_filter = lambda record: record["extra"].get("is_csv", False)

        self.logger.add(
            self.log_file_path,
            rotation=self.rotation_period,
            format="{message}",
            level="INFO",
            filter=csv_filter,
            encoding="utf-8"
        )

    def _write_header(self) -> None:
        """Write CSV header to the file."""
        self.logger.bind(is_csv=True).info("name,status,time,date")

    def log_status(self, name: str, status: str, time_str: str, date_str: str) -> None:
        """Log a person's status to the CSV file.

        Args:
            name: Name of the person
            status: Status (IN/OUT)
            time_str: Time string (HH:MM:SS)
            date_str: Date string (YYYY-MM-DD)
        """
        csv_message = f'{name},{status},{time_str},{date_str}'
        self.logger.bind(is_csv=True).info(csv_message)

    def get_log_path(self) -> str:
        """Get the path to the CSV log file.

        Returns:
            Path to the CSV log file
        """
        return self.log_file_path
