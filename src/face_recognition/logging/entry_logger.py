"""Entry logging for tracking and visualizing person entries and exits."""

import os
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional

import cv2
import numpy as np
from loguru import logger

from .csv_logger import CSVLogger


class EntryLogger:
    """Logger for tracking and recording person entries and exits.

    This class coordinates:
    - Person status tracking (IN/OUT)
    - Communication with API via APIClient
    - CSV logging via CSVLogger
    - Visualization of recent entries
    """

    def __init__(
        self,
        args,
        max_entries: int = 3
    ):
        """Initialize the entry logger.

        Args:
            args: Configuration arguments
            max_entries: Maximum number of recent entries to display on screen
        """
        self.FR_SLUG = os.getenv("FR_SLUG")
        self.args = args
        self.client_slug = args.client_slug
        self.recent_entries = deque(maxlen=max_entries)
        self.max_track_lifetime_seconds = getattr(
            args, 'max_track_lifetime_seconds', 120
        )

        # Initialize CSV logger
        log_file_path = f'/app/volumes/storage/{self.FR_SLUG}/logs/{self.client_slug}/status_info.csv'
        rotation_period = getattr(self.args, 'csv_log_rotation', '2 weeks')
        self.csv_logger = CSVLogger(
            log_file_path=log_file_path,
            rotation_period=rotation_period,
            logger_instance=args.logger
        )

        # No API in R&D mode
        self.current_users = args.db_names
        self.new_users: List = []
        self.deleted_users: List = []
        self.name_to_id: List = {}

        self.person_status: Dict[str, str] = {}

    def log_person_entry(
        self,
        name: str,
        status: str,
        appear_time: datetime,
        camera_name: str = "Unknown",
        camera_id: Optional[int] = None,
        proof_image: Optional[np.ndarray] = None
    ) -> bool:
        """Log a person's entry or exit.

        Args:
            name: Name of the person
            status: Entry/exit status (IN/OUT)
            appear_time: Time when the person appeared
            camera_name: Name of the camera that detected the person
            camera_id: ID of the camera that detected the person
            proof_image: Optional recognized frame image (numpy array)

        Returns:
            bool: True if the status was recorded, False if unchanged
        """
        previous_status = self.person_status.get(name)
        recorded = False

        # If status is the same as before, do nothing else
        if previous_status == status.upper():
            timestamp = appear_time.strftime("%Y-%m-%d %H:%M:%S")
            logger.debug(
                f"[{timestamp}] {name} | Status: {status} | "
                f"Camera: {camera_name} (unchanged)"
            )
            return recorded

        recorded = True

        # Update the cached status
        self.person_status[name] = status.upper()

        # Format the appearance time
        today_date = appear_time.strftime("%Y-%m-%d")
        today_time = appear_time.strftime("%H:%M:%S")

        # Log status change with color coding for console
        self._log_status_to_console(name, status, today_time)

        # Log to CSV file
        self.csv_logger.log_status(name, status, today_time, today_date)

        # Add to recent entries for visualization
        self.recent_entries.appendleft(f"{name} - {status} @ {today_time}")

        return recorded

    def _log_status_to_console(self, name: str, status: str, time_str: str) -> None:
        """Log status change to console with color coding.

        Args:
            name: Name of the person
            status: Status (IN/OUT)
            time_str: Time string
        """
        # ANSI color codes for terminal output
        BOLD = "\033[1m"
        BLUE = "\033[94m"
        YELLOW = "\033[93m"
        RESET = "\033[0m"

        if status.upper() == "IN":
            logger.info(f"{BOLD}{BLUE}STATUS   | {name} {status.upper()} at {time_str}{RESET}")
        else:
            logger.info(f"{BOLD}{YELLOW}STATUS   | {name} {status.upper()} at {time_str}{RESET}")

    def visualize_entries(self, frame: np.ndarray, max_text_width: int = 0) -> None:
        """Visualize recent entries on the frame.

        Args:
            frame: Frame to add visualization to
            max_text_width: Maximum width of the text display
        """
        padding = 10
        base_y = 30

        # Calculate max text width
        for entry in self.recent_entries:
            text_size = cv2.getTextSize(entry, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            max_text_width = max(max_text_width, text_size[0])

        # Draw entry boxes
        for i, entry in enumerate(self.recent_entries):
            top_right_x = frame.shape[1] - max_text_width - padding * 2
            cv2.rectangle(
                frame,
                (top_right_x, base_y - 25 + i * 35),
                (frame.shape[1] - 10, base_y + i * 35 + 5),
                (0, 0, 0),
                -1,
            )
            cv2.putText(
                frame,
                entry,
                (top_right_x + 5, base_y + i * 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

    def save_status_info(self, video_name: str = 'status_info') -> str:
        """Get the path to the status information log file.

        Args:
            video_name: Base name for the output CSV file (kept for compatibility)

        Returns:
            Text message indicating where the status information was saved
        """
        log_path = self.csv_logger.get_log_path()
        return f"Status information has been saved continuously to {log_path}"
