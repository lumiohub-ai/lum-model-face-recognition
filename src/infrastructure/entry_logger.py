"""Entry logging for tracking and visualizing person entries and exits."""

import os
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger

from infrastructure.api import APIClient
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

        # Track last seen location (camera) for each person
        self.person_last_camera: Dict[str, str] = {}

        # Initialize API client
        api_host = os.getenv("API_HOST", getattr(args, 'api_host', "http://localhost:7091/"))
        self.api_client = APIClient(
            api_host=api_host,
            email=args.email,
            password=args.password,
            client_slug=self.client_slug
        )

        # Initialize CSV logger
        log_file_path = f'/app/volumes/storage/{self.FR_SLUG}/logs/{self.client_slug}/status_info.csv'
        rotation_period = getattr(self.args, 'csv_log_rotation', '2 weeks')
        self.csv_logger = CSVLogger(
            log_file_path=log_file_path,
            rotation_period=rotation_period,
            logger_instance=args.logger
        )

        # Get user information from API
        self.current_users = args.db_names
        self.new_users, self.deleted_users, self.name_to_id = self.api_client.get_all_users(
            self.current_users
        )

        # Get initial person status
        self.person_status = self._get_last_status()

    def _get_last_status(self) -> Dict[str, str]:
        """Get the last status of each user from the API.

        Returns:
            Dictionary mapping user names to their last status (IN/OUT)
        """
        # Fetch IN users
        in_names = self.api_client.get_users_by_status("in")

        # Fetch OUT users
        out_names = self.api_client.get_users_by_status("out")

        person_status = {}

        # Set OUT status first
        for name in out_names:
            person_status[name] = "OUT"

        # IN overrides OUT if there's any overlap
        for name in in_names:
            person_status[name] = "IN"

        # Handle users without status
        for entry in self.name_to_id:
            name = entry["name"]
            if name not in person_status:
                logger.warning(
                    f"No status from API for user '{name}', defaulting to OUT"
                )
                person_status[name] = "OUT"

        return person_status

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
            proof_image: Optional annotated frame with person bbox as proof

        Returns:
            bool: True if the status was recorded, False if unchanged
        """
        previous_status = self.person_status.get(name)
        previous_camera = self.person_last_camera.get(name)
        recorded = False
        location_changed = previous_camera != camera_name

        # Send location data to API if location (camera) changed (if in production mode)
        if self.args.production and location_changed:
            self._send_location_data(name, status, appear_time, camera_name, camera_id)
            self.person_last_camera[name] = camera_name

        # If status is the same as before, do nothing else
        if previous_status == status.upper():
            timestamp = appear_time.strftime("%Y-%m-%d %H:%M:%S")
            logger.debug(
                f"[{timestamp}] {name} | Status: {status} | "
                f"Camera: {camera_name} ({'location changed' if location_changed else 'unchanged'})"
            )
            return recorded

        recorded = True

        # Update the cached status
        self.person_status[name] = status.upper()

        # Format the appearance time
        today_date = appear_time.strftime("%Y-%m-%d")
        today_time = appear_time.strftime("%H:%M:%S")

        # Send attendance data to API if in production mode
        if self.args.production:
            self._send_data_to_api(name, status, camera_id, camera_name, proof_image)

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

    def _send_data_to_api(
        self,
        name: str,
        status: str,
        camera_id: Optional[int] = None,
        camera_name: Optional[str] = None,
        proof_image: Optional[np.ndarray] = None
    ) -> None:
        """Send person entry/exit data to the API via Celery task.

        Args:
            name: Name of the person
            status: Entry/exit status (IN/OUT)
            camera_id: ID of the camera that detected the person
            camera_name: Name of the camera that detected the person
            proof_image: Optional annotated frame with person bbox as proof
        """
        user_id = next(
            (int(i['id']) for i in self.name_to_id if i['name'] == name),
            None
        )

        if user_id is None:
            logger.warning(f'User with name {name} not found in the database')
            return

        response = self.api_client.create_attendance_record(
            user_id=user_id,
            status=status,
            camera_id=camera_id,
            camera_name=camera_name,
            user_name=name,
            proof_image=proof_image
        )
        if response is None:
            logger.warning(f"Failed to queue attendance record for {name}")

    def _send_location_data(
        self,
        name: str,
        status: str,
        appear_time: datetime,
        camera_name: str,
        camera_id: Optional[int] = None
    ) -> None:
        """Send user location data to the API via Celery task.

        Args:
            name: Name of the person
            status: Entry/exit status (IN/OUT)
            appear_time: Time when the person appeared
            camera_name: Name of the camera that detected the person
            camera_id: ID of the camera that detected the person
        """
        # Get user ID
        user_id = next(
            (int(i['id']) for i in self.name_to_id if i['name'] == name),
            None
        )

        # Generate timestamp in ISO 8601 format with milliseconds
        timestamp = appear_time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

        response = self.api_client.send_user_location(
            user_name=name,
            camera_name=camera_name,
            timestamp=timestamp,
            status=status,
            user_id=user_id,
            camera_id=camera_id
        )

        if response is None:
            logger.warning(f"Failed to queue location data for {name}")

    def send_unrecognized_face(
        self,
        face: np.ndarray,
        status: str,
        camera_id: Optional[int] = None,
        camera_name: Optional[str] = None
    ) -> Optional[Any]:
        """Send unrecognized face image to the API via Celery task.

        Uploads image to GCS first, then queues Celery task.

        Args:
            face: Detected face image (numpy array)
            status: Status of the user ('IN' or 'OUT')
            camera_id: ID of the camera that detected the face
            camera_name: Name of the camera that detected the face

        Returns:
            True if task was queued successfully, False otherwise
        """
        # Upload face image to GCS
        image_url = None
        try:
            from infrastructure.storage import upload_proof_image
            image_url = upload_proof_image(
                image=face,
                prefix="unrecognized_faces",
                client_slug=self.client_slug
            )
            if image_url:
                logger.info(f"Uploaded unrecognized face to GCS: {image_url}")
            else:
                logger.warning("Failed to upload unrecognized face to GCS")
        except Exception as e:
            logger.error(f"Error uploading unrecognized face: {e}")

        return self.api_client.send_unrecognized_face(
            face=face,
            status=status,
            camera_id=camera_id,
            camera_name=camera_name,
            image_url=image_url
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
