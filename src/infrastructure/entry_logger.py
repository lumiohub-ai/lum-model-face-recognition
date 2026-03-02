"""Entry logging for tracking and visualizing person entries and exits."""

from collections import deque
from datetime import datetime, timezone
from typing import Dict, Optional

import numpy as np
from loguru import logger

from .csv_logger import CSVLogger


class EntryLogger:
    """Logger for tracking and recording person entries and exits.

    Uses:
    - Repository: Direct database reads (users, status)
    - Celery tasks: Write operations (attendance, location, unrecognized faces)
    - CSVLogger: Local CSV logging
    """

    def __init__(self, args, max_entries: int = 3):
        self.args = args
        self.client_slug = args.client_slug
        self.recent_entries = deque(maxlen=max_entries)
        self.max_track_lifetime_seconds = getattr(args, 'max_track_lifetime_seconds', 120)

        # Track last seen location (camera) for each person
        self.person_last_camera: Dict[str, str] = {}

        # Initialize repository for database reads
        from infrastructure.storage.repository import Repository
        self.repository = Repository(self.client_slug)

        # Initialize CSV logger
        log_file_path = f'/app/volumes/storage/person-tracking/logs/{self.client_slug}/status_info.csv'
        rotation_period = getattr(self.args, 'csv_log_rotation', '2 weeks')
        self.csv_logger = CSVLogger(
            log_file_path=log_file_path,
            rotation_period=rotation_period,
            logger_instance=args.logger
        )

        # Get user information from database
        self.current_users = args.db_names
        self.new_users, self.deleted_users, self.name_to_id = \
            self.repository.check_new_and_deleted_users(self.current_users)

        # Get initial person status
        self.person_status = self._get_last_status()

    def _get_last_status(self) -> Dict[str, str]:
        """Get the last status of each user from the database."""
        in_names = self.repository.get_users_by_status("in")
        out_names = self.repository.get_users_by_status("out")

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
                logger.warning(f"No status for user '{name}', defaulting to OUT")
                person_status[name] = "OUT"

        return person_status

    def reload_status(self) -> None:
        """Reload person status from the database."""
        logger.info("Reloading person status from database...")
        old_status = self.person_status.copy()
        self.person_status = self._get_last_status()

        # Log changes
        for name, new_status in self.person_status.items():
            old = old_status.get(name)
            if old != new_status:
                logger.info(f"Status updated for {name}: {old} -> {new_status}")

    def log_person_entry(
        self,
        name: str,
        status: str,
        appear_time: datetime,
        camera_name: str = "Unknown",
        camera_id: Optional[int] = None,
        proof_image: Optional[np.ndarray] = None
    ) -> bool:
        """Log a person's entry or exit."""
        previous_status = self.person_status.get(name)
        previous_camera = self.person_last_camera.get(name)
        recorded = False
        location_changed = previous_camera != camera_name

        # Send location data if location changed (production mode)
        if self.args.production and location_changed:
            self._send_location_data(name, status, appear_time, camera_name, camera_id)
            self.person_last_camera[name] = camera_name

        # If status unchanged, do nothing else
        if previous_status == status.upper():
            return recorded

        recorded = True
        self.person_status[name] = status.upper()

        today_date = appear_time.strftime("%Y-%m-%d")
        today_time = appear_time.strftime("%H:%M:%S")

        # Send attendance data (production mode)
        if self.args.production:
            self._send_attendance(name, status, camera_id, camera_name, proof_image)

        # Log to console and CSV
        self._log_status_to_console(name, status, today_time)
        self.csv_logger.log_status(name, status, today_time, today_date)
        self.recent_entries.appendleft(f"{name} - {status} @ {today_time}")

        return recorded

    def _log_status_to_console(self, name: str, status: str, time_str: str) -> None:
        """Log status change to console with color coding."""
        BOLD = "\033[1m"
        BLUE = "\033[94m"
        YELLOW = "\033[93m"
        RESET = "\033[0m"

        if status.upper() == "IN":
            logger.info(f"{BOLD}{BLUE}STATUS   | {name} {status.upper()} at {time_str}{RESET}")
        else:
            logger.info(f"{BOLD}{YELLOW}STATUS   | {name} {status.upper()} at {time_str}{RESET}")

    def _send_attendance(
        self,
        name: str,
        status: str,
        camera_id: Optional[int],
        camera_name: Optional[str],
        proof_image: Optional[np.ndarray]
    ) -> None:
        """Send attendance record via Celery task."""
        user_id = next((int(i['id']) for i in self.name_to_id if i['name'] == name), None)
        if user_id is None:
            logger.warning(f'User {name} not found in database')
            return

        # Upload proof image to GCS if provided
        proof_image_url = None
        if proof_image is not None:
            try:
                from infrastructure.storage import ImageFetcher
                proof_image_url = ImageFetcher().upload_image(proof_image, "attendance_proofs", self.client_slug)
            except Exception as e:
                logger.warning(f"Failed to upload proof image: {e}")

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

        try:
            from workers.detection_tasks import task_record_attendance
            task_record_attendance.delay(
                client_slug=self.client_slug,
                user_id=user_id,
                user_name=name,
                status=status,
                camera_id=camera_id,
                camera_name=camera_name,
                proof_image_url=proof_image_url,
                recorded_at=timestamp
            )
            logger.debug(f"[Celery] Queued attendance: {name} {status}")
        except Exception as e:
            logger.error(f"[Celery] Failed to queue attendance: {e}")

    def _send_location_data(
        self,
        name: str,
        status: str,
        appear_time: datetime,
        camera_name: str,
        camera_id: Optional[int]
    ) -> None:
        """Send user location via Celery task."""
        user_id = next((int(i['id']) for i in self.name_to_id if i['name'] == name), None)
        timestamp = appear_time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

        try:
            from workers.detection_tasks import task_update_user_location
            task_update_user_location.delay(
                client_slug=self.client_slug,
                user_id=user_id,
                user_name=name,
                camera_id=camera_id or 0,
                camera_name=camera_name,
                status=status,
                updated_at=timestamp
            )
            logger.debug(f"[Celery] Queued location: {name} at {camera_name}")
        except Exception as e:
            logger.error(f"[Celery] Failed to queue location: {e}")

    def send_unrecognized_face(
        self,
        face: np.ndarray,
        status: str,
        camera_id: Optional[int] = None,
        camera_name: Optional[str] = None
    ) -> bool:
        """Send unrecognized face via Celery task."""
        # Upload face image to GCS
        image_url = None
        try:
            from infrastructure.storage import ImageFetcher
            image_url = ImageFetcher().upload_image(face, "unrecognized_faces", self.client_slug)
            if image_url:
                logger.info(f"Uploaded unrecognized face to GCS: {image_url}")
        except Exception as e:
            logger.error(f"Error uploading unrecognized face: {e}")

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

        try:
            from workers.detection_tasks import task_save_unrecognized_face
            task_save_unrecognized_face.delay(
                client_slug=self.client_slug,
                camera_id=camera_id,
                camera_name=camera_name,
                status=status,
                image_url=image_url,
                detected_at=timestamp
            )
            logger.debug(f"[Celery] Queued unrecognized face from camera {camera_id}")
            return True
        except Exception as e:
            logger.error(f"[Celery] Failed to queue unrecognized face: {e}")
            return False

    def save_status_info(self) -> str:
        """Get the path to the status information log file."""
        return f"Status information saved to {self.csv_logger.get_log_path()}"
