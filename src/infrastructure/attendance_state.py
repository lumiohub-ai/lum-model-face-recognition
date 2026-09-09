"""Entry logging for tracking and visualizing person entries and exits."""

from collections import deque, OrderedDict
from datetime import datetime, timezone
from typing import Dict, Optional

import numpy as np
from loguru import logger

# Cap on the per-track attendance-dedup set (LSO-193). Track ids grow
# monotonically per camera, so the oldest entry is the longest-departed track
# and is safe to evict — this bounds memory without a track-removal hook.
_SENT_ATTENDANCE_CAP = 4096


class EntryLogger:
    """Logger for tracking and recording person entries and exits.

    Uses:
    - Repository: Direct database reads (users, status)
    - Celery tasks: Write operations (attendance, location, unrecognized faces)
    """

    def __init__(self, args, max_entries: int = 3):
        self.args = args
        self.client_slug = args.client_slug
        self.recent_entries = deque(maxlen=max_entries)
        self.max_track_lifetime_seconds = getattr(args, 'max_track_lifetime_seconds', 120)

        # Unrecognized-case gate (LSO-7): a case reaches the dashboard only if a
        # clear frontal, level face was seen. Below either threshold -> dropped
        # (logged, not uploaded/persisted).
        self.unrecognized_frontality_min = getattr(args, 'unrecognized_frontality_min', 0.6)
        self.unrecognized_pitch_min = getattr(args, 'unrecognized_pitch_min', 0.4)

        # Track last seen location (camera) for each person
        self.person_last_camera: Dict[str, str] = {}

        # Initialize repository for database reads
        from infrastructure.storage.backend_reader import Repository
        self.repository = Repository(self.client_slug)

        # Get user information from database
        self.current_users = args.db_names
        self.new_users, self.deleted_users, self.name_to_id = \
            self.repository.check_new_and_deleted_users(self.current_users)

        # LSO-193: attendance IN/OUT is deduped authoritatively in the DB
        # (record_attendance, FOR UPDATE) — the only state shared across the
        # split camera-workers. There is deliberately NO in-memory status cache
        # here: a per-worker cache drifts from the DB and from other workers and
        # silently drops transitions (and never sees manual edits). This set is
        # only a bounded per-track send throttle, not a status store.
        self._sent_attendance: "OrderedDict[tuple, None]" = OrderedDict()

    def prune_location_cache(self) -> None:
        """Drop `person_last_camera` entries for users no longer known.

        Called on an embedding/user reload. Preserves the one piece of cleanup
        the removed `reload_status()` used to do (LSO-193): without it,
        `person_last_camera` keeps stale camera names for deleted/renamed users
        for the life of the process. `name_to_id` is the current known set.
        """
        known = {e["name"] for e in self.name_to_id}
        stale = [n for n in self.person_last_camera if n not in known]
        for n in stale:
            del self.person_last_camera[n]
        if stale:
            logger.debug(f"Pruned {len(stale)} stale person_last_camera entries")

    def log_person_entry(
        self,
        name: str,
        status: str,
        appear_time: datetime,
        camera_name: str = "Unknown",
        camera_id: Optional[int] = None,
        proof_image: Optional[np.ndarray] = None,
        track_id: Optional[int] = None,
    ) -> bool:
        """Log a person's entry or exit.

        Emits at most one attendance event per (track, identity): one appearance
        sends once, not every recognition frame. Whether that event is a real
        IN/OUT transition or a duplicate is decided by the DB
        (`record_attendance`), the only guard shared across the split
        camera-workers — so a manual edit or another worker's write is always
        respected (LSO-193).
        """
        previous_camera = self.person_last_camera.get(name)
        recorded = False
        location_changed = previous_camera != camera_name

        # Send location data if location changed (production mode)
        if self.args.production and location_changed:
            self._send_location_data(name, status, appear_time, camera_name, camera_id)
            self.person_last_camera[name] = camera_name

        # Per-track send throttle (NOT a status cache). A track holds one
        # cam-direction status for its whole life, so keying on (track_id, name)
        # sends once per appearance; an identity correction mid-track re-sends
        # and the DB dedups it. track_id is None only on paths that don't carry
        # one — then always send and let the DB guard decide.
        if track_id is not None:
            key = (track_id, name)
            if key in self._sent_attendance:
                return recorded
            self._sent_attendance[key] = None
            if len(self._sent_attendance) > _SENT_ATTENDANCE_CAP:
                self._sent_attendance.popitem(last=False)

        recorded = True
        today_time = appear_time.strftime("%H:%M:%S")

        # Send attendance data (production mode). The DB suppresses duplicates.
        if self.args.production:
            self._send_attendance(name, status, camera_id, camera_name, proof_image)

        self._log_status_to_console(name, status, today_time)
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
            logger.exception(f"[Celery] Failed to queue attendance: {e}")

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
            logger.exception(f"[Celery] Failed to queue location: {e}")

    def send_unrecognized_face(
        self,
        face: np.ndarray,
        status: str,
        camera_id: Optional[int] = None,
        camera_name: Optional[str] = None,
        face_frontality: float = 0.0,
        face_pitch: float = 0.0,
        face_det_score: float = 0.0,
    ) -> bool:
        """Send unrecognized face via Celery task, gated on face orientation.

        Only a clear frontal, level face reaches the dashboard. Backs of heads,
        profiles, and looking-down faces score low and are dropped (logged as
        UNRECOGNIZED_DROPPED, no upload/persist). Returns False when dropped.
        """
        if face_frontality < self.unrecognized_frontality_min:
            logger.info(
                f"UNRECOGNIZED_DROPPED | camera={camera_id} "
                f"frontality={face_frontality:.3f} pitch={face_pitch:.3f} "
                f"det={face_det_score:.3f} (< {self.unrecognized_frontality_min} frontality)"
            )
            return False
        if face_pitch < self.unrecognized_pitch_min:
            logger.info(
                f"UNRECOGNIZED_DROPPED | camera={camera_id} "
                f"frontality={face_frontality:.3f} pitch={face_pitch:.3f} "
                f"det={face_det_score:.3f} (< {self.unrecognized_pitch_min} pitch)"
            )
            return False

        # Upload face image to GCS
        image_url = None
        try:
            from infrastructure.storage import ImageFetcher
            image_url = ImageFetcher().upload_image(face, "unrecognized_faces", self.client_slug)
            if image_url:
                logger.debug(f"Uploaded unrecognized face to GCS: {image_url}")
        except Exception as e:
            logger.exception(f"Error uploading unrecognized face: {e}")

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
            logger.exception(f"[Celery] Failed to queue unrecognized face: {e}")
            return False

