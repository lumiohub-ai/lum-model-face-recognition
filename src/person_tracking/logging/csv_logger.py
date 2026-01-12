"""
CSV Logger for Person Tracking Events.

Logs person tracking events to CSV files for:
- Offline analysis
- Auditing
- Compliance reporting
- Long-term storage
"""

import csv
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
from loguru import logger


class CSVLogger:
    """
    Logs person tracking events to CSV files.

    Creates timestamped CSV files with event data for each camera.
    Supports automatic file rotation and archiving.
    """

    def __init__(
        self,
        output_dir: str = "volumes/storage/person-tracking/logs",
        client_slug: str = "default",
        camera_id: int = 1
    ):
        """
        Initialize CSV Logger.

        Args:
            output_dir: Base output directory
            client_slug: Client organization slug
            camera_id: Camera identifier
        """
        self.output_dir = Path(output_dir)
        self.client_slug = client_slug
        self.camera_id = camera_id

        # Create output directory
        self.log_dir = self.output_dir / client_slug / f"camera_{camera_id}"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # CSV file paths
        self.events_file = self.log_dir / f"events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.summary_file = self.log_dir / "summary.csv"

        # CSV headers
        self.event_headers = [
            'timestamp',
            'track_id',
            'event_type',
            'person_name',
            'confidence',
            'duration_seconds',
            'camera_id'
        ]

        self.summary_headers = [
            'track_id',
            'person_name',
            'first_seen',
            'last_seen',
            'total_frames',
            'camera_id'
        ]

        # Initialize CSV files
        self._initialize_csv_files()

        # Stats
        self.events_logged = 0
        self.tracks_logged = set()

        logger.info(
            f"CSVLogger initialized: {self.log_dir} "
            f"(camera={camera_id}, client={client_slug})"
        )

    def _initialize_csv_files(self) -> None:
        """Initialize CSV files with headers."""
        # Events file
        if not self.events_file.exists():
            with open(self.events_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.event_headers)
                writer.writeheader()

        # Summary file (append mode, add header only if new)
        if not self.summary_file.exists():
            with open(self.summary_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.summary_headers)
                writer.writeheader()

    def log_event(
        self,
        track_id: int,
        event_type: str,
        person_name: Optional[str] = None,
        confidence: Optional[float] = None,
        duration: Optional[float] = None,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Log a person tracking event to CSV.

        Args:
            track_id: Track identifier
            event_type: Event type (e.g., 'identity_locked')
            person_name: Person name (if recognized)
            confidence: Confidence score
            duration: Duration in seconds
            timestamp: Event timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now()

        event_data = {
            'timestamp': timestamp.isoformat(),
            'track_id': track_id,
            'event_type': event_type,
            'person_name': person_name or '',
            'confidence': f"{confidence:.3f}" if confidence is not None else '',
            'duration_seconds': f"{duration:.2f}" if duration is not None else '',
            'camera_id': self.camera_id
        }

        # Write to CSV
        try:
            with open(self.events_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.event_headers)
                writer.writerow(event_data)

            self.events_logged += 1
            self.tracks_logged.add(track_id)

            logger.debug(
                f"Event logged: {event_type} | Track={track_id} | "
                f"Person={person_name or 'Unknown'}"
            )

        except Exception as e:
            logger.error(f"Failed to log event to CSV: {e}")

    def log_person_summary(
        self,
        track_id: int,
        person_name: Optional[str],
        first_seen: datetime,
        last_seen: datetime,
        total_frames: int
    ) -> None:
        """
        Log summary for a person (when they exit frame).

        Args:
            track_id: Track identifier
            person_name: Person name
            first_seen: First seen timestamp
            last_seen: Last seen timestamp
            total_frames: Total number of frames tracked
        """
        summary_data = {
            'track_id': track_id,
            'person_name': person_name or 'Unknown',
            'first_seen': first_seen.isoformat(),
            'last_seen': last_seen.isoformat(),
            'total_frames': total_frames,
            'camera_id': self.camera_id
        }

        try:
            with open(self.summary_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.summary_headers)
                writer.writerow(summary_data)

            logger.debug(
                f"Summary logged: Track={track_id} | "
                f"Person={person_name or 'Unknown'} | "
                f"Frames={total_frames}"
            )

        except Exception as e:
            logger.error(f"Failed to log summary to CSV: {e}")

    def log_batch_events(self, events: List[Dict[str, Any]]) -> None:
        """
        Log multiple events at once.

        Args:
            events: List of event dictionaries
        """
        for event in events:
            self.log_event(
                track_id=event.get('track_id'),
                event_type=event.get('event_type'),
                person_name=event.get('person_name'),
                confidence=event.get('confidence'),
                duration=event.get('duration'),
                timestamp=event.get('timestamp')
            )

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get CSV logger statistics.

        Returns:
            Dictionary with stats
        """
        return {
            'events_logged': self.events_logged,
            'unique_tracks': len(self.tracks_logged),
            'events_file': str(self.events_file),
            'summary_file': str(self.summary_file),
            'camera_id': self.camera_id,
            'client_slug': self.client_slug
        }

    def rotate_log_file(self) -> None:
        """
        Rotate log file (create new file with new timestamp).

        Call this periodically or when file size exceeds threshold.
        """
        # Create new events file
        self.events_file = self.log_dir / f"events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        # Initialize with headers
        with open(self.events_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.event_headers)
            writer.writeheader()

        logger.info(f"Log file rotated: {self.events_file}")

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"CSVLogger(camera={self.camera_id}, "
            f"events={self.events_logged}, "
            f"tracks={len(self.tracks_logged)})"
        )
