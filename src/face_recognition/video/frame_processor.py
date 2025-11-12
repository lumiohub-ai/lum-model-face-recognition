"""Frame processor for handling post-detection frame annotation and logging.

This module handles frame annotation, visualization, and logging AFTER face detection
and tracking have completed. It does NOT contain hot-path code.
"""

import datetime
from typing import Dict, List, Any, Optional
import cv2
import numpy as np
import pytz
from loguru import logger


class FrameProcessor:
    """Handles frame annotation and recognition result processing.

    This class is responsible for:
    - Adding visualizations to frames (counting lines, timestamps)
    - Processing recognition results (logging, saving frames)
    - Coordinating with EntryLogger for visualization and API submission

    Note: This class does NOT handle face detection or tracking (hot-path code).
    """

    def __init__(self, entry_logger, timezone: str = "UTC"):
        """Initialize the frame processor.

        Args:
            entry_logger: EntryLogger instance for logging and API communication
            timezone: Timezone for timestamp display
        """
        self.entry_logger = entry_logger
        self.timezone = timezone

    def add_counting_line(
        self,
        frame: np.ndarray,
        line_points: Optional[List[tuple]]
    ) -> None:
        """Draw a counting line on the frame.

        Args:
            frame: Frame to draw on (modified in-place)
            line_points: List of two points [(x1, y1), (x2, y2)]
        """
        if line_points and len(line_points) == 2:
            cv2.line(frame, line_points[0], line_points[1], (0, 255, 0), 3)

    def add_timestamp(
        self,
        frame: np.ndarray,
        position: tuple = (10, 30),
        font_scale: float = 0.7,
        color: tuple = (255, 255, 255),
        thickness: int = 2
    ) -> None:
        """Add timestamp to the frame.

        Args:
            frame: Frame to annotate (modified in-place)
            position: (x, y) position for the timestamp
            font_scale: Font scale for the text
            color: RGB color tuple
            thickness: Text thickness
        """
        try:
            tz = pytz.timezone(self.timezone)
            current_time = datetime.datetime.now(tz)
            timestamp_str = current_time.strftime("%Y-%m-%d %H:%M:%S")

            cv2.putText(
                frame,
                timestamp_str,
                position,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                thickness
            )
        except Exception as e:
            logger.warning(f"Failed to add timestamp to frame: {e}")

    def process_recognition_results(
        self,
        persons_recognized: Dict[str, List],
        camera_type: str,
        camera_name: str,
        camera_id: int,
        save_recognized_callback: Optional[callable] = None,
        save_recognized_enabled: bool = True
    ) -> None:
        """Process recognition results and handle logging/saving.

        Args:
            persons_recognized: Dictionary of recognized persons
                Format: {name: [track_id, appear_time, status, image, recognition_info]}
            camera_type: Camera type (IN/OUT/MANAGEMENT)
            camera_name: Human-readable camera name
            camera_id: Camera identifier
            save_recognized_callback: Optional callback for saving recognized frames
            save_recognized_enabled: Whether saving is enabled
        """
        for name, (track_id, appear_time, recognized_status, image, recognition_info) in persons_recognized.items():
            if recognized_status == 'recognized':
                self._handle_recognized_person(
                    name,
                    appear_time,
                    image,
                    camera_type,
                    camera_name,
                    camera_id,
                    save_recognized_callback,
                    save_recognized_enabled
                )
            elif recognized_status == 'unrecognized':
                # Only send if image is valid (not None)
                if image is not None:
                    self._handle_unrecognized_person(image, camera_type)

    def _handle_recognized_person(
        self,
        name: str,
        appear_time: datetime.datetime,
        image: np.ndarray,
        camera_type: str,
        camera_name: str,
        camera_id: int,
        save_callback: Optional[callable],
        save_enabled: bool
    ) -> None:
        """Handle a recognized person.

        Args:
            name: Person's name
            appear_time: Time when person appeared
            image: Person's image
            camera_type: Camera type
            camera_name: Camera name
            camera_id: Camera ID
            save_callback: Callback function to save the frame
            save_enabled: Whether saving is enabled
        """
        recorded = self.entry_logger.log_person_entry(
            name,
            camera_type,
            appear_time,
            camera_name,
            camera_id
        )

        logger.info(
            f"Person recognized: {name}, recorded={recorded}, "
            f"save_enabled={save_enabled}"
        )

        if save_enabled and recorded and save_callback:
            logger.info(f"Saving recognized frame for {name}")
            try:
                save_callback(name, image, camera_type)
            except Exception as e:
                logger.error(f"Failed to save recognized frame for {name}: {e}")

    def _handle_unrecognized_person(
        self,
        image: np.ndarray,
        camera_type: str
    ) -> None:
        """Handle an unrecognized or partially matched person.

        Args:
            image: Person's image
            camera_type: Camera type
        """
        try:
            self.entry_logger.send_unrecognized_face(face=image, status=camera_type)
        except Exception as e:
            logger.error(f"Failed to send unrecognized face to API: {e}")

    def annotate_frame(
        self,
        frame: np.ndarray,
        line_points: Optional[List[tuple]] = None,
        add_timestamp: bool = True,
        add_entries: bool = True
    ) -> np.ndarray:
        """Add all annotations to a frame.

        Args:
            frame: Frame to annotate (modified in-place)
            line_points: Optional counting line points
            add_timestamp: Whether to add timestamp
            add_entries: Whether to add entry/exit visualization

        Returns:
            Annotated frame (same as input, modified in-place)
        """
        # Add counting line
        if line_points:
            self.add_counting_line(frame, line_points)

        # Add entry/exit visualization
        if add_entries:
            try:
                self.entry_logger.visualize_entries(frame)
            except Exception as e:
                logger.warning(f"Failed to visualize entries: {e}")

        # Add timestamp
        if add_timestamp:
            self.add_timestamp(frame)

        return frame

    def send_frame_to_dashboard(
        self,
        frame: np.ndarray,
        camera_type: str,
        camera_id: int
    ) -> None:
        """Send annotated frame to the dashboard via API.

        Args:
            frame: Annotated frame
            camera_type: Camera type
            camera_id: Camera ID
        """
        try:
            self.entry_logger.send_annotated_frame(frame, camera_type, camera_id)
        except Exception as e:
            logger.warning(f"Failed to send frame to dashboard: {e}")


def create_frame_processor(entry_logger, timezone: str = "UTC") -> FrameProcessor:
    """Factory function to create a FrameProcessor.

    Args:
        entry_logger: EntryLogger instance
        timezone: Timezone for timestamps

    Returns:
        Configured FrameProcessor instance

    Example:
        >>> processor = create_frame_processor(entry_logger, "Asia/Tashkent")
        >>> processor.annotate_frame(frame, line_points=[(100, 200), (300, 400)])
    """
    return FrameProcessor(entry_logger, timezone)
