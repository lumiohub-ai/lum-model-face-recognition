"""Video stream handling module for processing camera streams and video files."""

import os
import cv2
import threading
import time
import logging
import gc
from typing import Any, Tuple


class StreamHandler:
    """Handles video stream input from various sources with robust error handling and reconnection.

    This class manages video capture from files, cameras or network streams, providing a
    reliable and thread-safe interface for reading frames even with unreliable sources.
    """
    def __init__(self, src: Any, logger: logging.Logger) -> None:
        """Initialize the stream handler with a video source.

        Args:
            src: Video source (file path, camera index, or network URL)
            logger: Logger instance for reporting stream status
        """
        self.src = src
        self.is_video = self.is_video_file(src)
        self.logger = logger

        # Configure RTSP options for better compatibility and smooth playback
        if isinstance(src, str) and src.startswith('rtsp://'):
            # Set FFmpeg options BEFORE creating VideoCapture
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = (
                'rtsp_transport;tcp|'        # Use TCP for reliability
                'buffer_size;1024000|'       # 1MB buffer for network stability
                'max_delay;500000|'          # Max 0.5s delay
                'fflags;nobuffer|'           # Minimize buffering for real-time
                'flags;low_delay'            # Low latency mode
            )
            self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
            # Set buffer size: 3 frames is optimal for real-time playback
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        else:
            self.cap = cv2.VideoCapture(src)
        self.stopped = False
        self.lock = threading.Lock()
        # NO QUEUE - use latest frame only to prevent jitter and lag
        self.latest_frame = None
        self.latest_ret = False
        self.reconnect_delay = 1  # Initial delay between reconnection attempts
        self.max_delay = 30  # Maximum delay between reconnection attempts
        self.last_gc_time = time.time()
        self.gc_interval = 60  # Run garbage collection every 60 seconds

        ret, frame = self.cap.read()
        if not ret:
            self.logger.warning(f"Unable to read from source: {src}, will try to reconnect")
            self._reconnect()
            ret, frame = self.cap.read()
            if not ret:
                raise ValueError(f"Unable to read from source after initial reconnection attempts: {src}")

        self.ret = ret
        self.frame = frame

        if self.is_video:
            self.last_frame = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) + 1)
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        else:
            # For streams, these values might not be accurate
            self.last_frame = float('inf')
            self.fps = 30  # Default assumption

        self.thread = None  # Store reference to thread

    def _reconnect(self) -> bool:
        """Attempt to reconnect to the video source infinitely until successful.

        Returns:
            True if reconnection was successful
        """
        self.logger.warning(f"Reconnecting to stream: {self.src}")
        if self.cap is not None:
            self.cap.release()

        current_delay = self.reconnect_delay
        attempt_count = 1

        while True:  # Infinite reconnection loop
            # Configure RTSP options for better compatibility
            if isinstance(self.src, str) and self.src.startswith('rtsp://'):
                # Set same FFmpeg options as initial connection
                os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = (
                    'rtsp_transport;tcp|'
                    'buffer_size;1024000|'
                    'max_delay;500000|'
                    'fflags;nobuffer|'
                    'flags;low_delay'
                )
                self.cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
            else:
                self.cap = cv2.VideoCapture(self.src)
            ret, _ = self.cap.read()
            if ret:
                self.logger.warning(f"Successfully reconnected to stream: {self.src}")
                return True

            attempt_count += 1
            # Implement exponential backoff with a maximum delay
            current_delay = min(current_delay * 1.5, self.max_delay)
            time.sleep(current_delay)

    @staticmethod
    def is_video_file(source: str) -> bool:
        """Determine if the source is a video file based on its extension.

        Args:
            source: Path to the potential video file

        Returns:
            True if the source is a video file, False otherwise
        """
        return isinstance(source, str) and source.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))

    def start(self) -> "StreamHandler":
        """Start the frame reading thread for non-video file sources.

        Returns:
            Self reference for method chaining
        """
        if not self.is_video:
            self.thread = threading.Thread(target=self.update, daemon=True)
            self.thread.start()
        return self

    def update(self) -> None:
        """Background thread function that continuously reads frames from the video source.

        This method runs in a separate thread for live streams, continuously reading frames
        and updating the frame queue with the most recent frame.
        """
        consecutive_failures = 0
        while True:
            with self.lock:
                if self.stopped:
                    break

            ret, frame = self.cap.read()
            if not ret:
                consecutive_failures += 1
                if consecutive_failures >= 3:  # Try to reconnect after 3 consecutive failures
                    self.logger.warning(f"Stream timeout triggered. Attempting to reconnect...")
                    self._reconnect()
                    consecutive_failures = 0
                time.sleep(0.5)  # Short delay before retry
                continue

            # Reset failure counter on successful read
            consecutive_failures = 0

            # LATEST FRAME ONLY: Always overwrite with newest frame (no queue accumulation)
            # This prevents jitter by ensuring we never show old frames
            with self.lock:
                self.latest_ret = ret
                self.latest_frame = frame

            # Periodically run garbage collection
            current_time = time.time()
            if current_time - self.last_gc_time > self.gc_interval:
                gc.collect()
                self.last_gc_time = current_time

    def read(self) -> Tuple[bool, Any]:
        """Read the next frame from the video source.

        Returns:
            Tuple containing a boolean indicating success and the frame (if successful)
        """
        if self.is_video:
            ret, frame = self.cap.read()
            if not ret and not self.stopped:
                # For video files that reached the end, we can just stop the stream
                self.logger.warning(f"End of video stream reached: {self.src}")
                self.stop()
                return False, None

            return ret, frame

        # LATEST FRAME ONLY: Return the most recent frame from background thread
        # No queue, no old frames, no jitter
        with self.lock:
            if self.latest_frame is not None:
                self.ret = self.latest_ret
                self.frame = self.latest_frame
            return self.ret, self.frame

    def get_first_frame(self) -> Any:
        """Get the first frame that was captured from the video source.

        Returns:
            The first frame captured from the video source
        """
        return self.frame

    def stop(self) -> None:
        """Stop the frame reading thread and release resources."""
        with self.lock:
            if self.stopped:
                return
            self.stopped = True

        if self.thread is not None:
            self.thread.join()
        self.cap.release()
        # cv2.destroyAllWindows()

    def __enter__(self) -> "StreamHandler":
        """Context manager entry method.

        Returns:
            Started StreamHandler instance
        """
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Context manager exit method that ensures resources are properly released."""
        self.stop()