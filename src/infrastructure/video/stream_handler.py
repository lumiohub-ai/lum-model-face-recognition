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
        self.src = self.resolve_local_source(src)
        self.is_video = self.is_video_file(self.src)
        self.logger = logger

        # Suppress FFmpeg/libav C-level logs — they bypass Python logging and
        # spam stderr with HEVC ref errors, POC warnings, etc.
        os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
        os.environ['OPENCV_FFMPEG_LOGLEVEL'] = '-8'  # AV_LOG_QUIET

        # Configure RTSP options for better compatibility and smooth playback
        if isinstance(self.src, str) and self.src.startswith('rtsp://'):
            # Set FFmpeg options BEFORE creating VideoCapture
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = (
                'rtsp_transport;tcp|'        # Use TCP for reliability
                'buffer_size;1024000|'       # 1MB buffer for network stability
                'max_delay;500000|'          # Max 0.5s delay
                'fflags;nobuffer|'           # Minimize buffering for real-time
                'flags;low_delay'            # Low latency mode
            )
            self.cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
            # Set buffer size: 3 frames is optimal for real-time playback
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        else:
            self.cap = cv2.VideoCapture(self.src)
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
            self.logger.warning(f"Unable to read from source: {self.src}, will try to reconnect")
            self._reconnect()
            ret, frame = self.cap.read()
            if not ret:
                raise ValueError(f"Unable to read from source after initial reconnection attempts: {self.src}")

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
        """Attempt to reconnect to the video source until successful or stopped.

        Returns:
            True if reconnection was successful, False if shutdown was requested
        """
        self.logger.warning(f"Reconnecting to stream: {self.src}")
        if self.cap is not None:
            self.cap.release()

        current_delay = self.reconnect_delay
        attempt_count = 1

        while True:
            # Exit reconnection loop if shutdown was requested
            if self.stopped:
                self.logger.info(f"Reconnection aborted (shutdown requested): {self.src}")
                return False

            # Configure RTSP options for better compatibility
            if isinstance(self.src, str) and self.src.startswith('rtsp://'):
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

            # Release failed capture before next attempt
            self.cap.release()

            attempt_count += 1
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

    @staticmethod
    def resolve_local_source(source: Any) -> Any:
        """Resolve mounted local video files before opening them with OpenCV."""
        if not isinstance(source, str):
            return source

        stripped = source.strip()
        if not stripped or stripped.startswith(("rtsp://", "http://", "https://")):
            return source

        configured_media_dirs = os.getenv("SO_MEDIA_DIRS")
        if configured_media_dirs:
            media_dirs = [p for p in configured_media_dirs.split(os.pathsep) if p]
        else:
            media_dirs = [
                os.getenv("SO_RAW_MEDIA_DIR", "/app/raw_media"),
                os.getenv("SO_NEW_MEDIA_DIR", "/app/new_media"),
            ]
        basename = os.path.basename(stripped)
        aliases = {
            "fitting_room1.mp4": "fitting_room_1.mp4",
        }

        candidates = [stripped]
        if basename:
            for media_dir in media_dirs:
                candidates.append(os.path.join(media_dir, basename))
                alias = aliases.get(basename)
                if alias:
                    candidates.append(os.path.join(media_dir, alias))

        for candidate in candidates:
            if candidate and os.path.exists(candidate):
                return candidate

        return source

    def snapshot(self) -> Any:
        """Return a clean frame for configuration purposes (Virtual Line / ROI).

        Uses the latest frame already buffered by the pipeline, which is always
        clean because it was decoded sequentially. Opening a fresh VideoCapture
        at position 0 causes HEVC decode artifacts on files whose first frames
        are P/B-frames without a preceding IDR frame.
        """
        with self.lock:
            return self.latest_frame if self.latest_frame is not None else self.frame

    def start(self) -> "StreamHandler":
        """Start the frame reading thread for live streams (RTSP, webcam).

        Video files are read directly by the pipeline in read(); they do not need
        a background thread because the pipeline drives the frame rate.
        """
        if not self.is_video:
            self.thread = threading.Thread(target=self.update, daemon=True)
            self.thread.start()
        return self

    def update(self) -> None:
        """Background thread: continuously read frames from live streams.

        Reconnects on failure and skips the first 30 post-reconnect frames so
        that partially-decoded HEVC frames never land in latest_frame.
        """
        consecutive_failures = 0
        frames_to_skip = 0

        while True:
            with self.lock:
                if self.stopped:
                    break

            ret, frame = self.cap.read()
            if not ret:
                consecutive_failures += 1
                if consecutive_failures >= 3:
                    self.logger.warning(f"Stream timeout triggered. Attempting to reconnect...")
                    if self.stopped or not self._reconnect():
                        break
                    consecutive_failures = 0
                    # After reconnect the HEVC decoder has no reference frames yet.
                    # Skip the first 30 frames (~1 s at 30 fps) so partial /
                    # black-macroblock frames never land in latest_frame.
                    frames_to_skip = 30
                time.sleep(0.5)
                continue

            consecutive_failures = 0

            # Discard post-reconnect frames until the HEVC decoder has rebuilt its RPS
            if frames_to_skip > 0:
                frames_to_skip -= 1
                continue

            # LATEST FRAME ONLY: always overwrite with the newest frame
            with self.lock:
                self.latest_ret = ret
                self.latest_frame = frame

            current_time = time.time()
            if current_time - self.last_gc_time > self.gc_interval:
                gc.collect()
                self.last_gc_time = current_time

    def read(self) -> Tuple[bool, Any]:
        """Read the next frame from the video source."""
        if self.is_video:
            ret, frame = self.cap.read()
            if not ret and not self.stopped:
                self.logger.info(f"End of video stream reached, looping from start: {self.src}")
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                if not ret:
                    self.logger.warning(f"Unable to loop video stream: {self.src}")
                    self.stop()
                    return False, None

            if ret:
                with self.lock:
                    self.latest_ret = ret
                    self.latest_frame = frame

            return ret, frame

        # Live streams: return latest frame from background thread
        with self.lock:
            if self.latest_frame is not None:
                self.ret = self.latest_ret
                self.frame = self.latest_frame
            return self.ret, self.frame

    def stop(self) -> None:
        """Stop the frame reading thread and release resources."""
        with self.lock:
            if self.stopped:
                return
            self.stopped = True

        if self.thread is not None:
            self.thread.join(timeout=5)
            if self.thread.is_alive():
                self.logger.warning(f"Stream thread did not exit cleanly within 5s: {self.src}")
        self.cap.release()
