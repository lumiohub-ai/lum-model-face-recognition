"""Video stream handling module for processing camera streams and video files."""

import os
import cv2
import threading
import time
import logging
import gc
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple


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

        # Suppress FFmpeg/libav C-level logs — they bypass Python logging and
        # spam stderr with HEVC ref errors, POC warnings, etc.
        os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
        os.environ['OPENCV_FFMPEG_LOGLEVEL'] = '-8'  # AV_LOG_QUIET

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
        self.connected = False
        self.ret = False
        self.frame = None
        self.thread = None  # Store reference to thread
        self._last_frame_at: Optional[float] = None
        self._reconnecting = False
        # cap.read() = grab() + retrieve(). grab() blocks waiting for the next
        # frame off the network/demuxer (so it tracks stream cadence, not CPU
        # cost); retrieve() is the actual decode. Timed separately so "the
        # stream stalled" and "decode got slow" aren't the same number.
        # Kept local rather than pushed into a shared MetricsCollector, since
        # that's constructed after StreamManager in engine.py and this class
        # has no reason to depend on report timing.
        self._read_ms: deque = deque(maxlen=50)
        self._decode_ms: deque = deque(maxlen=50)

        ret, frame = self.cap.read()
        if not ret:
            if self.is_video:
                if self.cap is not None:
                    self.cap.release()
                    self.cap = None
                raise ValueError(f"Unable to read from source: {src}")

            self.logger.warning(
                f"Unable to read from source: {src} — "
                "will reconnect in background without blocking startup"
            )
            if self.cap is not None:
                self.cap.release()
            self.cap = None
        else:
            self.connected = True
            self.ret = ret
            self.frame = frame
            self._mark_frame_received()

        if self.is_video and self.connected:
            self.last_frame = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) + 1)
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        else:
            # For streams, these values might not be accurate
            self.last_frame = float('inf')
            self.fps = 30  # Default assumption

    def _mark_frame_received(self) -> None:
        """Record a successful frame read for health reporting."""
        self._last_frame_at = time.time()
        self._reconnecting = False

    def _iso_last_frame_at(self) -> Optional[str]:
        if self._last_frame_at is None:
            return None
        return (
            datetime.fromtimestamp(self._last_frame_at, tz=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )

    def frame_age_sec(self) -> Optional[float]:
        """Seconds since the last successful frame read, or None if never received."""
        if self._last_frame_at is None:
            return None
        return time.time() - self._last_frame_at

    def get_health(self, stale_sec: float = 15.0) -> Dict[str, Optional[str]]:
        """Return stream health (state / last_frame_at / last_error) for monitoring."""
        now = time.time()
        last_frame_at = self._iso_last_frame_at()
        has_recent_frame = (
            self._last_frame_at is not None
            and (now - self._last_frame_at) < stale_sec
        )

        if self.stopped:
            return {
                "state": "offline",
                "last_frame_at": last_frame_at,
                "last_error": "stream_stopped",
            }
        if has_recent_frame:
            return {
                "state": "streaming",
                "last_frame_at": last_frame_at,
                "last_error": None,
            }
        reconnect_grace_sec = 45.0
        if self._reconnecting or (not self.connected and not self.is_video):
            if (
                self._last_frame_at is None
                or (now - self._last_frame_at) > reconnect_grace_sec
            ):
                return {
                    "state": "offline",
                    "last_frame_at": last_frame_at,
                    "last_error": "no_recent_frames",
                }
            return {
                "state": "connecting",
                "last_frame_at": last_frame_at,
                "last_error": "reconnecting",
            }
        return {
            "state": "offline",
            "last_frame_at": last_frame_at,
            "last_error": "no_recent_frames",
        }

    def _reconnect(self) -> bool:
        """Attempt to reconnect to the video source until successful or stopped.

        Returns:
            True if reconnection was successful, False if shutdown was requested
        """
        self.connected = False
        self._reconnecting = True
        self.logger.warning(f"Reconnecting to stream: {self.src}")
        if self.cap is not None:
            self.cap.release()
            self.cap = None

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

            ret, frame = self.cap.read()
            if ret:
                self.connected = True
                self._mark_frame_received()
                with self.lock:
                    self.latest_ret = ret
                    self.latest_frame = frame
                self.logger.warning(f"Successfully reconnected to stream: {self.src}")
                return True

            # Release failed capture before next attempt
            self.cap.release()
            self.cap = None

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

    def start(self) -> "StreamHandler":
        """Start the frame reading thread for non-video file sources.

        Returns:
            Self reference for method chaining
        """
        if not self.is_video and self.thread is None:
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

            if self.cap is None or not self.connected:
                if self.stopped or not self._reconnect():
                    break
                consecutive_failures = 0
                continue

            _grab_start = time.monotonic()
            ret = self.cap.grab()
            read_ms = (time.monotonic() - _grab_start) * 1000.0
            if ret:
                _decode_start = time.monotonic()
                ret, frame = self.cap.retrieve()
                decode_ms = (time.monotonic() - _decode_start) * 1000.0
            else:
                frame = None
                decode_ms = 0.0
            if not ret:
                consecutive_failures += 1
                if consecutive_failures >= 3:
                    self.logger.warning(f"Stream timeout triggered. Attempting to reconnect...")
                    if self.stopped or not self._reconnect():
                        break
                    consecutive_failures = 0
                time.sleep(0.5)
                continue

            # Reset failure counter on successful read
            consecutive_failures = 0
            self._mark_frame_received()
            self._read_ms.append(read_ms)
            self._decode_ms.append(decode_ms)

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

    def get_read_avg_ms(self) -> float:
        """Rolling average cap.grab() duration in ms (last 50 successful
        reads) - time blocked waiting for the next frame off the
        network/demuxer. A spike here means the STREAM stalled, not that
        decode got slow. 0.0 if none recorded yet."""
        if not self._read_ms:
            return 0.0
        return sum(self._read_ms) / len(self._read_ms)

    def get_decode_avg_ms(self) -> float:
        """Rolling average cap.retrieve() duration in ms (last 50 successful
        reads) - actual CPU cost of decoding a grabbed frame. 0.0 if none
        recorded yet."""
        if not self._decode_ms:
            return 0.0
        return sum(self._decode_ms) / len(self._decode_ms)

    def read(self) -> Tuple[bool, Any]:
        """Read the next frame from the video source.

        Returns:
            Tuple containing a boolean indicating success and the frame (if successful)
        """
        if self.is_video:
            if self.cap is None or not self.connected:
                return False, None

            ret, frame = self.cap.read()
            if not ret and not self.stopped:
                # For video files that reached the end, we can just stop the stream
                self.logger.warning(f"End of video stream reached: {self.src}")
                self.stop()
                return False, None

            if ret:
                self._mark_frame_received()
            return ret, frame

        # LATEST FRAME ONLY: Return the most recent frame from background thread
        # No queue, no old frames, no jitter
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

        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=5)
            if self.thread.is_alive():
                self.logger.warning(
                    f"Stream thread did not exit cleanly within 5s: {self.src} — "
                    "skipping capture release to avoid race with background thread"
                )
                return

        with self.lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.connected = False