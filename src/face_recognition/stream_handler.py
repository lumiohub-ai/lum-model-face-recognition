import cv2
import threading
import queue
import time
import logging
from typing import Any, Tuple


class StreamHandler:
    def __init__(self, src: Any, logger: logging.Logger) -> None:
        self.src = src
        self.is_video = self.is_video_file(src)
        self.logger = logger
        self.cap = cv2.VideoCapture(src)
        self.stopped = False
        self.lock = threading.Lock()
        self.frame_queue = queue.Queue(maxsize=1)  # Keep only the latest frame
        self.reconnect_delay = 1  # Initial delay between reconnection attempts
        self.max_delay = 30  # Maximum delay between reconnection attempts

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
        """Attempt to reconnect to the video source infinitely until successful"""
        if self.cap is not None:
            self.cap.release()
        
        current_delay = self.reconnect_delay
        attempt_count = 1
        
        while True:  # Infinite reconnection loop
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
        return isinstance(source, str) and source.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))

    def start(self) -> "StreamHandler":
        if not self.is_video:
            self.thread = threading.Thread(target=self.update, daemon=True)
            self.thread.start()
        return self

    def update(self) -> None:
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
            
            if not self.frame_queue.full():
                self.frame_queue.put((ret, frame))
            else:
                # Get rid of old frame
                try:
                    self.frame_queue.get_nowait()
                    self.frame_queue.put((ret, frame))
                except queue.Empty:
                    pass

    def read(self) -> Tuple[bool, Any]:
        if self.is_video:
            ret, frame = self.cap.read()
            if not ret and not self.stopped:
                # For video files that reached the end, we can try to restart
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
            return ret, frame
        
        try:
            if not self.frame_queue.empty():
                self.ret, self.frame = self.frame_queue.get(timeout=0.5)
            # If queue is empty but we have a last valid frame, return it
            return self.ret, self.frame
        except queue.Empty:
            # Queue is empty and no frame was received in time
            if not self.stopped:
                self.logger.warning("No frame available in queue")
            return self.ret, self.frame
        
    def get_first_frame(self) -> Any:
        return self.frame

    def stop(self) -> None:
        with self.lock:
            if self.stopped:
                return
            self.stopped = True
        if self.thread is not None:
            self.thread.join()
        self.cap.release()
        # cv2.destroyAllWindows()

    def __enter__(self) -> "StreamHandler":
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stop()