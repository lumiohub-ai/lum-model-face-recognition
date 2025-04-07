import cv2
import threading
import queue
from typing import Any, Tuple


class StreamHandler:
    def __init__(self, src: Any, buffer_size: int = 30) -> None:
        self.src = src
        self.is_video = self.is_video_file(src)
        self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            raise ValueError(f"Unable to open source: {src}")
        self.stopped = False
        self.lock = threading.Lock()
        self.buffer = queue.Queue(maxsize=buffer_size)
        self.buffer_size = buffer_size
        self.drop_count = 0  # Track dropped frames due to buffer overflow

        ret, frame = self.cap.read()
        if not ret:
            raise ValueError(f"Unable to read from source: {src}")
        self.ret = ret
        self.frame = frame
        self.last_frame = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) + 1)
        self.thread = None

    @staticmethod
    def is_video_file(source: str) -> bool:
        return isinstance(source, str) and source.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))

    def start(self) -> "StreamHandler":
        if not self.is_video:
            self.thread = threading.Thread(target=self.update, daemon=True)
            self.thread.start()
        return self

    def update(self) -> None:
        while True:
            with self.lock:
                if self.stopped:
                    break
            ret, frame = self.cap.read()
            if not ret:
                with self.lock:
                    self.stopped = True
                break
            try:
                if self.buffer.full():
                    _ = self.buffer.get_nowait()  # Drop oldest frame
                    self.drop_count += 1
                self.buffer.put_nowait((ret, frame))
            except queue.Full:
                pass

    def read(self) -> Tuple[bool, Any]:
        if self.is_video:
            return self.cap.read()
        try:
            self.ret, self.frame = self.buffer.get(timeout=1.0)
        except queue.Empty:
            print("[Warning] Frame buffer is empty. Using last known frame.")

        return self.ret, self.frame

    def _log_buffer_state(self) -> None:
        current_size = self.buffer.qsize()
        print(f"[Buffer] Size: {current_size}/{self.buffer_size} | Dropped Frames: {self.drop_count}")

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

    def __enter__(self) -> "StreamHandler":
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stop()
