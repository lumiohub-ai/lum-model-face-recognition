from typing import Any, Tuple
import threading
import queue
import cv2

class StreamHandler:
    def __init__(self, src: Any) -> None:
        self.src = src
        self.is_video = self.is_video_file(src)
        self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            raise ValueError(f"Unable to open source: {src}")
        self.stopped = False
        self.lock = threading.Lock()
        self.frame_queue = queue.Queue(maxsize=1)  # Keep only the latest frame

        ret, frame = self.cap.read()
        if not ret:
            raise ValueError(f"Unable to read from source: {src}")
        self.ret = ret
        self.frame = frame
        self.last_frame = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) + 1)
        self.thread = None  # Store reference to thread

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
            if not self.frame_queue.full():
                self.frame_queue.put((ret, frame))

    def read(self) -> Tuple[bool, Any]:
        if self.is_video:
            ret, frame = self.cap.read()
            return ret, frame
        if not self.frame_queue.empty():
            self.ret, self.frame = self.frame_queue.get()
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
        cv2.destroyAllWindows()

    def __enter__(self) -> "StreamHandler":
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stop()