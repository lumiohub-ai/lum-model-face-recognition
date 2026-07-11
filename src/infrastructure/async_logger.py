"""Async logger — non-blocking I/O via background worker threads.

Three workers handle DB attendance, GCS image uploads, and Redis stream events
so camera threads are never blocked by I/O.
"""

import queue
import threading
from typing import Optional

import numpy as np
from loguru import logger


class AsyncLogger:
    """Non-blocking logger with background db / gcs / redis workers.

    Public API (called from camera threads — returns immediately):
        log_entry(entry)      → queues attendance event for DB
        upload_image(data)    → queues image for GCS upload
        publish_event(event)  → queues event for Redis XADD

    If a queue is full the entry is dropped with a warning; the camera
    thread is never blocked.
    """

    def __init__(
        self,
        entry_logger,
        async_queue_size: int = 500,
    ):
        self._entry_logger = entry_logger
        self._running = False

        self._db_queue: queue.Queue = queue.Queue(maxsize=async_queue_size)
        self._gcs_queue: queue.Queue = queue.Queue(maxsize=200)
        self._redis_queue: queue.Queue = queue.Queue(maxsize=async_queue_size)

        self._db_thread: Optional[threading.Thread] = None
        self._gcs_thread: Optional[threading.Thread] = None
        self._redis_thread: Optional[threading.Thread] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the three background worker threads."""
        self._running = True
        self._db_thread = threading.Thread(
            target=self._db_worker, daemon=True, name="async-db"
        )
        self._gcs_thread = threading.Thread(
            target=self._gcs_worker, daemon=True, name="async-gcs"
        )
        self._redis_thread = threading.Thread(
            target=self._redis_worker, daemon=True, name="async-redis"
        )
        for t in (self._db_thread, self._gcs_thread, self._redis_thread):
            t.start()
        logger.info("AsyncLogger started (db / gcs / redis workers)")

    def stop(self, timeout: float = 5.0) -> None:
        """Signal workers to stop and wait for them to finish."""
        self._running = False
        for t in (self._db_thread, self._gcs_thread, self._redis_thread):
            if t and t.is_alive():
                t.join(timeout=timeout)
        logger.info("AsyncLogger stopped")

    # ── Public API (camera threads) ───────────────────────────────────────────

    def log_entry(self, entry: dict) -> None:
        """Queue an attendance/recognition event for background DB logging.

        Required keys:
            name (str|None), status (str), appear_time (datetime),
            camera_name (str), camera_id (int),
            recognized (bool), confidence (float),
            proof_image (np.ndarray|None), face_image (np.ndarray|None),
            application (list[str])
        """
        try:
            self._db_queue.put_nowait(entry)
        except queue.Full:
            logger.warning("AsyncLogger db_queue full — dropping log entry")

    def upload_image(self, data: dict) -> None:
        """Queue an image for GCS upload.

        Required keys: image (np.ndarray), folder (str), client_slug (str)
        Optional key:  callback (callable receiving the resulting URL)
        """
        try:
            self._gcs_queue.put_nowait(data)
        except queue.Full:
            logger.warning("AsyncLogger gcs_queue full — dropping image upload")

    def publish_event(self, event: dict) -> None:
        """Queue a Redis stream event.

        Required keys: fields (dict)
        Optional key:  stream (str, default 'ai:events')
        """
        try:
            self._redis_queue.put_nowait(event)
        except queue.Full:
            logger.warning("AsyncLogger redis_queue full — dropping redis event")

    # ── Background workers ────────────────────────────────────────────────────

    def _db_worker(self) -> None:
        # Keep draining after stop() clears _running until the queue is
        # actually empty (bounded by stop()'s join timeout) — otherwise
        # whatever was queued at shutdown (e.g. an end-of-day burst of
        # attendance events) is silently discarded instead of drained.
        while self._running or not self._db_queue.empty():
            try:
                entry = self._db_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._process_db_entry(entry)
            except Exception as e:
                logger.exception(f"AsyncLogger db_worker error: {e}")

    def _gcs_worker(self) -> None:
        while self._running or not self._gcs_queue.empty():
            try:
                data = self._gcs_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._process_gcs_upload(data)
            except Exception as e:
                logger.exception(f"AsyncLogger gcs_worker error: {e}")

    def _redis_worker(self) -> None:
        while self._running or not self._redis_queue.empty():
            try:
                event = self._redis_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._process_redis_event(event)
            except Exception as e:
                logger.exception(f"AsyncLogger redis_worker error: {e}")

    # ── Internal processing ───────────────────────────────────────────────────

    def _process_db_entry(self, entry: dict) -> None:
        name = entry.get("name")
        status = entry.get("status")
        appear_time = entry.get("appear_time")
        camera_name = entry.get("camera_name", "Unknown")
        camera_id = entry.get("camera_id")
        proof_image = entry.get("proof_image")
        application = entry.get("application", ["attendance"])
        recognized = entry.get("recognized", False)
        face_image = entry.get("face_image")

        if recognized and name:
            recorded = self._entry_logger.log_person_entry(
                name=name,
                status=status,
                appear_time=appear_time,
                camera_name=camera_name,
                camera_id=camera_id,
                proof_image=proof_image,
            )
            if recorded:
                logger.info(
                    f"ATTENDANCE | {name} {status} at {camera_name} | "
                    f"conf={entry.get('confidence', 0.0):.2f}"
                )
        else:
            if "unrecognized" in application and face_image is not None:
                if isinstance(face_image, np.ndarray) and face_image.size > 0:
                    self._entry_logger.send_unrecognized_face(
                        face=face_image,
                        status=status,
                        camera_id=camera_id,
                        camera_name=camera_name,
                    )
                    logger.info(
                        f"UNRECOGNIZED | Sent face from camera {camera_id} ({status})"
                    )

    def _process_gcs_upload(self, data: dict) -> None:
        image = data.get("image")
        folder = data.get("folder", "uploads")
        client_slug = data.get("client_slug", "")
        if image is None:
            return
        try:
            from infrastructure.storage import ImageFetcher

            url = ImageFetcher().upload_image(image, folder, client_slug)
            callback = data.get("callback")
            if callback:
                callback(url)
        except Exception as e:
            logger.exception(f"GCS upload failed: {e}")

    def _process_redis_event(self, event: dict) -> None:
        stream = event.get("stream", "ai:events")
        fields = event.get("fields", {})
        if not fields:
            return
        try:
            from messaging.redis_client import RedisClient
            RedisClient.get_instance().client.xadd(stream, fields)
        except Exception as e:
            logger.exception(f"Redis xadd failed: {e}")
