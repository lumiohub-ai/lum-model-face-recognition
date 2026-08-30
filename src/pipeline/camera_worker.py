"""Per-camera frame producer.

One thread per camera: read a frame, apply its ROI, drop it unless it is a
detection frame, then write it to shared memory and enqueue a Celery task.
Tracking, identity and logging all happen in workers.camera_tasks, in a
separate process.
"""

import dataclasses
import threading
import time
from typing import Any, Dict, Optional

from loguru import logger


#: Drop a queued frame older than this rather than processing it late.
_TASK_EXPIRES_S = 1.0


class CeleryCameraProducer:
    """Reads one camera's stream and hands each detection frame to Celery.

    Stays a thread rather than moving into the task because a task is
    stateless per call and cannot hold the StreamHandler's persistent RTSP
    connection open.
    """

    def __init__(
        self,
        camera_id: int,
        camera_config: Dict[str, Any],
        stream_handler,
        detection_interval: int = 2,
        metrics_collector=None,
    ):
        self.camera_id = camera_id
        self.camera_config = camera_config
        self.stream_handler = stream_handler
        self.detection_interval = max(1, detection_interval)
        self._metrics = metrics_collector

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._frame_num: int = 0

        self._frame_slot = None  # constructed in start(), not __init__ —
        # see start()'s comment on why shared-memory allocation waits until
        # the producer thread is actually about to run.

        logger.debug(
            f"CeleryCameraProducer[cam={camera_id}] created: "
            f"detect_every={detection_interval}"
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        # Constructed here, not __init__: __init__ can run in the engine's
        # construction thread well before this camera's thread starts (or a
        # reinit tears down and rebuilds several CeleryCameraProducers in a
        # row) — allocating the shared-memory slot exactly when the thread
        # that owns its writes starts keeps the slot's lifetime tied to the
        # thread's, matching CameraFrameSlot's single-writer assumption.
        from workers.frame_store import CameraFrameSlot

        self._frame_slot = CameraFrameSlot(self.camera_id)
        self._running = True
        name = f"cam-producer-{self.camera_id}"
        self._thread = threading.Thread(target=self.run, daemon=True, name=name)
        self._thread.start()
        logger.info(f"CeleryCameraProducer[cam={self.camera_id}] started")

    def stop(self, timeout: float = 5.0) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        if self._frame_slot is not None:
            self._frame_slot.close()
        logger.info(f"CeleryCameraProducer[cam={self.camera_id}] stopped")

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        logger.debug(f"CeleryCameraProducer[cam={self.camera_id}] loop started")
        while self._running:
            try:
                self._produce_one_frame()
            except Exception as e:
                logger.exception(f"CeleryCameraProducer[cam={self.camera_id}] error: {e}")
                time.sleep(0.01)

    def _produce_one_frame(self) -> None:
        from workers.camera_tasks import process_frame_task

        # ── Step 1: Read frame — identical to CameraWorker ─────────────────
        ret, frame = self.stream_handler.read()
        if not ret or frame is None:
            time.sleep(0.005)
            return

        self._frame_num += 1

        # ── Step 2: Apply ROI — identical to CameraWorker ──────────────────
        # Read live off camera_config rather than a cached self.roi: a config
        # reload mutates this dict in place (see engine.py's
        # reload_camera_configs), so this always sees the current ROI without
        # needing its own restart the way a stream_url change does.
        roi = self.camera_config.get("roi")
        if roi:
            x1, y1, x2, y2 = roi
            frame = frame[y1:y2, x1:x2]
            if frame.size == 0:
                return

        # ── Step 3: Frame-skip — identical to CameraWorker ─────────────────
        if self._frame_num % self.detection_interval != 0:
            return

        frame_num = self._frame_num
        if self._metrics is not None:
            self._metrics.record_frame(self.camera_id)

        # ── Step 4 (new): hand off via shared memory + Celery ──────────────
        frame_slot = self._frame_slot
        assert frame_slot is not None  # start() always sets this before this loop runs
        handle = frame_slot.write(frame)
        # dataclasses.asdict, not the handle itself: celery_app.py's
        # task_serializer='json' can't encode a FrameHandle instance —
        # see process_frame_task's docstring for why the fix lives at this
        # boundary rather than in the global Celery config.
        # expires: the thread path's bounded queues dropped frames under load
        # and stayed responsive; Celery queues are unbounded, so without this
        # an overloaded worker accumulates silent lag instead. 1s rather than
        # one frame interval (~130ms) because the goal is bounding a backlog,
        # not enforcing cadence — a healthy queue never expires anything, and
        # the frame's shared-memory slot is long overwritten by then anyway.
        process_frame_task.apply_async(
            kwargs={
                "camera_id": self.camera_id,
                "frame_handle": dataclasses.asdict(handle),
                "frame_num": frame_num,
            },
            expires=_TASK_EXPIRES_S,
        )
        # Deliberately fire-and-forget: this producer does not wait for the
        # task's result. Waiting here would recreate the exact synchronous
        # blocking (camera_worker.py's Step 4 wait on get_detections) that
        # this migration exists to remove — a slow or backlogged worker
        # would stall frame reading for this camera, when the entire point
        # of moving the CPU work off-thread is to let it run independently.

