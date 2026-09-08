"""Per-camera frame producer.

One thread per camera: read a frame, apply its ROI, drop it unless it is a
detection frame, then write it to shared memory and enqueue it on the
`yolo` queue as a `yolo.detect` request. YOLO batches it with co-arriving
requests from other cameras, forwards each frame's detections on to
`camera.track` (see workers/yolo_tasks.py), which is where tracking,
identity and logging all happen — statically pinned to one camera-worker per
camera_id via compose.yml, not this process.
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
        # Last StreamHandler frame_seq this producer acted on. The loop below
        # polls far faster than any camera delivers, and StreamHandler.read()
        # hands back the latest frame whether or not it's new — so without
        # this, the same pixels get counted, written to shared memory and
        # enqueued repeatedly. Measured live at 6 cameras: ~51% of enqueued
        # frames were duplicates, which doubled GPU work and cycled the
        # shm ring fast enough to drop frames that were never processed.
        self._last_frame_seq: int = 0

        # Detection frames emitted since construction, for consumers that
        # can't reach a MetricsCollector. In the decode worker `_metrics` is
        # None (that collector lives in another process now), so this counter
        # is what the fps gauge is ultimately derived from — decode_main
        # samples it per publish tick and turns the delta into a rate. Plain
        # int, no lock: single writer (the producer thread), and readers only
        # ever difference successive samples, so a torn read isn't possible
        # for a value written by one thread on CPython.
        self._frames_emitted: int = 0

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
        # self.run() is the sole writer to _frame_slot — if the thread is
        # still alive past the join timeout, skip the close rather than
        # unlink shm out from under a live write (BufferError).
        if self._thread and self._thread.is_alive():
            logger.warning(
                f"CeleryCameraProducer[cam={self.camera_id}] thread still "
                f"running after {timeout:.1f}s timeout, skipping shm close"
            )
        elif self._frame_slot is not None:
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
        from workers.celery_app import camera_queue_name
        from workers.yolo_tasks import detect_task

        # ── Step 1: Read frame, but only act on a NEW one ──────────────────
        # read_with_seq, not read(): this loop polls much faster than any
        # camera delivers frames, and read() returns the latest frame
        # regardless of whether this producer has already seen it. Acting on
        # an unchanged seq would re-run the whole pipeline (shm write, YOLO,
        # tracking) on pixels already processed.
        ret, frame, frame_seq = self.stream_handler.read_with_seq()
        if not ret or frame is None:
            time.sleep(0.005)
            return
        if frame_seq == self._last_frame_seq:
            # Same frame as last time — the stream hasn't produced a new one
            # yet. Sleep briefly rather than spinning hot on the CPU that
            # decoding needs.
            time.sleep(0.005)
            return
        self._last_frame_seq = frame_seq

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
        self._frames_emitted += 1
        if self._metrics is not None:
            self._metrics.record_frame(self.camera_id)

        # ── Step 4 (new): hand off via shared memory + Celery ──────────────
        frame_slot = self._frame_slot
        assert frame_slot is not None  # start() always sets this before this loop runs
        handle = frame_slot.write(frame)
        # dataclasses.asdict, not the handle itself: celery_app.py's
        # task_serializer='json' can't encode a FrameHandle instance —
        # see detect_task's docstring for why the fix lives at this
        # boundary rather than in the global Celery config.
        # expires / deadline: the thread path's bounded queues dropped
        # frames under load and stayed responsive; Celery queues are
        # unbounded, so without an expiry an overloaded worker accumulates
        # silent lag instead. 1s rather than one frame interval (~130ms)
        # because the goal is bounding a backlog, not enforcing cadence — a
        # healthy queue never expires anything, and the frame's
        # shared-memory slot is long overwritten by then anyway.
        #
        # Both `expires` AND `deadline` carry this same budget, not just
        # one: `expires` is Celery's own mechanism, evaluated on delivery to
        # THIS hop (yolo.detect); `deadline` is a plain epoch-seconds kwarg
        # that travels IN the payload to every later hop, because
        # celery-batches' Batches base class does not honour `expires` at
        # all (verified in the go/no-go spike — see
        # docs/LSO67_FOLLOWUP_QUEUE_DESIGN.md's "Verified constraints").
        deadline = time.time() + _TASK_EXPIRES_S
        detect_task.apply_async(
            kwargs={
                "camera_id": self.camera_id,
                "frame_handle": dataclasses.asdict(handle),
                "frame_num": frame_num,
                "next_queue": camera_queue_name(self.camera_id),
                "deadline": deadline,
            },
            queue="yolo",
            expires=_TASK_EXPIRES_S,
        )
        # Deliberately fire-and-forget: this producer does not wait for the
        # task's result. Waiting here would recreate the exact synchronous
        # blocking (camera_worker.py's Step 4 wait on get_detections) that
        # this migration exists to remove — a slow or backlogged worker
        # would stall frame reading for this camera, when the entire point
        # of moving the CPU work off-thread is to let it run independently.

