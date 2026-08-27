"""CameraWorker — per-camera processing thread.

Each camera gets one CameraWorker that:
  1. Reads frames from its StreamHandler
  2. Submits frames to the shared GPUInferenceWorker for YOLO detection
  3. Runs CPU-only tracking via CameraEngine.update_tracking()
  4. Submits person ROIs to the GPU worker for ArcFace embedding
  5. Runs CPU-only identity resolution via CameraEngine.finalize_identities()
  6. Hands attendance events to AsyncLogger (returns immediately)

This allows N camera threads to share the GPU in one batched worker.
"""

import dataclasses
import threading
import time
from typing import Any, Dict, List, Optional

import cv2
from loguru import logger


class CameraWorker:
    """Processes a single camera stream in a dedicated thread.

    Coordinates with the shared GPUInferenceWorker for GPU inference and
    uses its own CameraEngine for CPU-only tracking and identity resolution.
    """

    def __init__(
        self,
        camera_id: int,
        camera_config: Dict[str, Any],
        camera_engine,
        gpu_worker,
        async_logger,
        stream_handler,
        detection_interval: int = 2,
        recognition_interval: int = 5,
        annotator=None,
        video_writer: Optional[cv2.VideoWriter] = None,
        metrics_collector=None,
    ):
        """
        Args:
            camera_id:            DB camera id — the key for this camera's GPU
                                  queues and metrics series. Deliberately NOT a
                                  list position (LSO-130): positions shift when
                                  a camera leaves the set mid-run.
            camera_config:        Camera config dict (roi, cam_type, etc.)
            camera_engine:        CameraEngine instance for this camera (CPU-only calls)
            gpu_worker:           Shared GPUInferenceWorker instance
            async_logger:         AsyncLogger for non-blocking I/O
            stream_handler:       StreamHandler for reading frames
            detection_interval:   Run YOLO every N frames (1 = every frame)
            recognition_interval: Run ArcFace every N detection frames
            annotator:            Optional FrameAnnotator for save_video
            video_writer:         Optional cv2.VideoWriter for save_video
        """
        self.camera_id = camera_id
        self.camera_config = camera_config
        self.camera_engine = camera_engine
        self.gpu_worker = gpu_worker
        self.async_logger = async_logger
        self.stream_handler = stream_handler
        self.detection_interval = max(1, detection_interval)
        self.recognition_interval = max(1, recognition_interval)
        self.annotator = annotator
        self.video_writer = video_writer

        self.roi: Optional[List[int]] = camera_config.get("roi")
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Per-camera counters
        self._frame_num: int = 0
        self._detection_frame_num: int = 0

        # FPS tracking for annotation overlay
        self._fps: float = 0.0
        self._last_frame_time: float = 0.0

        # Metrics collector (optional)
        self._metrics = metrics_collector

        # Cache last known face bbox + score per track (persists between recognition frames)
        self._face_cache: Dict[int, Dict] = {}  # track_id -> {face_bbox, face_det_score}

        logger.debug(
            f"CameraWorker[cam={camera_id}] created: "
            f"camera_id={camera_config.get('camera_id')}, "
            f"detect_every={detection_interval}, "
            f"recog_every={recognition_interval}"
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the camera worker thread."""
        self._running = True
        name = f"cam-worker-{self.camera_id}"
        self._thread = threading.Thread(target=self.run, daemon=True, name=name)
        self._thread.start()
        logger.debug(f"CameraWorker[cam={self.camera_id}] thread started")

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the worker to stop and wait for it."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        logger.info(f"CameraWorker[cam={self.camera_id}] stopped")

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Main processing loop for this camera."""
        logger.debug(f"CameraWorker[cam={self.camera_id}] loop started")
        while self._running:
            try:
                self._process_one_frame()
            except Exception as e:
                logger.exception(f"CameraWorker[cam={self.camera_id}] error: {e}")
                time.sleep(0.01)
    ## This function is the main loop of the CameraWorker thread. It continuously processes frames from the camera stream until the worker is stopped. It calls the _process_one_frame() method to handle each frame, and if any exception occurs during processing, it logs the error and sleeps briefly before continuing.
    def _process_one_frame(self) -> None:
        # ── Step 1: Read frame ────────────────────────────────────────────────
        ret, frame = self.stream_handler.read()
        if not ret or frame is None:
            time.sleep(0.005)
            return

        self._frame_num += 1

        # ── Step 2: Apply ROI (before submitting to GPU worker) ───────────────
        if self.roi:
            x1, y1, x2, y2 = self.roi
            frame = frame[y1:y2, x1:x2]
            if frame.size == 0:
                return

        # ── Step 3: Frame-skip (detection_interval) ───────────────────────────
        if self._frame_num % self.detection_interval != 0:
            return

        self._detection_frame_num += 1
        frame_num = self._frame_num

        # Record this processed detection-frame for FPS monitoring
        if self._metrics is not None:
            self._metrics.record_frame(self.camera_id)

        # ── Step 4: Submit frame to GPU worker, wait for detections ───────────
        if not self.gpu_worker.submit_frame(self.camera_id, frame, frame_num):
            return
        detections = self.gpu_worker.get_detections(self.camera_id, frame_num)

        # ── Step 5: CPU tracking + person ROI extraction ──────────────────────
        active_tracks, removed_tracks, person_rois = (
            self.camera_engine.update_tracking(detections, frame, frame_num)
        )

        # ── Step 6: Submit face ROIs to GPU worker, wait for embeddings ───────
        run_recognition = (
            self._detection_frame_num % self.recognition_interval == 0
        )
        if run_recognition and person_rois:
            track_ids = [tid for tid, _, _o in person_rois]
            rois = [roi for _, roi, _o in person_rois]
            roi_offsets = {tid: off for tid, _, off in person_rois}
            face_seq = self.gpu_worker.submit_faces(self.camera_id, rois, track_ids)
            embeddings_map = (
                self.gpu_worker.get_embeddings(self.camera_id, face_seq)
                if face_seq is not None
                else {}
            )
        else:
            roi_offsets = {}
            embeddings_map = {}

        # ── Step 7: CPU identity resolution ───────────────────────────────────
        events = self.camera_engine.finalize_identities(
            active_tracks, removed_tracks, embeddings_map, frame, frame_num
        )

        # ── Step 7b: Emit real-time floor positions (~5Hz/track) ──────────────
        self.camera_engine.emit_positions(active_tracks)

        # ── Step 8: Non-blocking event logging ────────────────────────────────
        for event in events:
            self.async_logger.log_entry(event)

        # ── Step 9: Annotate + write video (only when save_video=True) ────────
        if self.annotator is not None and self.video_writer is not None:
            self._annotate_and_write(frame, active_tracks, embeddings_map, roi_offsets)

    def _annotate_and_write(
        self,
        frame,
        active_tracks: List[Dict],
        embeddings_map: Dict,
        roi_offsets: Dict,
    ) -> None:
        """Build per-person annotation state, annotate the frame, and write to disk."""
        now = time.time()
        if self._last_frame_time > 0:
            elapsed = now - self._last_frame_time
            self._fps = 1.0 / elapsed if elapsed > 0 else self._fps
        self._last_frame_time = now

        state_manager = self.camera_engine.state_manager
        identity_manager = self.camera_engine.identity_manager
        active_track_ids = {t["track_id"] for t in active_tracks}

        # Evict stale tracks from face cache
        for stale_id in list(self._face_cache.keys()):
            if stale_id not in active_track_ids:
                del self._face_cache[stale_id]

        person_states = []
        for track in active_tracks:
            track_id = track["track_id"]
            state = state_manager.get_state(track_id)
            emb_info = embeddings_map.get(track_id, {})

            # Offset face bbox/landmarks from ROI-space to full-frame coords
            face_bbox_roi = emb_info.get("face_bbox")
            if face_bbox_roi is not None and track_id in roi_offsets:
                rx, ry = roi_offsets[track_id]
                landmarks_roi = emb_info.get("face_landmarks")
                self._face_cache[track_id] = {
                    "face_bbox": [
                        rx + face_bbox_roi[0], ry + face_bbox_roi[1],
                        rx + face_bbox_roi[2], ry + face_bbox_roi[3],
                    ],
                    "face_det_score": emb_info.get("det_score"),
                    "face_landmarks": (
                        [[rx + p[0], ry + p[1]] for p in landmarks_roi]
                        if landmarks_roi is not None else None
                    ),
                }

            cached = self._face_cache.get(track_id, {})
            vote_status = identity_manager.get_voting_status(track_id)
            identity = state.identity if state else None
            identity_locked = state.identity_locked if state else False
            if not identity_locked and not identity:
                tentative = vote_status.get("top_identity")
                if tentative and vote_status.get("votes", 0) > 0:
                    identity = tentative

            person_states.append({
                "track_id": track_id,
                "global_id": track.get("global_track_id"),
                "bbox": track["bbox"],
                "face_bbox": cached.get("face_bbox"),
                "face_det_score": cached.get("face_det_score"),
                "face_landmarks": cached.get("face_landmarks"),
                "vote_count": vote_status.get("votes", 0),
                "required_votes": vote_status.get("required_votes", 0),
                "keypoints": track.get("keypoints"),
                "identity": identity,
                "identity_locked": identity_locked,
                "track_age": 0,
                "in_current_frame": True,
                "last_detected_action": state.last_detected_action if state else None,
            })

        annotated = self.annotator.annotate_frame(
            frame, person_states, fps=self._fps, roi_active=bool(self.roi)
        )
        self.video_writer.write(annotated)


#: Drop a queued frame older than this rather than processing it late.
_TASK_EXPIRES_S = 1.0


class CeleryCameraProducer:
    """LSO-67 Stage 1: the producer half of the Celery path for one flagged
    camera (configs/config.yaml's pipeline.celery_camera_ids).

    Runs CameraWorker's Steps 1-3 exactly (read frame, apply ROI, frame-skip
    by detection_interval) — everything a Celery task cannot do itself,
    because a task instance is stateless per call and cannot hold the
    StreamHandler's persistent RTSP connection open the way this thread
    does. What CameraWorker's Step 4 onward did in-process — submit to GPU,
    track, finalize identity, log — happens in workers.camera_tasks instead,
    in a separate OS process, which is the entire point of this migration.

    Deliberately NOT a CameraWorker subclass: sharing a base class across
    "does everything in-thread" and "reads a frame and hands off a handle"
    would blur exactly the boundary Stage 1 exists to draw. The duplicated
    read/ROI/frame-skip logic is 15 lines: the small, explicit copy is worth
    more than an abstraction that would need to grow parameters to express
    "except stop here" on one path.
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

        self.roi: Optional[List[int]] = camera_config.get("roi")
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
        logger.info(
            f"CeleryCameraProducer[cam={self.camera_id}] started "
            f"(routing frames to Celery — LSO-67 Stage 1)"
        )

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
        if self.roi:
            x1, y1, x2, y2 = self.roi
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

