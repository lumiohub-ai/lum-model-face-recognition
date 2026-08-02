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

    # frame_ms is the gap between consecutive detection-frame starts. A gap
    # this large means a stream stall/reconnect happened in between, not that
    # one frame genuinely took 5s — discard it rather than poisoning the
    # rolling stage-timing window with a single huge outlier.
    _MAX_SANE_FRAME_GAP_MS: float = 5000.0

    def __init__(
        self,
        camera_idx: int,
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
            camera_idx:           0-based index matching the GPU worker queue slot
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
        self.camera_idx = camera_idx
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

        # perf_counter() timestamp of the previous detection frame's *end*
        # (right before it was recorded), used to compute frame_ms for the
        # stage-timing breakdown (LSO-66) — see the comment at the recording
        # site in _process_one_frame() for why "end", not "start".
        self._last_frame_end: Optional[float] = None

        # Accumulated stream_handler.read() cost since the last recorded
        # detection frame — includes every skipped frame's own read, since
        # those happen on this same thread and are real time spent before the
        # next detection frame can start. Flushed into the "decode" stage and
        # reset to 0.0 each time a detection frame's stages get recorded.
        self._decode_ms_accum: float = 0.0

        # Metrics collector (optional)
        self._metrics = metrics_collector

        # Cache last known face bbox + score per track (persists between recognition frames)
        self._face_cache: Dict[int, Dict] = {}  # track_id -> {face_bbox, face_det_score}

        logger.debug(
            f"CameraWorker[{camera_idx}] created: "
            f"camera_id={camera_config.get('camera_id')}, "
            f"detect_every={detection_interval}, "
            f"recog_every={recognition_interval}"
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the camera worker thread."""
        self._running = True
        name = f"cam-worker-{self.camera_idx}"
        self._thread = threading.Thread(target=self.run, daemon=True, name=name)
        self._thread.start()
        logger.debug(f"CameraWorker[{self.camera_idx}] thread started")

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the worker to stop and wait for it."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        logger.info(f"CameraWorker[{self.camera_idx}] stopped")

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Main processing loop for this camera."""
        logger.debug(f"CameraWorker[{self.camera_idx}] loop started")
        while self._running:
            try:
                self._process_one_frame()
            except Exception as e:
                logger.exception(f"CameraWorker[{self.camera_idx}] error: {e}")
                time.sleep(0.01)
    ## This function is the main loop of the CameraWorker thread. It continuously processes frames from the camera stream until the worker is stopped. It calls the _process_one_frame() method to handle each frame, and if any exception occurs during processing, it logs the error and sleeps briefly before continuing.
    def _process_one_frame(self) -> None:
        # ── Step 1: Read frame ────────────────────────────────────────────────
        _t_read = time.perf_counter()
        ret, frame = self.stream_handler.read()
        self._decode_ms_accum += (time.perf_counter() - _t_read) * 1000
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
            self._metrics.record_frame(self.camera_idx)

        # ── Step 4: Submit frame to GPU worker, wait for detections ───────────
        _t0 = time.perf_counter()
        self.gpu_worker.submit_frame(self.camera_idx, frame, frame_num)
        detections, detect_timing = self.gpu_worker.get_detections(self.camera_idx)
        detect_ms = (time.perf_counter() - _t0) * 1000

        # ── Step 5: CPU tracking + person ROI extraction ──────────────────────
        active_tracks, removed_tracks, person_rois = (
            self.camera_engine.update_tracking(detections, frame, frame_num)
        )
        track_ms = self.camera_engine.stage_timings.get("track", 0.0)
        reid_ms = self.camera_engine.stage_timings.get("reid", 0.0)

        # ── Step 6: Submit face ROIs to GPU worker, wait for embeddings ───────
        _t0 = time.perf_counter()
        run_recognition = (
            self._detection_frame_num % self.recognition_interval == 0
        )
        if run_recognition and person_rois:
            track_ids = [tid for tid, _, _o in person_rois]
            rois = [roi for _, roi, _o in person_rois]
            roi_offsets = {tid: off for tid, _, off in person_rois}
            self.gpu_worker.submit_faces(self.camera_idx, rois, track_ids)
        else:
            roi_offsets = {}
            # Always send a submission to keep the GPU worker synchronised
            self.gpu_worker.submit_faces(self.camera_idx, [], [])

        embeddings_map, face_timing = self.gpu_worker.get_embeddings(self.camera_idx)
        face_ms = (time.perf_counter() - _t0) * 1000

        # ── Step 7: CPU identity resolution ───────────────────────────────────
        events = self.camera_engine.finalize_identities(
            active_tracks, removed_tracks, embeddings_map, frame, frame_num
        )
        match_ms = self.camera_engine.stage_timings.get("match", 0.0)
        identity_ms = self.camera_engine.stage_timings.get("identity", 0.0)

        # ── Step 7b: Emit real-time floor positions (~5Hz/track) ──────────────
        _t0 = time.perf_counter()
        self.camera_engine.emit_positions(active_tracks)
        publish_ms = (time.perf_counter() - _t0) * 1000

        # ── Step 8: Non-blocking event logging ────────────────────────────────
        for event in events:
            self.async_logger.log_entry(event)

        # ── Step 9: Annotate + write video (only when save_video=True) ────────
        if self.annotator is not None and self.video_writer is not None:
            self._annotate_and_write(frame, active_tracks, embeddings_map, roi_offsets)

        # ── Step 10: Record per-stage breakdown for the metrics dashboard ─────
        #
        # frame_ms is measured HERE, at the end, as the gap since the previous
        # detection frame *finished* — not at the top as the gap since it
        # *started*. It must be measured at the same point the stages below
        # are read out: they're this frame's own detect/track/.../publish
        # spans, and pairing them against a frame_ms captured at the top would
        # actually pair them with the *previous* frame's elapsed span instead
        # (this frame's work happens strictly after that top-of-frame mark).
        # That mismatch doesn't average out — the max(0, ...) clamp on "other"
        # below is one-sided, so it silently inflates the pct total instead of
        # settling back near 100%. Caught via a live run whose shares summed
        # to ~144%; see LSO-66 plan notes.
        t_frame_end = time.perf_counter()
        frame_ms: Optional[float] = None
        if self._last_frame_end is not None:
            gap_ms = (t_frame_end - self._last_frame_end) * 1000
            if gap_ms <= self._MAX_SANE_FRAME_GAP_MS:
                frame_ms = gap_ms
        self._last_frame_end = t_frame_end

        # Flush the accumulated read()/decode cost (this frame's own read plus
        # every skipped frame's read since the last recorded detection frame)
        # unconditionally, so it can't grow unbounded across a stretch where
        # metrics are disabled or frame_ms comes back None (e.g. right after
        # a stream reconnect).
        decode_ms = self._decode_ms_accum
        self._decode_ms_accum = 0.0

        if self._metrics is not None and frame_ms is not None:
            stages = {
                "decode": decode_ms,
                "detect": detect_ms,
                "detect_wait": detect_timing.get("wait_ms", 0.0),
                "detect_gpu": detect_timing.get("gpu_ms", 0.0),
                "track": track_ms,
                "reid": reid_ms,
                "face": face_ms,
                "face_wait": face_timing.get("wait_ms", 0.0),
                "face_gpu": face_timing.get("gpu_ms", 0.0),
                "match": match_ms,
                "identity": identity_ms,
                "publish": publish_ms,
            }
            primary_sum = (
                decode_ms + detect_ms + track_ms + reid_ms + face_ms
                + match_ms + identity_ms + publish_ms
            )
            stages["other"] = max(0.0, frame_ms - primary_sum)
            self._metrics.record_frame_stages(self.camera_idx, frame_ms, stages)

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
