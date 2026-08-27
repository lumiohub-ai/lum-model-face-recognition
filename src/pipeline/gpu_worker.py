"""GPU Inference Worker — serialises all GPU operations across camera threads.

Two independent GPU threads:
  - YOLO thread:    collect frames (any camera ready) → batch YOLO → distribute detections
  - ArcFace thread: collect faces  (any camera ready) → batch ArcFace → distribute embeddings

Camera threads are fully independent — a slow camera never blocks a fast one.
"""

import queue
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from loguru import logger

# frontality/pitch moved with _run_arcface_batch to workers/face_tasks.py —
# they annotate the result dict where it is built, so this module no longer
# imports lum_vision at all.


class GPUInferenceWorker:
    """Batched GPU inference worker shared across all camera threads.

    YOLO and ArcFace run in separate threads so face embedding for one camera
    never waits on another camera's CPU tracking.
    """

    def __init__(
        self,
        camera_ids: Sequence[int],
        metrics_collector=None,
    ):
        # This class holds NO models. YOLO runs in the `yolo` Celery worker,
        # SCRFD+ArcFace in the `face` worker, both reached by task. What
        # remains here is exactly what should: the per-camera queues, the
        # cross-camera batch collection that makes the batched GPU call worth
        # ~1.8x, and the request/response correlation that keeps a camera
        # from silently desyncing.
        self._running = False
        self._yolo_thread: Optional[threading.Thread] = None
        self._arcface_thread: Optional[threading.Thread] = None
        self._metrics = metrics_collector  # Optional[MetricsCollector]

        # Producer-side slot for the cross-camera YOLO batch. One per loop,
        # not per camera — see FrameBatchSlot's docstring for why
        # CameraFrameSlot cannot serve this.
        from workers.frame_store import FrameBatchSlot, RoiBatchSlot

        self._yolo_batch_slot = FrameBatchSlot("yolo")

        # The ArcFace loop flattens every camera's ROIs into ONE list before
        # inference (see _arcface_loop), so it needs a single loop-wide slot,
        # not one per camera. RoiBatchSlot keys its shared-memory block by
        # camera id, so this uses a negative sentinel: DB camera ids are
        # always positive, so it cannot collide with a real camera's slot —
        # including the per-camera ROI slots camera_tasks.py owns.
        self._ARCFACE_SLOT_ID = -1
        self._arcface_batch_slot = RoiBatchSlot(self._ARCFACE_SLOT_ID)

        # Queues are keyed by the DB camera id, NOT by position in the camera
        # list (LSO-130). A positional key silently re-points every later
        # camera's queues when one camera is removed from the middle of the
        # set, which is why the engine used to rebuild everything instead.
        self._camera_ids: List[int] = list(camera_ids)

        # Guards the four dicts below: add_camera/remove_camera mutate them
        # while the YOLO and ArcFace loops are iterating.
        self._queues_lock = threading.Lock()

        self._frame_in_queues: Dict[int, queue.Queue] = {}
        self._detection_out_queues: Dict[int, queue.Queue] = {}
        self._face_in_queues: Dict[int, queue.Queue] = {}
        self._embedding_out_queues: Dict[int, queue.Queue] = {}
        self._face_seq: Dict[int, int] = {}
        for cam_id in self._camera_ids:
            self._make_queues(cam_id)

        logger.debug(
            f"GPUInferenceWorker: {len(self._camera_ids)} camera(s) "
            f"ids={self._camera_ids}"
        )

    # ── Camera set management ─────────────────────────────────────────────────

    def _make_queues(self, camera_id: int) -> None:
        self._frame_in_queues[camera_id] = queue.Queue(maxsize=2)
        self._detection_out_queues[camera_id] = queue.Queue(maxsize=2)
        self._face_in_queues[camera_id] = queue.Queue(maxsize=4)
        self._embedding_out_queues[camera_id] = queue.Queue(maxsize=4)
        self._face_seq[camera_id] = 0

    @property
    def camera_ids(self) -> List[int]:
        with self._queues_lock:
            return list(self._camera_ids)

    def add_camera(self, camera_id: int) -> None:
        """Register queues for a camera joining the running set."""
        with self._queues_lock:
            if camera_id in self._frame_in_queues:
                logger.debug(f"GPUInferenceWorker: camera {camera_id} already registered")
                return
            self._make_queues(camera_id)
            self._camera_ids.append(camera_id)
        logger.info(f"GPUInferenceWorker: camera {camera_id} added")

    def remove_camera(self, camera_id: int) -> None:
        """Drop a camera's queues. Anything still queued is discarded — the
        camera's worker is stopping, so nothing would consume it."""
        with self._queues_lock:
            if camera_id not in self._frame_in_queues:
                logger.debug(f"GPUInferenceWorker: camera {camera_id} not registered")
                return
            for d in (
                self._frame_in_queues,
                self._detection_out_queues,
                self._face_in_queues,
                self._embedding_out_queues,
                self._face_seq,
            ):
                d.pop(camera_id, None)
            if camera_id in self._camera_ids:
                self._camera_ids.remove(camera_id)
        logger.info(f"GPUInferenceWorker: camera {camera_id} removed")

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        self._running = True
        self._yolo_thread = threading.Thread(
            target=self._yolo_loop, daemon=True, name="gpu-yolo"
        )
        self._arcface_thread = threading.Thread(
            target=self._arcface_loop, daemon=True, name="gpu-arcface"
        )
        self._yolo_thread.start()
        self._arcface_thread.start()
        logger.info("GPUInferenceWorker started")

    def stop(self, timeout: float = 5.0) -> None:
        self._running = False
        for t in (self._yolo_thread, self._arcface_thread):
            if t and t.is_alive():
                t.join(timeout=timeout)
        # Released only after both loops have stopped: the YOLO loop is the
        # sole writer to this slot, and unlinking it while that thread is
        # still mid-write would raise BufferError in the releasing thread.
        self.close()
        logger.info("GPUInferenceWorker stopped")

    # ── Camera-thread API ─────────────────────────────────────────────────────

    def _queue_for(
        self, queues: Dict[int, queue.Queue], camera_id: int, op: str
    ) -> Optional[queue.Queue]:
        """Look up a camera's queue, tolerating removal mid-flight.

        A camera worker can still be draining its last cycle after
        `remove_camera` has run, so a missing queue is expected during a
        set change — not an error worth raising into the worker thread.

        Deliberately does NOT take `_queues_lock`. This runs on every frame
        for every camera, and a single `dict.get` is atomic under CPython's
        GIL — it either sees the queue or it doesn't, never a torn state.
        The lock is only needed where a dict is *iterated* (`_collect_batch`),
        since iteration is what a concurrent mutation breaks. Revisit if this
        ever runs free-threaded, or if a compound read is added here.
        """
        q = queues.get(camera_id)
        if q is None:
            logger.debug(f"{op}: camera {camera_id} is no longer registered")
        return q

    def _await_response(self, q: queue.Queue, seq: int, timeout: float, op: str,
                        camera_id: int, default):
        """Return the payload tagged *seq*, discarding anything older.

        Responses are matched, not counted. A reply the caller gave up
        on stays in the queue and would otherwise be served to the next request
        forever, one frame behind — the desync is permanent because exactly one
        response is produced and one consumed per cycle from then on.
        """
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.warning(f"{op} timeout for camera {camera_id} seq={seq}")
                return default
            try:
                got_seq, payload = q.get(timeout=remaining)
            except queue.Empty:
                logger.warning(f"{op} timeout for camera {camera_id} seq={seq}")
                return default
            if got_seq == seq:
                return payload
            # An earlier request's reply, orphaned by a timeout or a dropped
            # frame. Drop it and keep waiting: that is what drains the backlog
            # instead of carrying it forward.
            logger.warning(
                f"{op}: discarding stale response for camera {camera_id} "
                f"(got seq={got_seq}, want seq={seq})"
            )
            if self._metrics is not None:
                self._metrics.record_stale_response(camera_id)
            if got_seq > seq:
                # Newer than what we asked for: our request was evicted from the
                # in-queue by submit_frame's drop-oldest, so no reply for it will
                # ever come. Give up now instead of burning the whole timeout.
                return default

    def submit_frame(
        self, camera_id: int, frame: np.ndarray, frame_num: int
    ) -> bool:
        """Submit a frame for YOLO detection (non-blocking; drops oldest if full).

        Returns False when the frame was dropped and no response will ever be
        produced for it, so the caller must skip its ``get_detections`` rather
        than consume some other frame's result.
        """
        q = self._queue_for(self._frame_in_queues, camera_id, "submit_frame")
        if q is None:
            return False
        try:
            q.put_nowait((frame, frame_num))
            return True
        except queue.Full:
            # A frame is being dropped either way — count it before the
            # replacement attempt, so the counter cannot under-report if
            # that retry also finds the queue full.
            if self._metrics is not None:
                self._metrics.record_drop(camera_id)
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            try:
                q.put_nowait((frame, frame_num))
                return True
            except queue.Full:
                return False

    def get_detections(
        self, camera_id: int, frame_num: int, timeout: float = 2.0
    ) -> List[Dict]:
        """Block until this camera's detections for *frame_num* are available."""
        q = self._queue_for(self._detection_out_queues, camera_id, "get_detections")
        if q is None:
            return []
        return self._await_response(
            q, frame_num, timeout, "get_detections", camera_id, []
        )

    def submit_faces(
        self,
        camera_id: int,
        person_rois: List[np.ndarray],
        track_ids: List[int],
    ) -> Optional[int]:
        """Submit person ROI crops for ArcFace embedding.

        Returns the sequence number to pass to ``get_embeddings``, or None if the
        camera is no longer registered. Unlike the detection path there is no
        caller-side id to reuse, so the worker issues one.
        """
        with self._queues_lock:
            q = self._face_in_queues.get(camera_id)
            if q is None:
                logger.debug(f"submit_faces: camera {camera_id} is no longer registered")
                return None
            seq = self._face_seq[camera_id] = self._face_seq.get(camera_id, 0) + 1
        q.put((person_rois, track_ids, seq))
        return seq

    def get_embeddings(
        self, camera_id: int, seq: int, timeout: float = 2.0
    ) -> Dict[int, Dict]:
        """Block until this camera's ArcFace results for *seq* are available."""
        q = self._queue_for(self._embedding_out_queues, camera_id, "get_embeddings")
        if q is None:
            return {}
        return self._await_response(
            q, seq, timeout, "get_embeddings", camera_id, {}
        )

    # ── YOLO loop ─────────────────────────────────────────────────────────────

    def _yolo_loop(self) -> None:
        logger.info("GPUInferenceWorker YOLO loop running")
        while self._running:
            try:
                batch = self._collect_frames()
                if not batch:
                    time.sleep(0.001)
                    continue

                # Sorted order is the alignment contract: FrameBatchSlot packs
                # in sorted-camera-id order and the worker returns results in
                # that same order, so index i below belongs to cam_ids[i].
                cam_ids = sorted(batch.keys())
                all_detections = self._run_yolo_batch(batch)

                for i, cam_id in enumerate(cam_ids):
                    # Same mid-flight-removal case as the submit/get paths, so
                    # use the same helper rather than repeating the reasoning.
                    out_q = self._queue_for(
                        self._detection_out_queues, cam_id, "yolo_distribute"
                    )
                    if out_q is not None:
                        # Echo the submitted frame_num so the waiting camera can
                        # tell this reply from an orphaned earlier one.
                        out_q.put((batch[cam_id][1], all_detections[i]))

            except Exception as e:
                logger.exception(f"GPUInferenceWorker YOLO error: {e}")

    # ── ArcFace loop ──────────────────────────────────────────────────────────

    def _arcface_loop(self) -> None:
        logger.info("GPUInferenceWorker ArcFace loop running")
        while self._running:
            try:
                face_batch = self._collect_faces()
                if not face_batch:
                    time.sleep(0.001)
                    continue

                all_crops: List[np.ndarray] = []
                crop_cam_ids: List[int] = []
                crop_track_ids: List[int] = []

                for cam_id, (rois, track_ids, _seq) in face_batch.items():
                    for roi, tid in zip(rois, track_ids):
                        all_crops.append(roi)
                        crop_cam_ids.append(cam_id)
                        crop_track_ids.append(tid)

                cam_results: Dict[int, Dict[int, Dict]] = {
                    cid: {} for cid in face_batch
                }
                if all_crops:
                    face_results = self._run_arcface_batch(all_crops)
                    for i, (cam_id, track_id) in enumerate(
                        zip(crop_cam_ids, crop_track_ids)
                    ):
                        cam_results[cam_id][track_id] = face_results[i]

                for cam_id, results in cam_results.items():
                    out_q = self._queue_for(
                        self._embedding_out_queues, cam_id, "arcface_distribute"
                    )
                    if out_q is not None:
                        out_q.put((face_batch[cam_id][2], results))

            except Exception as e:
                logger.exception(f"GPUInferenceWorker ArcFace error: {e}")

    # ── Collection helpers ────────────────────────────────────────────────────

    def _collect_batch(self, in_queues: Dict[int, queue.Queue]) -> Dict[int, Any]:
        """Block until at least one queue has an item, then drain any others ready now.

        Iterates a snapshot rather than the live dict: `add_camera`/`remove_camera`
        mutate these dicts from the engine's thread, and mutating a dict while
        another thread iterates it raises RuntimeError. Taking the snapshot each
        pass also means a camera added mid-wait is picked up on the next one.
        """
        batch: Dict[int, Any] = {}

        while self._running and not batch:
            with self._queues_lock:
                snapshot = list(in_queues.items())
            for cam_id, q in snapshot:
                try:
                    batch[cam_id] = q.get_nowait()
                except queue.Empty:
                    pass
            if not batch:
                time.sleep(0.001)

        with self._queues_lock:
            snapshot = list(in_queues.items())
        for cam_id, q in snapshot:
            if cam_id not in batch:
                try:
                    batch[cam_id] = q.get_nowait()
                except queue.Empty:
                    pass

        return batch

    def _collect_frames(self) -> Dict[int, Tuple[np.ndarray, int]]:
        return self._collect_batch(self._frame_in_queues)

    def _collect_faces(self) -> Dict[int, Tuple[List, List, int]]:
        return self._collect_batch(self._face_in_queues)

    # ── Inference helpers ─────────────────────────────────────────────────────

    # How long to wait for the YOLO worker's reply. Sits above the camera
    # side's own 2.0s get_detections timeout so that when the GPU worker is
    # merely slow, the camera's timeout fires first and drops one frame —
    # rather than this thread giving up and stranding a reply that the next
    # cycle would then have to discard as stale.
    YOLO_TASK_TIMEOUT_S = 3.0

    def _run_yolo_batch(
        self, batch: Dict[int, Tuple[np.ndarray, int]]
    ) -> List[List[Dict]]:
        """Run YOLO on a cross-camera batch via the `yolo` Celery worker.

        Takes the raw `{camera_id: (frame, frame_num)}` batch rather than a
        bare frame list: the frames must be packed into shared memory with
        their camera ids and frame numbers attached, and the packing order
        is what aligns the returned detections back to their cameras.

        Returns detections positionally aligned to `sorted(batch.keys())`,
        matching what the in-process version returned and what `_yolo_loop`
        already expects.

        Never raises. Any failure — dead worker, timeout, an exception inside
        the task — degrades to one empty detection list per frame, which ages
        each camera's tracks by a frame. That is what the tracker already
        does on a dropped frame, and it is strictly better than propagating
        an exception into the loop thread that owns every camera's queues.
        """
        if not batch:
            return []

        n_frames = len(batch)
        try:
            import dataclasses

            from workers.yolo_tasks import detect_batch_task

            t0 = time.time()
            handle = self._yolo_batch_slot.write(batch)
            async_result = detect_batch_task.delay(
                handle=dataclasses.asdict(handle)
            )
            detections = async_result.get(timeout=self.YOLO_TASK_TIMEOUT_S)
            duration_ms = (time.time() - t0) * 1000

            if self._metrics is not None:
                # Still recorded as "yolo ms", but note this is now
                # round-trip (pack + broker + inference + reply), not bare
                # inference — not comparable to older, in-process history.
                self._metrics.record_yolo_ms(duration_ms, batch_size=n_frames)

            if len(detections) != n_frames:
                # A malformed reply would silently misalign every camera's
                # detections. Refuse it.
                logger.error(
                    f"yolo.detect_batch returned {len(detections)} results for "
                    f"{n_frames} frames — discarding to avoid misrouting"
                )
                return [[] for _ in range(n_frames)]
            return detections
        except Exception as e:
            logger.warning(
                f"YOLO batch task failed ({type(e).__name__}: {e}) — "
                f"{n_frames} frame(s) get no detections this cycle"
            )
            return [[] for _ in range(n_frames)]

    # Sits above the camera side's own 2.0s get_embeddings timeout, for the
    # same reason as YOLO_TASK_TIMEOUT_S. The face path is the slower of the
    # two (SCRFD runs per-ROI and ArcFace per-crop, both un-batchable), so it
    # gets more headroom.
    FACE_TASK_TIMEOUT_S = 4.0

    def _run_arcface_batch(self, person_rois: List[np.ndarray]) -> List[Dict]:
        """Detect + embed faces for a flat, cross-camera list of person ROIs,
        via the `face` Celery worker.

        The inference itself — including the deliberate one-crop-per-call
        embedding loop — now lives in workers/face_tasks.py. What stays here
        is the transport: pack the ROIs into shared memory, dispatch, wait,
        and hand back one result per ROI in the order given, which is the
        alignment `_arcface_loop` scatters on.

        Never raises. Any failure yields the default "no face" dict for every
        ROI — the same thing the in-process version produced when detection
        failed, and what downstream already treats as "recognition didn't
        advance this cycle" rather than as corrupt identity data.
        """
        if not person_rois:
            return []

        n_rois = len(person_rois)

        def _blank() -> List[Dict]:
            return [
                {
                    "embedding": None,
                    "face_image": None,
                    "face_detected": False,
                    "det_score": 0.0,
                }
                for _ in range(n_rois)
            ]

        try:
            import dataclasses

            from workers.face_tasks import embed_batch_task

            t0 = time.time()
            # track_ids here are positional tags only: this batch is already
            # flattened across cameras by _arcface_loop, which re-associates
            # results by index, so the slot just needs stable per-ROI slots.
            handle = self._arcface_batch_slot.write(
                person_rois, list(range(n_rois))
            )
            async_result = embed_batch_task.delay(
                handle=dataclasses.asdict(handle)
            )
            results = async_result.get(timeout=self.FACE_TASK_TIMEOUT_S)
            duration_ms = (time.time() - t0) * 1000

            if self._metrics is not None:
                # Round-trip now (pack + broker + inference + reply), not bare
                # inference — not comparable to pre-Stage-2 history. The
                # det/embed split that record_arcface_det_ms and
                # record_arcface_embed_ms used to carry is logged inside the
                # task instead; only the total is observable from here.
                self._metrics.record_arcface_ms(duration_ms, batch_size=n_rois)

            if len(results) != n_rois:
                # A short or reordered reply would hand one person's embedding
                # to another person's track. Refuse it.
                logger.error(
                    f"face.embed_batch returned {len(results)} results for "
                    f"{n_rois} ROIs — discarding to avoid misrouting identities"
                )
                return _blank()
            return results
        except Exception as e:
            logger.warning(
                f"Face batch task failed ({type(e).__name__}: {e}) — "
                f"{n_rois} ROI(s) get no embeddings this cycle"
            )
            return _blank()

    # _parse_yolo_result moved to workers/yolo_tasks.py — it runs where
    # the ultralytics Results object exists, so that object
    # never has to cross the broker. Deliberately not left as a duplicate
    # here: two copies of detection parsing would drift.

    def close(self) -> None:
        """Release both batch slots. Called from stop(), after the loops that
        write to them have joined."""
        self._yolo_batch_slot.close()
        self._arcface_batch_slot.close()
