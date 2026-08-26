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

from lum_vision.face_detection import frontality, pitch


class GPUInferenceWorker:
    """Batched GPU inference worker shared across all camera threads.

    YOLO and ArcFace run in separate threads so face embedding for one camera
    never waits on another camera's CPU tracking.
    """

    def __init__(
        self,
        detector,
        face_detector,
        camera_ids: Sequence[int],
        metrics_collector=None,
    ):
        self._detector = detector
        self._face_detector = face_detector
        self._running = False
        self._yolo_thread: Optional[threading.Thread] = None
        self._arcface_thread: Optional[threading.Thread] = None
        self._metrics = metrics_collector  # Optional[MetricsCollector]

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

        Responses are matched, not counted (LSO-138). A reply the caller gave up
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

                cam_ids = sorted(batch.keys())
                frames = [batch[cid][0] for cid in cam_ids]
                all_detections = self._run_yolo_batch(frames)

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

    def _run_yolo_batch(self, frames: List[np.ndarray]) -> List[List[Dict]]:
        """Run YOLO on a batch of frames and return per-frame detections."""
        if not frames:
            return []
        try:
            t0 = time.time()
            # Here we inference the batch of frames using the YOLO model. The model is expected to return a list of results, one for each frame.
            results = self._detector.model(
                frames,
                conf=self._detector.confidence_threshold,
                iou=self._detector.iou_threshold,
                verbose=False,
                device=self._detector.device,
            )
            duration_ms = (time.time() - t0) * 1000
            if self._metrics is not None:
                self._metrics.record_yolo_ms(duration_ms, batch_size=len(frames))
            return [self._parse_yolo_result(r) for r in results]
        except Exception as e:
            logger.exception(f"YOLO batch inference failed: {e}")
            return [[] for _ in frames]

    def _run_arcface_batch(self, person_rois: List[np.ndarray]) -> List[Dict]:
        """Detect face and extract embedding for each person ROI.

        Detection and embedding are separate calls (LSO-117) so each can be
        timed on its own, but both run one ROI/crop at a time: SCRFD has no
        batch path (LSO-118), and embedding at a varying batch size is far
        slower than at a fixed one - see the comment on the embed loop below.
        """
        t0 = time.time()
        results: List[Dict] = []
        # (result_idx, face) for every ROI that had >=1 detected face with
        # landmarks; parallel to crops so embed_batch()'s output lines up.
        faces_to_embed: List[Tuple[int, Any]] = []
        crops: List[np.ndarray] = []

        # Timed separately from embedding below (LSO-117): detection is still
        # N per-ROI calls (SCRFD has no batch path - LSO-118), so this number
        # won't move with batch size the way embedding does. An offline
        # benchmark (lum-model-vision#16) found detection at ~93% of total
        # ArcFace time at N=362 faces - det_t0/det_ms_total is what confirms
        # (or updates) that under real production load.
        det_t0 = time.time()
        for i, roi in enumerate(person_rois):
            result: Dict = {
                "embedding": None,
                "face_image": None,
                "face_detected": False,
                "det_score": 0.0,
            }
            results.append(result)
            if roi is None or roi.size == 0:
                continue
            try:
                ## Detect + align faces in the ROI (no embedding yet - that
                ## happens once, batched, after this loop over all ROIs).
                faces = self._face_detector.detect_and_align(roi)
                if faces and faces[0].aligned_crop is not None:
                    faces_to_embed.append((i, faces[0]))
                    crops.append(faces[0].aligned_crop)
                # A face with no landmarks (aligned_crop is None) can't be
                # embedded, same as the old per-face path (alignment there
                # required kps too) - result stays the default no-face dict.
            except Exception as e:
                logger.debug(f"Face detection error on ROI: {e}")
        det_ms_total = (time.time() - det_t0) * 1000
        if self._metrics is not None:
            self._metrics.record_arcface_det_ms(det_ms_total, batch_size=len(person_rois))

        if crops:
            embed_t0 = time.time()
            # One crop per call, deliberately: ORT's CUDA EP re-plans on every
            # input-shape change, so a batch size that varies cycle-to-cycle
            # costs ~80ms vs ~2.6ms at a fixed shape. Batching all crops in one
            # call shipped as 0.3.0 and halved prod FPS. Don't reintroduce it.
            embeddings: List[Any] = []
            last_error: Optional[Exception] = None
            for crop in crops:
                try:
                    out = self._face_detector.embed_batch([crop])
                except Exception as e:
                    last_error = e
                    out = None
                embeddings.append(out[0] if out is not None and len(out) > 0 else None)
            failed = sum(1 for e in embeddings if e is None)
            if failed:
                # One line per cycle, not per face: a hard embedder failure
                # (CUDA OOM, model unloaded) would otherwise log once per face
                # per cycle across every camera.
                reason = last_error if last_error is not None else "embedder returned no result"
                logger.warning(
                    f"Face embedding failed for {failed}/{len(crops)} faces "
                    f"this cycle: {reason}"
                )
            if self._metrics is not None:
                self._metrics.record_arcface_embed_ms(
                    (time.time() - embed_t0) * 1000, batch_size=len(crops)
                )

            for (i, face), embedding in zip(faces_to_embed, embeddings):
                if embedding is None:
                    # Failed embedding stays "no face", as before 0.3.0 —
                    # downstream treats face_detected as "has an embedding".
                    continue
                roi = person_rois[i]
                # face.bbox / face.kps are already in ROI coordinates; the
                # detector's internal padding is undone before it returns.
                x1, y1, x2, y2 = face.bbox.astype(int)
                face_crop = roi[max(0, y1):y2, max(0, x1):x2]

                kps = None
                if hasattr(face, "kps") and face.kps is not None:
                    kps = face.kps.astype(int).tolist()

                results[i] = {
                    "embedding": embedding,
                    "face_image": face_crop if face_crop.size > 0 else None,
                    "face_detected": True,
                    "det_score": (
                        float(face.det_score)
                        if hasattr(face, "det_score")
                        else 0.0
                    ),
                    "face_bbox": [x1, y1, x2, y2],
                    "face_landmarks": kps,
                    # Orientation proxies from the package (single source of
                    # truth); the unrecognized-case gate reads these.
                    "frontality": frontality(kps),
                    "pitch": pitch(kps),
                }

        duration_ms = (time.time() - t0) * 1000
        if self._metrics is not None and results:
            self._metrics.record_arcface_ms(duration_ms, batch_size=len(results))
        return results

    @staticmethod
    def _parse_yolo_result(result) -> List[Dict]:
        """Convert a YOLO result object to a list of detection dicts."""
        detections = []
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return detections
        for idx in range(len(boxes)):
            cls_id = int(boxes.cls[idx].cpu().numpy())
            if cls_id != 0:
                continue
            bbox = boxes.xyxy[idx].cpu().numpy().tolist()
            conf = float(boxes.conf[idx].cpu().numpy())
            detections.append(
                {
                    "bbox": bbox,
                    "confidence": conf,
                    "keypoints": None,
                    "person_id": idx,
                }
            )
        return detections
