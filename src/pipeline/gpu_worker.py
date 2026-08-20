"""GPU Inference Worker — serialises all GPU operations across camera threads.

Two independent GPU threads:
  - YOLO thread:    collect frames (any camera ready) → batch YOLO → distribute detections
  - ArcFace thread: collect faces  (any camera ready) → batch ArcFace → distribute embeddings

Camera threads are fully independent — a slow camera never blocks a fast one.
"""

import queue
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

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
        num_cameras: int,
        metrics_collector=None,
    ):
        self._detector = detector
        self._face_detector = face_detector
        self._num_cameras = num_cameras
        self._running = False
        self._yolo_thread: Optional[threading.Thread] = None
        self._arcface_thread: Optional[threading.Thread] = None
        self._metrics = metrics_collector  # Optional[MetricsCollector]

        # Per-camera queues indexed by camera_idx (0-based)
        self._frame_in_queues: Dict[int, queue.Queue] = {
            i: queue.Queue(maxsize=2) for i in range(num_cameras)
        }
        self._detection_out_queues: Dict[int, queue.Queue] = {
            i: queue.Queue(maxsize=2) for i in range(num_cameras)
        }
        self._face_in_queues: Dict[int, queue.Queue] = {
            i: queue.Queue(maxsize=4) for i in range(num_cameras)
        }
        self._embedding_out_queues: Dict[int, queue.Queue] = {
            i: queue.Queue(maxsize=4) for i in range(num_cameras)
        }

        logger.debug(f"GPUInferenceWorker: {num_cameras} camera(s)")

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

    def submit_frame(
        self, camera_idx: int, frame: np.ndarray, frame_num: int
    ) -> None:
        """Submit a frame for YOLO detection (non-blocking; drops oldest if full)."""
        try:
            self._frame_in_queues[camera_idx].put_nowait((frame, frame_num))
        except queue.Full:
            try:
                self._frame_in_queues[camera_idx].get_nowait()
            except queue.Empty:
                pass
            self._frame_in_queues[camera_idx].put_nowait((frame, frame_num))
            if self._metrics is not None:
                self._metrics.record_drop(camera_idx)

    def get_detections(
        self, camera_idx: int, timeout: float = 2.0
    ) -> List[Dict]:
        """Block until YOLO detections are available for this camera."""
        try:
            return self._detection_out_queues[camera_idx].get(timeout=timeout)
        except queue.Empty:
            logger.warning(f"get_detections timeout for camera {camera_idx}")
            return []

    def submit_faces(
        self,
        camera_idx: int,
        person_rois: List[np.ndarray],
        track_ids: List[int],
    ) -> None:
        """Submit person ROI crops for ArcFace embedding."""
        self._face_in_queues[camera_idx].put((person_rois, track_ids))

    def get_embeddings(
        self, camera_idx: int, timeout: float = 2.0
    ) -> Dict[int, Dict]:
        """Block until ArcFace results are available for this camera."""
        try:
            return self._embedding_out_queues[camera_idx].get(timeout=timeout)
        except queue.Empty:
            logger.warning(f"get_embeddings timeout for camera {camera_idx}")
            return {}

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
                    self._detection_out_queues[cam_id].put(all_detections[i])

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

                for cam_id, (rois, track_ids) in face_batch.items():
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
                    self._embedding_out_queues[cam_id].put(results)

            except Exception as e:
                logger.exception(f"GPUInferenceWorker ArcFace error: {e}")

    # ── Collection helpers ────────────────────────────────────────────────────

    def _collect_batch(self, in_queues: Dict[int, queue.Queue]) -> Dict[int, Any]:
        """Block until at least one queue has an item, then drain any others ready now."""
        batch: Dict[int, Any] = {}

        while self._running and not batch:
            for cam_id, q in in_queues.items():
                try:
                    batch[cam_id] = q.get_nowait()
                except queue.Empty:
                    pass
            if not batch:
                time.sleep(0.001)

        for cam_id, q in in_queues.items():
            if cam_id not in batch:
                try:
                    batch[cam_id] = q.get_nowait()
                except queue.Empty:
                    pass

        return batch

    def _collect_frames(self) -> Dict[int, Tuple[np.ndarray, int]]:
        return self._collect_batch(self._frame_in_queues)

    def _collect_faces(self) -> Dict[int, Tuple[List, List]]:
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

        Detection is still per-ROI (SCRFD has no batch path in this
        insightface version), but embedding is now ONE detect_and_align() +
        embed_batch() call across every ROI in this batch, instead of the
        old per-ROI detect() (which internally did N separate single-face
        embedding calls for N faces) - LSO-117.
        """
        t0 = time.time()
        results: List[Dict] = []
        # (result_idx, face) for every ROI that had >=1 detected face with
        # landmarks; parallel to crops so embed_batch()'s output lines up.
        faces_to_embed: List[tuple] = []
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
            try:
                ## The one GPU call this method exists to make possible -
                ## every face found above, embedded together.
                embeddings = self._face_detector.embed_batch(crops)
            except Exception as e:
                logger.debug(f"Batch face embedding error: {e}")
                embeddings = None
            if self._metrics is not None:
                self._metrics.record_arcface_embed_ms(
                    (time.time() - embed_t0) * 1000, batch_size=len(crops)
                )

            if embeddings is not None:
                for (i, face), embedding in zip(faces_to_embed, embeddings):
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
