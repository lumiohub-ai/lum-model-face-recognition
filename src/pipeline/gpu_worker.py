"""GPU Inference Worker — serialises all GPU operations across camera threads.

Two independent GPU threads:
  - YOLO thread:    collect frames (any camera ready) → batch YOLO → distribute detections
  - ArcFace thread: collect faces  (any camera ready) → batch ArcFace → distribute embeddings

Camera threads are fully independent — a slow camera never blocks a fast one.
"""

import queue
import threading
import time
from collections import Counter
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
        t_submit = time.perf_counter()
        try:
            self._frame_in_queues[camera_idx].put_nowait((frame, frame_num, t_submit))
        except queue.Full:
            try:
                self._frame_in_queues[camera_idx].get_nowait()
            except queue.Empty:
                pass
            self._frame_in_queues[camera_idx].put_nowait((frame, frame_num, t_submit))
            if self._metrics is not None:
                self._metrics.record_drop(camera_idx)

    def get_detections(
        self, camera_idx: int, timeout: float = 2.0
    ) -> Tuple[List[Dict], Dict[str, float]]:
        """Block until YOLO detections are available for this camera.

        Returns (detections, timing) where timing is {"wait_ms", "gpu_ms"} —
        time this camera spent queued for the shared YOLO thread vs. its
        amortized share of the batch inference itself.
        """
        try:
            return self._detection_out_queues[camera_idx].get(timeout=timeout)
        except queue.Empty:
            logger.warning(f"get_detections timeout for camera {camera_idx}")
            return [], {}

    def submit_faces(
        self,
        camera_idx: int,
        person_rois: List[np.ndarray],
        track_ids: List[int],
    ) -> None:
        """Submit person ROI crops for ArcFace embedding."""
        self._face_in_queues[camera_idx].put(
            (person_rois, track_ids, time.perf_counter())
        )

    def get_embeddings(
        self, camera_idx: int, timeout: float = 2.0
    ) -> Tuple[Dict[int, Dict], Dict[str, float]]:
        """Block until ArcFace results are available for this camera.

        Returns (embeddings_map, timing) — see get_detections() for the
        wait_ms/gpu_ms shape.
        """
        try:
            return self._embedding_out_queues[camera_idx].get(timeout=timeout)
        except queue.Empty:
            logger.warning(f"get_embeddings timeout for camera {camera_idx}")
            return {}, {}

    # ── YOLO loop ─────────────────────────────────────────────────────────────

    def _yolo_loop(self) -> None:
        logger.info("GPUInferenceWorker YOLO loop running")
        while self._running:
            try:
                batch = self._collect_frames()
                if not batch:
                    time.sleep(0.001)
                    continue

                t_dequeue = time.perf_counter()
                cam_ids = sorted(batch.keys())
                frames = [batch[cid][0] for cid in cam_ids]
                all_detections, batch_ms = self._run_yolo_batch(frames)

                # Amortized, not exact — one batch call serves every camera in
                # cam_ids, so an equal split is the best per-camera estimate of
                # compute cost available without instrumenting the model itself.
                gpu_ms_per_cam = batch_ms / len(cam_ids) if cam_ids else 0.0

                for i, cam_id in enumerate(cam_ids):
                    t_submit = batch[cam_id][2]
                    wait_ms = max(0.0, (t_dequeue - t_submit) * 1000)
                    timing = {"wait_ms": wait_ms, "gpu_ms": gpu_ms_per_cam}
                    self._detection_out_queues[cam_id].put((all_detections[i], timing))

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

                t_dequeue = time.perf_counter()
                all_crops: List[np.ndarray] = []
                crop_cam_ids: List[int] = []
                crop_track_ids: List[int] = []

                for cam_id, (rois, track_ids, _t_submit) in face_batch.items():
                    for roi, tid in zip(rois, track_ids):
                        all_crops.append(roi)
                        crop_cam_ids.append(cam_id)
                        crop_track_ids.append(tid)

                cam_results: Dict[int, Dict[int, Dict]] = {
                    cid: {} for cid in face_batch
                }
                batch_ms = 0.0
                if all_crops:
                    face_results, batch_ms = self._run_arcface_batch(all_crops)
                    for i, (cam_id, track_id) in enumerate(
                        zip(crop_cam_ids, crop_track_ids)
                    ):
                        cam_results[cam_id][track_id] = face_results[i]

                # ArcFace runs serially per-ROI, so split compute time
                # proportionally to how many crops each camera contributed —
                # an equal per-camera split would be wrong when ROI counts differ.
                crops_per_cam = Counter(crop_cam_ids)
                ms_per_crop = (batch_ms / len(all_crops)) if all_crops else 0.0

                for cam_id, results in cam_results.items():
                    t_submit = face_batch[cam_id][2]
                    wait_ms = max(0.0, (t_dequeue - t_submit) * 1000)
                    gpu_ms = crops_per_cam.get(cam_id, 0) * ms_per_crop
                    timing = {"wait_ms": wait_ms, "gpu_ms": gpu_ms}
                    self._embedding_out_queues[cam_id].put((results, timing))

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

    def _collect_frames(self) -> Dict[int, Tuple[np.ndarray, int, float]]:
        return self._collect_batch(self._frame_in_queues)

    def _collect_faces(self) -> Dict[int, Tuple[List, List, float]]:
        return self._collect_batch(self._face_in_queues)

    # ── Inference helpers ─────────────────────────────────────────────────────

    def _run_yolo_batch(
        self, frames: List[np.ndarray]
    ) -> Tuple[List[List[Dict]], float]:
        """Run YOLO on a batch of frames; returns (per-frame detections, batch_ms)."""
        if not frames:
            return [], 0.0
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
                self._metrics.record_yolo_ms(duration_ms)
            return [self._parse_yolo_result(r) for r in results], duration_ms
        except Exception as e:
            logger.exception(f"YOLO batch inference failed: {e}")
            return [[] for _ in frames], 0.0

    def _run_arcface_batch(
        self, person_rois: List[np.ndarray]
    ) -> Tuple[List[Dict], float]:
        """Detect face and extract embedding for each person ROI; returns (results, batch_ms)."""
        t0 = time.time()
        results = []
        for roi in person_rois:
            result: Dict = {
                "embedding": None,
                "face_image": None,
                "face_detected": False,
                "det_score": 0.0,
            }
            if roi is None or roi.size == 0:
                results.append(result)
                continue
            try:
                ## Here we inference the face detection model on the ROI
                ## The function calls both RetinaFace and Arcface inside of InsideFace.
                faces = self._face_detector.detect(roi)
                if faces:
                    face = faces[0]
                    # face.bbox / face.kps are already in ROI coordinates; the
                    # detector's internal padding is undone before it returns.
                    x1, y1, x2, y2 = face.bbox.astype(int)
                    face_crop = roi[max(0, y1):y2, max(0, x1):x2]

                    kps = None
                    if hasattr(face, "kps") and face.kps is not None:
                        kps = face.kps.astype(int).tolist()

                    result = {
                        "embedding": face.embedding,
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
            except Exception as e:
                logger.debug(f"Face detection error on ROI: {e}")
            results.append(result)
        duration_ms = (time.time() - t0) * 1000
        if self._metrics is not None and results:
            self._metrics.record_arcface_ms(duration_ms)
        return results, duration_ms

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
