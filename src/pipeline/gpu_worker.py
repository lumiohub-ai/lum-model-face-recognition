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


class GPUInferenceWorker:
    """Batched GPU inference worker shared across all camera threads.

    YOLO and ArcFace run in separate threads so face embedding for one camera
    never waits on another camera's CPU tracking.

    If a TritonInferenceClient is provided, all GPU inference is routed through
    the Triton server (gRPC). Local detector/face_detector are used as fallback.
    """

    def __init__(
        self,
        detector=None,
        face_detector=None,
        num_cameras: int = 1,
        metrics_collector=None,
        triton_client=None,
    ):
        self._detector = detector
        self._face_detector = face_detector
        self._triton_client = triton_client
        self._num_cameras = num_cameras
        self._running = False
        self._yolo_thread: Optional[threading.Thread] = None
        self._arcface_thread: Optional[threading.Thread] = None
        self._metrics = metrics_collector  # Optional[MetricsCollector]

        backend = "triton" if triton_client is not None else "local"
        logger.debug(f"GPUInferenceWorker: backend={backend}")

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
        """Run YOLO on a batch of frames. Routes to Triton if client is configured."""
        if not frames:
            return []
        t0 = time.time()
        try:
            if self._triton_client is not None:
                results = self._triton_client.infer_yolo_batch(frames)
            else:
                raw = self._detector.model(
                    frames,
                    conf=self._detector.confidence_threshold,
                    iou=self._detector.iou_threshold,
                    verbose=False,
                    device=self._detector.device,
                )
                results = [self._parse_yolo_result(r) for r in raw]
            duration_ms = (time.time() - t0) * 1000
            if self._metrics is not None:
                self._metrics.record_yolo_ms(duration_ms)
            return results
        except Exception as e:
            logger.exception(f"YOLO batch inference failed: {e}")
            return [[] for _ in frames]

    def _run_arcface_batch(self, person_rois: List[np.ndarray]) -> List[Dict]:
        """Detect face and extract embedding for each person ROI.

        Routes to Triton if client is configured; otherwise uses local InsightFace.
        """
        t0 = time.time()
        try:
            if self._triton_client is not None:
                results = self._triton_client.infer_arcface_batch(person_rois)
            else:
                results = self._run_arcface_local(person_rois)
        except Exception as e:
            logger.exception(f"ArcFace batch inference failed: {e}")
            results = [
                {"embedding": None, "face_image": None, "face_detected": False, "det_score": 0.0}
                for _ in person_rois
            ]
        duration_ms = (time.time() - t0) * 1000
        if self._metrics is not None and results:
            self._metrics.record_arcface_ms(duration_ms)
        if person_rois and duration_ms > 50:
            logger.info(f"ArcFace batch: {len(person_rois)} ROI(s) in {duration_ms:.0f}ms")
        return results

    def _run_arcface_local(self, person_rois: List[np.ndarray]) -> List[Dict]:
        """Local InsightFace inference path (used when Triton is not configured)."""
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
                faces = self._face_detector.detect(roi)
                if faces:
                    face = faces[0]
                    roi_h, roi_w = roi.shape[:2]
                    pad_pct = self._face_detector.padding_percent
                    pad_w = int(roi_w * pad_pct / 100)
                    pad_h = int(roi_h * pad_pct / 100)

                    x1, y1, x2, y2 = face.bbox.astype(int)
                    x1 -= pad_w; y1 -= pad_h; x2 -= pad_w; y2 -= pad_h
                    face_crop = roi[max(0, y1):y2, max(0, x1):x2]

                    kps = None
                    if hasattr(face, "kps") and face.kps is not None:
                        kps = (face.kps.astype(int) - [pad_w, pad_h]).tolist()

                    result = {
                        "embedding": face.embedding,
                        "face_image": face_crop if face_crop.size > 0 else None,
                        "face_detected": True,
                        "det_score": float(face.det_score) if hasattr(face, "det_score") else 0.0,
                        "face_bbox": [x1, y1, x2, y2],
                        "face_landmarks": kps,
                    }
            except Exception as e:
                logger.debug(f"Face detection error on ROI: {e}")
            results.append(result)
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
