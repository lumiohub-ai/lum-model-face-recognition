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
        """Submit person ROI crops for ArcFace embedding (non-blocking; drops oldest if full)."""
        q = self._face_in_queues[camera_idx]
        try:
            q.put_nowait((person_rois, track_ids))
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            q.put_nowait((person_rois, track_ids))

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
                    q = self._detection_out_queues[cam_id]
                    try:
                        q.put_nowait(all_detections[i])
                    except queue.Full:
                        try:
                            q.get_nowait()
                        except queue.Empty:
                            pass
                        q.put_nowait(all_detections[i])

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
                    q = self._embedding_out_queues[cam_id]
                    try:
                        q.put_nowait(results)
                    except queue.Full:
                        try:
                            q.get_nowait()
                        except queue.Empty:
                            pass
                        q.put_nowait(results)

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
            return [self._parse_yolo_result(r) for r in results]
        except Exception as e:
            logger.exception(f"YOLO batch inference failed: {e}")
            return [[] for _ in frames]

    def _run_arcface_batch(self, person_rois: List[np.ndarray]) -> List[Dict]:
        """Detect face and extract embedding for each person ROI."""
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
                faces = self._face_detector.detect(roi)
                if faces:
                    # A person ROI can include nearby faces in crowded scenes.
                    # Use the strongest/largest face instead of InsightFace's
                    # first result so identity and attributes stay tied to the
                    # tracked person as often as possible.
                    face = max(faces, key=self._face_rank)
                    # face.bbox / face.kps are in padded-image coordinates.
                    # Subtract the padding offset to get back to ROI space.
                    roi_h, roi_w = roi.shape[:2]
                    pad_pct = self._face_detector.padding_percent
                    pad_w = int(roi_w * pad_pct / 100)
                    pad_h = int(roi_h * pad_pct / 100)

                    x1, y1, x2, y2 = face.bbox.astype(int)
                    x1 -= pad_w; y1 -= pad_h; x2 -= pad_w; y2 -= pad_h

                    # Add 40% padding around the face so RetinaFace can detect
                    # it when this crop is later used as a user enrollment photo
                    face_w = x2 - x1
                    face_h = y2 - y1
                    save_pad_x = int(face_w * 0.4)
                    save_pad_y = int(face_h * 0.4)
                    sx1 = max(0, x1 - save_pad_x)
                    sy1 = max(0, y1 - save_pad_y)
                    sx2 = min(roi_w, x2 + save_pad_x)
                    sy2 = min(roi_h, y2 + save_pad_y)
                    face_crop = roi[sy1:sy2, sx1:sx2]

                    kps = None
                    if hasattr(face, "kps") and face.kps is not None:
                        kps = (face.kps.astype(int) - [pad_w, pad_h]).tolist()

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
                        "face_width": max(0, face_w),
                        "face_height": max(0, face_h),
                        "face_landmarks": kps,
                        "gender": getattr(face, "sex", getattr(face, "gender", None)),
                        "age": int(round(face.age)) if getattr(face, "age", None) is not None else None,
                    }
            except Exception as e:
                logger.debug(f"Face detection error on ROI: {e}")
            results.append(result)
        duration_ms = (time.time() - t0) * 1000
        if self._metrics is not None and results:
            self._metrics.record_arcface_ms(duration_ms)
        return results

    @staticmethod
    def _face_rank(face) -> Tuple[float, float]:
        """Rank detected faces by confidence first, then face area."""
        det_score = float(getattr(face, "det_score", 0.0) or 0.0)
        try:
            x1, y1, x2, y2 = face.bbox.astype(float)
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        except Exception:
            area = 0.0
        return det_score, area

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
