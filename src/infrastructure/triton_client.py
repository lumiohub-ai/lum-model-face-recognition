"""Triton gRPC inference client for YOLO person detection and ArcFace recognition.

Replaces local ONNX inference (Ultralytics YOLO + InsightFace) with remote
calls to the lum-triton Triton Inference Server via gRPC.

Preprocessing/postprocessing mirrors exactly what InsightFace and Ultralytics do
so results are numerically equivalent to running the models locally.
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger

# ArcFace 112×112 canonical landmark positions (InsightFace standard)
_ARCFACE_DST = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def _norm_crop(img: np.ndarray, landmark: np.ndarray, image_size: int = 112) -> np.ndarray:
    """Align face to canonical image_size×image_size using 5-point landmarks.

    Matches insightface.utils.face_align.norm_crop() output without requiring
    the insightface package at runtime.
    """
    dst = _ARCFACE_DST * (image_size / 112.0)
    M, _ = cv2.estimateAffinePartial2D(
        landmark.astype(np.float32), dst, method=cv2.LMEDS
    )
    return cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)


try:
    import tritonclient.grpc as grpcclient
    from tritonclient.grpc import InferInput, InferRequestedOutput
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False

# SCRFD (scrfd_10g_kps_dynamic_u8) config: 3 FPN strides, 2 anchors/location, 640×640 input
_SCRFD_STRIDES = [8, 16, 32]
_SCRFD_NUM_ANCHORS = 2
_SCRFD_INPUT_SIZE = (640, 640)

# Triton output names — must match arcface_det config.pbtxt (v3 model)
_DET_OUTPUT_NAMES = [
    "score_8", "score_16", "score_32",
    "bbox_8",  "bbox_16",  "bbox_32",
    "kps_8",   "kps_16",   "kps_32",
]


class TritonInferenceClient:
    """gRPC client for YOLO + ArcFace inference on NVIDIA Triton.

    Thread-safe: tritonclient.grpc.InferenceServerClient uses channel-level locking.
    """

    def __init__(
        self,
        url: str,
        confidence_threshold: float = 0.5,
        face_det_threshold: float = 0.5,
        face_nms_threshold: float = 0.4,
        face_padding_percent: float = 20.0,
    ):
        if not _TRITON_AVAILABLE:
            raise RuntimeError(
                "tritonclient[grpc] is not installed. "
                "Add `tritonclient[grpc]~=2.47.0` to requirements.txt."
            )
        self._url = url
        self.confidence_threshold = confidence_threshold
        self.face_det_threshold = face_det_threshold
        self.face_nms_threshold = face_nms_threshold
        self.face_padding_percent = max(0.0, face_padding_percent)
        self._center_cache: Dict[Tuple, np.ndarray] = {}

        self._client = grpcclient.InferenceServerClient(url=url, verbose=False)
        logger.info(f"TritonInferenceClient connected: {url}")

    # ── YOLO person detection ─────────────────────────────────────────────────

    def infer_yolo_batch(self, frames: List[np.ndarray]) -> List[List[Dict]]:
        """Batch YOLO person detection. One Triton call for all frames.

        Returns list (one per frame) of detection dicts:
          {'bbox': [x1,y1,x2,y2], 'confidence': float, 'keypoints': None, 'person_id': int}
        """
        if not frames:
            return []

        preprocessed, metas = [], []
        for frame in frames:
            img, scale, pw, ph = self._preprocess_yolo(frame)
            preprocessed.append(img)
            metas.append((scale, pw, ph))

        batch = np.stack(preprocessed, axis=0)  # [B,3,640,640] uint8

        try:
            inp = InferInput("images", list(batch.shape), "UINT8")
            inp.set_data_from_numpy(batch)
            response = self._client.infer(
                model_name="yolo_person",
                inputs=[inp],
                outputs=[InferRequestedOutput("output0")],
            )
            raw = response.as_numpy("output0")  # [B, 300, 6]
        except Exception as e:
            logger.exception(f"Triton yolo_person infer failed: {e}")
            return [[] for _ in frames]

        return [
            self._postprocess_yolo(raw[i], *metas[i])
            for i in range(len(frames))
        ]

    def _preprocess_yolo(
        self, frame: np.ndarray, input_size: int = 640
    ) -> Tuple[np.ndarray, float, int, int]:
        """Letterbox for YOLO. Returns (img_chw_uint8, scale, pad_w, pad_h).

        /255.0 normalization now happens inside the Triton model graph, not
        here — sending uint8 instead of float32 cuts the wire payload 4x.
        """
        h, w = frame.shape[:2]
        scale = min(input_size / h, input_size / w)
        new_h, new_w = int(round(h * scale)), int(round(w * scale))
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_h = (input_size - new_h) // 2
        pad_w = (input_size - new_w) // 2
        canvas = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
        canvas[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

        # BGR→RGB, HWC→CHW (still uint8)
        img = canvas[:, :, ::-1]
        return np.ascontiguousarray(img.transpose(2, 0, 1)), scale, pad_w, pad_h

    def _postprocess_yolo(
        self, raw: np.ndarray, scale: float, pad_w: int, pad_h: int
    ) -> List[Dict]:
        """Parse YOLO output [300,6] → detections in original frame coords."""
        detections = []
        for idx, (x1, y1, x2, y2, conf, cls_id) in enumerate(raw):
            if conf < self.confidence_threshold or int(cls_id) != 0:
                continue
            detections.append({
                "bbox": [
                    float((x1 - pad_w) / scale),
                    float((y1 - pad_h) / scale),
                    float((x2 - pad_w) / scale),
                    float((y2 - pad_h) / scale),
                ],
                "confidence": float(conf),
                "keypoints": None,
                "person_id": idx,
            })
        return detections

    # ── ArcFace face detection + recognition ──────────────────────────────────

    def infer_arcface_batch(self, person_rois: List[np.ndarray]) -> List[Dict]:
        """Detect face + extract embedding for each person ROI.

        Fix 1 + Fix 2: all N ROIs are detected in ONE arcface_det call, and
        all M detected faces are embedded in ONE arcface_rec call, instead of
        N+M sequential round-trips.

        Returns list of result dicts (one per input ROI):
          {'embedding': np.ndarray[512]|None, 'face_image': ndarray|None,
           'face_detected': bool, 'det_score': float,
           'face_bbox': [x1,y1,x2,y2], 'face_landmarks': [[x,y],...]}
        """
        _empty: Dict = {
            "embedding": None, "face_image": None,
            "face_detected": False, "det_score": 0.0,
        }

        # Skip None/empty ROIs; remember their original positions
        valid_indices = [
            i for i, r in enumerate(person_rois)
            if r is not None and r.size > 0
        ]
        if not valid_indices:
            return [_empty.copy() for _ in person_rois]

        valid_rois = [person_rois[i] for i in valid_indices]

        # ── Fix 2: all face detections in one Triton call ──────────────────
        try:
            face_detections = self._detect_faces_batch(valid_rois)
        except Exception as e:
            logger.debug(f"Triton arcface_det batch failed: {e}")
            return [_empty.copy() for _ in person_rois]

        # ── Fix 1: all embeddings in one Triton call ───────────────────────
        # Collect (roi, kps) pairs for ROIs that have a detected face
        face_roi_indices: List[int] = []    # index into valid_rois
        roi_kps_pairs: List[Tuple[np.ndarray, np.ndarray]] = []
        for i, det in enumerate(face_detections):
            if det is not None:
                _bbox, kps, _score = det
                face_roi_indices.append(i)
                roi_kps_pairs.append((valid_rois[i], kps))

        embeddings: List[Optional[np.ndarray]] = []
        if roi_kps_pairs:
            try:
                embeddings = self._get_embeddings_batch(roi_kps_pairs)
            except Exception as e:
                logger.debug(f"Triton arcface_rec batch failed: {e}")
                embeddings = [None] * len(roi_kps_pairs)

        # Map embedding back to valid_roi index
        emb_by_valid_idx: Dict[int, Optional[np.ndarray]] = {
            vi: emb for vi, emb in zip(face_roi_indices, embeddings)
        }

        # Assemble per-valid-roi results
        det_results = []
        for i, (roi, det) in enumerate(zip(valid_rois, face_detections)):
            if det is None:
                det_results.append(_empty.copy())
                continue
            bbox, kps, det_score = det
            x1, y1, x2, y2 = bbox.astype(int)
            face_crop = roi[max(0, y1):y2, max(0, x1):x2]
            det_results.append({
                "embedding": emb_by_valid_idx.get(i),
                "face_image": face_crop if face_crop.size > 0 else None,
                "face_detected": True,
                "det_score": float(det_score),
                "face_bbox": [int(x1), int(y1), int(x2), int(y2)],
                "face_landmarks": kps.reshape(5, 2).tolist(),
            })

        # Map back to original per-input-roi order
        results: List[Dict] = [_empty.copy() for _ in person_rois]
        for orig_idx, det_result in zip(valid_indices, det_results):
            results[orig_idx] = det_result
        return results

    # ── internal batch helpers ────────────────────────────────────────────────

    def _detect_faces_batch(
        self, rois: List[np.ndarray]
    ) -> List[Optional[Tuple[np.ndarray, np.ndarray, float]]]:
        """Run SCRFD on all ROIs in one Triton call.

        Returns list of (bbox_xyxy, kps_flat10, score) per ROI, or None for
        ROIs with no detected face.  Coordinates are in the original ROI space.
        """
        blobs, det_scales, pad_ws, pad_hs = [], [], [], []
        for roi in rois:
            pad_w, pad_h = 0, 0
            if self.face_padding_percent > 0:
                h, w = roi.shape[:2]
                pad_h = int(h * self.face_padding_percent / 100)
                pad_w = int(w * self.face_padding_percent / 100)
                roi = cv2.copyMakeBorder(
                    roi, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_REPLICATE
                )
            blob, det_scale = self._preprocess_det(roi)
            blobs.append(blob)
            det_scales.append(det_scale)
            pad_ws.append(pad_w)
            pad_hs.append(pad_h)

        batch = np.concatenate(blobs, axis=0)  # [B, 3, 640, 640] uint8

        inp = InferInput("input.1", list(batch.shape), "UINT8")
        inp.set_data_from_numpy(batch)
        response = self._client.infer(
            model_name="arcface_det",
            inputs=[inp],
            outputs=[InferRequestedOutput(n) for n in _DET_OUTPUT_NAMES],
        )
        net_outs = [response.as_numpy(n) for n in _DET_OUTPUT_NAMES]
        # net_outs[i] shape: [B, N_anchors, C]

        results = []
        for b_idx in range(len(rois)):
            per_img = [out[b_idx] for out in net_outs]  # each [N_anchors, C]
            scores_all, bboxes_all, kpss_all = self._decode_scrfd(per_img)
            face = self._pick_best_face(
                scores_all, bboxes_all, kpss_all,
                det_scales[b_idx], pad_ws[b_idx], pad_hs[b_idx],
            )
            results.append(face)
        return results

    def _get_embeddings_batch(
        self, roi_kps_pairs: List[Tuple[np.ndarray, np.ndarray]]
    ) -> List[Optional[np.ndarray]]:
        """Align all faces and extract embeddings in one arcface_rec call.

        Args:
            roi_kps_pairs: list of (roi_bgr, kps_flat10) — original ROI (no padding)
                           and the 10-element flat keypoints array in ROI coords.
        Returns:
            list of L2-normalised [512] embeddings (or None on failure).
        """
        crops = []
        for roi, kps_flat in roi_kps_pairs:
            aligned = _norm_crop(roi, kps_flat.reshape(5, 2))      # [112,112,3] uint8
            crops.append(aligned.transpose(2, 0, 1))               # [3,112,112]

        batch = np.ascontiguousarray(np.stack(crops, axis=0))      # [B,3,112,112] uint8

        inp = InferInput("input.1", list(batch.shape), "UINT8")
        inp.set_data_from_numpy(batch)
        response = self._client.infer(
            model_name="arcface_rec",
            inputs=[inp],
            outputs=[InferRequestedOutput("683")],
        )
        embs = response.as_numpy("683")  # [B, 512]

        results = []
        for emb in embs:
            norm = np.linalg.norm(emb)
            results.append(emb / norm if norm > 0 else emb)
        return results

    # ── SCRFD pre/postprocessing ──────────────────────────────────────────────

    def _preprocess_det(
        self, img: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Preprocess one ROI for SCRFD. Returns (blob [1,3,640,640] uint8, det_scale).

        Aspect-ratio-preserving resize → zero-pad → NCHW. The (x-127.5)/128
        normalisation happens inside the Triton model graph.
        """
        h, w = img.shape[:2]
        ih, iw = _SCRFD_INPUT_SIZE

        im_ratio = float(h) / w
        model_ratio = float(ih) / iw
        if im_ratio > model_ratio:
            new_h, new_w = ih, int(ih / im_ratio)
        else:
            new_w, new_h = iw, int(iw * im_ratio)
        det_scale = float(new_h) / h

        resized = cv2.resize(img, (new_w, new_h))
        canvas = np.zeros((ih, iw, 3), dtype=np.uint8)
        canvas[:new_h, :new_w] = resized

        blob = canvas.transpose(2, 0, 1)[np.newaxis, :]  # [1,3,640,640] uint8
        return np.ascontiguousarray(blob), det_scale

    def _decode_scrfd(
        self, net_outs: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """Decode 9 SCRFD FPN outputs for a single image → (scores, bboxes, kps).

        net_outs ordering matches _DET_OUTPUT_NAMES:
          [score_8, score_16, score_32, bbox_8, bbox_16, bbox_32, kps_8, kps_16, kps_32]
        Each tensor has shape [N_anchors, C] (already sliced per-image by caller).
        """
        fmc = len(_SCRFD_STRIDES)  # 3
        ih, iw = _SCRFD_INPUT_SIZE
        scores_list, bboxes_list, kpss_list = [], [], []

        for idx, stride in enumerate(_SCRFD_STRIDES):
            scores = net_outs[idx]             # [n_anchors, 1]
            bbox_preds = net_outs[idx + fmc]   # [n_anchors, 4]
            kps_preds = net_outs[idx + fmc*2]  # [n_anchors, 10]

            feat_h, feat_w = ih // stride, iw // stride
            key = (feat_h, feat_w, stride)
            if key not in self._center_cache:
                centers = np.stack(
                    np.mgrid[:feat_h, :feat_w][::-1], axis=-1
                ).astype(np.float32) * stride
                centers = centers.reshape(-1, 2)
                centers = np.stack([centers] * _SCRFD_NUM_ANCHORS, axis=1).reshape(-1, 2)
                self._center_cache[key] = centers
            ac = self._center_cache[key]

            bboxes = np.stack([
                ac[:, 0] - bbox_preds[:, 0] * stride,
                ac[:, 1] - bbox_preds[:, 1] * stride,
                ac[:, 0] + bbox_preds[:, 2] * stride,
                ac[:, 1] + bbox_preds[:, 3] * stride,
            ], axis=-1)

            kps_parts = []
            for k in range(0, kps_preds.shape[1], 2):
                kps_parts.append(ac[:, 0] + kps_preds[:, k] * stride)
                kps_parts.append(ac[:, 1] + kps_preds[:, k + 1] * stride)
            kpss = np.stack(kps_parts, axis=-1)

            scores_list.append(scores[:, 0])
            bboxes_list.append(bboxes)
            kpss_list.append(kpss)

        return scores_list, bboxes_list, kpss_list

    def _pick_best_face(
        self,
        scores_all: List[np.ndarray],
        bboxes_all: List[np.ndarray],
        kpss_all: List[np.ndarray],
        det_scale: float,
        pad_w: int,
        pad_h: int,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
        """NMS + pick highest-scoring face. Returns (bbox_xyxy, kps_flat10, score)
        in original ROI coordinates, or None if no face passes threshold.
        """
        if not scores_all:
            return None

        scores = np.concatenate(scores_all)
        bboxes = np.concatenate(bboxes_all)
        kpss = np.concatenate(kpss_all)

        pos = np.where(scores >= self.face_det_threshold)[0]
        if len(pos) == 0:
            return None

        scores, bboxes, kpss = scores[pos], bboxes[pos], kpss[pos]
        keep = self._nms(bboxes, scores, self.face_nms_threshold)
        if len(keep) == 0:
            return None

        best = keep[np.argmax(scores[keep])]
        bbox = bboxes[best] / det_scale
        kps = kpss[best] / det_scale

        if pad_w or pad_h:
            bbox -= [pad_w, pad_h, pad_w, pad_h]
            kps[0::2] -= pad_w
            kps[1::2] -= pad_h

        return bbox, kps, float(scores[best])

    @staticmethod
    def _nms(bboxes: np.ndarray, scores: np.ndarray, threshold: float) -> np.ndarray:
        """Greedy NMS. Returns indices of kept boxes in score-descending order."""
        x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            order = order[1:][iou <= threshold]
        return np.array(keep, dtype=np.int32)
