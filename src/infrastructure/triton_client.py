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

# SCRFD (det_10g.onnx) config: 3 FPN strides, 2 anchors per location, fixed 640×640 input
_SCRFD_STRIDES = [8, 16, 32]
_SCRFD_NUM_ANCHORS = 2
_SCRFD_INPUT_SIZE = (640, 640)

# Triton model output names — must match config.pbtxt declarations
_DET_OUTPUT_NAMES = ["448", "471", "494", "451", "474", "497", "454", "477", "500"]


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

        batch = np.stack(preprocessed, axis=0).astype(np.float32)  # [B,3,640,640]

        try:
            inp = InferInput("images", list(batch.shape), "FP32")
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
        """Letterbox + normalize for YOLO. Returns (img_chw_float32, scale, pad_w, pad_h)."""
        h, w = frame.shape[:2]
        scale = min(input_size / h, input_size / w)
        new_h, new_w = int(round(h * scale)), int(round(w * scale))
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_h = (input_size - new_h) // 2
        pad_w = (input_size - new_w) // 2
        canvas = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
        canvas[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

        # BGR→RGB, normalize [0,1], HWC→CHW
        img = canvas[:, :, ::-1].astype(np.float32) / 255.0
        return img.transpose(2, 0, 1), scale, pad_w, pad_h

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

        Matches the result format from the local FaceDetector.detect() path:
          {'embedding': np.ndarray[512]|None, 'face_image': ndarray|None,
           'face_detected': bool, 'det_score': float,
           'face_bbox': [x1,y1,x2,y2], 'face_landmarks': [[x,y],...]}
        """
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
                face = self._detect_face(roi)
                if face is None:
                    results.append(result)
                    continue

                bbox, kps, det_score = face
                x1, y1, x2, y2 = bbox.astype(int)
                face_crop = roi[max(0, y1):y2, max(0, x1):x2]

                embedding = self._get_embedding(roi, kps)

                result = {
                    "embedding": embedding,
                    "face_image": face_crop if face_crop.size > 0 else None,
                    "face_detected": True,
                    "det_score": float(det_score),
                    "face_bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "face_landmarks": kps.reshape(5, 2).tolist(),
                }
            except Exception as e:
                logger.debug(f"ArcFace pipeline error on ROI: {e}")
            results.append(result)
        return results

    def _detect_face(
        self, roi: np.ndarray
    ) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
        """Run SCRFD face detection via Triton arcface_det.

        Returns (bbox_xyxy in ROI coords, kps_flat10 in ROI coords, score), or None.
        Applies the same border padding as local FaceDetector to improve edge-face detection.
        """
        # Add padding around the ROI (same as FaceDetector._add_padding)
        pad_w, pad_h = 0, 0
        if self.face_padding_percent > 0:
            h, w = roi.shape[:2]
            pad_h = int(h * self.face_padding_percent / 100)
            pad_w = int(w * self.face_padding_percent / 100)
            roi = cv2.copyMakeBorder(
                roi, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_REPLICATE
            )

        blob, det_scale = self._preprocess_det(roi)

        try:
            inp = InferInput("input.1", list(blob.shape), "FP32")
            inp.set_data_from_numpy(blob)
            response = self._client.infer(
                model_name="arcface_det",
                inputs=[inp],
                outputs=[InferRequestedOutput(n) for n in _DET_OUTPUT_NAMES],
            )
            net_outs = [response.as_numpy(n) for n in _DET_OUTPUT_NAMES]
        except Exception as e:
            logger.debug(f"Triton arcface_det infer failed: {e}")
            return None

        scores_all, bboxes_all, kpss_all = self._decode_scrfd(net_outs)
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

        # Remove the padding offset so coordinates are in the original ROI space
        if pad_w or pad_h:
            bbox -= [pad_w, pad_h, pad_w, pad_h]
            kps[0::2] -= pad_w  # x components
            kps[1::2] -= pad_h  # y components

        return bbox, kps, scores[best]

    def _preprocess_det(
        self, img: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Preprocess ROI for SCRFD det_10g. Returns (blob [1,3,640,640], det_scale).

        Matches InsightFace's SCRFD.forward() preprocessing exactly:
        aspect-ratio-preserving resize → zero-pad → (x-127.5)/128 → NCHW.
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

        blob = (canvas.astype(np.float32) - 127.5) / 128.0
        blob = blob.transpose(2, 0, 1)[np.newaxis, :]  # NHWC→NCHW, add batch
        return blob, det_scale

    def _decode_scrfd(
        self, net_outs: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """Decode 9 SCRFD FPN outputs → (scores, bboxes, keypoints) per FPN level.

        Output tensor ordering matches _DET_OUTPUT_NAMES:
          [score_s8, score_s16, score_s32, bbox_s8, bbox_s16, bbox_s32,
           kps_s8, kps_s16, kps_s32]
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
                # mgrid[::-1] → (x_grid, y_grid); each location gets 2 anchors
                centers = np.stack(
                    np.mgrid[:feat_h, :feat_w][::-1], axis=-1
                ).astype(np.float32) * stride
                centers = centers.reshape(-1, 2)
                centers = np.stack([centers] * _SCRFD_NUM_ANCHORS, axis=1).reshape(-1, 2)
                self._center_cache[key] = centers
            ac = self._center_cache[key]

            # SCRFD predictions are stride-normalized — multiply by stride to get pixels.
            # distance2bbox: (l,t,r,b) distances from anchor center
            bboxes = np.stack([
                ac[:, 0] - bbox_preds[:, 0] * stride,
                ac[:, 1] - bbox_preds[:, 1] * stride,
                ac[:, 0] + bbox_preds[:, 2] * stride,
                ac[:, 1] + bbox_preds[:, 3] * stride,
            ], axis=-1)

            # distance2kps: (dx,dy) offsets for each of 5 keypoints → flat [n, 10]
            kps_parts = []
            for k in range(0, kps_preds.shape[1], 2):
                kps_parts.append(ac[:, 0] + kps_preds[:, k] * stride)
                kps_parts.append(ac[:, 1] + kps_preds[:, k + 1] * stride)
            kpss = np.stack(kps_parts, axis=-1)

            scores_list.append(scores[:, 0])
            bboxes_list.append(bboxes)
            kpss_list.append(kpss)

        return scores_list, bboxes_list, kpss_list

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

    def _get_embedding(
        self, roi: np.ndarray, kps_flat: np.ndarray
    ) -> Optional[np.ndarray]:
        """Align face to 112×112 via norm_crop, then get ArcFace embedding from Triton."""
        # kps_flat is [10] in (x1,y1,x2,y2,...) order — reshape to (5,2) for norm_crop
        aligned = _norm_crop(roi, kps_flat.reshape(5, 2))

        # InsightFace ArcFace preprocessing: BGR, (x-127.5)/127.5 → [-1,1], NHWC→NCHW
        blob = (aligned.astype(np.float32) - 127.5) / 127.5
        blob = blob.transpose(2, 0, 1)[np.newaxis, :]  # [1,3,112,112]

        try:
            inp = InferInput("input.1", list(blob.shape), "FP32")
            inp.set_data_from_numpy(blob)
            response = self._client.infer(
                model_name="arcface_rec",
                inputs=[inp],
                outputs=[InferRequestedOutput("683")],
            )
            emb = response.as_numpy("683")[0]  # [512]
        except Exception as e:
            logger.debug(f"Triton arcface_rec infer failed: {e}")
            return None

        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb
