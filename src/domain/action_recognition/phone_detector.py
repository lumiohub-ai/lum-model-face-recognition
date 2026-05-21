"""Phone object detection within person crop images.

Uses a YOLO model (COCO class 67 = cell phone) to detect phones inside
padded person crops before sending to the VLM.  Providing hard physical
evidence of a visible phone dramatically reduces false positives from
dark clothing, tools, pockets, and machine parts.
"""

import os
import types
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from loguru import logger

# COCO class 67 = "cell phone"
COCO_PHONE_CLASS_ID = 67


class PhoneDetector:
    """Detects cell phones inside padded person-crop images using YOLO.

    The detector runs COCO class-67 inference on a small crop (already
    resized to ≤512 px by the caller) and returns:
        phone_detected      bool
        phone_confidence    float
        phone_bbox_in_crop  [x1,y1,x2,y2] in crop-space, or None
        phone_location_hint "hand" | "face" | "ear" | "body" | "none"
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        confidence_threshold: float = 0.25,
    ):
        """
        Args:
            model_path: Path to YOLO .pt file.  Defaults to yolo26s.pt from
                        YOLO_CONFIG_DIR env var (same model as PersonDetector).
            device: 'cuda' / 'cpu' / None (auto).
            confidence_threshold: Minimum YOLO score to accept a phone detection.
        """
        self.confidence_threshold = confidence_threshold
        self._model = None

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        if model_path is None:
            yolo_cache = os.environ.get('YOLO_CONFIG_DIR', '')
            candidate = Path(yolo_cache) / 'yolo26s.pt' if yolo_cache else None
            if candidate and candidate.exists():
                model_path = str(candidate)
            else:
                model_path = 'yolo26s.pt'

        self._load_model(model_path)

    def _load_model(self, model_path: str) -> None:
        try:
            from ultralytics import YOLO
            self._model = YOLO(model_path)
            self._model.to(self.device)
            self._patch_fused_model()
            # Patch fuse() to be a no-op: yolo26 (NMS-free) may have already been
            # fused by PersonDetector; calling fuse() again raises AttributeError: bn.
            orig_fuse = self._model.model.fuse

            def _safe_fuse(*args, **kwargs):
                try:
                    return orig_fuse(*args, **kwargs)
                except AttributeError as e:
                    if "'Conv' object has no attribute 'bn'" not in str(e):
                        raise
                    self._patch_fused_model()
                    return self._model.model

            self._model.model.fuse = _safe_fuse
            logger.info(
                f"PhoneDetector: loaded '{model_path}' on {self.device} "
                f"(conf_threshold={self.confidence_threshold})"
            )
        except Exception as e:
            logger.error(f"PhoneDetector: failed to load '{model_path}': {e}")
            self._model = None

    def _patch_fused_model(self) -> None:
        """Make already-fused YOLO Conv blocks safe for Ultralytics fuse calls."""
        if self._model is None or getattr(self._model, 'model', None) is None:
            return

        patched = 0
        for module in self._model.model.modules():
            if module.__class__.__name__ == 'Conv' and not hasattr(module, 'bn'):
                def _noop_fuse(conv_self):
                    return conv_self

                module.fuse = types.MethodType(_noop_fuse, module)
                patched += 1

        if patched:
            logger.debug(f"PhoneDetector: patched {patched} already-fused Conv blocks")

    @property
    def available(self) -> bool:
        return self._model is not None

    def detect(self, person_crop: np.ndarray) -> Dict:
        """Detect a cell phone inside a person-crop image.

        Returns a dict with phone detection results.  If the model is not
        loaded or inference fails, returns a safe no-phone result.
        """
        if self._model is None or person_crop is None or person_crop.size == 0:
            return self._no_phone()

        try:
            # Run without class filter — yolo26 (NMS-free) doesn't support
            # Ultralytics' classes= kwarg and raises AttributeError: bn.
            # Filter to COCO class 67 (cell phone) manually below.
            results = self._predict(person_crop)

            h, w = person_crop.shape[:2]
            best_conf = 0.0
            best_bbox = None

            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    if cls_id != COCO_PHONE_CLASS_ID:
                        continue
                    conf = float(box.conf[0])
                    if conf > best_conf:
                        best_conf = conf
                        best_bbox = [int(v) for v in box.xyxy[0].tolist()]

            if best_bbox is None:
                return self._no_phone()

            return {
                'phone_detected': True,
                'phone_confidence': round(best_conf, 3),
                'phone_bbox_in_crop': best_bbox,
                'phone_location_hint': self._infer_location(best_bbox, h, w),
            }

        except Exception as e:
            logger.warning(f"PhoneDetector inference error: {type(e).__name__}: {e}")
            return self._no_phone()

    def _predict(self, person_crop: np.ndarray):
        try:
            return self._model(
                [person_crop],
                conf=self.confidence_threshold,
                iou=0.45,
                verbose=False,
                device=self.device,
            )
        except AttributeError as e:
            if "'Conv' object has no attribute 'bn'" not in str(e):
                raise
            logger.warning("PhoneDetector: repairing fused YOLO model after missing-bn error")
            self._patch_fused_model()
            return self._model(
                [person_crop],
                conf=self.confidence_threshold,
                iou=0.45,
                verbose=False,
                device=self.device,
            )

    @staticmethod
    def _no_phone() -> Dict:
        return {
            'phone_detected': False,
            'phone_confidence': 0.0,
            'phone_bbox_in_crop': None,
            'phone_location_hint': 'none',
        }

    @staticmethod
    def _infer_location(bbox: list, crop_h: int, crop_w: int) -> str:
        """Classify where in the crop the phone centre falls.

        The padded person crop is approximately:
          top 20 %         → head / face area
          20-30 % near edge → ear (phone-calling posture)
          20-65 %          → hand / torso (screen use)
          below 65 %       → lower body / pocket / carrying
        """
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        rel_y = cy / max(crop_h, 1)
        rel_x = cx / max(crop_w, 1)

        if rel_y < 0.20:
            return 'face'
        if rel_y < 0.30 and (rel_x < 0.18 or rel_x > 0.82):
            return 'ear'
        if rel_y < 0.65:
            return 'hand'
        return 'body'
