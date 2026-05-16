"""Action Recognition using Ollama with Gemma 3 model.

Evidence-based pipeline:
  1. Resize person crop (≤512 px) for fast inference.
  2. Run phone object detector (YOLO COCO class 67) on the crop.
  3. Build a VLM prompt that includes phone-detector evidence.
  4. Parse the VLM JSON (activity, confidence, phone_visible, phone_location, reason).
  5. Apply phone-gating: if no phone detected AND VLM confidence < 0.90,
     reject using_phone / phone_calling.
  6. Apply action-specific temporal voting (phone actions need 2/4 votes).
  7. Optionally save debug crops for false-positive inspection.
"""

import json
import os
import queue
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Callable, Deque

import cv2
import numpy as np
import base64
from loguru import logger
import ollama

# Actions that require extra voting evidence before being displayed
_PHONE_ACTIONS = frozenset({'using_phone', 'phone_calling'})

# Temporal voting parameters
_VOTE_WINDOW = 4          # keep last 4 VLM results per track
_VOTE_REQUIRED_PHONE = 2  # phone labels need 2/4 consensus
_VOTE_REQUIRED_DEFAULT = 1  # other actions update immediately

# If the phone object detector found no phone, require at least this VLM
# confidence before allowing using_phone / phone_calling.
_NO_PHONE_VLM_THRESHOLD = 0.90

# Per-action minimum confidence (applied AFTER phone-gating)
_MIN_CONFIDENCE = {
    'using_phone': 0.70,
    'phone_calling': 0.70,
    'chatting': 0.55,
    'working': 0.45,
    'carrying': 0.50,
    'walking': 0.45,
    'idle': 0.35,
}

# Fallback keywords: model outputs not in the action list that map to idle
_IDLE_FALLBACK_KEYWORDS = frozenset({
    'none', 'other', 'unclear', 'unknown',
    'not_using_phone', 'handling', 'moving', 'passing', 'standing',
})


class ActionRecognizer:
    """Recognizes person actions using Ollama with Gemma 3 model.

    Uses a two-stage pipeline:
      - PhoneDetector (YOLO) for hard evidence of a visible phone
      - VLM (Gemma 3 via Ollama) for activity classification

    Phone labels (using_phone, phone_calling) are gated on the phone
    detector and require 2 consecutive positive detections before display.
    """

    def __init__(
        self,
        ollama_api_url: Optional[str] = None,
        client_slug: Optional[str] = None,
        enabled: bool = True,
        check_interval_seconds: int = 8,
        max_queue_size: int = 20,
        num_workers: int = 2,
        model_name: str = 'gemma3:4b',
        inference_timeout: int = 30,
        actions: Dict = None,
        phone_detector=None,
        debug_save_dir: Optional[str] = None,
    ):
        self.ollama_api_url = ollama_api_url
        self.client_slug = client_slug
        self.enabled = enabled
        self.check_interval_seconds = check_interval_seconds
        self.max_queue_size = max_queue_size
        self.num_workers = num_workers
        self.model_name = model_name
        self.inference_timeout = inference_timeout
        self.phone_detector = phone_detector

        self.actions_config = actions or {}
        self.action_mapping = self._build_action_mapping()

        # Debug crop saving (non-fatal if directory creation fails)
        self.debug_save_dir = debug_save_dir
        if debug_save_dir:
            try:
                Path(debug_save_dir).mkdir(parents=True, exist_ok=True)
                logger.info(f"ActionRecognizer: debug crops → {debug_save_dir}")
            except Exception as e:
                logger.warning(f"ActionRecognizer: could not create debug dir '{debug_save_dir}': {e}")
                self.debug_save_dir = None

        # Ollama client
        self._ollama_client = ollama.Client(
            host=self.ollama_api_url, timeout=inference_timeout
        )

        # Async processing queue
        self.inference_queue = queue.Queue(maxsize=max_queue_size)
        self.result_callbacks: Dict[str, Callable] = {}

        # Temporal voting: action history per track_id
        self._action_history: Dict[int, Deque] = {}

        # Worker threads
        self.workers: List[threading.Thread] = []
        self.running = False

        # Performance metrics
        self.total_inferences = 0
        self.total_inference_time = 0.0
        self.total_api_errors = 0
        self.total_timeouts = 0

        phone_det_status = 'available' if (phone_detector and phone_detector.available) else 'disabled'
        logger.info(
            f"ActionRecognizer initialized | enabled={enabled} | "
            f"ollama_api={ollama_api_url} | model={model_name} | "
            f"interval={check_interval_seconds}s | workers={num_workers} | "
            f"timeout={inference_timeout}s | actions={len(self.actions_config)} | "
            f"phone_detector={phone_det_status}"
        )

    # ── Public lifecycle ──────────────────────────────────────────────────────

    def start_workers(self) -> None:
        if not self.enabled:
            logger.info("Action recognition disabled, not starting workers")
            return
        if self.running:
            return
        self.running = True
        for i in range(self.num_workers):
            t = threading.Thread(
                target=self._worker_loop,
                name=f"ActionRecognizer-Worker-{i}",
                daemon=True,
            )
            t.start()
            self.workers.append(t)

    def stop_workers(self) -> None:
        if not self.running:
            return
        logger.info("Stopping action recognition workers...")
        self.running = False
        for _ in self.workers:
            try:
                self.inference_queue.put(None, timeout=1.0)
            except queue.Full:
                pass
        for t in self.workers:
            t.join(timeout=5.0)
        self.workers.clear()
        logger.info("Action recognition workers stopped")

    def recognize_async(
        self,
        image: np.ndarray,
        request_id: str,
        callback: Optional[Callable] = None,
        metadata: Optional[Dict] = None,
    ) -> bool:
        """Queue image for async recognition.  Returns False if queue is full."""
        if not self.enabled:
            return False
        try:
            if callback:
                self.result_callbacks[request_id] = callback
            self.inference_queue.put(
                {'image': image, 'request_id': request_id, 'metadata': metadata or {}},
                block=False,
            )
            return True
        except queue.Full:
            logger.warning(
                f"Action recognition queue full ({self.max_queue_size}), dropping request"
            )
            return False

    def get_queue_size(self) -> int:
        return self.inference_queue.qsize()

    def get_metrics(self) -> Dict:
        avg = (
            self.total_inference_time / self.total_inferences
            if self.total_inferences > 0 else 0.0
        )
        return {
            'total_inferences': self.total_inferences,
            'total_time': self.total_inference_time,
            'average_time': avg,
            'total_errors': self.total_api_errors,
            'total_timeouts': self.total_timeouts,
            'queue_size': self.get_queue_size(),
            'workers_running': self.running,
        }

    # ── Internal worker loop ──────────────────────────────────────────────────

    def _worker_loop(self) -> None:
        while self.running:
            try:
                item = self.inference_queue.get(timeout=1.0)
                if item is None:
                    break
                self._process_inference_request(item)
                self.inference_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.exception(f"Action recognition worker error: {e}")

    def _process_inference_request(self, item: Dict) -> None:
        image = item['image']
        request_id = item['request_id']
        metadata = item.get('metadata', {})
        track_id = metadata.get('track_id', -1)
        global_id = metadata.get('global_id')
        camera_id = metadata.get('camera_id', '?')

        # ── Step 1: resize crop (≤512 px on longest side) ────────────────────
        h, w = image.shape[:2]
        max_dim = 512
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            image = cv2.resize(
                image, (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )
        crop_h, crop_w = image.shape[:2]

        # ── Step 2: phone object detection ───────────────────────────────────
        phone_info: Dict = {'phone_detected': False, 'phone_confidence': 0.0,
                            'phone_bbox_in_crop': None, 'phone_location_hint': 'none'}
        if self.phone_detector and self.phone_detector.available:
            phone_info = self.phone_detector.detect(image)

        logger.debug(
            f"PHONE_DETECTION | cam={camera_id} track={track_id} global={global_id} | "
            f"crop={crop_h}x{crop_w} | phone_det={phone_info['phone_detected']} "
            f"conf={phone_info['phone_confidence']:.2f} "
            f"location={phone_info['phone_location_hint']}"
        )

        # ── Step 3: VLM inference ────────────────────────────────────────────
        t0 = time.time()
        vlm_result = self._recognize_via_api(image, phone_info)
        inference_time = time.time() - t0

        self.total_inferences += 1
        self.total_inference_time += inference_time

        raw_action = vlm_result.get('action') if vlm_result else None
        vlm_confidence = vlm_result.get('confidence', 0.0) if vlm_result else 0.0
        vlm_phone_visible = vlm_result.get('phone_visible', False) if vlm_result else False
        vlm_phone_location = vlm_result.get('phone_location', 'none') if vlm_result else 'none'

        logger.debug(
            f"VLM_RESULT | cam={camera_id} track={track_id} | "
            f"raw_action={raw_action} conf={vlm_confidence:.2f} | "
            f"phone_visible={vlm_phone_visible} phone_location={vlm_phone_location}"
        )

        # ── Step 4: phone-gating ─────────────────────────────────────────────
        gated_action = raw_action
        rejection_reason = None

        if raw_action in _PHONE_ACTIONS:
            phone_confirmed = (
                phone_info['phone_detected']
                or vlm_phone_visible
            )
            if not phone_confirmed:
                if vlm_confidence < _NO_PHONE_VLM_THRESHOLD:
                    gated_action = 'working' if 'working' in self.actions_config else 'idle'
                    rejection_reason = (
                        f"rejected_{raw_action}_no_phone_object"
                        f"_vlm_conf_{vlm_confidence:.2f}"
                    )
            elif raw_action == 'phone_calling':
                # calling requires phone near face/ear
                location_ok = phone_info['phone_location_hint'] in ('ear', 'face')
                if not location_ok and vlm_phone_location not in ('ear', 'face'):
                    gated_action = 'using_phone'  # downgrade, not reject entirely

        # ── Step 5: temporal voting ──────────────────────────────────────────
        voted_action = self.get_voted_action(track_id, gated_action)

        if voted_action in _PHONE_ACTIONS and not (
            phone_info['phone_detected'] or vlm_phone_visible
        ):
            voted_action = None
            rejection_reason = rejection_reason or f"rejected_{voted_action}_not_enough_votes"

        # Log final decision
        if rejection_reason:
            logger.info(
                f"VOTE_RESULT | cam={camera_id} track={track_id} | "
                f"voted={voted_action} | gated={gated_action} raw={raw_action} | "
                f"REJECTED: {rejection_reason}"
            )
        else:
            display_str = 'displayed' if voted_action else 'suppressed_no_consensus'
            logger.debug(
                f"VOTE_RESULT | cam={camera_id} track={track_id} | "
                f"voted={voted_action} | gated={gated_action} raw={raw_action} | "
                f"{display_str}"
            )

        # ── Step 6: debug crop saving ────────────────────────────────────────
        if self.debug_save_dir and (
            raw_action in _PHONE_ACTIONS
            or phone_info['phone_detected']
        ):
            self._save_debug_crop(
                image, camera_id, track_id,
                raw_action or 'none', vlm_confidence,
                phone_info['phone_detected'],
                voted_action,
            )

        # ── Step 7: build result and call callback ───────────────────────────
        activity_type = self.action_mapping.get(voted_action, 'unknown')
        result_data = {
            'action': voted_action,
            'activity_type': activity_type,
            'confidence': vlm_confidence,
            'raw_output': vlm_result.get('raw_output') if vlm_result else None,
            'inference_time': inference_time,
            'phone_detected': phone_info['phone_detected'],
            'phone_confidence': phone_info['phone_confidence'],
            'phone_location': phone_info['phone_location_hint'],
            'metadata': metadata,
        }

        if metadata.get('user_id') and metadata.get('camera_id') and voted_action:
            try:
                self._post_activity_to_backend(
                    user_id=metadata['user_id'],
                    camera_id=metadata['camera_id'],
                    activity_type=activity_type,
                    proof_image=image,
                    metadata=metadata,
                )
            except Exception as e:
                logger.exception(f"Failed to post activity to backend: {e}")

        callback = self.result_callbacks.pop(request_id, None)
        if callback:
            callback(result_data)

    # ── VLM inference ─────────────────────────────────────────────────────────

    def _recognize_via_api(
        self, image: np.ndarray, phone_info: Dict
    ) -> Optional[Dict]:
        """Send image to Ollama for activity classification.

        Builds a dynamic prompt that includes phone-detector evidence so the
        VLM can make an informed, evidence-anchored decision.
        """
        try:
            _, buffer = cv2.imencode(
                '.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85]
            )
            image_b64 = base64.b64encode(buffer).decode('utf-8')

            prompt = self._build_dynamic_prompt(phone_info)

            response = self._ollama_client.generate(
                model=self.model_name,
                prompt=prompt,
                images=[image_b64],
                stream=False,
            )

            raw_output = (
                getattr(response, 'response', None) or response['response']
            ).strip()

            action, confidence, phone_visible, phone_location = (
                self._parse_json_response(raw_output)
            )

            logger.info(
                f"Ollama response: '{raw_output[:100]}' → "
                f"action={action} conf={confidence:.2f} "
                f"phone_visible={phone_visible}"
            )

            return {
                'action': action,
                'confidence': confidence,
                'phone_visible': phone_visible,
                'phone_location': phone_location,
                'raw_output': raw_output,
            }

        except Exception as e:
            err_name = type(e).__name__
            if 'timeout' in err_name.lower():
                self.total_timeouts += 1
                logger.warning(
                    f"Ollama timeout after {self.inference_timeout}s: {err_name}: {e}"
                )
            else:
                self.total_api_errors += 1
                logger.warning(f"Ollama API error: {err_name}: {e}")
            return None

    def _build_dynamic_prompt(self, phone_info: Dict) -> str:
        """Build a VLM prompt that embeds phone-detector evidence.

        When the phone detector found a phone, the VLM is asked to verify
        and classify it.  When no phone was found, the VLM is explicitly
        instructed to raise its evidence bar for phone labels.
        """
        action_names = list(self.actions_config.keys())
        labels_str = ', '.join(action_names)

        lines = [
            "You are analyzing a CCTV person crop from a factory or workshop.",
            f"Choose exactly one activity label from: {labels_str}",
            "",
        ]

        # Embed phone detector evidence
        if phone_info.get('phone_detected'):
            conf = phone_info.get('phone_confidence', 0.0)
            loc = phone_info.get('phone_location_hint', 'unknown')
            lines += [
                "PHONE DETECTION RESULT (YOLO object detector):",
                f"  A phone-like object was detected in this crop "
                f"(confidence {conf:.2f}, location: {loc}).",
                "  Use this as evidence when deciding whether using_phone or phone_calling applies.",
                "  Confirm visually — if you see a rectangular device consistent with the "
                "phone detector finding, select the appropriate phone label.",
                "",
            ]
        else:
            lines += [
                "PHONE DETECTION RESULT (YOLO object detector):",
                "  No phone-like object was detected in this crop.",
                "  Only choose using_phone or phone_calling if a phone is UNMISTAKABLY "
                "visible with confidence >= 0.90.",
                "  If in doubt, do NOT choose using_phone — prefer working, carrying, "
                "walking, or idle instead.",
                "",
            ]

        lines.append("Activity rules:")
        for action_name, cfg in self.actions_config.items():
            desc = cfg.get('description', '')
            lines.append(f"  - {action_name}: {desc}")

        lines += [
            "",
            "IMPORTANT — do NOT choose using_phone or phone_calling:",
            "  - because the person is looking down.",
            "  - because a hand is near the chest, face, or pocket.",
            "  - for dark shirt areas, shadows, tools, wood pieces, machine parts, or pockets.",
            "  - when the object in hand is clearly a tool, board, box, or material.",
            "",
            "Respond ONLY with valid JSON, no extra text:",
            '{"activity": "...", "confidence": 0.0, '
            '"phone_visible": false, "phone_location": "none", "reason": "..."}',
            "",
            f"Use only these labels: {labels_str}",
            'If uncertain and no phone is visible, use '
            '{"activity": "idle", "confidence": 0.5, "phone_visible": false, '
            '"phone_location": "none", "reason": "uncertain"}',
        ]

        return "\n".join(lines)

    # ── JSON parsing ──────────────────────────────────────────────────────────

    def _parse_json_response(self, raw_output: str):
        """Parse VLM JSON response.

        Returns (action, confidence, phone_visible, phone_location).
        """
        text = raw_output.strip()

        # Strip markdown fences
        if '```' in text:
            for part in text.split('```'):
                part = part.strip().lstrip('json').strip()
                if part.startswith('{'):
                    text = part
                    break

        # Isolate JSON object
        start, end = text.find('{'), text.rfind('}')
        if start != -1 and end > start:
            text = text[start:end + 1]

        try:
            data = json.loads(text)
            raw_action = str(data.get('activity', '')).strip().lower()
            confidence = float(data.get('confidence', 0.5))
            phone_visible = bool(data.get('phone_visible', False))
            phone_location = str(data.get('phone_location', 'none')).lower()
        except (json.JSONDecodeError, ValueError, TypeError):
            raw_action = raw_output.strip().lower().rstrip('.,!;:')
            confidence = 0.5
            phone_visible = False
            phone_location = 'none'

        action = self._match_action_name(raw_action)

        # Apply per-action minimum confidence
        min_conf = _MIN_CONFIDENCE.get(action, 0.35)
        if action in _PHONE_ACTIONS and not phone_visible and confidence < min_conf:
            # VLM itself says phone not visible → downgrade
            fallback = 'working' if 'working' in self.actions_config else 'idle'
            return fallback, confidence, phone_visible, phone_location

        if confidence < min_conf:
            fallback = 'idle' if 'idle' in self.actions_config else None
            return fallback, confidence, phone_visible, phone_location

        return action, confidence, phone_visible, phone_location

    def _match_action_name(self, raw: str) -> Optional[str]:
        """Match raw VLM string to a configured action name."""
        if not raw:
            return 'idle' if 'idle' in self.actions_config else None

        # Exact match (with and without underscores)
        for name in self.actions_config:
            if raw == name.lower() or raw == name.lower().replace('_', ' '):
                return name

        # Fallback keywords that mean idle / not meaningful
        if any(w in raw for w in _IDLE_FALLBACK_KEYWORDS):
            return 'idle' if 'idle' in self.actions_config else None

        # Substring — longest configured name wins
        best, best_len = None, 0
        for name in self.actions_config:
            a = name.lower()
            a_sp = a.replace('_', ' ')
            if a in raw or a_sp in raw:
                if len(a) > best_len:
                    best, best_len = name, len(a)
        return best

    # ── Temporal voting ───────────────────────────────────────────────────────

    def get_voted_action(
        self, track_id: int, new_action: Optional[str]
    ) -> Optional[str]:
        """Record new result and return the majority from the last N results.

        Phone actions (using_phone, phone_calling) require 2/4 votes.
        All other actions require 1/4 (displayed after first detection).
        """
        if track_id not in self._action_history:
            self._action_history[track_id] = deque(maxlen=_VOTE_WINDOW)
        self._action_history[track_id].append(new_action)

        history = self._action_history[track_id]
        counts: Dict[Optional[str], int] = {}
        for a in history:
            if a is not None:
                counts[a] = counts.get(a, 0) + 1

        if not counts:
            return None

        best = max(counts, key=lambda k: counts[k])
        vote_req = (
            _VOTE_REQUIRED_PHONE if best in _PHONE_ACTIONS
            else _VOTE_REQUIRED_DEFAULT
        )

        if len(history) < vote_req:
            # Not enough history yet — phone waits, others show immediately
            return None if best in _PHONE_ACTIONS else new_action

        return best if counts[best] >= vote_req else None

    # ── Debug helpers ─────────────────────────────────────────────────────────

    def _save_debug_crop(
        self,
        image: np.ndarray,
        camera_id,
        track_id: int,
        raw_action: str,
        confidence: float,
        phone_detected: bool,
        voted_action: Optional[str],
    ) -> None:
        if not self.debug_save_dir:
            return
        try:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            phone_tag = 'phoneTrue' if phone_detected else 'phoneFalse'
            voted_tag = voted_action or 'suppressed'
            fname = (
                f"cam{camera_id}_track{track_id}_{ts}_"
                f"raw{raw_action}_conf{int(confidence * 100)}_"
                f"{phone_tag}_voted{voted_tag}.jpg"
            )
            path = os.path.join(self.debug_save_dir, fname)
            cv2.imwrite(path, image)
        except Exception as e:
            logger.debug(f"Failed to save debug crop: {e}")

    # ── Action mapping ────────────────────────────────────────────────────────

    def _build_action_mapping(self) -> Dict[str, str]:
        mapping: Dict[Optional[str], str] = {None: 'unknown'}
        for name, cfg in self.actions_config.items():
            mapping[name] = cfg.get('backend_type', name)
        return mapping

    # ── Backend posting ───────────────────────────────────────────────────────

    def _post_activity_to_backend(
        self,
        user_id: int,
        camera_id: int,
        activity_type: str,
        proof_image: np.ndarray,
        metadata: dict,
    ) -> None:
        if not self.client_slug:
            return
        try:
            from datetime import datetime as _dt, timezone
            from workers.detection_tasks import task_record_activity
            from infrastructure.storage import ImageFetcher

            proof_url = None
            if proof_image is not None:
                try:
                    proof_url = ImageFetcher().upload_image(
                        proof_image, 'activity_proofs', self.client_slug
                    )
                except Exception as e:
                    logger.warning(f"Failed to upload activity proof: {e}")

            ts = _dt.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
            task_record_activity.delay(
                client_slug=self.client_slug,
                user_id=int(user_id) if user_id else 0,
                user_name=metadata.get('user_name', 'Unknown'),
                activity_type=activity_type,
                camera_id=camera_id,
                confidence=None,
                proof_image_url=proof_url,
                detected_at=ts,
            )
            logger.info(f"Activity queued: user={user_id} type={activity_type}")
        except Exception as e:
            logger.exception(f"Failed to queue activity: {e}")
            raise

    def __del__(self):
        self.stop_workers()
