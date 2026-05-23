"""Action Recognition using Ollama with Gemma 3 model.

Evidence-based pipeline:
  1. Resize person crop (≤896 px) for fast inference.
  2. Run phone object detector (YOLO COCO class 67) on the crop.
  3. Build a VLM prompt that includes phone-detector evidence.
  4. Parse the VLM JSON (activity, confidence, phone_visible, phone_location, reason).
  5. Apply phone-gating: if no phone detected AND VLM confidence is weak,
     reject using_phone / phone_calling.
  6. Apply action-specific temporal voting (phone actions need 2/4 votes).
  7. Optionally save debug crops for false-positive inspection.
"""

import itertools
import json
import os
import queue
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Callable, Deque, Hashable, Tuple

# Priority constants for the inference queue. Lower = served first.
PRIORITY_PHONE = 0     # YOLO already flagged a phone-shaped object in the crop
PRIORITY_GENERAL = 10  # Periodic activity poll for a locked identity
PRIORITY_UNKNOWN = 20  # Activity poll for an unrecognized person; evicted first under load

import cv2
import numpy as np
import base64
from loguru import logger
import ollama

# Actions that require extra voting evidence before being displayed
_PHONE_ACTIONS = frozenset({'using_phone', 'phone_calling'})

# Temporal voting parameters
_VOTE_WINDOW = 4          # keep last 4 VLM results per track
_VOTE_REQUIRED_PHONE = 1  # surface phone events on the first confirmed detection; phone-gating filters false positives
_VOTE_REQUIRED_DEFAULT = 1  # other actions update immediately
_PHONE_OVERRIDE_CONFIDENCE = 0.30

# If the phone object detector found no phone, require at least this VLM
# confidence before allowing using_phone. Set high: without YOLO confirmation
# the VLM alone on CCTV crops produces too many false positives.
_NO_PHONE_VLM_THRESHOLD = 0.80

# Per-action minimum confidence (applied AFTER phone-gating)
_MIN_CONFIDENCE = {
    'using_phone': 0.65,  # raised — factory clothing/tools cause false positives at 0.55
    'phone_calling': 0.65,
    'chatting': 0.55,
    'working': 0.40,  # low threshold — working is the correct default in this environment
    'carrying': 0.50,
    'walking': 0.45,
    'idle': 0.70,  # high — standing still near machinery is working, not idle
}

# Fallback keywords: model outputs not in the action list that map to idle.
# Removed 'standing', 'moving', 'passing' — in a workshop these are working postures.
_IDLE_FALLBACK_KEYWORDS = frozenset({
    'none', 'other', 'unclear', 'unknown', 'not_using_phone', 'handling',
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
        max_queue_delay_seconds: float = 12.0,
        actions: Dict = None,
        phone_detector=None,
        debug_save_dir: Optional[str] = None,
        phone_precheck_interval_seconds: float = 2.0,
        general_poll_interval_seconds: float = 30.0,
        unknown_poll_interval_seconds: float = 45.0,
    ):
        self.ollama_api_url = ollama_api_url
        self.client_slug = client_slug
        self.enabled = enabled
        self.check_interval_seconds = check_interval_seconds
        self.max_queue_size = max_queue_size
        self.num_workers = num_workers
        self.model_name = model_name
        self.inference_timeout = inference_timeout
        self.max_queue_delay_seconds = max_queue_delay_seconds
        self.phone_detector = phone_detector
        self.phone_precheck_interval_seconds = phone_precheck_interval_seconds
        self.general_poll_interval_seconds = general_poll_interval_seconds
        self.unknown_poll_interval_seconds = unknown_poll_interval_seconds

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

        # Async processing queue (priority-ordered: phone candidates first)
        self.inference_queue: queue.PriorityQueue = queue.PriorityQueue(maxsize=max_queue_size)
        self._queue_seq = itertools.count()  # monotonic tiebreaker for FIFO within same priority
        self.result_callbacks: Dict[str, Callable] = {}

        # Deduplication: tracks already sitting in the queue (not yet processed)
        self._pending_keys: set = set()
        self._pending_lock = threading.Lock()

        # Temporal voting: action history per (camera_id, track_id). Local track
        # IDs repeat across cameras, and this recognizer instance is shared.
        self._action_history: Dict[Hashable, Deque] = {}

        # Activity-session aggregator: collapses a run of identical voted_action
        # results for the same (camera, track) into a single backend event.
        # Without this, a 60 s phone session produces 8 separate records.
        self._activity_sessions: Dict[Hashable, Dict] = {}
        self._session_idle_close_seconds = 45.0
        self._session_lock = threading.Lock()

        # Worker threads
        self.workers: List[threading.Thread] = []
        self.running = False

        # Performance metrics
        self.total_inferences = 0
        self.total_inference_time = 0.0
        self.total_api_errors = 0
        self.total_timeouts = 0

        # Server availability backoff — avoids flooding logs when Gemma starts late
        self._consecutive_connection_errors = 0
        self._server_backoff_until = 0.0

        phone_det_status = 'available' if (phone_detector and phone_detector.available) else 'disabled'
        logger.info(
            f"ActionRecognizer initialized | enabled={enabled} | "
            f"ollama_api={ollama_api_url} | model={model_name} | "
            f"interval={check_interval_seconds}s | workers={num_workers} | "
            f"timeout={inference_timeout}s | max_queue_delay={max_queue_delay_seconds}s | "
            f"actions={len(self.actions_config)} | "
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
        # Sentinel: (-1, seq, {'_shutdown': True}) sorts before any real entry
        for _ in self.workers:
            try:
                self.inference_queue.put(
                    (-1, next(self._queue_seq), {'_shutdown': True}),
                    timeout=1.0,
                )
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
        priority: int = PRIORITY_GENERAL,
    ) -> bool:
        """Queue image for async recognition. Returns False if rejected.

        priority: PRIORITY_PHONE (0) for YOLO-confirmed phone candidates,
                  PRIORITY_GENERAL (10) for periodic polls. Lower = served first.
                  HIGH-priority puts evict the oldest LOW-priority item if the
                  queue is full.
        """
        if not self.enabled:
            return False

        meta = metadata or {}
        meta.setdefault('queued_at', time.time())
        meta['priority'] = priority
        # Deduplicate: skip if this (camera, track) is already waiting in the queue
        cam_id = meta.get('camera_id', '')
        track_id = meta.get('track_id', -1)
        pending_key = (cam_id, track_id)
        with self._pending_lock:
            if pending_key in self._pending_keys:
                return False
            self._pending_keys.add(pending_key)

        item = {'image': image, 'request_id': request_id, 'metadata': meta,
                'pending_key': pending_key}
        seq = next(self._queue_seq)
        entry: Tuple[int, int, Dict] = (priority, seq, item)

        if callback:
            self.result_callbacks[request_id] = callback

        try:
            self.inference_queue.put(entry, block=False)
            return True
        except queue.Full:
            # Evict the most-recent lower-priority entry to make room. Phone
            # candidates evict either GENERAL or UNKNOWN; GENERAL polls evict
            # UNKNOWN polls. This protects high-value events under load.
            if self._evict_lowest_priority(min_priority=priority + 1):
                try:
                    self.inference_queue.put(entry, block=False)
                    return True
                except queue.Full:
                    pass
            with self._pending_lock:
                self._pending_keys.discard(pending_key)
            self.result_callbacks.pop(request_id, None)
            logger.warning(
                f"Action recognition queue full ({self.max_queue_size}), dropping "
                f"priority={priority} request"
            )
            return False

    def _evict_lowest_priority(self, min_priority: int) -> bool:
        """Remove one queued entry with priority >= min_priority. Returns True on success.

        Pops the highest-priority-number (lowest-priority) entry; ties broken by newest
        sequence so older low-priority work is preserved over newer low-priority work.
        Touches PriorityQueue internals under its mutex.
        """
        q = self.inference_queue
        with q.mutex:
            if not q.queue:
                return False
            # heap not sorted; scan to find a victim with priority >= min_priority
            victim_idx = -1
            victim_key = None  # (priority, seq) — we want max
            for i, entry in enumerate(q.queue):
                prio, seq, _ = entry
                if prio < min_priority:
                    continue
                key = (prio, seq)
                if victim_key is None or key > victim_key:
                    victim_key = key
                    victim_idx = i
            if victim_idx < 0:
                return False
            _, _, evicted_item = q.queue.pop(victim_idx)
            # Re-heapify after arbitrary removal
            import heapq
            heapq.heapify(q.queue)
            q.unfinished_tasks -= 1
            q.not_full.notify()
        # Cleanup pending-key and callback for the evicted entry (outside the queue mutex)
        evicted_pending = evicted_item.get('pending_key')
        evicted_req = evicted_item.get('request_id')
        if evicted_pending is not None:
            with self._pending_lock:
                self._pending_keys.discard(evicted_pending)
        if evicted_req:
            self.result_callbacks.pop(evicted_req, None)
        return True

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
            # Hold off while the server is known unreachable so stale items age
            # out via max_queue_delay_seconds instead of generating a flood of
            # connection errors.
            backoff_remaining = self._server_backoff_until - time.time()
            if backoff_remaining > 0:
                time.sleep(min(backoff_remaining, 1.0))
                continue
            try:
                entry = self.inference_queue.get(timeout=1.0)
                if entry is None:
                    break
                # PriorityQueue: entry is (priority, seq, item_dict)
                _, _, item = entry
                if item.get('_shutdown'):
                    break
                pending_key = item.get('pending_key')
                try:
                    self._process_inference_request(item)
                finally:
                    if pending_key is not None:
                        with self._pending_lock:
                            self._pending_keys.discard(pending_key)
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
        queued_at = float(metadata.get('queued_at') or time.time())
        queue_delay = time.time() - queued_at

        if queue_delay > self.max_queue_delay_seconds:
            logger.warning(
                f"ACTION_STALE_QUEUE_DROP | cam={camera_id} track={track_id} | "
                f"request={request_id} queue_delay={queue_delay:.2f}s "
                f"> {self.max_queue_delay_seconds:.2f}s"
            )
            callback = self.result_callbacks.pop(request_id, None)
            if callback:
                callback({
                    'action': None,
                    'activity_type': 'unknown',
                    'confidence': 0.0,
                    'raw_output': None,
                    'inference_time': 0.0,
                    'stale': True,
                    'stale_reason': 'queue_delay',
                    'metadata': metadata,
                })
            return

        # ── Step 1: focus on upper body, then resize (≤896 px on longest side) ─
        # Phones are almost always in the upper 65% of the person box (hands,
        # chest, face). Cropping before resize gives the VLM a denser view of
        # the region where a phone would appear.
        h, w = image.shape[:2]
        upper_body = image[:int(h * 0.75), :]
        image = upper_body

        h, w = image.shape[:2]
        # Person-crop is downsampled aggressively — only used for general-activity
        # context where 512 px is sufficient. Phone-focus regions get their own,
        # larger budget below so small phones (~12-18 px in the source) stay legible.
        max_dim = 512
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            image = cv2.resize(
                image, (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )
        crop_h, crop_w = image.shape[:2]

        # ── Step 2: phone object detection ───────────────────────────────────
        # Prefer native-resolution focus regions. A fixed "hand" slice misses
        # calling poses, so the camera pipeline may pass multiple regions:
        # upper body/face+hands first, then torso/hands, then the person crop.
        phone_info: Dict = {'phone_detected': False, 'phone_confidence': 0.0,
                            'phone_bbox_in_crop': None, 'phone_location_hint': 'none'}
        hand_region = metadata.get('hand_region')
        phone_regions = metadata.get('phone_regions') or []
        if hand_region is not None and hand_region.size > 0 and not phone_regions:
            phone_regions = [{'name': 'hand', 'image': hand_region}]
        use_focus_for_vlm = False
        vlm_region = None
        vlm_region_name = 'person_crop'

        # If the camera engine already ran YOLO, reuse that result and skip a
        # second YOLO pass here. Saves ~20-50ms per inference and avoids divergent
        # phone-detector answers between camera_engine and recognizer.
        precomputed = metadata.get('precomputed_phone_info')
        if precomputed is not None:
            phone_info = dict(precomputed)
            vlm_region, vlm_region_name = self._select_focus_region(phone_regions)
            use_focus_for_vlm = vlm_region is not None
        elif self.phone_detector and self.phone_detector.available:
            for region in phone_regions:
                region_img = region.get('image') if isinstance(region, dict) else region
                region_name = region.get('name', 'region') if isinstance(region, dict) else 'region'
                if region_img is None or region_img.size == 0:
                    continue
                candidate_info = self.phone_detector.detect(region_img)
                if candidate_info['phone_detected']:
                    phone_info = candidate_info
                    vlm_region = region_img
                    vlm_region_name = region_name
                    if region_name in ('face_hands', 'upper_body'):
                        phone_info['phone_location_hint'] = self._refine_region_location(
                            phone_info['phone_location_hint'], region_name
                        )
                    use_focus_for_vlm = True
                    break
            if not phone_info['phone_detected']:
                phone_info = self.phone_detector.detect(image)
                vlm_region, vlm_region_name = self._select_focus_region(phone_regions)
                if vlm_region is not None:
                    use_focus_for_vlm = True
        else:
            vlm_region, vlm_region_name = self._select_focus_region(phone_regions)
            if vlm_region is not None:
                use_focus_for_vlm = True

        logger.debug(
            f"PHONE_DETECTION | cam={camera_id} track={track_id} | "
            f"crop={crop_h}x{crop_w} "
            f"regions={self._describe_regions(phone_regions)} | "
            f"phone_det={phone_info['phone_detected']} "
            f"conf={phone_info['phone_confidence']:.2f} "
            f"vlm_src={vlm_region_name if use_focus_for_vlm else 'person_crop'}"
        )

        # ── Step 3: VLM inference ────────────────────────────────────────────
        # Asymmetric resolution budget:
        #   • Phone-focus regions (face_hands / hand / upper_body) → 896 px so
        #     small phones (12-18 px in source) remain legible at VLM scale.
        #     This is the case driving most of our work; do not regress it.
        #   • General person crop → 512 px (set above). Plenty for working /
        #     walking / idle classification, cheaper inference.
        if use_focus_for_vlm and vlm_region is not None and vlm_region.size > 0:
            phone_focus = vlm_region_name in ('face_hands', 'hand', 'upper_body', 'multi_focus')
            focus_max_dim = 896 if phone_focus else max_dim
            hr_h, hr_w = vlm_region.shape[:2]
            if max(hr_h, hr_w) > focus_max_dim:
                scale = focus_max_dim / max(hr_h, hr_w)
                vlm_input = cv2.resize(
                    vlm_region, (int(hr_w * scale), int(hr_h * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            else:
                vlm_input = vlm_region
        else:
            vlm_input = image

        t0 = time.time()
        vlm_result = self._recognize_via_api(
            vlm_input,
            phone_info,
            focus_region_name=vlm_region_name if use_focus_for_vlm else 'person_crop',
        )
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
        phone_confirmed = phone_info['phone_detected'] or vlm_phone_visible

        if phone_confirmed and raw_action not in _PHONE_ACTIONS:
            location = phone_info['phone_location_hint'] or vlm_phone_location
            if location in ('ear', 'face'):
                gated_action = 'phone_calling'
            elif phone_info['phone_confidence'] >= _PHONE_OVERRIDE_CONFIDENCE or vlm_phone_visible:
                gated_action = 'using_phone'

        if raw_action in _PHONE_ACTIONS:
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
        vote_key = (camera_id, track_id)
        voted_action = self.get_voted_action(vote_key, gated_action)
        if gated_action in _PHONE_ACTIONS and phone_confirmed and vlm_confidence >= 0.70:
            voted_action = gated_action

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
            or vlm_phone_visible
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
            'queued_at': queued_at,
            'completed_at': time.time(),
            'queue_delay': queue_delay,
            'metadata': metadata,
        }

        if metadata.get('camera_id') and voted_action:
            should_post = self._begin_or_extend_session(
                camera_id=metadata['camera_id'],
                track_id=track_id,
                user_id=metadata.get('user_id') or 0,
                activity_type=activity_type,
            )
            if should_post:
                try:
                    self._post_activity_to_backend(
                        user_id=metadata.get('user_id') or 0,
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

    @staticmethod
    def _refine_region_location(location: str, region_name: str) -> str:
        if region_name == 'face_hands' and location in ('face', 'ear', 'hand'):
            return location
        if region_name == 'face_hands' and location == 'body':
            return 'hand'
        if region_name == 'upper_body' and location == 'body':
            return 'hand'
        return location

    @staticmethod
    def _describe_regions(regions) -> str:
        if not regions:
            return 'none'
        parts = []
        for region in regions:
            if isinstance(region, dict):
                img = region.get('image')
                name = region.get('name', 'region')
            else:
                img = region
                name = 'region'
            if img is not None and getattr(img, 'size', 0) > 0:
                h, w = img.shape[:2]
                parts.append(f"{name}:{w}x{h}")
        return ','.join(parts) if parts else 'none'

    @staticmethod
    def _select_focus_region(regions):
        """Pick the single best focus region for the VLM.

        Small VLMs anchor on the dominant panel of a montage, which usually drowns
        out the very region that contained the phone. Send one clean crop instead.
        Preference order: face_hands → hand → upper_body → first available.
        """
        by_name: Dict[str, np.ndarray] = {}
        first = None
        for region in regions or []:
            img = region.get('image') if isinstance(region, dict) else region
            name = region.get('name', 'region') if isinstance(region, dict) else 'region'
            if img is None or getattr(img, 'size', 0) == 0:
                continue
            by_name.setdefault(name, img)
            if first is None:
                first = (img, name)
        for pref in ('face_hands', 'hand', 'upper_body'):
            if pref in by_name:
                return by_name[pref], pref
        if first is not None:
            return first
        return None, 'person_crop'

    @staticmethod
    def _build_focus_montage(person_crop: np.ndarray, regions) -> Optional[np.ndarray]:
        valid_regions = []
        for region in regions:
            img = region.get('image') if isinstance(region, dict) else region
            if img is not None and getattr(img, 'size', 0) > 0:
                valid_regions.append(img)

        if not valid_regions:
            return None

        panels = [person_crop] + valid_regions[:3]
        target_h = 360
        resized = []
        for panel in panels:
            h, w = panel.shape[:2]
            if h <= 0 or w <= 0:
                continue
            scale = target_h / h
            panel = cv2.resize(
                panel,
                (max(1, int(w * scale)), target_h),
                interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
            )
            resized.append(panel)

        if len(resized) <= 1:
            return None

        sep = np.full((target_h, 8, 3), 255, dtype=np.uint8)
        montage = resized[0]
        for panel in resized[1:]:
            montage = np.hstack((montage, sep, panel))

        max_w = 1280
        if montage.shape[1] > max_w:
            scale = max_w / montage.shape[1]
            montage = cv2.resize(
                montage,
                (max_w, max(1, int(montage.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )

        return montage

    # ── VLM inference ─────────────────────────────────────────────────────────

    def _recognize_via_api(
        self, image: np.ndarray, phone_info: Dict, focus_region_name: str = 'person_crop'
    ) -> Optional[Dict]:
        """Send image to Ollama for activity classification.

        Builds a dynamic prompt that includes phone-detector evidence so the
        VLM can make an informed, evidence-anchored decision.
        """
        try:
            _, buffer = cv2.imencode(
                '.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 92]
            )
            image_b64 = base64.b64encode(buffer).decode('utf-8')

            prompt = self._build_dynamic_prompt(phone_info, focus_region_name=focus_region_name)

            response = self._ollama_client.generate(
                model=self.model_name,
                prompt=prompt,
                images=[image_b64],
                stream=False,
                options={'num_predict': 64},  # JSON {activity,confidence,phone_visible,phone_location,reason} fits in ~50 tokens; 64 = margin without paying for unused budget
            )

            raw_output = (
                getattr(response, 'response', None) or response['response']
            ).strip()

            # Server responded — clear any backoff state
            self._consecutive_connection_errors = 0
            self._server_backoff_until = 0.0

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
            err_str = str(e).lower()
            is_conn_error = (
                'connectionerror' in err_name.lower()
                or 'connecterror' in err_name.lower()
                or 'failed to connect' in err_str
                or 'connection refused' in err_str
                or 'cannot connect' in err_str
            )
            if 'timeout' in err_name.lower() or 'timeout' in err_str:
                self.total_timeouts += 1
                self._consecutive_connection_errors = 0
                logger.warning(
                    f"Ollama timeout after {self.inference_timeout}s: {err_name}: {e}"
                )
            elif is_conn_error:
                self.total_api_errors += 1
                self._consecutive_connection_errors += 1
                # Exponential backoff: 1s, 2s, 4s, 8s, 16s, 30s (cap)
                backoff = min(30.0, 2.0 ** min(self._consecutive_connection_errors - 1, 4))
                self._server_backoff_until = time.time() + backoff
                # Log verbosely on 1st failure, then only every 10th to avoid spam
                if self._consecutive_connection_errors <= 2 or self._consecutive_connection_errors % 10 == 0:
                    logger.warning(
                        f"Ollama server not reachable (attempt "
                        f"#{self._consecutive_connection_errors}), backing off "
                        f"{backoff:.0f}s | {e}"
                    )
                else:
                    logger.debug(
                        f"Ollama connection error #{self._consecutive_connection_errors}: {e}"
                    )
            else:
                self.total_api_errors += 1
                self._consecutive_connection_errors = 0
                logger.warning(f"Ollama API error: {err_name}: {e}")
            return None

    def _build_dynamic_prompt(self, phone_info: Dict, focus_region_name: str = 'person_crop') -> str:
        """Build a concise VLM prompt with phone-detector evidence."""
        action_names = list(self.actions_config.keys())
        labels_str = ', '.join(action_names)

        lines = [
            "Classify what the person in this image is doing.",
            f"Choose exactly one label: {labels_str}",
            "",
            "Label meanings:",
        ]

        for name, cfg in self.actions_config.items():
            desc = cfg.get('description', '')
            lines.append(f"  {name} – {desc}")

        if phone_info.get('phone_detected'):
            conf = phone_info.get('phone_confidence', 0.0)
            loc = phone_info.get('phone_location_hint', 'unknown')
            lines += [
                "",
                f"YOLO detected a phone-like object (confidence {conf:.2f}, location: {loc}).",
                "If you clearly see a distinct rectangular screen in the person's hand, choose using_phone.",
                "If the object is ambiguous, a shadow, or clothing, choose working instead.",
            ]
        else:
            lines += [
                "",
                "YOLO found no phone. Do NOT choose using_phone unless a distinct rectangular",
                "phone screen is unmistakably visible in the hand — not a shadow, tool, or clothing.",
                "Default to working for anyone near machines, wood, tools, or materials.",
            ]

        lines += [
            "",
            "Reply with JSON only:",
            '{"activity": "<label>", "confidence": <0.0-1.0>, "phone_visible": <true/false>, '
            '"phone_location": "<none|hand|ear|face>", "reason": "<one sentence>"}',
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
            # In a workshop context, low-confidence guesses default to working rather than idle
            fallback = 'working' if 'working' in self.actions_config else None
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
        self, track_id: Hashable, new_action: Optional[str]
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

        for phone_action in ('phone_calling', 'using_phone'):
            if counts.get(phone_action, 0) >= _VOTE_REQUIRED_PHONE:
                return phone_action

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

    # ── Session aggregation ───────────────────────────────────────────────────

    def _begin_or_extend_session(
        self,
        camera_id,
        track_id: int,
        user_id: int,
        activity_type: str,
    ) -> bool:
        """Return True if this is the start of a new activity session (i.e. the
        caller should post to the backend), False if it merely extends a session
        that is already open.

        A session is identified by (camera_id, track_id, user_id, activity_type).
        It stays open while same-activity results arrive at least every
        _session_idle_close_seconds; the first contradicting activity, or a long
        gap, closes it and opens a new one on the next call.
        """
        key = (camera_id, track_id, user_id)
        now = time.time()
        with self._session_lock:
            sess = self._activity_sessions.get(key)
            if sess is not None:
                if (
                    sess['activity'] == activity_type
                    and (now - sess['last_seen']) <= self._session_idle_close_seconds
                ):
                    sess['last_seen'] = now
                    sess['hits'] += 1
                    return False
                # Activity changed or idle timeout — close old, open new
            self._activity_sessions[key] = {
                'activity': activity_type,
                'started_at': now,
                'last_seen': now,
                'hits': 1,
            }

            # Opportunistic GC: drop sessions idle past 5× idle window
            cutoff = now - self._session_idle_close_seconds * 5
            stale = [k for k, s in self._activity_sessions.items() if s['last_seen'] < cutoff]
            for k in stale:
                if k != key:
                    self._activity_sessions.pop(k, None)
        return True

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
            uname = metadata.get('user_name')
            if not uname:
                gid = metadata.get('global_id') or metadata.get('track_id')
                uname = f"Unrecognized worker (track {gid})" if gid is not None else "Unrecognized worker"
            task_record_activity.delay(
                client_slug=self.client_slug,
                user_id=int(user_id) if user_id else 0,
                user_name=uname,
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
