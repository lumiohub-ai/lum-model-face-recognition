"""Async wrapper around the lum_vision action recognizer.

The model package exposes a blocking ``recognize()`` and deliberately owns no
threads. This module supplies the concurrency and the side effects — queueing,
worker threads, GCS proof upload and the Celery activity task — which are the
application's concerns, not the model's.

It also owns the per-identity throttle. That lives here rather than on
``CameraEngine`` because one worker is shared by every camera in an engine, so
a person visible on several cameras is classified once per interval rather than
once per camera.
"""

import queue
import threading
import time
from typing import Callable, Dict, List, Optional

import numpy as np
from loguru import logger

from lum_vision import ActionRecognizer


class ActionRecognitionWorker:
    """Runs action recognition off the camera threads and reports results.

    Presents the same surface the camera engine used before the models were
    extracted: ``enabled``, ``check_interval_seconds`` and ``recognize_async``.
    """

    def __init__(
        self,
        recognizer: ActionRecognizer,
        client_slug: Optional[str] = None,
        max_queue_size: int = 50,
        num_workers: int = 1,
        async_logger=None,
        metrics_collector=None,
        min_crop_height: int = 0,
        min_crop_width: int = 0,
        min_crop_area: int = 0,
    ):
        """Initialize the worker pool.

        Args:
            recognizer: Synchronous recognizer from lum_vision
            client_slug: Organization slug for Celery activity tasks
            max_queue_size: Maximum queued inference requests
            num_workers: Number of background worker threads
            async_logger: Optional AsyncLogger for off-thread GCS proof upload.
                          May also be supplied later via set_async_logger().
            metrics_collector: Optional MetricsCollector for inference stats
            min_crop_height: Skip crops shorter than this, in pixels (0 disables)
            min_crop_width: Skip crops narrower than this, in pixels (0 disables)
            min_crop_area: Skip crops smaller than this, in pixels² (0 disables)
        """
        self.recognizer = recognizer
        self.client_slug = client_slug
        self.max_queue_size = max_queue_size
        self.num_workers = num_workers
        self._async_logger = async_logger
        self._metrics = metrics_collector
        self.min_crop_height = min_crop_height
        self.min_crop_width = min_crop_width
        self.min_crop_area = min_crop_area

        self.inference_queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self.result_callbacks: Dict[str, Callable] = {}
        self.workers: List[threading.Thread] = []
        self.running = False

        # Per-identity throttle, shared across every camera in this engine.
        # Redis-backed, not an in-process dict (LSO-67): the throttle's whole
        # job is cross-CAMERA ("classify a person once per interval, not once
        # per camera"), and cameras now run as Celery tasks in separate
        # processes where a threading.Lock protects nothing. See
        # workers/identity_throttle.py.
        from workers.identity_throttle import RedisIdentityThrottle

        self._throttle = RedisIdentityThrottle(
            interval_seconds=self.check_interval_seconds
        )

        # Counters (queue-side; model-side counters live on the recognizer)
        self.total_queued = 0
        self.total_dropped = 0
        self.total_posted = 0
        self.total_too_small = 0

    @property
    def enabled(self) -> bool:
        """Whether action recognition is turned on."""
        return self.recognizer.enabled

    @property
    def check_interval_seconds(self) -> int:
        """Minimum seconds between checks for the same person."""
        return self.recognizer.check_interval_seconds

    def set_async_logger(self, async_logger) -> None:
        """Supply the AsyncLogger used for off-thread proof upload.

        The engine builds its AsyncLogger after this worker, so the reference
        arrives late. Until it does, uploads fall back to running inline.
        """
        self._async_logger = async_logger

    def set_metrics_collector(self, metrics_collector) -> None:
        """Supply the metrics collector (also built after this worker)."""
        self._metrics = metrics_collector

    # ── Throttle ──────────────────────────────────────────────────────────────

    def reserve_check(self, identity: str, now: Optional[float] = None) -> bool:
        """Claim the right to classify ``identity`` now.

        Atomic across processes, not just threads: returns True at most once
        per ``check_interval_seconds`` per identity no matter how many camera
        workers ask, which is what stops a person seen on three cameras from
        triggering three VLM inferences. Callers that then fail to queue must
        call :meth:`cancel_check` so the person isn't skipped for a whole
        interval.

        Args:
            identity: Recognized person name
            now: Ignored. Kept for signature compatibility with the previous
                 in-process implementation, whose tests injected a clock;
                 expiry is now the Redis server's own, so there is no local
                 clock to override.

        Returns:
            True if the caller should run inference for this identity
        """
        return self._throttle.reserve(identity)

    def cancel_check(self, identity: str) -> None:
        """Release a reservation whose work never got queued."""
        self._throttle.cancel(identity)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start_workers(self) -> None:
        """Start background worker threads for async inference."""
        if not self.enabled:
            logger.info("Action recognition disabled, not starting workers")
            return

        if self.running:
            logger.warning("Workers already running")
            return

        self.running = True

        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"ActionRecognizer-Worker-{i}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)

        logger.info(f"Action recognition workers started ({self.num_workers})")

    def stop_workers(self, timeout: float = 5.0) -> None:
        """Stop all worker threads gracefully."""
        if not self.running:
            return

        logger.info("Stopping action recognition workers...")
        self.running = False

        for worker in self.workers:
            if worker.is_alive():
                worker.join(timeout=timeout)

        self.workers.clear()
        self.result_callbacks.clear()
        logger.info(f"Action recognition workers stopped | {self.get_stats()}")

    def _worker_loop(self) -> None:
        """Background worker loop for processing inference queue."""
        while self.running:
            try:
                item = self.inference_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._process_inference_request(item)
            except Exception as e:
                logger.exception(f"Error in action recognition worker: {e}")

    # ── Inference ─────────────────────────────────────────────────────────────

    def _process_inference_request(self, item: Dict) -> None:
        """Run one inference and dispatch its result.

        Args:
            item: Dictionary containing inference request data
        """
        image = item['image']
        request_id = item['request_id']
        metadata = item.get('metadata', {})

        # Claim the callback up front: an early return or a raised exception
        # must not leave it (and the person crop it closes over) in the dict.
        callback = self.result_callbacks.pop(request_id, None)

        # recognize() collapses every failure into None, so read the model's own
        # counter to tell a timeout (the common case) from a hard error.
        timeouts_before = getattr(self.recognizer, "total_timeouts", 0)

        started = time.time()
        result = self.recognizer.recognize(image, metadata=metadata)
        elapsed_ms = (time.time() - started) * 1000.0

        if result is None:
            timed_out = getattr(self.recognizer, "total_timeouts", 0) > timeouts_before
            self._record_metric(elapsed_ms, "timeout" if timed_out else "error")
            return

        if result.action is None:
            # The model answered but we could not map it. Recording it would put
            # a meaningless "unknown" row and a proof image behind the dashboard.
            self._record_metric(elapsed_ms, "parse_fail")
            logger.debug(
                f"Action unrecognized, not recorded | raw={result.raw_output!r} | "
                f"identity={metadata.get('identity')}"
            )
            return

        self._record_metric(elapsed_ms, "ok")

        if metadata.get('user_id') and metadata.get('camera_id'):
            try:
                self._post_activity_to_backend(
                    user_id=metadata['user_id'],
                    camera_id=metadata['camera_id'],
                    activity_type=result.activity_type,
                    proof_image=image,  # Send the person crop as proof
                    metadata=metadata
                )
            except Exception as e:
                logger.exception(f"Failed to post activity to backend: {e}")

        if callback:
            callback({
                'action': result.action,
                'activity_type': result.activity_type,
                'raw_output': result.raw_output,
                'inference_time': result.inference_time,
                'metadata': result.metadata,
            })

        logger.debug(
            f"Action recognized: {result.action} ({result.activity_type}) | "
            f"time={result.inference_time:.3f}s | "
            f"user_id={metadata.get('user_id')} | "
            f"track_id={metadata.get('track_id')}"
        )

    def _record_metric(self, elapsed_ms: float, status: str) -> None:
        """Report one inference to the metrics collector, if wired."""
        if self._metrics is None:
            return
        try:
            self._metrics.record_action_inference(
                elapsed_ms, status, queue_depth=self.inference_queue.qsize()
            )
        except Exception as e:  # metrics must never break inference
            logger.debug(f"Failed to record action metric: {e}")

    def _post_activity_to_backend(
        self,
        user_id: int,
        camera_id: int,
        activity_type: str,
        proof_image: np.ndarray,
        metadata: dict
    ) -> None:
        """Post activity to backend API.

        The proof upload is handed to the AsyncLogger's GCS worker so this
        thread isn't blocked by it; the Celery task is fired from the upload
        callback, since it needs the resulting URL.

        Args:
            user_id: User ID
            camera_id: Camera ID
            activity_type: Activity type (backend format)
            proof_image: Person crop image to send as proof
            metadata: Additional metadata
        """
        if not self.client_slug:
            logger.warning("No client_slug configured, skipping backend post")
            return

        from datetime import datetime, timezone

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        # camera_engine sends the recognized name under 'identity'.
        user_name = metadata.get('identity') or 'Unknown'

        def fire(proof_image_url: Optional[str]) -> None:
            """Queue the Celery task once the proof URL is known (or known absent)."""
            try:
                from workers.detection_tasks import task_record_activity

                task_record_activity.delay(
                    client_slug=self.client_slug,
                    user_id=int(user_id) if user_id else 0,
                    user_name=user_name,
                    activity_type=activity_type,
                    camera_id=camera_id,
                    confidence=None,
                    proof_image_url=proof_image_url,
                    detected_at=timestamp,
                )
                self.total_posted += 1
                logger.info(
                    f"Activity queued to Celery: user_id={user_id}, type={activity_type}"
                )
            except Exception as e:
                logger.exception(f"Failed to queue activity: {e}")

        queued = False
        if self._async_logger is not None and proof_image is not None:
            queued = self._async_logger.upload_image({
                'image': proof_image,
                'folder': 'activity_proofs',
                'client_slug': self.client_slug,
                'callback': fire,
            })

        if not queued:
            # No AsyncLogger, or its GCS queue was full. Record the activity
            # anyway — losing the proof image beats losing the activity.
            if proof_image is not None and self._async_logger is not None:
                logger.warning("GCS queue full — recording activity without proof image")
            fire(None)

    def _crop_too_small(self, image: Optional[np.ndarray]) -> bool:
        """Whether a crop is too small to be worth an inference.

        A distant figure a few dozen pixels tall carries no readable posture,
        so the VLM answers from the background instead — a wrong activity that
        still costs a full inference and a GCS proof upload. The defaults in
        config.yaml sit just under the smallest crop the eval set in
        notebooks/eval/action could still label (207px tall).
        """
        if image is None:
            return True
        if not (self.min_crop_height or self.min_crop_width or self.min_crop_area):
            return False

        height, width = image.shape[:2]
        return (
            height < self.min_crop_height
            or width < self.min_crop_width
            or height * width < self.min_crop_area
        )

    def should_skip_small(self, image: Optional[np.ndarray]) -> bool:
        """``_crop_too_small``, counting the skip. Callers gate on this.

        Deliberately side-effecting: callers check the size *before* reserving
        the per-identity throttle slot, so this is the only place that sees
        every skipped crop and can keep the counter honest.
        """
        if self._crop_too_small(image):
            self.total_too_small += 1
            return True
        return False

    def recognize_async(
        self,
        image: np.ndarray,
        request_id: str,
        callback: Optional[Callable] = None,
        metadata: Optional[Dict] = None
    ) -> bool:
        """Queue image for async action recognition.

        Args:
            image: Person crop image (numpy array)
            request_id: Unique request identifier
            callback: Optional callback function to receive results
            metadata: Optional metadata (should include user_id, camera_id, identity, track_id)

        Returns:
            True if queued successfully, False if queue is full
        """
        if not self.enabled:
            return False

        # Safety net for direct callers; camera_engine gates earlier, before it
        # reserves the throttle slot, so in the normal path this never fires.
        if self.should_skip_small(image):
            return False

        try:
            # Register callback if provided
            if callback:
                self.result_callbacks[request_id] = callback

            # Queue inference request
            self.inference_queue.put({
                'image': image,
                'request_id': request_id,
                'metadata': metadata or {}
            }, block=False)

            self.total_queued += 1
            return True

        except queue.Full:
            self.result_callbacks.pop(request_id, None)
            self.total_dropped += 1
            logger.warning(
                f"Action recognition queue full ({self.max_queue_size}), "
                "dropping request"
            )
            return False

    def get_stats(self) -> Dict[str, float]:
        """Model counters plus queue-side counters, for monitoring."""
        stats = dict(self.recognizer.get_stats())
        stats.update(
            queue_depth=self.inference_queue.qsize(),
            total_queued=self.total_queued,
            total_dropped=self.total_dropped,
            total_posted=self.total_posted,
            total_too_small=self.total_too_small,
            pending_callbacks=len(self.result_callbacks),
            # Replaces the old `tracked_identities` gauge: the throttle's keys
            # now live in Redis with their own expiry, and counting them would
            # cost a SCAN per stats call. Whether the throttle is degraded is
            # the fact worth surfacing anyway -- when it is, every action check
            # is being declined.
            throttle_degraded=self._throttle.degraded,
        )
        return stats
