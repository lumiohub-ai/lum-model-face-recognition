"""Async wrapper around the lum_vision action recognizer.

The model package exposes a blocking ``recognize()`` and deliberately owns no
threads. This module supplies the concurrency and the side effects — queueing,
worker threads, GCS proof upload and the Celery activity task — which are the
application's concerns, not the model's.
"""

import queue
import threading
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
    ):
        """Initialize the worker pool.

        Args:
            recognizer: Synchronous recognizer from lum_vision
            client_slug: Organization slug for Celery activity tasks
            max_queue_size: Maximum queued inference requests
            num_workers: Number of background worker threads
        """
        self.recognizer = recognizer
        self.client_slug = client_slug
        self.max_queue_size = max_queue_size
        self.num_workers = num_workers

        self.inference_queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self.result_callbacks: Dict[str, Callable] = {}
        self.workers: List[threading.Thread] = []
        self.running = False

    @property
    def enabled(self) -> bool:
        """Whether action recognition is turned on."""
        return self.recognizer.enabled

    @property
    def check_interval_seconds(self) -> int:
        """Minimum seconds between checks for the same person."""
        return self.recognizer.check_interval_seconds

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

    def stop_workers(self) -> None:
        """Stop all worker threads gracefully."""
        if not self.running:
            return

        logger.info("Stopping action recognition workers...")
        self.running = False

        # Send sentinel values to unblock workers
        for _ in self.workers:
            try:
                self.inference_queue.put(None, timeout=1.0)
            except queue.Full:
                pass

        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=5.0)

        self.workers.clear()
        logger.info("Action recognition workers stopped")

    def _worker_loop(self) -> None:
        """Background worker loop for processing inference queue."""
        while self.running:
            try:
                # Get item from queue (blocking)
                item = self.inference_queue.get(timeout=1.0)

                # Sentinel value to stop worker
                if item is None:
                    break

                self._process_inference_request(item)
                self.inference_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logger.exception(f"Error in action recognition worker: {e}")

    def _process_inference_request(self, item: Dict) -> None:
        """Run one inference and dispatch its result.

        Args:
            item: Dictionary containing inference request data
        """
        try:
            image = item['image']
            request_id = item['request_id']
            metadata = item.get('metadata', {})

            result = self.recognizer.recognize(image, metadata=metadata)
            if result is None:
                return

            # Post to backend if we have user_id and camera_id
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

            # Call callback if registered
            callback = self.result_callbacks.pop(request_id, None)
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

        except Exception as e:
            logger.exception(f"Failed to process inference request: {e}")

    def _post_activity_to_backend(
        self,
        user_id: int,
        camera_id: int,
        activity_type: str,
        proof_image: np.ndarray,
        metadata: dict
    ) -> None:
        """Post activity to backend API.

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

        try:
            from datetime import datetime, timezone
            from workers.detection_tasks import task_record_activity
            from infrastructure.storage import ImageFetcher

            # Upload proof image to GCS if provided
            proof_image_url = None
            if proof_image is not None:
                try:
                    proof_image_url = ImageFetcher().upload_image(proof_image, "activity_proofs", self.client_slug)
                except Exception as e:
                    logger.warning(f"Failed to upload activity proof image: {e}")

            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

            # Queue Celery task directly
            task_record_activity.delay(
                client_slug=self.client_slug,
                user_id=int(user_id) if user_id else 0,
                user_name=metadata.get('user_name', 'Unknown'),
                activity_type=activity_type,
                camera_id=camera_id,
                confidence=None,
                proof_image_url=proof_image_url,
                detected_at=timestamp
            )

            logger.info(f"Activity queued to Celery: user_id={user_id}, type={activity_type}")

        except Exception as e:
            logger.exception(f"Failed to queue activity: {e}")
            raise

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
            metadata: Optional metadata (should include user_id, camera_id, person_name, track_id, timestamp)

        Returns:
            True if queued successfully, False if queue is full
        """
        if not self.enabled:
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

            return True

        except queue.Full:
            self.result_callbacks.pop(request_id, None)
            logger.warning(
                f"Action recognition queue full ({self.max_queue_size}), "
                "dropping request"
            )
            return False
