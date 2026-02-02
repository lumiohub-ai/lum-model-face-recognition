"""Action Recognition using Ollama with Gemma 3 model.

This module provides action recognition for detecting person activities:
- sleeping
- using phone
- working with computer
- talking with someone

Uses Ollama API with gemma3:4b model for inference.
"""

import os
import queue
import threading
import time
from typing import Optional, Dict, List, Callable
import base64
import io

import numpy as np
import cv2
from loguru import logger
import ollama


class ActionRecognizer:
    """Recognizes person actions using Ollama with Gemma 3 model.

    Features:
    - Ollama Python client for LLM inference
    - Async queue-based processing
    - Automatic backend activity posting
    - Rate limiting per person
    """

    # VLM action to backend activity_type mapping
    ACTION_MAPPING = {
        "sleeping": "sleeping",
        "using phone": "phone_usage",
        "working with computer": "working",
        "talking with someone": "talking",
        "not_focusing": "not_focusing",
        "idle": "unknown",
        None: "unknown"
    }

    def __init__(
        self,
        ollama_api_url: Optional[str] = None,
        api_client = None,  # APIClient instance for backend posting
        enabled: bool = True,
        check_interval_seconds: int = 30,
        max_queue_size: int = 50,
        num_workers: int = 1,
        model_name: str = "gemma3:4b",
        inference_timeout: int = 30
    ):
        """Initialize action recognizer.

        Args:
            ollama_api_url: URL of Ollama API service (e.g., http://localhost:11435)
            api_client: APIClient instance for posting to backend
            enabled: Enable/disable action recognition
            check_interval_seconds: Interval between action checks per person
            max_queue_size: Maximum queued inference requests
            num_workers: Number of background worker threads
            model_name: Ollama model to use for inference
            inference_timeout: Timeout for Ollama API calls in seconds (default: 30s)
        """
        self.ollama_api_url = ollama_api_url or os.getenv("OLLAMA_API_URL", "http://localhost:11435")
        self.api_client = api_client
        self.enabled = enabled
        self.check_interval_seconds = check_interval_seconds
        self.max_queue_size = max_queue_size
        self.num_workers = num_workers
        self.model_name = model_name
        self.inference_timeout = inference_timeout

        # Configure Ollama client with timeout
        if self.ollama_api_url:
            ollama.Client(host=self.ollama_api_url, timeout=inference_timeout)

        # Async processing queue
        self.inference_queue = queue.Queue(maxsize=max_queue_size)
        self.result_callbacks: Dict[str, Callable] = {}

        # Worker threads
        self.workers: List[threading.Thread] = []
        self.running = False

        # Performance metrics
        self.total_inferences = 0
        self.total_inference_time = 0.0
        self.total_api_errors = 0
        self.total_timeouts = 0

        logger.info(
            f"ActionRecognizer initialized | enabled={enabled} | "
            f"ollama_api={self.ollama_api_url} | model={model_name} | "
            f"interval={check_interval_seconds}s | workers={num_workers} | "
            f"timeout={inference_timeout}s"
        )

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

        logger.info(f"Started {self.num_workers} action recognition worker threads")

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

                # Process inference request
                self._process_inference_request(item)

                # Mark task as done
                self.inference_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in action recognition worker: {e}")

    def _process_inference_request(self, item: Dict) -> None:
        """Process a single inference request.

        Args:
            item: Dictionary containing inference request data
        """
        try:
            image = item['image']
            request_id = item['request_id']
            metadata = item.get('metadata', {})

            # Run inference via VLM API
            start_time = time.time()
            result = self.recognize_via_api(image)
            inference_time = time.time() - start_time

            # Update metrics
            self.total_inferences += 1
            self.total_inference_time += inference_time

            # Map VLM action to backend activity_type
            vlm_action = result.get('action') if result else None
            activity_type = self.ACTION_MAPPING.get(vlm_action, "unknown")

            # Prepare result
            result_data = {
                'action': vlm_action,
                'activity_type': activity_type,
                'raw_output': result.get('raw_output') if result else None,
                'inference_time': inference_time,
                'metadata': metadata
            }

            # Post to backend if we have user_id and camera_id
            if self.api_client and metadata.get('user_id') and metadata.get('camera_id'):
                try:
                    self._post_activity_to_backend(
                        user_id=metadata['user_id'],
                        camera_id=metadata['camera_id'],
                        activity_type=activity_type,
                        proof_image=image,  # Send the person crop as proof
                        metadata=metadata
                    )
                except Exception as e:
                    logger.error(f"Failed to post activity to backend: {e}")

            # Call callback if registered
            callback = self.result_callbacks.get(request_id)
            if callback:
                callback(result_data)
                # Clean up callback
                del self.result_callbacks[request_id]

            logger.info(
                f"Action recognized: {vlm_action} ({activity_type}) | "
                f"time={inference_time:.3f}s | "
                f"user_id={metadata.get('user_id')} | "
                f"track_id={metadata.get('track_id')}"
            )

        except Exception as e:
            logger.error(f"Failed to process inference request: {e}")
            self.total_api_errors += 1

    def recognize_via_api(self, image: np.ndarray) -> Optional[Dict]:
        """Send image to Ollama for action recognition.

        Args:
            image: Person crop image (numpy array, BGR format from OpenCV)

        Returns:
            Dictionary with action and raw model output, or None if failed
        """
        try:
            # Convert image to base64 for Ollama
            # Ollama expects images in JPEG format
            _, buffer = cv2.imencode('.jpg', image)
            image_base64 = base64.b64encode(buffer).decode('utf-8')

            # Create prompt for action recognition
            prompt = """Classify what the person in this image is doing. Pick the best match:

1. sleeping - head resting on desk, leaning back with eyes closed, or slumped over
2. using phone - holding a phone, looking down at a device in hand
3. working with computer - sitting at a desk facing a screen or typing
4. talking with someone - facing another person, gesturing, or in conversation
5. idle - standing still, sitting without doing anything specific, looking around

Respond with ONLY one of these exact phrases:
- "sleeping"
- "using phone"
- "working with computer"
- "talking with someone"
- "idle"

Just the phrase, no explanation."""

            # Call Ollama API with timeout
            client = ollama.Client(host=self.ollama_api_url, timeout=self.inference_timeout)
            response = client.generate(
                model=self.model_name,
                prompt=prompt,
                images=[image_base64],
                stream=False
            )

            # Parse response
            raw_output = response.get('response', '').strip().lower()

            # Extract action from response - check for "idle"/"none" first to avoid
            # false matches from loose substring matching (e.g. "not using phone")
            action = None
            if raw_output.startswith('idle') or raw_output.startswith('none') or 'none of' in raw_output:
                action = 'idle'
            elif raw_output.startswith('sleeping') or raw_output == 'sleeping':
                action = 'sleeping'
            elif raw_output.startswith('using phone') or raw_output == 'using phone':
                action = 'using phone'
            elif raw_output.startswith('working with computer') or raw_output == 'working with computer':
                action = 'working with computer'
            elif raw_output.startswith('talking with someone') or raw_output == 'talking with someone':
                action = 'talking with someone'

            logger.debug(f"Ollama response: {raw_output} -> action: {action}")

            return {
                'action': action,
                'raw_output': raw_output
            }

        except TimeoutError as e:
            self.total_timeouts += 1
            logger.warning(f"Ollama API timeout after {self.inference_timeout}s: {e}")
            return None
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            return None

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
        if not self.api_client:
            logger.warning("No API client configured, skipping backend post")
            return

        try:
            # Post to backend using existing send_activities method
            self.api_client.send_activities(
                activity_type=activity_type,
                camera_id=camera_id,
                user_id=user_id,
                confidence_score=None,  # confidence_score (not available yet)
                proof_image=proof_image   # Send person crop as proof image
            )

            logger.info(f"Activity posted to backend: user_id={user_id}, type={activity_type}")

        except Exception as e:
            logger.error(f"Failed to post activity to backend: {e}")
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
            logger.warning(
                f"Action recognition queue full ({self.max_queue_size}), "
                "dropping request"
            )
            return False

    def get_queue_size(self) -> int:
        """Get current queue size."""
        return self.inference_queue.qsize()

    def get_metrics(self) -> Dict:
        """Get performance metrics."""
        avg_time = (
            self.total_inference_time / self.total_inferences
            if self.total_inferences > 0
            else 0.0
        )

        return {
            'total_inferences': self.total_inferences,
            'total_time': self.total_inference_time,
            'average_time': avg_time,
            'total_errors': self.total_api_errors,
            'total_timeouts': self.total_timeouts,
            'queue_size': self.get_queue_size(),
            'workers_running': self.running,
            'ollama_api_url': self.ollama_api_url,
            'model_name': self.model_name,
            'inference_timeout': self.inference_timeout
        }

    def __del__(self):
        """Cleanup on deletion."""
        self.stop_workers()
