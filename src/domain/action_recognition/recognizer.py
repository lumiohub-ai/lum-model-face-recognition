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
    - Configurable actions via config.yaml
    """

    # Default action mapping (used if no config provided)
    DEFAULT_ACTIONS = {
        "sleeping": {"backend_type": "sleeping", "description": "head resting on desk, leaning back with eyes closed, or slumped over"},
        "using phone": {"backend_type": "phone_usage", "description": "holding a phone, looking down at a device in hand"},
        "working with computer": {"backend_type": "working", "description": "sitting at a desk facing a screen or typing"},
        "talking with someone": {"backend_type": "talking", "description": "facing another person, gesturing, or in conversation"},
        "not_focusing": {"backend_type": "not_focusing", "description": "distracted, looking away from work, wandering attention"},
        "idle": {"backend_type": "unknown", "description": "standing still, sitting without doing anything specific, looking around"},
    }

    def __init__(
        self,
        ollama_api_url: Optional[str] = None,
        client_slug: Optional[str] = None,
        enabled: bool = True,
        check_interval_seconds: int = 30,
        max_queue_size: int = 50,
        num_workers: int = 1,
        model_name: str = "gemma3:4b",
        inference_timeout: int = 30,
        actions: Optional[Dict] = None
    ):
        """Initialize action recognizer.

        Args:
            ollama_api_url: URL of Ollama API service (e.g., http://localhost:11434)
            client_slug: Organization slug for Celery tasks
            enabled: Enable/disable action recognition
            check_interval_seconds: Interval between action checks per person
            max_queue_size: Maximum queued inference requests
            num_workers: Number of background worker threads
            model_name: Ollama model to use for inference
            inference_timeout: Timeout for Ollama API calls in seconds (default: 30s)
            actions: Dictionary of actions from config (action_name -> {backend_type, description})
        """
        self.ollama_api_url = ollama_api_url or os.getenv("SO_OLLAMA_API_URL", "http://localhost:11434")
        self.client_slug = client_slug or os.getenv("SO_CLIENT_SLUG")
        self.enabled = enabled
        self.check_interval_seconds = check_interval_seconds
        self.max_queue_size = max_queue_size
        self.num_workers = num_workers
        self.model_name = model_name
        self.inference_timeout = inference_timeout

        # Build action mapping from config or use defaults
        self.actions_config = actions or self.DEFAULT_ACTIONS
        self.action_mapping = self._build_action_mapping()
        self.prompt_template = self._build_prompt_template()

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
            f"timeout={inference_timeout}s | actions={len(self.actions_config)}"
        )

    def _build_action_mapping(self) -> Dict[str, str]:
        """Build VLM action to backend activity_type mapping from config.

        Returns:
            Dictionary mapping action names to backend activity types
        """
        mapping = {None: "unknown"}
        for action_name, config in self.actions_config.items():
            backend_type = config.get("backend_type", action_name)
            mapping[action_name] = backend_type
        return mapping

    def _build_prompt_template(self) -> str:
        """Build VLM prompt from configured actions.

        Returns:
            Prompt string with numbered action list and descriptions
        """
        lines = ["Classify what the person in this image is doing. Pick the best match:\n"]

        # Build numbered action list with descriptions
        for i, (action_name, config) in enumerate(self.actions_config.items(), 1):
            description = config.get("description", "")
            lines.append(f"{i}. {action_name} - {description}")

        # Add response instructions
        lines.append("\nRespond with ONLY one of these exact phrases:")
        for action_name in self.actions_config.keys():
            lines.append(f'- "{action_name}"')
        lines.append("\nJust the phrase, no explanation.")

        return "\n".join(lines)

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
            activity_type = self.action_mapping.get(vlm_action, "unknown")

            # Prepare result
            result_data = {
                'action': vlm_action,
                'activity_type': activity_type,
                'raw_output': result.get('raw_output') if result else None,
                'inference_time': inference_time,
                'metadata': metadata
            }

            # Post to backend if we have user_id and camera_id
            if metadata.get('user_id') and metadata.get('camera_id'):
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

            # Call Ollama API with timeout using dynamic prompt
            client = ollama.Client(host=self.ollama_api_url, timeout=self.inference_timeout)
            response = client.generate(
                model=self.model_name,
                prompt=self.prompt_template,
                images=[image_base64],
                stream=False
            )

            # Parse response
            raw_output = response.get('response', '').strip().lower()

            # Extract action from response using configured actions
            action = self._parse_action_response(raw_output)

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

    def _parse_action_response(self, raw_output: str) -> Optional[str]:
        """Parse VLM response to extract action using configured actions.

        Args:
            raw_output: Raw response from VLM (lowercase, stripped)

        Returns:
            Matched action name or None if no match
        """
        # Check for "none" or "none of" patterns first (fallback to idle if configured)
        if raw_output.startswith('none') or 'none of' in raw_output:
            if 'idle' in self.actions_config:
                return 'idle'
            return None

        # Check each configured action
        for action_name in self.actions_config.keys():
            action_lower = action_name.lower()
            if raw_output.startswith(action_lower) or raw_output == action_lower:
                return action_name

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
            logger.error(f"Failed to queue activity: {e}")
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
