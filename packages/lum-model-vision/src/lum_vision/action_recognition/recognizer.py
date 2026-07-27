"""Action recognition using Ollama with a vision-language model.

Classifies what a person in a crop is doing (sleeping, using phone, working,
talking, ...). Actions are configurable — see :class:`~lum_vision.config.ActionConfig`.

This module is deliberately synchronous. Queueing, worker threads and result
dispatch belong to the caller, which knows its own concurrency budget and
shutdown ordering; a library that spawns threads can do neither well.
"""

import base64
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import cv2
import numpy as np
import ollama
from loguru import logger

from ..config import ActionConfig


@dataclass
class ActionResult:
    """Outcome of a single action-recognition inference."""

    action: Optional[str]
    """Matched action name from the configured set, or None if unrecognized."""

    activity_type: str
    """``action`` mapped through the configured backend_type, or "unknown"."""

    raw_output: str = ""
    inference_time: float = 0.0
    metadata: Dict = field(default_factory=dict)


class ActionRecognizer:
    """Recognizes person actions from an image crop via an Ollama VLM.

    Call :meth:`recognize` directly; it blocks for the duration of one
    inference and is safe to call from multiple threads.
    """

    def __init__(self, config: ActionConfig):
        """Initialize the recognizer.

        Args:
            config: Ollama endpoint, model, timeout and the action set
        """
        self.config = config
        self.enabled = config.enabled
        self.model_name = config.model_name
        self.inference_timeout = config.inference_timeout
        self.check_interval_seconds = config.check_interval_seconds

        self.actions_config = config.actions or {}
        self.action_mapping = self._build_action_mapping()
        self.prompt_template = self._build_prompt_template()

        # Create Ollama client once (reused across all inference calls)
        self._ollama_client = ollama.Client(
            host=config.ollama_api_url, timeout=config.inference_timeout
        )

        # Performance metrics
        self.total_inferences = 0
        self.total_inference_time = 0.0
        self.total_api_errors = 0
        self.total_timeouts = 0

        logger.info(
            f"ActionRecognizer initialized | enabled={self.enabled} | "
            f"ollama_api={config.ollama_api_url} | model={self.model_name} | "
            f"interval={self.check_interval_seconds}s | "
            f"timeout={self.inference_timeout}s | actions={len(self.actions_config)}"
        )

    def _build_action_mapping(self) -> Dict[Optional[str], str]:
        """Build VLM action to backend activity_type mapping from config.

        Returns:
            Dictionary mapping action names to backend activity types
        """
        mapping: Dict[Optional[str], str] = {None: "unknown"}
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

    def recognize(
        self, image: np.ndarray, metadata: Optional[Dict] = None
    ) -> Optional[ActionResult]:
        """Classify the action in a person crop. Blocks for one inference.

        Args:
            image: Person crop image (numpy array, BGR format)
            metadata: Optional caller context, echoed back on the result

        Returns:
            An :class:`ActionResult`, or None if inference failed
        """
        start_time = time.time()
        result = self.recognize_via_api(image)
        inference_time = time.time() - start_time

        if result is None:
            return None

        self.total_inferences += 1
        self.total_inference_time += inference_time

        action = result.get("action")
        return ActionResult(
            action=action,
            activity_type=self.action_mapping.get(action, "unknown"),
            raw_output=result.get("raw_output", ""),
            inference_time=inference_time,
            metadata=metadata or {},
        )

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

            # Call Ollama API using shared client
            ## INFER: Here we send the image to the Ollama API for inference. The model is expected to return a response containing the recognized action.
            response = self._ollama_client.generate(
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
            self.total_api_errors += 1
            logger.exception(f"Ollama API error: {e}")
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

    def get_stats(self) -> Dict[str, float]:
        """Return inference counters for monitoring."""
        avg = (
            self.total_inference_time / self.total_inferences
            if self.total_inferences
            else 0.0
        )
        return {
            "total_inferences": self.total_inferences,
            "avg_inference_time": avg,
            "total_api_errors": self.total_api_errors,
            "total_timeouts": self.total_timeouts,
        }
