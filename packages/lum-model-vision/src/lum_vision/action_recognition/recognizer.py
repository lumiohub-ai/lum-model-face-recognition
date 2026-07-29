"""Action recognition using Ollama with a vision-language model.

Classifies what a person in a crop is doing (sleeping, using phone, working,
talking, ...). Actions are configurable — see :class:`~lum_vision.config.ActionConfig`.

This module is deliberately synchronous. Queueing, worker threads and result
dispatch belong to the caller, which knows its own concurrency budget and
shutdown ordering; a library that spawns threads can do neither well.
"""

import base64
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import cv2
import httpx
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

        # Case-insensitive name lookup: the model echoes an enum member back, but
        # casing is not guaranteed even under a constrained schema.
        self._action_lookup = {name.lower(): name for name in self.actions_config}
        self.response_format = self._build_response_format()

        # Create Ollama client once (reused across all inference calls)
        self._ollama_client = ollama.Client(
            host=config.ollama_api_url, timeout=config.inference_timeout
        )

        # Performance metrics
        self.total_inferences = 0
        self.total_inference_time = 0.0
        self.total_api_errors = 0
        self.total_timeouts = 0
        self.total_parse_failures = 0

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

    def _build_response_format(self) -> Optional[Dict[str, Any]]:
        """JSON schema constraining the reply to a configured action.

        Returns None when no actions are configured — an ``enum`` of just
        ``["none"]`` would force every answer to "none", so in that case the
        caller must omit ``format=`` entirely and fall back to string parsing.
        """
        if not self.actions_config:
            return None
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [*self.actions_config, "none"],
                }
            },
            "required": ["action"],
        }

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

        # Add response instructions. The schema in _build_response_format is what
        # actually constrains the reply; naming the options here keeps the model
        # from having to infer them from the schema alone.
        lines.append('\nRespond with JSON: {"action": "<one of the names above>"}')
        lines.append('Use "none" only if no option applies.')

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
            kwargs: Dict[str, Any] = {
                "model": self.model_name,
                "prompt": self.prompt_template,
                "images": [image_base64],
                "stream": False,
                # Classification, not generation — sampling only adds variance.
                "options": {"temperature": 0},
            }
            if self.response_format is not None:
                kwargs["format"] = self.response_format
            response = self._ollama_client.generate(**kwargs)

            # Case is preserved: the schema enum is case-exact, and the fallback
            # path lowercases for itself.
            raw_output = response.get('response', '').strip()

            # Extract action from response using configured actions
            action = self._parse_action_response(raw_output)

            logger.debug(f"Ollama response: {raw_output} -> action: {action}")

            return {
                'action': action,
                'raw_output': raw_output
            }

        except httpx.TimeoutException as e:
            self.total_timeouts += 1
            logger.warning(f"Ollama API timeout after {self.inference_timeout}s: {e}")
            return None
        except httpx.ConnectError as e:
            # Ollama being down is an expected operational state, not a defect —
            # a traceback per inference would bury the logs.
            self.total_api_errors += 1
            logger.warning(f"Ollama unreachable at {self.config.ollama_api_url}: {e}")
            return None
        except ollama.ResponseError as e:
            self.total_api_errors += 1
            logger.warning(f"Ollama returned an error (model '{self.model_name}'): {e}")
            return None
        except Exception as e:
            self.total_api_errors += 1
            logger.exception(f"Ollama API error: {e}")
            return None

    def _parse_action_response(self, raw_output: str) -> Optional[str]:
        """Parse VLM response to extract action using configured actions.

        Args:
            raw_output: Raw response from VLM (stripped, original case)

        Returns:
            Matched action name, or None if the model picked "none" or replied
            with something outside the configured set
        """
        try:
            value = json.loads(raw_output)["action"]
            matched = self._action_lookup.get(str(value).strip().lower())
            if matched is None and str(value).strip().lower() != "none":
                # Valid JSON, but an action we never offered — worth counting
                # separately from "the model correctly rejected every option".
                self.total_parse_failures += 1
            return matched
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

        # Fallback: the model (or a stub) ignored the schema and replied in prose.
        self.total_parse_failures += 1
        return self._match_free_text(raw_output.lower())

    def _match_free_text(self, lowered: str) -> Optional[str]:
        """Best-effort match of an unconstrained reply against the action set."""
        for action_name in self.actions_config:
            if lowered.startswith(action_name.lower()):
                return action_name
        # Prefix match misses replies like "the person is using phone".
        for action_name in self.actions_config:
            if action_name.lower() in lowered:
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
            "total_parse_failures": self.total_parse_failures,
        }
