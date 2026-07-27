"""ActionRecognizer is synchronous and side-effect free.

Previously this class queued Celery tasks and uploaded to GCS from inside a
worker thread, which made it untestable without infrastructure.
"""

import numpy as np
import pytest

from lum_vision import ActionConfig, ActionRecognizer

ACTIONS = {
    "sleeping": {"backend_type": "sleeping", "description": "head on desk"},
    "using phone": {"backend_type": "phone_usage", "description": "holding a phone"},
    "idle": {"backend_type": "unknown", "description": "standing still"},
}


class FakeOllama:
    """Stands in for ollama.Client."""

    def __init__(self, response: str):
        self.response = response
        self.calls = 0

    def generate(self, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs
        return {"response": self.response}


@pytest.fixture
def image():
    return np.zeros((64, 32, 3), dtype=np.uint8)


def build(response: str) -> ActionRecognizer:
    recognizer = ActionRecognizer(ActionConfig(enabled=True, actions=ACTIONS))
    recognizer._ollama_client = FakeOllama(response)
    return recognizer


def test_recognizes_action_and_maps_to_backend_type(image):
    result = build("using phone").recognize(image)

    assert result.action == "using phone"
    assert result.activity_type == "phone_usage"
    assert result.inference_time >= 0


def test_unrecognized_response_maps_to_unknown(image):
    result = build("dancing on the ceiling").recognize(image)

    assert result.action is None
    assert result.activity_type == "unknown"


def test_none_response_falls_back_to_idle_when_configured(image):
    assert build("none of the above").recognize(image).action == "idle"


def test_metadata_is_echoed_back_untouched(image):
    meta = {"track_id": 7, "camera_id": 2}
    assert build("sleeping").recognize(image, metadata=meta).metadata == meta


def test_returns_none_when_inference_fails(image):
    recognizer = build("sleeping")

    class Boom:
        def generate(self, **kwargs):
            raise RuntimeError("ollama is down")

    recognizer._ollama_client = Boom()

    assert recognizer.recognize(image) is None
    assert recognizer.total_api_errors == 1


def test_counters_track_successful_inferences(image):
    recognizer = build("idle")
    recognizer.recognize(image)
    recognizer.recognize(image)

    stats = recognizer.get_stats()
    assert stats["total_inferences"] == 2
    assert stats["avg_inference_time"] >= 0


def test_prompt_lists_every_configured_action(image):
    recognizer = build("idle")
    recognizer.recognize(image)

    prompt = recognizer._ollama_client.last_kwargs["prompt"]
    for action in ACTIONS:
        assert action in prompt


def test_recognizer_has_no_queue_or_worker_api():
    """Concurrency belongs to the caller now."""
    recognizer = ActionRecognizer(ActionConfig())

    for removed in ("start_workers", "stop_workers", "recognize_async", "inference_queue"):
        assert not hasattr(recognizer, removed)
