"""ActionRecognizer is synchronous and side-effect free.

Previously this class queued Celery tasks and uploaded to GCS from inside a
worker thread, which made it untestable without infrastructure.
"""

import json

import httpx
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


def test_none_response_yields_no_action(image):
    """"none" means the model rejected every option — distinct from idle."""
    result = build(json.dumps({"action": "none"})).recognize(image)

    assert result.action is None
    assert result.activity_type == "unknown"


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


# ── Structured output ─────────────────────────────────────────────────────────


def test_json_response_is_parsed(image):
    result = build(json.dumps({"action": "using phone"})).recognize(image)

    assert result.action == "using phone"
    assert result.activity_type == "phone_usage"


def test_json_action_matching_is_case_insensitive(image):
    assert build(json.dumps({"action": "Using Phone"})).recognize(image).action == "using phone"


def test_json_action_outside_the_enum_counts_as_a_parse_failure(image):
    recognizer = build(json.dumps({"action": "flying"}))

    result = recognizer.recognize(image)

    assert result.action is None
    assert recognizer.total_parse_failures == 1


def test_none_is_not_counted_as_a_parse_failure(image):
    recognizer = build(json.dumps({"action": "none"}))

    recognizer.recognize(image)

    assert recognizer.total_parse_failures == 0


def test_non_json_reply_falls_back_to_text_matching(image):
    """A model that ignores the schema must still be usable."""
    recognizer = build("sleeping")

    result = recognizer.recognize(image)

    assert result.action == "sleeping"
    assert recognizer.total_parse_failures == 1


def test_prose_reply_is_matched_by_substring(image):
    """Prefix matching alone missed replies that lead with filler."""
    assert build("The person is using phone.").recognize(image).action == "using phone"


def test_format_schema_constrains_reply_to_configured_actions(image):
    recognizer = build(json.dumps({"action": "idle"}))
    recognizer.recognize(image)

    schema = recognizer._ollama_client.last_kwargs["format"]

    assert schema["properties"]["action"]["enum"] == [*ACTIONS, "none"]
    assert recognizer._ollama_client.last_kwargs["options"]["temperature"] == 0


def test_no_format_sent_when_no_actions_are_configured(image):
    """An enum of just ["none"] would force every answer to "none"."""
    recognizer = ActionRecognizer(ActionConfig(enabled=True))
    recognizer._ollama_client = FakeOllama("anything")

    recognizer.recognize(image)

    assert recognizer.response_format is None
    assert "format" not in recognizer._ollama_client.last_kwargs


# ── Failure modes ─────────────────────────────────────────────────────────────


def test_timeout_is_counted_separately_from_api_errors(image):
    """httpx.TimeoutException is not a builtin TimeoutError — it used to slip through."""
    recognizer = build("sleeping")

    class TimingOut:
        def generate(self, **kwargs):
            raise httpx.ReadTimeout("timed out")

    recognizer._ollama_client = TimingOut()

    assert recognizer.recognize(image) is None
    assert recognizer.total_timeouts == 1
    assert recognizer.total_api_errors == 0


def test_unreachable_host_is_counted_as_an_api_error(image):
    recognizer = build("sleeping")

    class Unreachable:
        def generate(self, **kwargs):
            raise httpx.ConnectError("connection refused")

    recognizer._ollama_client = Unreachable()

    assert recognizer.recognize(image) is None
    assert recognizer.total_api_errors == 1
    assert recognizer.total_timeouts == 0
