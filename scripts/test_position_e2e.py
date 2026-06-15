"""End-to-end smoke test for the position-emit pipeline.

Bypasses the per-frame engine. Exercises: HomographyRegistry (with a hand-set
entry, no DB needed) -> CameraEngine.emit_positions -> MDAPublisher
-> real Redis Pub/Sub -> subscriber.

Run:
    SO_REDIS_HOST=localhost SO_REDIS_PORT=6400 PYTHONPATH=src \
        python scripts/test_position_e2e.py
"""

import json
import os
import sys
import threading
import time
import types

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

import redis  # noqa: E402

from domain.calibration.homography_registry import HomographyRegistry  # noqa: E402
from messaging.channels import EVENT_CHANNELS, EVENT_TYPES  # noqa: E402
from messaging.publisher import MDAPublisher  # noqa: E402
from pipeline.camera_engine import CameraEngine  # noqa: E402


CHANNEL = EVENT_CHANNELS["LOCATION"]
CLIENT_SLUG = "position-smoke-test"
CAMERA_ID = 7777
MAP_ID = 99


class StubIdentityManager:
    def is_identity_locked(self, _):
        return False

    def get_locked_identity(self, _):
        return None


def subscriber_loop(events_seen, expected, stop):
    r = redis.Redis(
        host=os.getenv("SO_REDIS_HOST", "localhost"),
        port=int(os.getenv("SO_REDIS_PORT", "6400")),
        decode_responses=True,
    )
    pubsub = r.pubsub()
    pubsub.subscribe(CHANNEL)
    deadline = time.time() + 6.0
    for msg in pubsub.listen():
        if msg.get("type") != "message":
            continue
        try:
            payload = json.loads(msg["data"])
        except (TypeError, ValueError):
            continue
        if payload.get("client_slug") != CLIENT_SLUG:
            continue
        if payload.get("event_type") != EVENT_TYPES["USER_LOCATION_UPDATED"]:
            continue
        if "position" not in payload:
            continue  # ignore entry/exit variants
        events_seen.append(payload)
        if len(events_seen) >= expected:
            break
        if time.time() > deadline:
            break
    pubsub.close()
    stop.set()


def main() -> int:
    # Synthetic H = scale 2 + translate 10 (same as Phase 1 fixture)
    H = np.array([[2, 0, 10], [0, 2, 10], [0, 0, 1]], dtype=np.float64)
    registry = HomographyRegistry()
    registry._cache[(CLIENT_SLUG, CAMERA_ID)] = (H, MAP_ID)

    engine = types.SimpleNamespace()
    engine.client_slug = CLIENT_SLUG
    engine.camera_id = CAMERA_ID
    engine.homography_registry = registry
    engine.identity_manager = StubIdentityManager()
    engine.name_to_id_map = {}
    engine._publisher = MDAPublisher(CLIENT_SLUG)
    engine._position_last_emit = {}
    engine.emit_positions = types.MethodType(CameraEngine.emit_positions, engine)

    events: list = []
    stop = threading.Event()
    sub = threading.Thread(
        target=subscriber_loop, args=(events, 1, stop), daemon=True
    )
    sub.start()
    time.sleep(0.5)  # let SUBSCRIBE land

    bbox = [0.0, 0.0, 100.0, 100.0]
    # foot point = (50, 100), projected via H -> (110, 210)
    print(f"[1/2] emitting one position via engine.emit_positions ...")
    engine.emit_positions([{"track_id": 12345, "bbox": bbox}])

    sub.join(timeout=6.0)

    if not events:
        print("FAIL — no event arrived on events:location")
        return 1

    evt = events[0]
    print("[2/2] event received:")
    print(json.dumps(evt, indent=2))

    assert evt["camera_id"] == CAMERA_ID
    assert evt["map_id"] == MAP_ID
    assert evt["track_id"] == 12345
    assert evt["user_id"] is None
    assert evt["user_name"] is None
    pos = evt["position"]
    assert abs(pos["x"] - 110.0) < 1e-6, f"x={pos['x']}"
    assert abs(pos["y"] - 210.0) < 1e-6, f"y={pos['y']}"

    print(
        "\nOK — position event published with correct map_id, "
        "foot-point projection, and anonymous-track fields"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
