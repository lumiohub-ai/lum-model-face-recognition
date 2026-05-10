"""End-to-end smoke test for the floor-homography handler.

Exercises: compute_homography (math) -> MDAPublisher -> real Redis Pub/Sub ->
subscriber. Skips the stream-consumer leg (the running container hasn't been
rebuilt with the new dispatch branch).

Run from repo root:
    SO_REDIS_HOST=localhost SO_REDIS_PORT=6400 PYTHONPATH=src python scripts/test_homography_e2e.py
"""

import json
import os
import sys
import threading
import time

import numpy as np
import redis

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from domain.calibration.homography import compute_homography  # noqa: E402
from messaging.channels import EVENT_CHANNELS  # noqa: E402
from messaging.publisher import MDAPublisher  # noqa: E402


CHANNEL = EVENT_CHANNELS["CALIBRATION"]
CLIENT_SLUG = "homography-smoke-test"
CAMERA_ID = 9999
COMMAND_ID = "smoke-test-001"


def subscribe_until(events_seen: list, expected: int, stop: threading.Event) -> None:
    r = redis.Redis(
        host=os.getenv("SO_REDIS_HOST", "localhost"),
        port=int(os.getenv("SO_REDIS_PORT", "6400")),
        decode_responses=True,
    )
    pubsub = r.pubsub()
    pubsub.subscribe(CHANNEL)
    deadline = time.time() + 5.0
    for msg in pubsub.listen():
        if msg["type"] != "message":
            continue
        try:
            payload = json.loads(msg["data"])
        except (TypeError, ValueError):
            payload = msg["data"]
        if isinstance(payload, dict) and payload.get("client_slug") == CLIENT_SLUG:
            events_seen.append(payload)
            if len(events_seen) >= expected:
                break
        if time.time() > deadline:
            break
    pubsub.close()
    stop.set()


def main() -> int:
    src_pts = [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]]
    H_true = np.array(
        [[1.2, 0.1, 30.0], [0.05, 0.9, -10.0], [1e-4, 2e-4, 1.0]], dtype=np.float64
    )
    src_arr = np.asarray(src_pts).reshape(-1, 1, 2).astype(np.float64)
    import cv2
    dst_pts = cv2.perspectiveTransform(src_arr, H_true).reshape(-1, 2).tolist()

    events: list = []
    stop = threading.Event()
    sub_thread = threading.Thread(
        target=subscribe_until, args=(events, 2, stop), daemon=True
    )
    sub_thread.start()
    time.sleep(0.5)  # let SUBSCRIBE land

    publisher = MDAPublisher(CLIENT_SLUG)

    print(f"[1/3] computing homography over {len(src_pts)} point pairs ...")
    result = compute_homography(src_pts, dst_pts)
    print(
        f"      method={result['method']}, "
        f"reprojection_error={result['reprojection_error']:.6f}"
    )

    print(f"[2/3] publishing HomographyComputed to {CHANNEL} ...")
    publisher.publish_homography_computed(
        command_id=COMMAND_ID,
        camera_id=CAMERA_ID,
        homography_matrix=result["homography_matrix"],
        reprojection_error=result["reprojection_error"],
        per_point_errors=result["per_point_errors"],
        method=result["method"],
        inlier_mask=result["inlier_mask"],
    )

    print(f"[3/3] publishing HomographyFailed to {CHANNEL} ...")
    publisher.publish_homography_failed(
        command_id=COMMAND_ID + "-fail",
        camera_id=CAMERA_ID,
        error="synthetic failure for smoke test",
    )

    sub_thread.join(timeout=6.0)

    if len(events) < 2:
        print(f"FAIL — expected 2 events, got {len(events)}")
        for e in events:
            print(json.dumps(e, indent=2))
        return 1

    success_evt = next(
        (e for e in events if e.get("event_type") == "HomographyComputed"), None
    )
    failure_evt = next(
        (e for e in events if e.get("event_type") == "HomographyFailed"), None
    )

    assert success_evt is not None, "missing HomographyComputed"
    assert failure_evt is not None, "missing HomographyFailed"
    assert success_evt["camera_id"] == CAMERA_ID
    assert success_evt["command_id"] == COMMAND_ID
    assert len(success_evt["homography_matrix"]) == 3
    assert len(success_evt["homography_matrix"][0]) == 3
    assert success_evt["reprojection_error"] < 1e-3
    assert success_evt["method"] == "DLT"
    assert "inlier_mask" not in success_evt  # DLT path
    assert failure_evt["error"] == "synthetic failure for smoke test"

    H_recovered = np.array(success_evt["homography_matrix"])
    H_recovered_norm = H_recovered / H_recovered[2, 2]
    H_true_norm = H_true / H_true[2, 2]
    np.testing.assert_allclose(H_recovered_norm, H_true_norm, atol=1e-4)

    print("\nOK — both events received, matrix recovered to 1e-4, payload validated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
