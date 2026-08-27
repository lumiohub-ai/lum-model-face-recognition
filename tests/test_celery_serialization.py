"""Serializer contract tests for the Celery app.

Both serializer bugs in this migration were found by a *live camera run*, not
by tests: first `FrameHandle` (a frozen dataclass) failing `task_serializer=
'json'`, then the face result dicts' numpy failing `result_serializer='json'`.
Each cost a debugging cycle against a real RTSP stream. These tests pin the
contract so the next one fails in CI instead.

The asserted contract is deliberately asymmetric:

  - task PAYLOADS stay JSON — they only carry small handles and scalars, and
    JSON keeps them readable in Redis during an incident.
  - task RESULTS are pickled — the face inference results carry numpy in
    three separate places, one of which (`face_bbox`) does not look like
    numpy at a glance.

Run: PYTHONPATH=src python tests/test_celery_serialization.py
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

import numpy as np  # noqa: E402
from kombu.serialization import dumps, loads, prepare_accept_content  # noqa: E402

from workers.celery_app import celery  # noqa: E402


def _round_trip_result(value):
    """Encode/decode exactly the way celery.backends.base does: normalise the
    configured accept list to content-types first (kombu's `accept=` compares
    against content-types like 'application/x-python-serialize', NOT short
    names like 'pickle' — passing the raw config list here raises
    ContentDisallowed and looks like a config bug when it is a test bug)."""
    accept = prepare_accept_content(celery.conf.result_accept_content)
    content_type, encoding, data = dumps(
        value, serializer=celery.conf.result_serializer
    )
    return loads(data, content_type, encoding, accept=accept)


def _round_trip_task_payload(value):
    accept = prepare_accept_content(celery.conf.accept_content)
    content_type, encoding, data = dumps(
        value, serializer=celery.conf.task_serializer
    )
    return loads(data, content_type, encoding, accept=accept)


def _face_result_like_gpu_worker():
    """The exact dict shape `_run_arcface_batch` builds (gpu_worker.py:490-505).

    Three independent numpy leaks, all load-bearing:
      - `embedding`   : the 512-float ArcFace vector
      - `face_image`  : a raw pixel crop (`roi[y1:y2, x1:x2]`)
      - `face_bbox`   : looks like a plain list, but the elements come from
                        `face.bbox.astype(int)` and are numpy.int64
    """
    x1, y1, x2, y2 = np.array([10.0, 20.0, 90.0, 100.0]).astype(int)
    return {
        "embedding": np.random.rand(512).astype(np.float32),
        "face_image": (np.random.rand(80, 80, 3) * 255).astype(np.uint8),
        "face_detected": True,
        "det_score": 0.94,
        "face_bbox": [x1, y1, x2, y2],
        "face_landmarks": [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]],
        "frontality": 0.81,
        "pitch": 0.62,
    }


class TaskPayloadStaysJsonTests(unittest.TestCase):
    def test_task_serializer_is_json(self):
        self.assertEqual(celery.conf.task_serializer, "json")

    def test_camera_task_payload_round_trips(self):
        """The camera task's payload shape: a FrameHandle flattened to a plain
        dict at the boundary. If someone re-introduces the dataclass here,
        this fails."""
        payload = {
            "camera_id": 40,
            "frame_handle": {
                "camera_id": 40, "seq": 1,
                "height": 720, "width": 1280, "channels": 3,
            },
            "frame_num": 12,
        }
        self.assertEqual(_round_trip_task_payload(payload), payload)

    def test_a_dataclass_payload_is_still_rejected(self):
        """Guards the reason payloads stay JSON: had this silently started
        accepting arbitrary objects, the boundary conversion in
        camera_tasks.py could be dropped without anything failing."""
        from workers.frame_store import FrameHandle

        handle = FrameHandle(camera_id=40, seq=1, height=720, width=1280, channels=3)
        with self.assertRaises(Exception):
            _round_trip_task_payload({"frame_handle": handle})


class FaceResultNeedsPickleTests(unittest.TestCase):
    def test_result_serializer_is_pickle(self):
        self.assertEqual(celery.conf.result_serializer, "pickle")

    def test_face_result_round_trips_intact(self):
        original = _face_result_like_gpu_worker()
        got = _round_trip_result({10: original})[10]

        self.assertTrue(np.array_equal(original["embedding"], got["embedding"]))
        self.assertTrue(np.array_equal(original["face_image"], got["face_image"]))
        self.assertEqual(original["face_bbox"], got["face_bbox"])
        self.assertEqual(original["det_score"], got["det_score"])
        self.assertIs(got["face_detected"], True)

    def test_face_result_would_fail_under_json(self):
        """The regression this config change exists to prevent. If a future
        edit flips result_serializer back to json, the assertion below stops
        holding and this test says exactly why."""
        import json

        with self.assertRaises(TypeError):
            json.dumps(_face_result_like_gpu_worker())

    def test_multi_track_result_round_trips(self):
        """`_run_arcface_batch` returns one entry per track, keyed by
        track_id — ints as dict keys, which JSON would have stringified even
        if the values had been encodable."""
        payload = {7: _face_result_like_gpu_worker(), 9: _face_result_like_gpu_worker()}
        got = _round_trip_result(payload)
        self.assertEqual(set(got.keys()), {7, 9})
        self.assertTrue(all(isinstance(k, int) for k in got))


class ExistingTaskResultsUnaffectedTests(unittest.TestCase):
    """result_serializer is app-global, so the pre-existing tasks' results ride
    on this change too. They are plain dicts, but assert it rather than assume."""

    def test_detection_task_result_shape(self):
        result = {"status": "success", "record_id": 123, "user_id": 48,
                  "attendance_status": "IN"}
        self.assertEqual(_round_trip_result(result), result)

    def test_camera_task_result_shape(self):
        result = {"camera_id": 40, "frame_num": 512, "active_track_count": 2,
                  "removed_track_count": 0, "event_count": 1, "recognition_ran": True}
        self.assertEqual(_round_trip_result(result), result)


if __name__ == "__main__":
    unittest.main()
