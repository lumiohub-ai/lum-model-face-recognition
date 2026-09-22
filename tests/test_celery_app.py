"""Unit tests for celery_app.py's queue/routing config and task_base's DLQ
key routing, both of which the per-camera-queue switch touches.

Run: PYTHONPATH=src python -m pytest tests/test_celery_app.py
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))


class CameraQueueNameTests(unittest.TestCase):
    """camera_queue_name routes each camera to its own cam.<id> queue
    (LSO-218). Which worker consumes that queue is a runtime Redis lease
    (pipeline/camera_lease.py + camera_boot.py), not a crc32 hash, so the
    contract left here is only that the name is per-camera and stable — the
    full routing contract lives in test_camera_routing.py."""

    def test_returns_the_per_camera_queue_form(self):
        from workers.celery_app import camera_queue_name

        for camera_id in (22, 1, 0, 999):
            self.assertEqual(camera_queue_name(camera_id), f"cam.{camera_id}")

    def test_same_camera_id_always_maps_to_the_same_queue(self):
        from workers.celery_app import camera_queue_name

        first = camera_queue_name(42)
        for _ in range(5):
            self.assertEqual(camera_queue_name(42), first)


class TrackTaskHasNoStaticQueueTests(unittest.TestCase):
    def test_camera_track_decorator_carries_no_queue(self):
        """camera.track is dispatched exclusively via yolo.detect's
        send_task(queue=camera_queue_name(...)) — a static queue= on the
        decorator would silently override every per-camera routing
        attempt, since Celery's task-level queue takes precedence over
        whatever apply_async/send_task passes."""
        from workers.camera_tasks import track_task

        # Celery only sets a `queue` attribute on the task object when the
        # decorator was given one — omitted entirely, not None, when it
        # wasn't (verified directly against the installed Celery version).
        self.assertFalse(hasattr(track_task, "queue"))

    def test_yolo_detect_keeps_its_static_queue(self):
        """Unlike camera.track, yolo.detect IS a shared, stateless queue —
        every yolo-worker replica consumes the same 'yolo' queue, so a
        static queue= here is correct, not a routing bug."""
        from workers.yolo_tasks import detect_task

        self.assertEqual(detect_task.queue, "yolo")


class CameraFramesQueueRemovedTests(unittest.TestCase):
    def test_camera_frames_is_no_longer_a_declared_queue(self):
        """The old shared camera_frames queue is gone — camera.track has no
        queue of its own to declare; only the per-camera cam.<id> queues
        (never declared in task_queues; see camera_queue_name's docstring)
        and the DLQ entry (a Redis key name, unrelated to routing — see
        DlqRoutingTests below) still mention 'camera_frames'."""
        from workers.celery_app import celery

        declared_names = {q.name for q in celery.conf.task_queues}
        self.assertNotIn("camera_frames", declared_names)

    def test_no_camera_star_route_remains(self):
        from workers.celery_app import celery

        self.assertNotIn("camera.*", celery.conf.task_routes)
        self.assertNotIn("workers.camera_tasks.*", celery.conf.task_routes)


class DlqRoutingTests(unittest.TestCase):
    """send_to_dlq routes by a lowercase substring match on the task name
    alone (task_base.py) — not by any Queue object. These pin that the new
    task names (yolo.detect, camera.track) still land in the DLQ key their
    old counterparts (yolo.detect_batch, camera.process_frame) did, since a
    substring match is easy to silently break with an unrelated rename."""

    def _dlq_key_for(self, task_name):
        from workers.task_base import send_to_dlq

        fake_client = mock.MagicMock()
        with mock.patch("messaging.redis_client.RedisClient") as fake_redis_cls:
            fake_redis_cls.get_instance.return_value.client = fake_client
            send_to_dlq(task_name, "task-id-1", (), {}, RuntimeError("boom"), "traceback")

        self.assertEqual(fake_client.lpush.call_count, 1)
        key = fake_client.lpush.call_args[0][0]
        return key

    def test_yolo_detect_routes_to_dlq_yolo(self):
        self.assertEqual(self._dlq_key_for("yolo.detect"), "dlq:yolo")

    def test_camera_track_routes_to_dlq_camera_frames(self):
        """Not renamed to dlq:camera_track — the key name is a pre-existing
        Redis convention, unrelated to the Celery task's own name, and
        changing it would only fragment operational tooling/runbooks that
        already point at dlq:camera_frames for anything camera-side."""
        self.assertEqual(self._dlq_key_for("camera.track"), "dlq:camera_frames")

    def test_face_embed_batch_routes_to_dlq_face(self):
        self.assertEqual(self._dlq_key_for("face.embed_batch"), "dlq:face")


class RedactUrlCredentialsTests(unittest.TestCase):
    """_redact_url_credentials strips broker-URL credentials before they ever
    reach a log line (LSO-220) — the one piece of that fix with real security
    relevance, so it gets a direct unit test rather than relying only on live
    verification."""

    def test_url_without_credentials_is_unchanged(self):
        from workers.celery_app import _redact_url_credentials

        url = "redis://redis:6379/0"
        self.assertEqual(_redact_url_credentials(url), url)

    def test_password_only_is_stripped(self):
        from workers.celery_app import _redact_url_credentials

        result = _redact_url_credentials("redis://:testpass@localhost:6379/0")
        self.assertEqual(result, "redis://localhost:6379/0")
        self.assertNotIn("testpass", result)

    def test_user_and_password_are_stripped(self):
        from workers.celery_app import _redact_url_credentials

        result = _redact_url_credentials("redis://user:testpass@localhost:6379/0")
        self.assertEqual(result, "redis://localhost:6379/0")
        self.assertNotIn("user", result)
        self.assertNotIn("testpass", result)

    def test_no_port_is_handled(self):
        from workers.celery_app import _redact_url_credentials

        result = _redact_url_credentials("redis://:testpass@localhost/0")
        self.assertEqual(result, "redis://localhost/0")

    def test_unparseable_url_does_not_raise(self):
        from workers.celery_app import _redact_url_credentials

        # A URL with an invalid (non-numeric) port raises ValueError from
        # urlsplit's .port property access, not from urlsplit() itself —
        # exercised here to confirm the try/except actually catches it.
        result = _redact_url_credentials("redis://user:pass@localhost:notaport/0")
        self.assertEqual(result, "<unparseable>")


if __name__ == "__main__":
    unittest.main()
