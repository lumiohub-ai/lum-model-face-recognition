"""Unit tests for SmartOfficeEngine.reload_camera_configs' same-camera-set
branch (LSO-155).

SmartOfficeEngine.__init__ is heavy (DB queries, model loading, video
streams) — this uses the same __new__-bypass pattern as test_camera_tasks.py
to build an engine with only the attributes reload_camera_configs and
_restart_camera_stream actually touch, so the test stays fast and mock-only.

Run: PYTHONPATH=src python -m pytest tests/test_engine_reload.py
"""

import importlib.util
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

# pipeline.engine unconditionally imports infrastructure -> storage ->
# embedding_sync, which imports lum_vision at module level — this isn't
# lazy, so there's no way to reach SmartOfficeEngine without lum_vision
# installed. Same skip pattern test_gpu_rpc.py already uses for the same
# reason, rather than mocking sys.modules for an entire private package.
_LUM_VISION_AVAILABLE = importlib.util.find_spec("lum_vision") is not None


class FakeProducer:
    """Stands in for CeleryCameraProducer — no real thread/stream_handler."""

    def __init__(self, camera_id, camera_config, stream_handler, **kwargs):
        self.camera_id = camera_id
        self.camera_config = camera_config
        self.stream_handler = stream_handler
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self, timeout=5.0):
        self.stopped = True


class FakeStreamManager:
    def __init__(self):
        self.streams = {}
        self.removed = []
        self.added = []

    def remove_stream(self, camera_id):
        self.removed.append(camera_id)
        self.streams.pop(camera_id, None)

    def add_stream(self, config):
        camera_id = config.get("camera_id")
        self.added.append(camera_id)
        self.streams[camera_id] = f"stream-for-{camera_id}"
        return self.streams[camera_id]


def _make_engine(camera_configs, camera_workers):
    from pipeline.engine import SmartOfficeEngine

    engine = SmartOfficeEngine.__new__(SmartOfficeEngine)
    engine.client_slug = "test-client"
    engine.applications = ["attendance"]
    engine.camera_configs = camera_configs
    engine.camera_workers = camera_workers
    engine.stream_manager = FakeStreamManager()
    engine._detection_interval = 2
    engine.metrics = None
    engine.needs_reinit = False
    return engine


@unittest.skipUnless(
    _LUM_VISION_AVAILABLE,
    "lum_vision not importable in this environment - pipeline.engine cannot "
    "be imported without it (same constraint as test_gpu_rpc.py)",
)
class SameCameraSetReloadTests(unittest.TestCase):
    """old_ids == new_ids branch of reload_camera_configs."""

    def _patch_common(self, new_configs):
        return mock.patch.multiple(
            "pipeline.engine",
            load_cameras_from_db=mock.Mock(return_value=new_configs),
            RedisClient=mock.Mock(),
            CeleryCameraProducer=FakeProducer,
        )

    def test_stream_url_change_restarts_only_that_cameras_producer(self):
        """LSO-155's actual failure: a rename/re-IP kept the same camera_id
        but changed stream_url, and the old code never restarted anything —
        the camera kept streaming from the dead URL forever."""
        cam1_config = {"camera_id": 1, "stream_url": "rtsp://old-ip/1"}
        cam2_config = {"camera_id": 2, "stream_url": "rtsp://stable/2"}
        producer1 = FakeProducer(1, cam1_config, "old-stream-1")
        producer2 = FakeProducer(2, cam2_config, "stream-2")
        engine = _make_engine([cam1_config, cam2_config], [producer1, producer2])

        new_cam1 = {"camera_id": 1, "stream_url": "rtsp://new-ip/1"}
        new_cam2 = {"camera_id": 2, "stream_url": "rtsp://stable/2"}

        with self._patch_common([new_cam1, new_cam2]):
            result = engine.reload_camera_configs()

        self.assertTrue(result)
        # Camera 1's stream_url changed: its producer must have been stopped
        # and replaced, and StreamManager told to reconnect it.
        self.assertTrue(producer1.stopped)
        self.assertEqual(engine.stream_manager.removed, [1])
        self.assertEqual(engine.stream_manager.added, [1])
        new_producer1 = next(w for w in engine.camera_workers if w.camera_id == 1)
        self.assertIsNot(new_producer1, producer1)
        self.assertTrue(new_producer1.started)
        self.assertEqual(new_producer1.camera_config["stream_url"], "rtsp://new-ip/1")

        # Camera 2's stream_url did NOT change: must be untouched.
        self.assertFalse(producer2.stopped)
        self.assertNotIn(2, engine.stream_manager.removed)
        self.assertNotIn(2, engine.stream_manager.added)
        self.assertIs(
            next(w for w in engine.camera_workers if w.camera_id == 2), producer2
        )

    def test_non_stream_field_change_updates_config_without_restart(self):
        """A field change that isn't stream_url (e.g. application) must
        still reach the in-place-mutated config dict, but must NOT trigger
        a stream restart — that would drop frames for no reason."""
        cam_config = {
            "camera_id": 1,
            "stream_url": "rtsp://stable/1",
            "application": ["attendance"],
        }
        producer = FakeProducer(1, cam_config, "stream-1")
        engine = _make_engine([cam_config], [producer])

        new_config = {
            "camera_id": 1,
            "stream_url": "rtsp://stable/1",
            "application": ["attendance", "action_recognition"],
        }

        with self._patch_common([new_config]):
            engine.reload_camera_configs()

        self.assertFalse(producer.stopped)
        self.assertEqual(engine.stream_manager.removed, [])
        self.assertEqual(engine.stream_manager.added, [])
        # In-place mutation means the ORIGINAL config dict object (which
        # producer.camera_config still points at) reflects the new value —
        # this is what makes CeleryCameraProducer's live roi/application
        # reads work without their own restart.
        self.assertEqual(
            producer.camera_config["application"],
            ["attendance", "action_recognition"],
        )
        self.assertIs(producer.camera_config, cam_config)

    def test_camera_set_changed_branch_is_unaffected(self):
        """The pre-existing full-engine-restart path (different camera_id
        set) must still take the old branch, not the new per-field diff."""
        cam1_config = {"camera_id": 1, "stream_url": "rtsp://a/1"}
        producer1 = FakeProducer(1, cam1_config, "stream-1")
        engine = _make_engine([cam1_config], [producer1])
        engine._running = True

        new_configs = [{"camera_id": 2, "stream_url": "rtsp://b/2"}]

        with self._patch_common(new_configs):
            with mock.patch.object(engine, "stop") as mock_stop:
                result = engine.reload_camera_configs()

        self.assertTrue(result)
        self.assertTrue(engine.needs_reinit)
        mock_stop.assert_called_once()
        # The per-camera restart helper must never fire on this branch.
        self.assertFalse(producer1.stopped)
        self.assertEqual(engine.stream_manager.removed, [])


if __name__ == "__main__":
    unittest.main()
