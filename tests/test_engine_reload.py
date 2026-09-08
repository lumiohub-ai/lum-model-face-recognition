"""Unit tests for SmartOfficeEngine.reload_camera_configs' same-camera-set
branch (LSO-155).

SmartOfficeEngine.__init__ is heavy (DB queries, model loading) — this uses
the same __new__-bypass pattern as test_camera_tasks.py to build an engine
with only the attributes reload_camera_configs actually touches, so the
test stays fast and mock-only.

This engine no longer owns any StreamManager or CeleryCameraProducer —
decoding moved into decode_main.py's DecodeWorker (docs/FOLLOW_UPS.md item
5). reload_camera_configs' job here is now only: keep `self.camera_configs`
correct (in-place mutation, so anything else holding the same dict object
sees updates for free) and, on a real camera-set change, flag
`needs_reinit` and stop. Restarting a camera's stream after its stream_url
changes is DecodeWorker's job now (test_decode_worker.py's
SyncConfigTests) — this file no longer covers that.

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


def _make_engine(camera_configs):
    from pipeline.engine import SmartOfficeEngine

    engine = SmartOfficeEngine.__new__(SmartOfficeEngine)
    engine.client_slug = "test-client"
    engine.applications = ["attendance"]
    engine.camera_configs = camera_configs
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
        )

    def test_a_field_change_reaches_the_in_place_mutated_config_dict(self):
        """A change to any field (stream_url included — this process
        doesn't act on it anymore, DecodeWorker does) must still reach the
        ORIGINAL config dict object, not a rebound one: anything else
        holding this same dict (a decode worker's own bookkeeping, this
        engine's own _camera_ids() calls) must see the update for free.
        This is the in-place-mutation guarantee LSO-155 depends on."""
        cam_config = {
            "camera_id": 1,
            "stream_url": "rtsp://old-ip/1",
            "application": ["attendance"],
        }
        engine = _make_engine([cam_config])

        new_config = {
            "camera_id": 1,
            "stream_url": "rtsp://new-ip/1",
            "application": ["attendance", "action_recognition"],
        }

        with self._patch_common([new_config]):
            result = engine.reload_camera_configs()

        self.assertTrue(result)
        # Same object, updated fields — not a new dict.
        self.assertIs(engine.camera_configs[0], cam_config)
        self.assertEqual(cam_config["stream_url"], "rtsp://new-ip/1")
        self.assertEqual(
            cam_config["application"], ["attendance", "action_recognition"]
        )

    def test_a_removed_field_is_popped_from_the_in_place_dict(self):
        """update-then-remove, never clear()-then-update: a field present
        in the old config but absent from the new one must be popped, not
        left stale — but the dict is never cleared first (a concurrent
        reader must never see an empty config mid-reload)."""
        cam_config = {"camera_id": 1, "stream_url": "rtsp://a/1", "roi": [0, 0, 10, 10]}
        engine = _make_engine([cam_config])

        new_config = {"camera_id": 1, "stream_url": "rtsp://a/1"}

        with self._patch_common([new_config]):
            engine.reload_camera_configs()

        self.assertNotIn("roi", cam_config)

    def test_camera_set_changed_branch_is_unaffected(self):
        """The pre-existing full-engine-restart path (different camera_id
        set) must still take the old branch, not the per-field diff."""
        cam1_config = {"camera_id": 1, "stream_url": "rtsp://a/1"}
        engine = _make_engine([cam1_config])
        engine._running = True

        new_configs = [{"camera_id": 2, "stream_url": "rtsp://b/2"}]

        with self._patch_common(new_configs):
            with mock.patch.object(engine, "stop") as mock_stop:
                result = engine.reload_camera_configs()

        self.assertTrue(result)
        self.assertTrue(engine.needs_reinit)
        mock_stop.assert_called_once()
        self.assertEqual(engine.camera_configs, new_configs)


if __name__ == "__main__":
    unittest.main()
