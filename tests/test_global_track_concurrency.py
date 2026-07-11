"""Concurrency + lifecycle-cleanup tests for GlobalTrackManager.

Covers the fixes for:
  - CRITICAL: unsynchronized shared state (dict-changed-size crash + duplicate
    global IDs from a racy `counter += 1`).
  - HIGH: `local_to_global` leak / stale global ids on track removal & archival.
  - MEDIUM: `enabled` init fallback (env var / default-off) never left undefined.

These run pure — no GPU. The ReID extractor is stubbed to return random unit
vectors so every assignment falls through to "create new global id", which is
exactly the path that hammers the shared counter and dict.

Run: PYTHONPATH=src python tests/test_global_track_concurrency.py
"""

import os
import sys
import threading
import time
import unittest
from datetime import datetime, timedelta

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from domain.person_tracking.global_track import GlobalTrackManager  # noqa: E402


def _make_manager(**overrides):
    """Manager with global tracking on and the GPU ReID extractor stubbed out."""
    mgr = GlobalTrackManager(app_config={"enable_global_tracking": True})
    # Orthogonal-ish random unit vectors → cosine sim ≈ 0 ≪ threshold, so each
    # call creates a brand-new global track (max contention on counter + dict).
    rng = np.random.default_rng(0)
    lock = threading.Lock()

    def _fake_extract(_crop):
        with lock:  # rng isn't thread-safe; the manager lock doesn't cover this
            v = rng.standard_normal(512).astype(np.float32)
        return v / np.linalg.norm(v)

    mgr._extract_body_embedding = _fake_extract  # type: ignore[assignment]
    mgr._body_reid_model = object()  # non-None so lazy init is skipped
    for k, v in overrides.items():
        setattr(mgr, k, v)
    return mgr


def _good_crop():
    # h=200, w=80 → aspect 2.5 (in [1.5, 4.0]), area 16000 (≥ 3200): passes.
    return np.zeros((200, 80, 3), dtype=np.uint8)


class TestConcurrency(unittest.TestCase):
    def setUp(self):
        # Force aggressive thread switching so the race is reliably exercised.
        # Without the manager lock this reproduces both failure modes:
        # "dictionary changed size during iteration" and lost counter updates
        # (duplicate/dropped global ids). Restored in tearDown.
        self._orig_switch = sys.getswitchinterval()
        sys.setswitchinterval(1e-6)

    def tearDown(self):
        sys.setswitchinterval(self._orig_switch)

    def test_no_duplicate_global_ids_under_thread_storm(self):
        """Many camera threads assigning while a validator sweeps: no crash,
        every created global id is unique, and the dict count matches creates."""
        mgr = _make_manager()
        n_threads = 8
        per_thread = 60
        errors = []

        def worker(cam_id):
            try:
                for local_id in range(per_thread):
                    mgr.assign_global_id(
                        camera_id=cam_id,
                        local_track_id=local_id,
                        person_crop=_good_crop(),
                        detection_confidence=0.9,
                        frame_num=local_id,
                    )
            except Exception as e:  # pragma: no cover - failure path
                errors.append(e)

        stop = threading.Event()

        def validator():
            try:
                while not stop.is_set():
                    mgr.periodic_validation()
                    mgr.get_cache_stats()
            except Exception as e:  # pragma: no cover - failure path
                errors.append(e)

        val_thread = threading.Thread(target=validator, daemon=True)
        val_thread.start()

        threads = [threading.Thread(target=worker, args=(c,)) for c in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        stop.set()
        val_thread.join(timeout=5)

        self.assertEqual(errors, [], f"threads raised: {errors}")

        # Every (camera, local) pair is unique and each produced a distinct new
        # global track. A racy counter would collide ids → fewer dict entries.
        expected = n_threads * per_thread
        self.assertEqual(len(mgr.global_tracks), expected)
        self.assertEqual(len(set(mgr.global_tracks.keys())), expected)

        # Mapping is consistent with the created tracks.
        mapped = {gid for cam in mgr.local_to_global.values() for gid in cam.values()}
        self.assertEqual(mapped, set(mgr.global_tracks.keys()))


class TestLocalToGlobalCleanup(unittest.TestCase):
    def test_mapping_removed_on_track_removed(self):
        mgr = _make_manager()
        gid = mgr.assign_global_id(
            camera_id=1, local_track_id=7, person_crop=_good_crop(),
            detection_confidence=0.9, frame_num=0,
        )
        self.assertIn(7, mgr.local_to_global[1])
        self.assertEqual(mgr.get_global_id(1, 7), gid)

        mgr.on_track_removed(camera_id=1, local_track_id=7)

        # Mapping is gone → no leak, no stale global id lookup.
        self.assertNotIn(7, mgr.local_to_global[1])
        self.assertIsNone(mgr.get_global_id(1, 7))

    def test_mapping_purged_on_archive(self):
        mgr = _make_manager()
        gid = mgr.assign_global_id(
            camera_id=2, local_track_id=3, person_crop=_good_crop(),
            detection_confidence=0.9, frame_num=0,
        )
        # Force the track to look long-inactive so archival collects it.
        track = mgr.global_tracks[gid]
        track.last_seen = datetime.now() - timedelta(minutes=30)
        track.mark_camera_inactive(2)

        archived = mgr.cleanup_inactive_global_tracks(max_inactive_min=10.0)

        self.assertEqual(archived, 1)
        self.assertNotIn(gid, mgr.global_tracks)
        # The dangling local→global entry must be purged, not left pointing at a
        # popped track.
        self.assertNotIn(3, mgr.local_to_global[2])


class TestEnabledFallback(unittest.TestCase):
    def test_env_var_used_when_no_config_signal(self):
        mgr = GlobalTrackManager(app_config={})
        mgr.config = {}  # simulate no yaml
        # Re-run the init decision with a cleared config via a fresh instance is
        # awkward; instead assert the documented behaviors directly.
        os.environ["ENABLE_GLOBAL_TRACKING"] = "true"
        try:
            m = GlobalTrackManager(app_config=None, config_path="/nonexistent.yaml")
            self.assertTrue(m.enabled)
        finally:
            del os.environ["ENABLE_GLOBAL_TRACKING"]

    def test_defaults_off_when_no_signal_anywhere(self):
        os.environ.pop("ENABLE_GLOBAL_TRACKING", None)
        m = GlobalTrackManager(app_config=None, config_path="/nonexistent.yaml")
        self.assertFalse(m.enabled)  # never left undefined → no AttributeError


if __name__ == "__main__":
    unittest.main(verbosity=2)
