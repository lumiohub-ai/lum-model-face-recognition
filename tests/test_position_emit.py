"""Unit tests for the foot-point + projection + throttle path.

Tests the pure pieces in isolation — no Redis, no DB, no engine. We construct
the registry with synthetic cache entries and a stub publisher, then call
CameraEngine.emit_positions directly with hand-built active_tracks.

Run: PYTHONPATH=src python tests/test_position_emit.py
"""

import os
import sys
import time
import types
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from domain.calibration.homography_registry import HomographyRegistry  # noqa: E402


def _bbox_foot(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, y2)


def _project(H, pt):
    arr = np.array([[[pt[0], pt[1]]]], dtype=np.float64)
    return cv2.perspectiveTransform(arr, H).reshape(2).tolist()


class StubPublisher:
    def __init__(self):
        self.events = []

    def publish_user_location_updated_position(
        self, camera_id, map_id, track_id, user_id, user_name, x, y
    ):
        self.events.append(
            {
                "camera_id": camera_id,
                "map_id": map_id,
                "track_id": track_id,
                "user_id": user_id,
                "user_name": user_name,
                "x": x,
                "y": y,
            }
        )
        return True


class StubIdentityManager:
    def __init__(self, locked_for=None):
        self._locked = locked_for or {}

    def is_identity_locked(self, track_id):
        return track_id in self._locked

    def get_locked_identity(self, track_id):
        return self._locked.get(track_id)


def make_engine(H, map_id, locked_for=None, name_to_id=None):
    """Build a minimal object with the attributes emit_positions touches."""
    registry = HomographyRegistry()
    registry._cache[("test-slug", 7)] = (H, map_id)

    engine = types.SimpleNamespace()
    engine.client_slug = "test-slug"
    engine.camera_id = 7
    engine.homography_registry = registry
    engine.identity_manager = StubIdentityManager(locked_for=locked_for)
    engine.name_to_id_map = name_to_id or {}
    engine._publisher = StubPublisher()
    engine._position_last_emit = {}

    from pipeline.camera_engine import CameraEngine
    engine.emit_positions = types.MethodType(CameraEngine.emit_positions, engine)
    return engine


class FootPointTests(unittest.TestCase):
    def test_foot_point_math(self):
        self.assertEqual(_bbox_foot([10, 20, 110, 220]), (60.0, 220.0))
        self.assertEqual(_bbox_foot([0, 0, 100, 100]), (50.0, 100.0))


class IdentityHomographyTests(unittest.TestCase):
    def test_identity_h_emits_foot_point_unchanged(self):
        H = np.eye(3, dtype=np.float64)
        engine = make_engine(H, map_id=42)
        engine.emit_positions(
            [{"track_id": 1, "bbox": [10.0, 20.0, 110.0, 220.0]}]
        )
        events = engine._publisher.events
        self.assertEqual(len(events), 1)
        evt = events[0]
        self.assertEqual(evt["camera_id"], 7)
        self.assertEqual(evt["map_id"], 42)
        self.assertEqual(evt["track_id"], 1)
        self.assertAlmostEqual(evt["x"], 60.0, places=9)
        self.assertAlmostEqual(evt["y"], 220.0, places=9)
        self.assertIsNone(evt["user_id"])
        self.assertIsNone(evt["user_name"])


class ScaleTranslateProjectionTests(unittest.TestCase):
    def test_scale_translate(self):
        # Same fixture as Phase 1: scale 2, translate +10
        H = np.array([[2, 0, 10], [0, 2, 10], [0, 0, 1]], dtype=np.float64)
        engine = make_engine(H, map_id=1)
        engine.emit_positions(
            [{"track_id": 5, "bbox": [0.0, 0.0, 100.0, 100.0]}]
        )
        evt = engine._publisher.events[0]
        # foot point = (50, 100); H · (50, 100) = (110, 210)
        self.assertAlmostEqual(evt["x"], 110.0, places=6)
        self.assertAlmostEqual(evt["y"], 210.0, places=6)


class ThrottleTests(unittest.TestCase):
    def test_emits_once_within_200ms(self):
        engine = make_engine(np.eye(3), map_id=1)
        bbox = [0.0, 0.0, 100.0, 100.0]
        engine.emit_positions([{"track_id": 9, "bbox": bbox}])
        engine.emit_positions([{"track_id": 9, "bbox": bbox}])  # immediate
        self.assertEqual(len(engine._publisher.events), 1)

    def test_emits_again_after_cooldown(self):
        engine = make_engine(np.eye(3), map_id=1)
        bbox = [0.0, 0.0, 100.0, 100.0]
        engine.emit_positions([{"track_id": 9, "bbox": bbox}])
        # Simulate cooldown elapsed by rewinding last_emit
        engine._position_last_emit[9] = time.monotonic() - 0.25
        engine.emit_positions([{"track_id": 9, "bbox": bbox}])
        self.assertEqual(len(engine._publisher.events), 2)


class RecognizedIdentityTests(unittest.TestCase):
    def test_locked_identity_populates_user_id_and_name(self):
        engine = make_engine(
            np.eye(3),
            map_id=1,
            locked_for={3: {"name": "Aziza K."}},
            name_to_id={"Aziza K.": 42},
        )
        engine.emit_positions(
            [{"track_id": 3, "bbox": [0.0, 0.0, 100.0, 100.0]}]
        )
        evt = engine._publisher.events[0]
        self.assertEqual(evt["user_id"], 42)
        self.assertEqual(evt["user_name"], "Aziza K.")


class MissingHomographyTests(unittest.TestCase):
    def test_no_h_no_emit(self):
        registry = HomographyRegistry()
        # Cache a negative entry to short-circuit DB
        registry._cache[("test-slug", 7)] = None

        engine = types.SimpleNamespace()
        engine.client_slug = "test-slug"
        engine.camera_id = 7
        engine.homography_registry = registry
        engine.identity_manager = StubIdentityManager()
        engine.name_to_id_map = {}
        engine._publisher = StubPublisher()
        engine._position_last_emit = {}

        from pipeline.camera_engine import CameraEngine
        engine.emit_positions = types.MethodType(
            CameraEngine.emit_positions, engine
        )

        engine.emit_positions(
            [{"track_id": 1, "bbox": [0.0, 0.0, 100.0, 100.0]}]
        )
        self.assertEqual(engine._publisher.events, [])


if __name__ == "__main__":
    unittest.main()
