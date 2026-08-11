"""Unit tests for CameraEngine._pick_best_signals — gate-passing frame selection (LSO-7).

Guards against picking a max-product frame that fails a threshold when another
frame in the track passes both (which would drop a valid unrecognized case).

Run: PYTHONPATH=src python tests/test_face_signals.py
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from pipeline.camera_engine import CameraEngine  # noqa: E402

FR_MIN, PI_MIN = 0.6, 0.4


def crop(fr, pi, det=0.9):
    return {"frontality": fr, "pitch": pi, "det_score": det, "frame": None, "bbox": None}


class TestPickBestSignals(unittest.TestCase):
    def test_prefers_passing_over_higher_product(self):
        # A: higher product (0.35) but fails pitch; B: lower product (0.25), passes both.
        crops = {0: crop(0.99, 0.35), 1: crop(0.61, 0.41)}
        fr, _det, pi, cd = CameraEngine._pick_best_signals(crops, FR_MIN, PI_MIN)
        self.assertEqual((fr, pi), (0.61, 0.41))
        self.assertIsNotNone(cd)

    def test_best_among_multiple_passing(self):
        crops = {0: crop(0.7, 0.5), 1: crop(0.9, 0.8), 2: crop(0.65, 0.45)}
        fr, _det, pi, _cd = CameraEngine._pick_best_signals(crops, FR_MIN, PI_MIN)
        self.assertEqual((fr, pi), (0.9, 0.8))

    def test_falls_back_to_best_effort_when_none_pass(self):
        # None pass; return the best-product frame so the gate drops it (auditable).
        crops = {0: crop(0.5, 0.3), 1: crop(0.55, 0.2)}
        fr, _det, pi, cd = CameraEngine._pick_best_signals(crops, FR_MIN, PI_MIN)
        self.assertEqual((fr, pi), (0.5, 0.3))
        self.assertIsNotNone(cd)

    def test_none_and_malformed_crops_skipped(self):
        crops = {0: crop(None, None), 1: {"foo": "bar"}, 2: crop(0.7, 0.5)}
        fr, _det, pi, _cd = CameraEngine._pick_best_signals(crops, FR_MIN, PI_MIN)
        self.assertEqual((fr, pi), (0.7, 0.5))

    def test_empty_history_returns_none_crop(self):
        _fr, _det, _pi, cd = CameraEngine._pick_best_signals({}, FR_MIN, PI_MIN)
        self.assertIsNone(cd)


if __name__ == "__main__":
    unittest.main()
