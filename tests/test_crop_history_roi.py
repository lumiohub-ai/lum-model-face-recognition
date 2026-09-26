"""LSO-224: crop_history keeps the padded person ROI, not the whole frame.

Both readers must still get the same pixels they got from the full frame:
the exact bbox for the proof image, and crop_person_roi(expand=0.1) for the
unrecognized card.

Run: PYTHONPATH=src python -m pytest tests/test_crop_history_roi.py
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from lum_vision import crop_person_roi  # noqa: E402
from pipeline.camera_engine import CameraEngine  # noqa: E402


def _frame(h=1080, w=1920):
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


class CropHistoryRoiTests(unittest.TestCase):
    def _entry(self, frame, bbox):
        return {"face": None, "bbox": bbox, **CameraEngine._person_crop(frame, bbox)}

    def test_proof_image_matches_full_frame_slice(self):
        frame = _frame()
        for bbox in ([800.4, 300.7, 1000.2, 900.9], [0, 0, 150, 400], [1800, 700, 1920, 1080]):
            x1, y1, x2, y2 = map(int, bbox)
            np.testing.assert_array_equal(
                CameraEngine._read_crop_image(self._entry(frame, bbox)), frame[y1:y2, x1:x2]
            )

    def test_card_image_matches_padded_roi(self):
        frame = _frame()
        bbox = [800.4, 300.7, 1000.2, 900.9]
        expected, _ = crop_person_roi(frame, np.asarray(bbox, dtype=float), expand=0.1)
        engine = CameraEngine.__new__(CameraEngine)
        np.testing.assert_array_equal(
            engine._unrecognized_card_image(self._entry(frame, bbox), track_id=1), expected
        )

    def test_entry_holds_only_the_person_not_the_frame(self):
        frame = _frame()
        entry = self._entry(frame, [800, 300, 1000, 900])
        self.assertLess(entry["person"].nbytes, frame.nbytes / 10)
        self.assertIsNone(entry["person"].base)  # a copy, not a view pinning the frame

    def test_missing_bbox_keeps_no_image(self):
        entry = self._entry(_frame(), None)
        self.assertIsNone(entry["person"])
        self.assertIsNone(CameraEngine._read_crop_image(entry))


if __name__ == "__main__":
    unittest.main()
