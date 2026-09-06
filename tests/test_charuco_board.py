"""Tests for dashboard-provided ChArUco board specifications."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from domain.calibration.charuco_board import (  # noqa: E402
    DEFAULT_CHARUCO_BOARD_SPEC,
    normalize_charuco_board_spec,
)


class CharucoBoardSpecTests(unittest.TestCase):
    def test_missing_spec_uses_reference_default(self):
        self.assertEqual(normalize_charuco_board_spec(None), DEFAULT_CHARUCO_BOARD_SPEC)

    def test_normalizes_numeric_values(self):
        result = normalize_charuco_board_spec(
            {
                "squares_x": "6",
                "squares_y": "8",
                "square_length_m": "0.05",
                "marker_length_m": "0.03",
                "dictionary": "DICT_5X5_100",
            }
        )
        self.assertEqual(result["squares_x"], 6)
        self.assertEqual(result["square_length_m"], 0.05)

    def test_rejects_non_positive_lengths(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            normalize_charuco_board_spec({"marker_length_m": 0})

    def test_rejects_marker_at_least_square(self):
        with self.assertRaisesRegex(ValueError, "smaller"):
            normalize_charuco_board_spec(
                {"square_length_m": 0.02, "marker_length_m": 0.02}
            )

    def test_rejects_unknown_dictionary(self):
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            normalize_charuco_board_spec({"dictionary": "DICT_UNKNOWN"})

    def test_rejects_board_larger_than_dictionary_capacity(self):
        with self.assertRaisesRegex(ValueError, "requires"):
            normalize_charuco_board_spec(
                {"squares_x": 11, "squares_y": 11, "dictionary": "DICT_4X4_50"}
            )


if __name__ == "__main__":
    unittest.main()