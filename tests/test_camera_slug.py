"""Slug drift-guard (LSO-30).

The camera-name -> MediaMTX path rule is shared across repos. Canonical contract
vectors: mediamtx-edge/generator/slug_vectors.json. If this fails, this repo's
slug has drifted from the rest of the stack.

Run with PYTHONPATH=src:  python -m unittest tests.test_camera_slug
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from config.camera_slug import mediamtx_path  # noqa: E402

# Mirror of slug_vectors.json → "valid" (keep in sync).
VALID = {
    "Vision_1": "vision1",
    "Vision_2": "vision2",
    "In_1": "in1",
    "IN_2": "in2",
    "Out-1": "out1",
    "Out_2": "out2",
    "Software": "software",
    "Software_2": "software2",
    "Meeting room": "meetingroom",
    "NLP": "nlp",
    "already-slugged": "alreadyslugged",
    "A.B_C-D": "abcd",
    "Café 3": "caf3",
}


class TestMediamtxSlug(unittest.TestCase):
    def test_canonical_vectors(self):
        for name, expected in VALID.items():
            self.assertEqual(mediamtx_path(name), expected, name)

    def test_idempotent(self):
        for name in VALID:
            self.assertEqual(mediamtx_path(mediamtx_path(name)), mediamtx_path(name))


if __name__ == "__main__":
    unittest.main()
