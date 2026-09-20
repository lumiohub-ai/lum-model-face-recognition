"""LSO-188: which edge path the AI reads is a per-site switch.

`SO_EDGE_PATH_KEY=slug` (default) keeps today's rtsp://<edge>/<slug(name)>,
which every edge serves. `id` reads the rtsp://<edge>/<cameras.id> alias that
config-sync >= 0.4.10 publishes next to the slug, so a rename can't move the
URL. The default MUST stay slug: 0.8.1 shipped the id form unconditionally
against edges that had no id paths and read zero frames.

Run: PYTHONPATH=src python -m unittest tests.test_camera_loader
"""

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from config.camera_loader import _resolve_stream_url  # noqa: E402

BASE = "rtsp://host.docker.internal:8554"


def _settings(key):
    p = patch("config.camera_loader.settings")
    s = p.start()
    s.edge_rtsp_base = BASE
    s.edge_path_key = key
    return p


class TestSlugDefault(unittest.TestCase):
    def setUp(self):
        self.p = _settings("slug")

    def tearDown(self):
        self.p.stop()

    def test_reads_slug_path(self):
        self.assertEqual(_resolve_stream_url({"id": 42, "name": "CEO Room"}), f"{BASE}/ceoroom")

    def test_rename_moves_the_url(self):
        # Known limitation of slug mode — the reason the id alias exists.
        before = _resolve_stream_url({"id": 42, "name": "CEO Room camera"})
        after = _resolve_stream_url({"id": 42, "name": "CEO Room"})
        self.assertNotEqual(before, after)

    def test_missing_name_skips(self):
        self.assertIsNone(_resolve_stream_url({"id": 42}))


class TestIdMode(unittest.TestCase):
    def setUp(self):
        self.p = _settings("id")

    def tearDown(self):
        self.p.stop()

    def test_reads_id_alias(self):
        self.assertEqual(_resolve_stream_url({"id": 42, "name": "CEO Room"}), f"{BASE}/42")

    def test_rename_does_not_move_the_url(self):
        before = _resolve_stream_url({"id": 42, "name": "CEO Room camera"})
        after = _resolve_stream_url({"id": 42, "name": "CEO Room"})
        self.assertEqual(before, after)

    def test_missing_id_skips_rather_than_falls_back(self):
        self.assertIsNone(_resolve_stream_url({"name": "CEO Room"}))


class TestGuards(unittest.TestCase):
    def test_unknown_key_raises(self):
        p = _settings("name")
        try:
            with self.assertRaises(RuntimeError):
                _resolve_stream_url({"id": 42, "name": "CEO Room"})
        finally:
            p.stop()

    def test_missing_base_raises(self):
        with patch("config.camera_loader.settings") as s:
            s.edge_rtsp_base = None
            s.edge_path_key = "slug"
            with self.assertRaises(RuntimeError):
                _resolve_stream_url({"id": 42, "name": "CEO Room"})


if __name__ == "__main__":
    unittest.main()
