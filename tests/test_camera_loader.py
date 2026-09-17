"""LSO-188: the edge stream URL keys on camera id, not name.

A rename used to recreate the edge MediaMTX path under the new name slug and
delete the old one, so `_resolve_stream_url`'s URL pointed at a path that no
longer existed. Keying on id means a rename never moves this URL.

Run: PYTHONPATH=src python -m unittest tests.test_camera_loader
"""

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from config.camera_loader import _resolve_stream_url  # noqa: E402


class TestResolveStreamUrl(unittest.TestCase):
    def test_keys_on_id_not_name(self):
        with patch("config.camera_loader.settings") as settings:
            settings.edge_rtsp_base = "rtsp://host.docker.internal:8554"
            url = _resolve_stream_url({"id": 42, "name": "CEO Room"})
        self.assertEqual(url, "rtsp://host.docker.internal:8554/42")

    def test_rename_does_not_change_the_url(self):
        with patch("config.camera_loader.settings") as settings:
            settings.edge_rtsp_base = "rtsp://host.docker.internal:8554"
            before = _resolve_stream_url({"id": 42, "name": "CEO Room camera"})
            after = _resolve_stream_url({"id": 42, "name": "CEO Room"})
        self.assertEqual(before, after)

    def test_missing_id_skips_rather_than_falls_back(self):
        with patch("config.camera_loader.settings") as settings:
            settings.edge_rtsp_base = "rtsp://host.docker.internal:8554"
            url = _resolve_stream_url({"name": "CEO Room"})
        self.assertIsNone(url)

    def test_missing_base_raises(self):
        with patch("config.camera_loader.settings") as settings:
            settings.edge_rtsp_base = None
            with self.assertRaises(RuntimeError):
                _resolve_stream_url({"id": 42, "name": "CEO Room"})


if __name__ == "__main__":
    unittest.main()
