"""LSO-130: metrics series are keyed by DB camera id, not list position.

Before this, a camera's history silently became a *different* camera's
whenever the camera set changed order — position 2 meant one camera on
Monday and another on Tuesday. These tests pin the id keying so a future
change can't quietly reintroduce a positional list.
"""

import unittest

from infrastructure.metrics_collector import MetricsCollector


class TestMetricsKeyedByCameraId(unittest.TestCase):
    def test_snapshot_uses_the_ids_it_was_given(self):
        m = MetricsCollector()
        for cam_id in (7, 22, 5):
            m.record_frame(cam_id)

        snap = m.snapshot([7, 22, 5])

        # Keys are stringified ids, not 0..n-1 positions.
        self.assertEqual(sorted(snap["cameras"], key=int), ["5", "7", "22"])

    def test_snapshot_defaults_to_the_cameras_actually_seen(self):
        m = MetricsCollector()
        m.record_frame(31)
        self.assertEqual(list(m.snapshot()["cameras"]), ["31"])

    def test_fps_and_drops_are_attributed_per_id(self):
        m = MetricsCollector()
        m.record_frame(7)
        m.record_drop(22)
        m.record_drop(22)

        snap = m.snapshot([7, 22])
        self.assertEqual(snap["cameras"]["22"]["frame_drops"], 2)
        self.assertEqual(snap["cameras"]["7"]["frame_drops"], 0)

    def test_an_unknown_id_does_not_borrow_another_cameras_series(self):
        """A positional scheme would have returned index 0's data here."""
        m = MetricsCollector()
        m.record_frame(7)
        snap = m.snapshot([999])
        self.assertEqual(snap["cameras"]["999"]["fps"], 0.0)

    def test_low_fps_alert_reports_the_camera_id(self):
        m = MetricsCollector()
        # Two frames far apart -> a real but very low fps.
        m.record_frame(22)
        m._frame_ts[22].append(m._frame_ts[22][0] + 100.0)

        alerts = m.check_alerts(camera_ids=[22], fps_threshold=1.0)
        low = [a for a in alerts if a["type"] == "LOW_FPS"]
        self.assertTrue(low, "expected a LOW_FPS alert")
        self.assertEqual(low[0]["camera_id"], 22)


if __name__ == "__main__":
    unittest.main()
