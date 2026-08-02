"""Unit tests for MetricsCollector's per-camera stage breakdown (LSO-66).

Pure CPU, no GPU/DB/network — exercises record_frame_stages() and the
snapshot() aggregation (pct shares, p95, wait/gpu sub-fields) in isolation.

Run: PYTHONPATH=src python tests/test_metrics_collector.py
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from infrastructure.metrics_collector import MetricsCollector  # noqa: E402


def stage_record(frame_ms=100.0, **overrides):
    """A full stage dict (every primary + aux key present) for one frame."""
    stages = {
        "decode": 5.0,
        "detect": 30.0, "detect_wait": 20.0, "detect_gpu": 8.0,
        "track": 10.0,
        "reid": 15.0,
        "face": 10.0, "face_wait": 6.0, "face_gpu": 3.0,
        "match": 5.0,
        "identity": 5.0,
        "publish": 2.0,
        "other": 18.0,
    }
    stages.update(overrides)
    return frame_ms, stages


class SnapshotEmptyTests(unittest.TestCase):
    def test_camera_with_no_frames_yet_does_not_blow_up(self):
        metrics = MetricsCollector()
        snap = metrics.snapshot(camera_indices=[0])

        cam = snap["cameras"]["0"]
        self.assertEqual(cam["frame_ms"], 0.0)
        self.assertEqual(cam["stages"], {})

    def test_unrequested_camera_is_absent(self):
        metrics = MetricsCollector()
        metrics.record_frame_stages(0, *stage_record())
        snap = metrics.snapshot(camera_indices=[1])

        self.assertNotIn("0", snap["cameras"])
        self.assertIn("1", snap["cameras"])


class StageAggregationTests(unittest.TestCase):
    def test_single_frame_pct_matches_ms_over_frame_ms(self):
        metrics = MetricsCollector()
        metrics.record_frame_stages(0, *stage_record(frame_ms=100.0, detect=30.0))

        snap = metrics.snapshot(camera_indices=[0])
        detect = snap["cameras"]["0"]["stages"]["detect"]

        self.assertAlmostEqual(detect["ms"], 30.0, places=1)
        self.assertAlmostEqual(detect["pct"], 30.0, places=1)

    def test_primary_stage_shares_sum_to_frame_ms(self):
        """Shares must sum to ~100% by construction (see LSO-66 plan)."""
        metrics = MetricsCollector()
        for _ in range(5):
            metrics.record_frame_stages(0, *stage_record())

        snap = metrics.snapshot(camera_indices=[0])
        stages = snap["cameras"]["0"]["stages"]
        primary = ("decode", "detect", "track", "reid", "face", "match", "identity", "publish", "other")

        total_pct = sum(stages[s]["pct"] for s in primary)
        self.assertAlmostEqual(total_pct, 100.0, delta=0.5)

        total_ms = sum(stages[s]["ms"] for s in primary)
        self.assertAlmostEqual(total_ms, snap["cameras"]["0"]["frame_ms"], delta=0.5)

    def test_mean_is_averaged_across_the_rolling_window(self):
        metrics = MetricsCollector()
        metrics.record_frame_stages(0, *stage_record(frame_ms=100.0, detect=20.0))
        metrics.record_frame_stages(0, *stage_record(frame_ms=100.0, detect=40.0))

        snap = metrics.snapshot(camera_indices=[0])
        self.assertAlmostEqual(snap["cameras"]["0"]["stages"]["detect"]["ms"], 30.0, places=1)

    def test_decode_stage_is_a_primary_stage_with_no_wait_gpu_subfields(self):
        """decode covers stream_handler.read() — a plain primary stage like track/reid,
        not one of the GPU-worker stages that has a wait/gpu split."""
        metrics = MetricsCollector()
        metrics.record_frame_stages(0, *stage_record(decode=7.0))

        stages = metrics.snapshot(camera_indices=[0])["cameras"]["0"]["stages"]
        self.assertAlmostEqual(stages["decode"]["ms"], 7.0, places=1)
        self.assertAlmostEqual(stages["decode"]["pct"], 7.0, places=1)
        self.assertNotIn("wait_ms", stages["decode"])
        self.assertNotIn("gpu_ms", stages["decode"])

    def test_wait_and_gpu_subfields_present_for_detect_and_face_only(self):
        metrics = MetricsCollector()
        metrics.record_frame_stages(0, *stage_record())

        stages = metrics.snapshot(camera_indices=[0])["cameras"]["0"]["stages"]
        self.assertIn("wait_ms", stages["detect"])
        self.assertIn("gpu_ms", stages["detect"])
        self.assertIn("wait_ms", stages["face"])
        self.assertNotIn("wait_ms", stages["track"])
        self.assertNotIn("wait_ms", stages["other"])

    def test_p95_reflects_a_bimodal_distribution(self):
        """reid-style stage: mostly cheap, occasionally expensive (OSNet)."""
        metrics = MetricsCollector()
        for _ in range(9):
            metrics.record_frame_stages(0, *stage_record(reid=2.0))
        metrics.record_frame_stages(0, *stage_record(reid=40.0))

        stages = metrics.snapshot(camera_indices=[0])["cameras"]["0"]["stages"]
        # Mean is dragged down by the 9 cheap frames; p95 should still surface the spike.
        self.assertLess(stages["reid"]["ms"], 10.0)
        self.assertGreaterEqual(stages["reid"]["p95_ms"], 40.0)

    def test_missing_stage_key_defaults_to_zero(self):
        """A stage absent from the recorded dict (e.g. reid disabled) contributes 0, not KeyError."""
        metrics = MetricsCollector()
        frame_ms, stages = stage_record()
        del stages["reid"]
        metrics.record_frame_stages(0, frame_ms, stages)

        snap_stages = metrics.snapshot(camera_indices=[0])["cameras"]["0"]["stages"]
        self.assertEqual(snap_stages["reid"]["ms"], 0.0)
        self.assertEqual(snap_stages["reid"]["pct"], 0.0)

    def test_rolling_window_respects_latency_buffer_size(self):
        metrics = MetricsCollector()
        for i in range(metrics.LATENCY_BUFFER + 10):
            metrics.record_frame_stages(0, *stage_record(detect=float(i)))

        # Only the most recent LATENCY_BUFFER records should count toward the mean.
        expected_values = list(
            range(10, metrics.LATENCY_BUFFER + 10)
        )  # first 10 evicted
        expected_mean = sum(expected_values) / len(expected_values)

        stages = metrics.snapshot(camera_indices=[0])["cameras"]["0"]["stages"]
        self.assertAlmostEqual(stages["detect"]["ms"], expected_mean, places=1)

    def test_cameras_are_independent(self):
        metrics = MetricsCollector()
        metrics.record_frame_stages(0, *stage_record(detect=10.0))
        metrics.record_frame_stages(1, *stage_record(detect=90.0))

        snap = metrics.snapshot(camera_indices=[0, 1])
        self.assertAlmostEqual(snap["cameras"]["0"]["stages"]["detect"]["ms"], 10.0, places=1)
        self.assertAlmostEqual(snap["cameras"]["1"]["stages"]["detect"]["ms"], 90.0, places=1)


class StreamDecodeTests(unittest.TestCase):
    """record_stream_read() — the background capture thread's own decode cost,
    tracked separately from the detection-frame stage breakdown (LSO-66 follow-up)."""

    def test_absent_by_default(self):
        metrics = MetricsCollector()
        metrics.record_frame_stages(0, *stage_record())

        stream = metrics.snapshot(camera_indices=[0])["cameras"]["0"]["stream"]
        self.assertEqual(stream["decode_ms"], 0.0)
        self.assertEqual(stream["native_fps"], 0.0)

    def test_records_mean_and_p95(self):
        metrics = MetricsCollector()
        for _ in range(9):
            metrics.record_stream_read(0, 5.0)
        metrics.record_stream_read(0, 50.0)

        stream = metrics.snapshot(camera_indices=[0])["cameras"]["0"]["stream"]
        self.assertLess(stream["decode_ms"], 10.0)
        self.assertGreaterEqual(stream["decode_p95_ms"], 50.0)

    def test_independent_of_detection_frame_stage_breakdown(self):
        """The background thread's own rate has nothing to do with the
        pipeline's detection_interval-throttled fps — they must not be mixed."""
        metrics = MetricsCollector()
        metrics.record_frame(0)  # pipeline-side fps tracking
        metrics.record_stream_read(0, 8.0)  # background capture thread

        snap = metrics.snapshot(camera_indices=[0])["cameras"]["0"]
        self.assertAlmostEqual(snap["stream"]["decode_ms"], 8.0, places=1)
        # Both exist independently; recording one doesn't populate the other.
        self.assertEqual(snap["stages"], {})

    def test_cameras_are_independent(self):
        metrics = MetricsCollector()
        metrics.record_stream_read(0, 5.0)
        metrics.record_stream_read(1, 25.0)

        snap = metrics.snapshot(camera_indices=[0, 1])
        self.assertAlmostEqual(snap["cameras"]["0"]["stream"]["decode_ms"], 5.0, places=1)
        self.assertAlmostEqual(snap["cameras"]["1"]["stream"]["decode_ms"], 25.0, places=1)

    def test_global_average_is_pooled_across_cameras_not_mean_of_means(self):
        """inference.stream_decode_avg_ms — same pooled-average semantics as
        yolo_avg_ms/arcface_avg_ms, for the Inference Latency chart."""
        metrics = MetricsCollector()
        metrics.record_stream_read(0, 10.0)
        metrics.record_stream_read(0, 10.0)
        metrics.record_stream_read(1, 100.0)  # one camera, one sample — should pull the pool up

        snap = metrics.snapshot(camera_indices=[0, 1])
        # Pooled: (10+10+100)/3 = 40.0 — NOT mean-of-per-camera-means ((10+100)/2 = 55.0)
        self.assertAlmostEqual(snap["inference"]["stream_decode_avg_ms"], 40.0, places=1)

    def test_global_average_zero_when_nothing_recorded(self):
        metrics = MetricsCollector()
        metrics.record_frame_stages(0, *stage_record())

        snap = metrics.snapshot(camera_indices=[0])
        self.assertEqual(snap["inference"]["stream_decode_avg_ms"], 0.0)


if __name__ == "__main__":
    unittest.main()
