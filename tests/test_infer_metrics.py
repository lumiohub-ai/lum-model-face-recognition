"""Worker-side latency and fps reporting across the Celery process split.

The Celery migration left `MetricsCollector.record_yolo_ms` and friends
without callers — detection and embedding moved into their own worker
processes, so the dashboard's latency panels went blank while still
rendering (docs/FOLLOW_UPS.md item 10). `pipeline/infer_metrics.py` bridges
that the same way `decode_metrics.py` already bridged `read_ms`/`decode_ms`.

These tests pin the properties that make the bridge trustworthy rather than
merely non-crashing: an idle stage must not keep reporting its last busy
average, a restarted worker must not emit a negative or garbage number, and
a Redis outage must never propagate into the inference path.

Run: PYTHONPATH=src python tests/test_infer_metrics.py
"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from pipeline.infer_metrics import (  # noqa: E402
    InferLatencyReporter,
    _RateReader,
)


class FakeRedis:
    """SET/GET/scan_iter over a dict. No TTL expiry — the TTL *argument* is
    asserted separately; expiry itself is Redis's behaviour, not ours."""

    def __init__(self):
        self.store = {}
        self.ttls = {}

    def set(self, key, value, ex=None):
        self.store[key] = value
        self.ttls[key] = ex

    def get(self, key):
        return self.store.get(key)

    def scan_iter(self, match=None, count=None):
        prefix = match.rstrip("*")
        return [k for k in list(self.store) if k.startswith(prefix)]


class DeadRedis:
    def set(self, *a, **k):
        raise RuntimeError("redis down")

    def get(self, *a, **k):
        raise RuntimeError("redis down")

    def scan_iter(self, *a, **k):
        raise RuntimeError("redis down")


def _reporter(redis, stage="yolo"):
    # interval_s=0 so every record() publishes: the throttle is asserted on
    # its own below, and leaving it on would make every other test timing-
    # dependent.
    return InferLatencyReporter(stage, redis_client=redis, interval_s=0)


class RateDerivationTests(unittest.TestCase):
    def test_first_read_has_nothing_to_difference_against(self):
        """One cumulative sample is not a rate.

        Dividing lifetime totals instead would report a process-lifetime
        average that drifts ever more slowly toward the truth — the trap
        _BatchStats already avoids by using a window.
        """
        r = FakeRedis()
        _reporter(r).record(batch_ms=100.0, batch_count=1)
        self.assertEqual(_RateReader("yolo", cache_ttl_s=0).read(redis_client=r), {})

    def test_averages_cover_the_interval_between_reads(self):
        """The window is between reads, NOT the worker's lifetime.

        The baseline here is deliberately slow (200ms/batch) and the interval
        fast (50ms/batch), so a lifetime average would land at 80.0 and only
        the interval delta gives 50.0. A baseline of zero would let both
        behaviours produce the same number and prove nothing.
        """
        r = FakeRedis()
        rep, reader = _reporter(r), _RateReader("yolo", cache_ttl_s=0)
        rep.record(batch_ms=200.0, batch_count=1, frame_ms=200.0, frame_count=5)
        reader.read(redis_client=r)  # establish the baseline

        for _ in range(4):  # 4 batches x 50ms over 20 frames
            rep.record(batch_ms=50.0, batch_count=1, frame_ms=50.0, frame_count=5)

        self.assertEqual(
            reader.read(redis_client=r),
            {"batch_avg_ms": 50.0, "frame_avg_ms": 10.0},
        )

    def test_an_idle_stage_reports_no_data_not_its_last_average(self):
        """A stalled worker must not look like a fast one."""
        r = FakeRedis()
        rep, reader = _reporter(r), _RateReader("yolo", cache_ttl_s=0)
        rep.record(batch_ms=200.0, batch_count=1)
        reader.read(redis_client=r)
        rep.record(batch_ms=50.0, batch_count=1)
        self.assertEqual(reader.read(redis_client=r)["batch_avg_ms"], 50.0)

        self.assertEqual(reader.read(redis_client=r), {})

    def test_a_restarted_worker_is_skipped_rather_than_reported_negative(self):
        """Counters reset to 0 on restart, so the delta goes negative."""
        r = FakeRedis()
        rep, reader = _reporter(r), _RateReader("yolo", cache_ttl_s=0)
        rep.record(batch_ms=500.0, batch_count=5)
        reader.read(redis_client=r)

        fresh = _reporter(r)
        fresh._worker_id = rep._worker_id  # same key, counters from zero
        fresh.record(batch_ms=10.0, batch_count=1)

        self.assertEqual(reader.read(redis_client=r), {})


class FleetAggregationTests(unittest.TestCase):
    def test_every_worker_process_publishes_under_its_own_key(self):
        """Per-process keys, not one shared hash: a hash has no per-field
        TTL, so a dead worker's numbers would be averaged in forever."""
        r = FakeRedis()
        a, b = _reporter(r), _reporter(r)
        b._worker_id = "other-host:999"
        a.record(batch_ms=1.0, batch_count=1)
        b.record(batch_ms=1.0, batch_count=1)
        self.assertEqual(len(r.store), 2)

    def test_averages_span_the_whole_fleet(self):
        r = FakeRedis()
        a, b = _reporter(r), _reporter(r)
        b._worker_id = "other-host:999"
        reader = _RateReader("yolo", cache_ttl_s=0)
        # Non-zero baselines, so a lifetime average could not coincide with
        # the interval one (see test_averages_cover_the_interval_between_reads).
        a.record(batch_ms=400.0, batch_count=2)
        b.record(batch_ms=400.0, batch_count=2)
        reader.read(redis_client=r)

        a.record(batch_ms=30.0, batch_count=1)
        b.record(batch_ms=90.0, batch_count=1)

        # (30 + 90) / 2 batches — not one worker's number, and not 30+90.
        self.assertEqual(reader.read(redis_client=r)["batch_avg_ms"], 60.0)

    def test_stages_do_not_bleed_into_each_other(self):
        r = FakeRedis()
        _reporter(r, "yolo").record(batch_ms=10.0, batch_count=1)
        _reporter(r, "face").record(det_ms=10.0, det_count=1)
        self.assertEqual(_RateReader("face", cache_ttl_s=0).read(redis_client=r), {})
        self.assertEqual(
            sorted(k.rsplit(":", 2)[0] for k in r.store),
            ["infer:latency:face", "infer:latency:yolo"],
        )


class DegradationTests(unittest.TestCase):
    def test_publishing_never_raises_into_the_inference_path(self):
        """This runs inside the yolo flush and the face task's finally — a
        metrics failure must not become the batch's result."""
        _reporter(DeadRedis()).record(batch_ms=1.0, batch_count=1)

    def test_reading_a_dead_redis_is_no_data_not_an_error(self):
        self.assertEqual(
            _RateReader("yolo", cache_ttl_s=0).read(redis_client=DeadRedis()), {}
        )

    def test_malformed_payload_is_ignored(self):
        r = FakeRedis()
        r.store["infer:latency:yolo:corrupt"] = "{not json"
        self.assertEqual(_RateReader("yolo", cache_ttl_s=0).read(redis_client=r), {})

    def test_entries_carry_a_ttl_so_dead_workers_expire(self):
        r = FakeRedis()
        _reporter(r).record(batch_ms=1.0, batch_count=1)
        self.assertEqual(set(r.ttls.values()), {15})


class ThrottleTests(unittest.TestCase):
    def test_writes_are_throttled_off_the_hot_path(self):
        """At a 10ms flush interval, publishing per batch would be ~100
        Redis round-trips/sec/worker to move a number sampled every few
        seconds."""
        r = FakeRedis()
        rep = InferLatencyReporter("yolo", redis_client=r, interval_s=60)
        for _ in range(50):
            rep.record(batch_ms=1.0, batch_count=1)

        self.assertEqual(len(r.store), 1)
        # The 49 unpublished batches are accumulated, not dropped — the next
        # publish carries them.
        self.assertEqual(json.loads(list(r.store.values())[0])["batch_count"], 1)
        self.assertEqual(rep._totals["batch_count"], 50)


if __name__ == "__main__":
    unittest.main()
