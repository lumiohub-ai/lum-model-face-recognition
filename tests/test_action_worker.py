"""Unit tests for ActionRecognitionWorker — queue, throttle and side effects.

Tests the application-side wrapper in isolation: no Ollama, no GCS, no Celery,
no cameras. The recognizer is a stub, the AsyncLogger is a stub that records
what was queued, and the Celery task is injected into sys.modules so the lazy
import inside _post_activity_to_backend picks up a fake.

Run: PYTHONPATH=src python tests/test_action_worker.py
"""

import os
import sys
import time
import types
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from pipeline.action_worker import ActionRecognitionWorker  # noqa: E402


class StubResult:
    """Stands in for lum_vision.ActionResult."""

    def __init__(self, action="using phone", activity_type="phone_usage"):
        self.action = action
        self.activity_type = activity_type
        self.raw_output = '{"action": "using phone"}'
        self.inference_time = 0.5
        self.metadata = {}


class StubRecognizer:
    """Stands in for lum_vision.ActionRecognizer."""

    def __init__(self, result=None, raises=None, enabled=True, interval=60,
                 times_out=False):
        self.enabled = enabled
        self.check_interval_seconds = interval
        self._result = result
        self._raises = raises
        self._times_out = times_out
        self.total_timeouts = 0
        self.calls = 0

    def recognize(self, image, metadata=None):
        self.calls += 1
        if self._raises is not None:
            raise self._raises
        if self._times_out:
            self.total_timeouts += 1
        return self._result

    def get_stats(self):
        return {"total_inferences": self.calls, "total_timeouts": 0}


class StubAsyncLogger:
    """Records upload requests instead of touching GCS."""

    def __init__(self, accept=True):
        self.accept = accept
        self.uploads = []

    def upload_image(self, data):
        if not self.accept:
            return False
        self.uploads.append(data)
        return True

    def run_callback(self, url):
        """Simulate the GCS worker completing the most recent upload."""
        self.uploads[-1]["callback"](url)


class StubCeleryTask:
    def __init__(self):
        self.calls = []

    def delay(self, **kwargs):
        self.calls.append(kwargs)


def install_fake_celery(task):
    """Point the lazy `from workers.detection_tasks import ...` at a fake."""
    module = types.ModuleType("workers.detection_tasks")
    module.task_record_activity = task
    pkg = sys.modules.get("workers") or types.ModuleType("workers")
    sys.modules["workers"] = pkg
    sys.modules["workers.detection_tasks"] = module
    return module


def build(recognizer=None, async_logger=None, **kwargs):
    return ActionRecognitionWorker(
        recognizer=recognizer or StubRecognizer(StubResult()),
        client_slug="test-slug",
        async_logger=async_logger,
        **kwargs,
    )


def request(image=None, request_id="req-1", **metadata):
    meta = {"user_id": 7, "camera_id": 2, "identity": "Alice", "track_id": 3}
    meta.update(metadata)
    return {
        "image": np.zeros((8, 8, 3), dtype=np.uint8) if image is None else image,
        "request_id": request_id,
        "metadata": meta,
    }


class CallbackLifetimeTests(unittest.TestCase):
    """The callback dict must never retain entries — it closes over crops."""

    def test_failed_inference_pops_the_callback(self):
        worker = build(StubRecognizer(result=None))
        worker.result_callbacks["req-1"] = lambda r: None

        worker._process_inference_request(request())

        self.assertEqual(worker.result_callbacks, {})

    def test_raised_inference_pops_the_callback(self):
        worker = build(StubRecognizer(raises=RuntimeError("ollama is down")))
        worker.result_callbacks["req-1"] = lambda r: None

        with self.assertRaises(RuntimeError):
            worker._process_inference_request(request())

        self.assertEqual(worker.result_callbacks, {})

    def test_unparsed_action_pops_the_callback(self):
        worker = build(StubRecognizer(StubResult(action=None, activity_type="unknown")))
        worker.result_callbacks["req-1"] = lambda r: None

        worker._process_inference_request(request())

        self.assertEqual(worker.result_callbacks, {})

    def test_worker_loop_survives_a_raising_inference(self):
        """One bad item must not kill the thread — it drains the whole queue."""
        recognizer = StubRecognizer(raises=RuntimeError("boom"))
        worker = build(recognizer)
        worker.start_workers()
        self.addCleanup(worker.stop_workers, 2.0)

        for i in range(3):
            worker.recognize_async(
                image=np.zeros((8, 8, 3), dtype=np.uint8),
                request_id=f"req-{i}",
                callback=lambda r: None,
                metadata={"identity": "Alice"},
            )

        deadline = time.time() + 3.0
        while recognizer.calls < 3 and time.time() < deadline:
            time.sleep(0.02)

        self.assertEqual(recognizer.calls, 3, "thread died on the first exception")
        self.assertTrue(worker.workers[0].is_alive())
        self.assertEqual(worker.result_callbacks, {})


class ResultDispatchTests(unittest.TestCase):
    def test_successful_result_invokes_callback(self):
        worker = build()
        received = []
        worker.result_callbacks["req-1"] = received.append
        worker._async_logger = StubAsyncLogger()

        worker._process_inference_request(request())

        self.assertEqual(len(received), 1)
        self.assertEqual(received[0]["action"], "using phone")
        self.assertEqual(received[0]["activity_type"], "phone_usage")

    def test_unparsed_action_is_not_recorded(self):
        task = StubCeleryTask()
        install_fake_celery(task)
        stub_logger = StubAsyncLogger()
        worker = build(
            StubRecognizer(StubResult(action=None, activity_type="unknown")),
            async_logger=stub_logger,
        )

        worker._process_inference_request(request())

        self.assertEqual(stub_logger.uploads, [])
        self.assertEqual(task.calls, [])


class BackendPostTests(unittest.TestCase):
    def setUp(self):
        self.task = StubCeleryTask()
        install_fake_celery(self.task)

    def test_upload_is_queued_and_celery_waits_for_the_url(self):
        stub_logger = StubAsyncLogger()
        worker = build(async_logger=stub_logger)

        worker._process_inference_request(request())

        self.assertEqual(len(stub_logger.uploads), 1)
        self.assertEqual(stub_logger.uploads[0]["folder"], "activity_proofs")
        self.assertEqual(self.task.calls, [], "Celery must wait for the proof URL")

        stub_logger.run_callback("https://gcs/proof.jpg")

        self.assertEqual(len(self.task.calls), 1)
        self.assertEqual(self.task.calls[0]["proof_image_url"], "https://gcs/proof.jpg")

    def test_failed_upload_still_records_the_activity(self):
        stub_logger = StubAsyncLogger()
        worker = build(async_logger=stub_logger)

        worker._process_inference_request(request())
        stub_logger.run_callback(None)

        self.assertEqual(len(self.task.calls), 1)
        self.assertIsNone(self.task.calls[0]["proof_image_url"])

    def test_full_gcs_queue_falls_back_to_recording_without_proof(self):
        worker = build(async_logger=StubAsyncLogger(accept=False))

        worker._process_inference_request(request())

        self.assertEqual(len(self.task.calls), 1)
        self.assertIsNone(self.task.calls[0]["proof_image_url"])

    def test_no_async_logger_still_records_the_activity(self):
        worker = build(async_logger=None)

        worker._process_inference_request(request())

        self.assertEqual(len(self.task.calls), 1)

    def test_user_name_comes_from_the_identity_metadata_key(self):
        worker = build(async_logger=StubAsyncLogger(accept=False))

        worker._process_inference_request(request(identity="Alice"))

        self.assertEqual(self.task.calls[0]["user_name"], "Alice")


class ThrottleTests(unittest.TestCase):
    def test_first_check_is_allowed_and_the_second_is_not(self):
        worker = build()

        self.assertTrue(worker.reserve_check("Alice", now=1000.0))
        self.assertFalse(worker.reserve_check("Alice", now=1030.0))

    def test_check_is_allowed_again_after_the_interval(self):
        worker = build()

        worker.reserve_check("Alice", now=1000.0)

        self.assertTrue(worker.reserve_check("Alice", now=1061.0))

    def test_two_cameras_yield_exactly_one_reservation(self):
        """The whole point of moving the throttle off CameraEngine."""
        worker = build()

        results = [worker.reserve_check("Alice", now=1000.0) for _ in range(2)]

        self.assertEqual(results.count(True), 1)

    def test_distinct_identities_are_independent(self):
        worker = build()

        self.assertTrue(worker.reserve_check("Alice", now=1000.0))
        self.assertTrue(worker.reserve_check("Bob", now=1000.0))

    def test_cancel_restores_eligibility_immediately(self):
        worker = build()
        worker.reserve_check("Alice", now=1000.0)

        worker.cancel_check("Alice")

        self.assertTrue(worker.reserve_check("Alice", now=1001.0))

    def test_stale_entries_are_pruned_above_the_cap(self):
        worker = build()
        for i in range(worker.MAX_TRACKED + 1):
            worker.reserve_check(f"person-{i}", now=1000.0)

        # Far enough ahead that every existing entry is stale.
        worker.reserve_check("trigger", now=1000.0 + worker.check_interval_seconds * 3)

        self.assertLessEqual(len(worker._last_check), worker.MAX_TRACKED)


class QueueTests(unittest.TestCase):
    def test_queue_full_drops_the_request_and_leaves_no_callback(self):
        worker = build(max_queue_size=1)
        worker.inference_queue.put(request())

        queued = worker.recognize_async(
            image=np.zeros((8, 8, 3), dtype=np.uint8),
            request_id="req-2",
            callback=lambda r: None,
        )

        self.assertFalse(queued)
        self.assertEqual(worker.result_callbacks, {})
        self.assertEqual(worker.total_dropped, 1)

    def test_disabled_worker_queues_nothing_and_starts_no_threads(self):
        worker = build(StubRecognizer(StubResult(), enabled=False))

        queued = worker.recognize_async(
            image=np.zeros((8, 8, 3), dtype=np.uint8), request_id="req-1"
        )
        worker.start_workers()

        self.assertFalse(queued)
        self.assertEqual(worker.workers, [])
        self.assertFalse(worker.running)


class LifecycleTests(unittest.TestCase):
    def test_start_and_stop_round_trip(self):
        worker = build()

        worker.start_workers()
        self.assertTrue(worker.running)
        self.assertEqual(len(worker.workers), 1)

        worker.stop_workers(timeout=2.0)

        self.assertFalse(worker.running)
        self.assertEqual(worker.workers, [])
        self.assertEqual(worker.result_callbacks, {})

    def test_stop_is_idempotent(self):
        worker = build()
        worker.start_workers()
        worker.stop_workers(timeout=2.0)

        worker.stop_workers(timeout=2.0)  # must not raise

        self.assertEqual(worker.workers, [])


class StubMetrics:
    def __init__(self):
        self.records = []

    def record_action_inference(self, ms, status, queue_depth=0):
        self.records.append(status)


class MetricsTests(unittest.TestCase):
    def _run(self, recognizer):
        metrics = StubMetrics()
        worker = build(recognizer, async_logger=StubAsyncLogger(accept=False))
        worker.set_metrics_collector(metrics)
        install_fake_celery(StubCeleryTask())
        worker._process_inference_request(request())
        return metrics.records

    def test_success_records_ok(self):
        self.assertEqual(self._run(StubRecognizer(StubResult())), ["ok"])

    def test_timeout_is_distinguished_from_a_hard_error(self):
        """recognize() returns None for both; the model counter tells them apart."""
        self.assertEqual(
            self._run(StubRecognizer(result=None, times_out=True)), ["timeout"]
        )

    def test_hard_failure_records_error(self):
        self.assertEqual(self._run(StubRecognizer(result=None)), ["error"])

    def test_unparsed_action_records_parse_fail(self):
        self.assertEqual(
            self._run(StubRecognizer(StubResult(action=None, activity_type="unknown"))),
            ["parse_fail"],
        )


class StatsTests(unittest.TestCase):
    def test_stats_merge_model_and_queue_counters(self):
        worker = build(async_logger=StubAsyncLogger(accept=False))
        install_fake_celery(StubCeleryTask())
        worker._process_inference_request(request())

        stats = worker.get_stats()

        self.assertEqual(stats["total_inferences"], 1)  # from the recognizer
        self.assertEqual(stats["total_posted"], 1)      # from the worker
        self.assertEqual(stats["queue_depth"], 0)
        self.assertEqual(stats["pending_callbacks"], 0)


# Values shipped in configs/config.yaml. Pinned here so a config edit that
# forgets these tests shows up as a failure rather than silently changing
# which crops reach the VLM.
SHIPPED_GATE = dict(min_crop_height=200, min_crop_width=80, min_crop_area=0)


def crop(h, w):
    return np.zeros((h, w, 3), dtype=np.uint8)


class CropSizeGateTests(unittest.TestCase):
    """The size gate in front of action recognition.

    A crop too small to read posture makes the VLM answer from the background —
    a wrong activity that still costs a full 4B-parameter inference and a GCS
    proof upload. Thresholds came from a labelling study, so they are easy to
    get subtly wrong on a future edit.
    """

    def test_disabled_by_default(self):
        worker = build()
        self.assertFalse(worker.should_skip_small(crop(1, 1)))
        self.assertEqual(worker.total_too_small, 0)

    def test_none_image_is_always_skipped(self):
        # Guards the enabled and disabled cases: there is nothing to infer from.
        self.assertTrue(build().should_skip_small(None))
        self.assertTrue(build(**SHIPPED_GATE).should_skip_small(None))

    def test_each_dimension_gates_independently(self):
        worker = build(min_crop_height=200, min_crop_width=80, min_crop_area=0)
        self.assertTrue(worker.should_skip_small(crop(199, 100)))   # height short
        self.assertTrue(worker.should_skip_small(crop(300, 79)))    # width short
        self.assertFalse(worker.should_skip_small(crop(200, 80)))   # both exactly at

    def test_thresholds_are_inclusive_at_the_boundary(self):
        worker = build(min_crop_height=200, min_crop_width=80, min_crop_area=16000)
        self.assertFalse(worker.should_skip_small(crop(200, 80)))   # 200*80 == 16000
        self.assertTrue(worker.should_skip_small(crop(199, 80)))

    def test_area_gate_is_off_by_default(self):
        # The eval crops are upper-body (median aspect h/w 1.19), so any area
        # threshold tight enough to bite is already implied by min_crop_height.
        # A non-zero value only rejects narrow crops -- min_crop_width's job --
        # and 30000 would have skipped this 336x133 crop the eval set labels
        # readable.
        worker = build(**SHIPPED_GATE)
        self.assertFalse(worker.should_skip_small(crop(336, 133)))

    def test_area_gate_still_works_when_configured(self):
        worker = build(min_crop_height=0, min_crop_width=0, min_crop_area=30000)
        self.assertTrue(worker.should_skip_small(crop(200, 80)))     # 16000
        self.assertFalse(worker.should_skip_small(crop(300, 100)))   # 30000

    def test_counter_increments_once_per_skipped_crop(self):
        worker = build(**SHIPPED_GATE)
        for _ in range(3):
            worker.should_skip_small(crop(10, 10))
        worker.should_skip_small(crop(400, 200))  # passes, must not count
        self.assertEqual(worker.total_too_small, 3)
        self.assertEqual(worker.get_stats()["total_too_small"], 3)

    def test_recognize_async_rejects_a_small_crop_without_queueing(self):
        # camera_engine gates earlier, so on the normal path this safety net
        # never fires — a direct caller must still not reach the queue.
        worker = build(**SHIPPED_GATE)
        self.assertFalse(worker.recognize_async(crop(10, 10), request_id="r1"))
        self.assertEqual(worker.inference_queue.qsize(), 0)
        self.assertEqual(worker.total_too_small, 1)

    def test_recognize_async_accepts_a_large_enough_crop(self):
        worker = build(**SHIPPED_GATE)
        self.assertTrue(worker.recognize_async(crop(400, 200), request_id="r1"))
        self.assertEqual(worker.inference_queue.qsize(), 1)
        self.assertEqual(worker.total_too_small, 0)


if __name__ == "__main__":
    unittest.main()
