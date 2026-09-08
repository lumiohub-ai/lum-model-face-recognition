"""Unit tests for workers/yolo_tasks.py's run_detect_batch.

Tests the pure batching logic directly — no Celery Batches machinery, no
broker — with real CameraFrameSlot writes so attach_and_read's actual
gone/torn-frame handling is exercised, a fake detector standing in for the
real YOLO model, and a recording `dispatch` callable standing in for
`send_task("camera.track", ...)`.

Run: PYTHONPATH=src python -m pytest tests/test_yolo_tasks.py
"""

import dataclasses
import os
import sys
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

import numpy as np  # noqa: E402


def _frame(h=32, w=32):
    return (np.random.rand(h, w, 3) * 255).astype(np.uint8)


def _simple_request(camera_id, frame_handle, frame_num, deadline=None, next_queue=None):
    from celery_batches import SimpleRequest

    kwargs = {
        "camera_id": camera_id,
        "frame_handle": dataclasses.asdict(frame_handle),
        "frame_num": frame_num,
    }
    if deadline is not None:
        kwargs["deadline"] = deadline
    if next_queue is not None:
        kwargs["next_queue"] = next_queue

    return SimpleRequest(
        id=f"req-{camera_id}-{frame_num}",
        name="yolo.detect",
        args=(),
        kwargs=kwargs,
        delivery_info={},
        hostname="test-worker",
        ignore_result=True,
        reply_to=None,
        correlation_id=None,
        request_dict={},
    )


class FakeBoxes:
    """Stands in for ultralytics' Results.boxes — enough for
    _parse_yolo_result to iterate: one class-0 (person) detection."""

    def __init__(self, n=1):
        import torch

        self._n = n
        self.cls = torch.zeros(n)
        self.xyxy = torch.tensor([[0.0, 0.0, 10.0, 10.0]] * n)
        self.conf = torch.tensor([0.9] * n)

    def __len__(self):
        return self._n


class FakeResult:
    def __init__(self, n=1):
        self.boxes = FakeBoxes(n)


class FakeDetector:
    """Records every call it receives; returns one FakeResult per input
    frame by default, or raises if configured to."""

    def __init__(self, raise_on_call=False):
        self.confidence_threshold = 0.5
        self.iou_threshold = 0.5
        self.device = "cpu"
        self.calls = []
        self._raise_on_call = raise_on_call

    def model(self, frames, conf, iou, verbose, device):
        self.calls.append(list(frames))
        if self._raise_on_call:
            raise RuntimeError("simulated model failure")
        return [FakeResult() for _ in frames]


class RecordingDispatch:
    def __init__(self):
        self.calls = []

    def __call__(self, kwargs, queue, expires):
        self.calls.append({"kwargs": kwargs, "queue": queue, "expires": expires})


class RunDetectBatchTests(unittest.TestCase):
    def setUp(self):
        from workers import frame_store

        self.slots = {}

    def tearDown(self):
        for slot in self.slots.values():
            slot.close()

    def _write(self, camera_id, frame=None):
        from workers import frame_store

        slot = self.slots.get(camera_id)
        if slot is None:
            slot = frame_store.CameraFrameSlot(camera_id=camera_id)
            self.slots[camera_id] = slot
        return slot.write(frame if frame is not None else _frame())

    def test_one_model_call_serves_every_request_in_the_batch(self):
        from workers.yolo_tasks import run_detect_batch

        h1 = self._write(1)
        h2 = self._write(2)
        requests = [
            _simple_request(1, h1, frame_num=1, deadline=time.time() + 10),
            _simple_request(2, h2, frame_num=1, deadline=time.time() + 10),
        ]
        detector = FakeDetector()
        dispatch = RecordingDispatch()

        run_detect_batch(requests, detector, dispatch=dispatch, now=time.time)

        self.assertEqual(len(detector.calls), 1)
        self.assertEqual(len(detector.calls[0]), 2)
        self.assertEqual(len(dispatch.calls), 2)

    def test_detections_are_forwarded_per_index_aligned_request(self):
        from workers.yolo_tasks import run_detect_batch

        h1 = self._write(1)
        h2 = self._write(2)
        requests = [
            _simple_request(1, h1, frame_num=1, deadline=time.time() + 10),
            _simple_request(2, h2, frame_num=1, deadline=time.time() + 10),
        ]
        detector = FakeDetector()
        dispatch = RecordingDispatch()

        run_detect_batch(requests, detector, dispatch=dispatch, now=time.time)

        by_camera = {c["kwargs"]["camera_id"]: c for c in dispatch.calls}
        self.assertEqual(set(by_camera), {1, 2})
        for cam_id, call in by_camera.items():
            dets = call["kwargs"]["detections"]
            self.assertEqual(len(dets), 1)
            self.assertEqual(dets[0]["bbox"], [0.0, 0.0, 10.0, 10.0])

    def test_a_gone_frame_is_skipped_but_other_requests_still_process(self):
        """attach_and_read returning None (slot vanished, or recycled before
        this task ran) must not block the rest of the batch."""
        from workers.frame_store import FrameHandle
        from workers.yolo_tasks import run_detect_batch

        phantom = FrameHandle(
            camera_id=999, seq=1, segment=1, instance_id=1,
            height=32, width=32, channels=3,
        )
        h2 = self._write(2)
        requests = [
            _simple_request(999, phantom, frame_num=1, deadline=time.time() + 10),
            _simple_request(2, h2, frame_num=1, deadline=time.time() + 10),
        ]
        detector = FakeDetector()
        dispatch = RecordingDispatch()

        run_detect_batch(requests, detector, dispatch=dispatch, now=time.time)

        self.assertEqual(len(detector.calls[0]), 1)  # only the real frame
        self.assertEqual(len(dispatch.calls), 1)
        self.assertEqual(dispatch.calls[0]["kwargs"]["camera_id"], 2)

    def test_a_past_deadline_request_is_skipped_before_the_model_call(self):
        from workers.yolo_tasks import run_detect_batch

        h1 = self._write(1)
        h2 = self._write(2)
        requests = [
            _simple_request(1, h1, frame_num=1, deadline=time.time() - 1),  # expired
            _simple_request(2, h2, frame_num=1, deadline=time.time() + 10),
        ]
        detector = FakeDetector()
        dispatch = RecordingDispatch()

        run_detect_batch(requests, detector, dispatch=dispatch, now=time.time)

        self.assertEqual(len(detector.calls[0]), 1)  # only camera 2's frame
        self.assertEqual(len(dispatch.calls), 1)
        self.assertEqual(dispatch.calls[0]["kwargs"]["camera_id"], 2)

    def test_no_deadline_never_expires(self):
        from workers.yolo_tasks import run_detect_batch

        h1 = self._write(1)
        requests = [_simple_request(1, h1, frame_num=1)]  # no deadline kwarg
        detector = FakeDetector()
        dispatch = RecordingDispatch()

        run_detect_batch(requests, detector, dispatch=dispatch, now=time.time)

        self.assertEqual(len(dispatch.calls), 1)
        self.assertIsNone(dispatch.calls[0]["expires"])

    def test_model_exception_forwards_empty_detections_rather_than_dropping_the_batch(self):
        from workers.yolo_tasks import run_detect_batch

        h1 = self._write(1)
        h2 = self._write(2)
        requests = [
            _simple_request(1, h1, frame_num=1, deadline=time.time() + 10),
            _simple_request(2, h2, frame_num=1, deadline=time.time() + 10),
        ]
        detector = FakeDetector(raise_on_call=True)
        dispatch = RecordingDispatch()

        run_detect_batch(requests, detector, dispatch=dispatch, now=time.time)

        self.assertEqual(len(dispatch.calls), 2)
        for call in dispatch.calls:
            self.assertEqual(call["kwargs"]["detections"], [])

    def test_empty_batch_makes_no_model_call(self):
        from workers.yolo_tasks import run_detect_batch

        detector = FakeDetector()
        dispatch = RecordingDispatch()

        run_detect_batch([], detector, dispatch=dispatch, now=time.time)

        self.assertEqual(detector.calls, [])
        self.assertEqual(dispatch.calls, [])

    def test_expires_forwarded_is_the_remaining_budget_not_a_fresh_window(self):
        from workers.yolo_tasks import run_detect_batch

        h1 = self._write(1)
        deadline = time.time() + 5.0
        requests = [_simple_request(1, h1, frame_num=1, deadline=deadline)]
        detector = FakeDetector()
        dispatch = RecordingDispatch()

        run_detect_batch(requests, detector, dispatch=dispatch, now=time.time)

        self.assertEqual(len(dispatch.calls), 1)
        remaining = dispatch.calls[0]["expires"]
        self.assertGreater(remaining, 0)
        self.assertLessEqual(remaining, 5.0)

    def test_next_queue_defaults_to_slot_queue_when_absent(self):
        # When a request carries no next_queue, the fallback must route through
        # camera_queue_name (the slot queue, LSO-186), never a literal cam.<id>
        # that no worker consumes.
        from workers.yolo_tasks import run_detect_batch
        from workers.celery_app import camera_queue_name

        h1 = self._write(1)
        requests = [_simple_request(1, h1, frame_num=1, deadline=time.time() + 10)]
        detector = FakeDetector()
        dispatch = RecordingDispatch()

        run_detect_batch(requests, detector, dispatch=dispatch, now=time.time)

        self.assertEqual(dispatch.calls[0]["queue"], camera_queue_name(1))

    def test_next_queue_is_honoured_when_present(self):
        from workers.yolo_tasks import run_detect_batch

        h1 = self._write(1)
        requests = [
            _simple_request(1, h1, frame_num=1, deadline=time.time() + 10, next_queue="cam.1")
        ]
        detector = FakeDetector()
        dispatch = RecordingDispatch()

        run_detect_batch(requests, detector, dispatch=dispatch, now=time.time)

        self.assertEqual(dispatch.calls[0]["queue"], "cam.1")


if __name__ == "__main__":
    unittest.main()
