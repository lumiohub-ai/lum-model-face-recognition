"""Unit tests for CeleryCameraProducer's enqueue path.

Covers the two properties that are invisible until production load: the
producer is the only frame-skip gate, and every enqueued frame carries an
expiry so an overloaded worker sheds stale frames instead of accumulating
silent lag.

Run: PYTHONPATH=src python tests/test_camera_producer.py
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

import numpy as np  # noqa: E402


def _frame(h=32, w=32):
    return (np.random.rand(h, w, 3) * 255).astype(np.uint8)


class FakeStream:
    def __init__(self, frames):
        self._frames = list(frames)

    def read(self):
        if not self._frames:
            return False, None
        return True, self._frames.pop(0)


class FakeTask:
    def __init__(self):
        self.calls = []

    def apply_async(self, kwargs=None, **options):
        self.calls.append({"kwargs": kwargs, "options": options})


class FakeSlot:
    def __init__(self):
        self._seq = 0

    def write(self, frame):
        from workers.frame_store import FrameHandle

        self._seq += 1
        return FrameHandle(
            camera_id=1, seq=self._seq, segment=self._seq % 8, instance_id=1,
            height=frame.shape[0], width=frame.shape[1], channels=3,
        )


class _PatchedTask:
    """CeleryCameraProducer imports process_frame_task inside the method, so
    swapping the module attribute is enough to intercept the enqueue."""

    def __init__(self, task):
        self.task = task
        self._saved = None

    def __enter__(self):
        from workers import camera_tasks

        self._saved = camera_tasks.process_frame_task
        camera_tasks.process_frame_task = self.task
        return self.task

    def __exit__(self, *exc):
        from workers import camera_tasks

        camera_tasks.process_frame_task = self._saved
        return False


def _make_producer(frames, detection_interval=1):
    from pipeline.camera_worker import CeleryCameraProducer

    producer = CeleryCameraProducer(
        camera_id=1,
        camera_config={"camera_id": 1},
        stream_handler=FakeStream(frames),
        detection_interval=detection_interval,
    )
    producer._frame_slot = FakeSlot()
    return producer


class EnqueueTests(unittest.TestCase):
    def test_every_enqueued_frame_carries_an_expiry(self):
        """Without this the queue is unbounded: an overloaded worker grows
        latency silently instead of dropping frames the way the bounded
        in-process queues used to."""
        from pipeline.camera_worker import _TASK_EXPIRES_S

        task = FakeTask()
        producer = _make_producer([_frame()])
        with _PatchedTask(task):
            producer._produce_one_frame()

        self.assertEqual(len(task.calls), 1)
        self.assertEqual(task.calls[0]["options"]["expires"], _TASK_EXPIRES_S)

    def test_payload_carries_the_handle_as_a_plain_dict(self):
        """task_serializer='json' cannot encode a FrameHandle dataclass."""
        task = FakeTask()
        producer = _make_producer([_frame()])
        with _PatchedTask(task):
            producer._produce_one_frame()

        kwargs = task.calls[0]["kwargs"]
        self.assertEqual(kwargs["camera_id"], 1)
        self.assertIsInstance(kwargs["frame_handle"], dict)
        self.assertEqual(kwargs["frame_handle"]["seq"], kwargs["frame_num"])

    def test_detection_interval_is_gated_here_and_only_here(self):
        """The task no longer skips on its own counter, so this gate is the
        only one; at interval=2 exactly half the frames must be enqueued."""
        task = FakeTask()
        producer = _make_producer([_frame() for _ in range(6)], detection_interval=2)
        with _PatchedTask(task):
            for _ in range(6):
                producer._produce_one_frame()

        self.assertEqual(len(task.calls), 3)
        self.assertEqual([c["kwargs"]["frame_num"] for c in task.calls], [2, 4, 6])

    def test_a_failed_read_enqueues_nothing(self):
        task = FakeTask()
        producer = _make_producer([])
        with _PatchedTask(task):
            producer._produce_one_frame()

        self.assertEqual(task.calls, [])

    def test_roi_is_read_live_from_camera_config_not_cached(self):
        """LSO-155: engine.reload_camera_configs mutates camera_config in
        place on a same-camera-set reload, rather than rebinding it — a
        cached self.roi captured once at construction would never see that
        update. The producer must read roi from camera_config on every
        frame instead."""
        from pipeline.camera_worker import CeleryCameraProducer

        camera_config = {"camera_id": 1}
        producer = CeleryCameraProducer(
            camera_id=1,
            camera_config=camera_config,
            stream_handler=FakeStream([_frame(64, 64), _frame(64, 64)]),
            detection_interval=1,
        )
        producer._frame_slot = FakeSlot()
        task = FakeTask()

        with _PatchedTask(task):
            producer._produce_one_frame()  # no roi set yet
            self.assertEqual(task.calls[0]["kwargs"]["frame_num"], 1)

            # Simulate reload_camera_configs' in-place mutation.
            camera_config["roi"] = [0, 0, 10, 10]
            producer._produce_one_frame()

        # The second call must have used the new roi — the cropped frame's
        # handle carries the cropped height/width, not the original 64x64.
        second_handle = task.calls[1]["kwargs"]["frame_handle"]
        self.assertEqual(second_handle["height"], 10)
        self.assertEqual(second_handle["width"], 10)


if __name__ == "__main__":
    unittest.main()
