"""Unit tests for camera_tasks.py (LSO-67, Stage 1).

_CameraContext.__init__ does real construction (DB queries via FaceMatcher/
EntryLogger, AsyncLogger's background threads) that these tests deliberately
never exercise — same reasoning as tests/test_action_worker.py's fake-the-
heavy-imports pattern. Instead, _CameraContext.__new__ is used to build an
instance with every real dependency replaced by a fake, and process_frame()
is tested directly: this is the method with actual logic (the port of
CameraWorker._process_one_frame's Steps 4-9), while __init__ is wiring.

Run: PYTHONPATH=src python tests/test_camera_tasks.py
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

import numpy as np  # noqa: E402


def _random_frame(h, w, c=3):
    return (np.random.rand(h, w, c) * 255).astype(np.uint8)


class FakeCameraEngine:
    """Records every call; returns caller-configured results so tests can
    drive both the recognition-runs and recognition-skipped branches."""

    def __init__(self, active_tracks=None, removed_tracks=None, person_rois=None,
                 events=None):
        self.calls = []
        self._active_tracks = active_tracks if active_tracks is not None else []
        self._removed_tracks = removed_tracks if removed_tracks is not None else []
        self._person_rois = person_rois if person_rois is not None else []
        self._events = events if events is not None else []

    def update_tracking(self, detections, frame, frame_num):
        self.calls.append(("update_tracking", detections, frame.shape, frame_num))
        return self._active_tracks, self._removed_tracks, self._person_rois

    def finalize_identities(self, active_tracks, removed_tracks, embeddings_map, frame, frame_num):
        self.calls.append(("finalize_identities", embeddings_map, frame_num))
        return self._events

    def emit_positions(self, active_tracks):
        self.calls.append(("emit_positions", len(active_tracks)))


class FakeGpuWorkerClient:
    def __init__(self, detections=None, embeddings=None):
        self.calls = []
        self._detections = detections if detections is not None else []
        self._embeddings = embeddings if embeddings is not None else {}

    def detect(self, camera_id, frame_handle, frame_num):
        self.calls.append(("detect", camera_id, frame_num))
        return self._detections

    def embed(self, camera_id, roi_batch_handle):
        self.calls.append(("embed", camera_id, roi_batch_handle.seq))
        return self._embeddings


class FakeAsyncLogger:
    def __init__(self):
        self.logged = []

    def log_entry(self, event):
        self.logged.append(event)


class FakeRoiSlot:
    def __init__(self):
        self.writes = []
        self._seq = 0

    def write(self, rois, track_ids):
        from workers.frame_store import RoiBatchHandle, RoiHandle

        self._seq += 1
        self.writes.append((list(rois), list(track_ids)))
        rois_meta = tuple(
            RoiHandle(track_id=tid, offset=0, height=r.shape[0], width=r.shape[1], channels=3)
            for tid, r in zip(track_ids, rois)
        )
        return RoiBatchHandle(camera_id=1, seq=self._seq, rois=rois_meta)


def _make_context(
    camera_engine=None,
    gpu_worker_client=None,
    async_logger=None,
    roi_slot=None,
    detection_interval=2,
    recognition_interval=3,
):
    """Builds a _CameraContext without running __init__ - see module
    docstring. Every attribute process_frame() touches is set explicitly;
    anything it touches that ISN'T set here will raise AttributeError,
    which is the point - it keeps this test honest about exactly what
    process_frame depends on."""
    from workers.camera_tasks import _CameraContext, _ROI_SLOTS

    ctx = _CameraContext.__new__(_CameraContext)
    ctx.camera_id = 1
    ctx.detection_interval = detection_interval
    ctx.recognition_interval = recognition_interval
    ctx.camera_engine = camera_engine if camera_engine is not None else FakeCameraEngine()
    ctx.gpu_worker_client = (
        gpu_worker_client if gpu_worker_client is not None else FakeGpuWorkerClient()
    )
    ctx.async_logger = async_logger if async_logger is not None else FakeAsyncLogger()
    ctx._frame_num = 0
    ctx._detection_frame_num = 0

    slot = roi_slot if roi_slot is not None else FakeRoiSlot()
    _ROI_SLOTS[1] = slot
    return ctx, slot


class ProcessFrameFrameSkipTests(unittest.TestCase):
    def setUp(self):
        from workers import frame_store

        self.slot = frame_store.CameraFrameSlot(camera_id=1)

    def tearDown(self):
        self.slot.close()
        from workers.camera_tasks import _ROI_SLOTS

        _ROI_SLOTS.pop(1, None)

    def test_frame_below_detection_interval_is_skipped_without_calling_gpu(self):
        """detection_interval=2: the 1st call (_frame_num becomes 1) must
        skip GPU work entirely, matching CameraWorker's Step 3 frame-skip."""
        ctx, _ = _make_context(detection_interval=2)
        handle = self.slot.write(_random_frame(64, 64))
        result = ctx.process_frame(handle, frame_num=handle.seq)
        self.assertIsNone(result)
        self.assertEqual(ctx.gpu_worker_client.calls, [])

    def test_frame_at_detection_interval_boundary_is_processed(self):
        ctx, _ = _make_context(detection_interval=2)
        handle1 = self.slot.write(_random_frame(64, 64))
        ctx.process_frame(handle1, frame_num=handle1.seq)  # frame 1: skipped
        handle2 = self.slot.write(_random_frame(64, 64))
        result = ctx.process_frame(handle2, frame_num=handle2.seq)  # frame 2: processed
        self.assertIsNotNone(result)
        self.assertEqual(result["frame_num"], handle2.seq)

    def test_missing_frame_slot_returns_none_without_advancing_frame_count(self):
        """attach_and_read returning None (camera removed, or this reply is
        for an already-superseded frame) must be a no-op, not a crash - and
        must bail out BEFORE incrementing _frame_num, mirroring
        CameraWorker._process_one_frame's Step 1 (stream_handler.read()
        failing returns before any counter increments)."""
        from workers.frame_store import FrameHandle

        ctx, _ = _make_context(detection_interval=1)
        phantom = FrameHandle(camera_id=999, seq=1, height=64, width=64, channels=3)
        result = ctx.process_frame(phantom, frame_num=1)
        self.assertIsNone(result)
        self.assertEqual(ctx._frame_num, 0)  # never touched — bailed before the increment


class ProcessFrameRecognitionGatingTests(unittest.TestCase):
    def setUp(self):
        from workers import frame_store

        self.slot = frame_store.CameraFrameSlot(camera_id=1)

    def tearDown(self):
        self.slot.close()
        from workers.camera_tasks import _ROI_SLOTS

        _ROI_SLOTS.pop(1, None)

    def test_recognition_runs_and_embeds_the_correct_rois_and_track_ids(self):
        """detection_interval=1, recognition_interval=1: every processed
        frame should run recognition and submit exactly the ROIs/track_ids
        update_tracking returned - not a subset, not reordered."""
        roi_a = _random_frame(20, 10)
        roi_b = _random_frame(30, 15)
        engine = FakeCameraEngine(
            active_tracks=[{"track_id": 5}, {"track_id": 9}],
            person_rois=[(5, roi_a, (0, 0)), (9, roi_b, (0, 0))],
            events=[{"kind": "attendance"}],
        )
        gpu_client = FakeGpuWorkerClient(
            detections=[{"bbox": [0, 0, 1, 1]}],
            embeddings={5: {"embedding": [0.1]}, 9: {"embedding": [0.2]}},
        )
        ctx, roi_slot = _make_context(
            camera_engine=engine, gpu_worker_client=gpu_client,
            detection_interval=1, recognition_interval=1,
        )
        handle = self.slot.write(_random_frame(64, 64))
        result = ctx.process_frame(handle, frame_num=handle.seq)

        self.assertTrue(result["recognition_ran"])
        self.assertEqual(len(roi_slot.writes), 1)
        written_rois, written_track_ids = roi_slot.writes[0]
        self.assertEqual(written_track_ids, [5, 9])
        self.assertTrue(np.array_equal(written_rois[0], roi_a))
        self.assertTrue(np.array_equal(written_rois[1], roi_b))
        self.assertEqual(("embed", 1, 1), gpu_client.calls[-1])

    def test_recognition_skipped_interval_does_not_call_embed(self):
        """recognition_interval=3: only every 3rd detection-frame should
        submit for embedding - the rest must skip the embed RPC entirely,
        not send an empty batch (that workaround was removed by LSO-138 for
        the in-process path; this bridge should not need to reintroduce it,
        since gpu_worker_rpc's detect/embed are independent round-trips,
        not paired queue operations that need a keepalive)."""
        engine = FakeCameraEngine(
            active_tracks=[{"track_id": 1}],
            person_rois=[(1, _random_frame(10, 10), (0, 0))],
        )
        gpu_client = FakeGpuWorkerClient()
        ctx, roi_slot = _make_context(
            camera_engine=engine, gpu_worker_client=gpu_client,
            detection_interval=1, recognition_interval=3,
        )
        for _ in range(2):
            handle = self.slot.write(_random_frame(64, 64))
            result = ctx.process_frame(handle, frame_num=handle.seq)
            self.assertFalse(result["recognition_ran"])
        self.assertEqual(roi_slot.writes, [])
        self.assertNotIn("embed", [c[0] for c in gpu_client.calls])

    def test_recognition_due_but_no_person_rois_does_not_call_embed(self):
        """run_recognition=True with an empty person_rois list must still
        skip the embed call - there is nothing to embed, and calling embed
        with an empty batch is a distinct, deliberate case handled by
        RoiBatchSlot's empty-write path elsewhere, not needed when there
        are simply no active tracks this cycle."""
        engine = FakeCameraEngine(active_tracks=[], person_rois=[])
        gpu_client = FakeGpuWorkerClient()
        ctx, roi_slot = _make_context(
            camera_engine=engine, gpu_worker_client=gpu_client,
            detection_interval=1, recognition_interval=1,
        )
        handle = self.slot.write(_random_frame(64, 64))
        ctx.process_frame(handle, frame_num=handle.seq)
        self.assertEqual(roi_slot.writes, [])
        self.assertNotIn("embed", [c[0] for c in gpu_client.calls])


class ProcessFrameEventLoggingTests(unittest.TestCase):
    def setUp(self):
        from workers import frame_store

        self.slot = frame_store.CameraFrameSlot(camera_id=1)

    def tearDown(self):
        self.slot.close()
        from workers.camera_tasks import _ROI_SLOTS

        _ROI_SLOTS.pop(1, None)

    def test_finalize_identities_events_reach_async_logger(self):
        """Step 8 of the ported flow: every event finalize_identities
        returns must be handed to async_logger.log_entry, non-blocking -
        the port's whole point is to keep this fire-and-forget, not
        introduce a synchronous DB write into the per-frame path."""
        events = [{"kind": "attendance", "user": "alice"}, {"kind": "activity"}]
        engine = FakeCameraEngine(events=events)
        async_logger = FakeAsyncLogger()
        ctx, _ = _make_context(
            camera_engine=engine, async_logger=async_logger,
            detection_interval=1, recognition_interval=1,
        )
        handle = self.slot.write(_random_frame(64, 64))
        ctx.process_frame(handle, frame_num=handle.seq)
        self.assertEqual(async_logger.logged, events)

    def test_emit_positions_is_called_with_active_tracks(self):
        engine = FakeCameraEngine(active_tracks=[{"track_id": 1}, {"track_id": 2}])
        ctx, _ = _make_context(camera_engine=engine, detection_interval=1, recognition_interval=1)
        handle = self.slot.write(_random_frame(64, 64))
        ctx.process_frame(handle, frame_num=handle.seq)
        emit_call = next(c for c in engine.calls if c[0] == "emit_positions")
        self.assertEqual(emit_call[1], 2)


class PersonDetectorStubTests(unittest.TestCase):
    """See camera_tasks._PersonDetectorStub's docstring: CameraEngine/
    PersonTracker must only ever read .confidence_threshold/.device off
    person_detector - this pins that contract from the stub's side."""

    def test_stub_exposes_only_the_two_attributes_camera_engine_reads(self):
        from workers.camera_tasks import _PersonDetectorStub

        stub = _PersonDetectorStub(confidence_threshold=0.45, device="cpu")
        self.assertEqual(stub.confidence_threshold, 0.45)
        self.assertEqual(stub.device, "cpu")

    def test_stub_raises_on_any_method_call_rather_than_silently_no_op(self):
        from workers.camera_tasks import _PersonDetectorStub

        stub = _PersonDetectorStub(confidence_threshold=0.45, device="cpu")
        with self.assertRaises(AttributeError):
            stub.model([])  # a real PersonDetector's inference entry point


if __name__ == "__main__":
    unittest.main()
