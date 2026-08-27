"""Unit tests for gpu_worker_rpc.py.

Covers the Unix-socket RPC bridge to GPUInferenceWorker: submit_frame+
get_detections collapsed into one detect() round-trip, submit_faces+
get_embeddings collapsed into one embed() round-trip, frame/ROI pixels
crossing via frame_store's shared-memory handles (never through the socket
itself), and the failure-mode degradation to [] / {} that a struggling main
process must produce instead of blocking or fabricating a plausible-looking
result.

Server + client run against a genuinely separate socket per test and, where
producer/consumer resource_tracker interaction matters, as real subprocess
pairs the same way tests/test_frame_store.py verifies frame_store itself —
that module's own tests already cover the shared-memory correctness this
bridge depends on, so these tests focus on the RPC layer on top of it.

Run: PYTHONPATH=src python tests/test_gpu_worker_rpc.py
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

import numpy as np  # noqa: E402

from workers import frame_store  # noqa: E402
from workers.gpu_worker_rpc import (  # noqa: E402
    GpuWorkerRpcClient,
    GpuWorkerRpcServer,
)


def _free_socket_path(tag: str) -> str:
    """A distinct, tag-suffixed path per test - tests run against real Unix
    sockets, and reusing one path across tests racing in the same process
    would let one test's leftover socket answer another's client.

    Hashes `tag` rather than embedding it verbatim: AF_UNIX's sun_path is
    capped at ~108 bytes on Linux, and a descriptive unittest method name
    (this suite's own test_embed_of_an_empty_batch_still_calls_... is 66
    chars alone) combined with a PID suffix and a stable prefix silently
    exceeds that — observed directly as `OSError: AF_UNIX path too long`
    while writing this suite, not a hypothetical.
    """
    import hashlib

    digest = hashlib.sha1(tag.encode()).hexdigest()[:10]
    return f"/tmp/gwrpc_{digest}_{os.getpid()}.sock"


def _random_frame(h, w, c=3):
    return (np.random.rand(h, w, c) * 255).astype(np.uint8)


class FakeGpuWorker:
    """Stands in for GPUInferenceWorker - just the four methods this bridge
    calls, recording every call for assertions, and honouring the real
    contracts the bridge depends on: submit_frame returns a bool drop
    report (False means no reply will ever come), and submit_faces
    returns the seq IT issues from its own counter - deliberately started
    far from RoiBatchSlot's counter (which begins at 1) so any bridge code
    that confuses the two counters fails these tests instead of passing by
    coincidence."""

    FACE_SEQ_START = 4710

    def __init__(self, detections=None, embeddings=None, accept_frames=True):
        self.calls = []
        self._detections = detections if detections is not None else []
        self._embeddings = embeddings if embeddings is not None else {}
        self._accept_frames = accept_frames
        self._face_seq = self.FACE_SEQ_START

    def submit_frame(self, camera_id, frame, frame_num):
        self.calls.append(("submit_frame", camera_id, frame.shape, frame_num))
        return self._accept_frames

    def get_detections(self, camera_id, frame_num):
        self.calls.append(("get_detections", camera_id, frame_num))
        return self._detections

    def submit_faces(self, camera_id, person_rois, track_ids) -> "int | None":
        # Annotated Optional to match the real GPUInferenceWorker.submit_faces
        # contract (None = camera not registered), which the subclass below
        # exercises.
        self.calls.append(("submit_faces", camera_id, len(person_rois), list(track_ids)))
        self._face_seq += 1
        return self._face_seq

    def get_embeddings(self, camera_id, seq):
        self.calls.append(("get_embeddings", camera_id, seq))
        return self._embeddings


class UnregisteredCameraGpuWorker(FakeGpuWorker):
    """submit_faces returns None - GPUInferenceWorker's 'camera no longer
    registered' case."""

    def submit_faces(self, camera_id, person_rois, track_ids):
        self.calls.append(("submit_faces", camera_id, len(person_rois), list(track_ids)))
        return None


class BrokenGpuWorker:
    def submit_frame(self, *a, **kw):
        raise RuntimeError("simulated bug in GPUInferenceWorker")


class DetectPathTests(unittest.TestCase):
    def setUp(self):
        self.socket_path = _free_socket_path(self._testMethodName)
        self.gpu_worker = FakeGpuWorker(
            detections=[{"bbox": [1, 2, 3, 4], "confidence": 0.9, "person_id": 0}]
        )
        self.server = GpuWorkerRpcServer(self.gpu_worker, socket_path=self.socket_path)
        self.server.start()
        self.client = GpuWorkerRpcClient(socket_path=self.socket_path, timeout_s=2.0)
        self.slot = frame_store.CameraFrameSlot(camera_id=1)

    def tearDown(self):
        self.server.stop()
        self.slot.close()
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

    def test_detect_round_trips_and_returns_the_real_detections(self):
        frame = _random_frame(480, 640)
        handle = self.slot.write(frame)
        detections = self.client.detect(camera_id=1, frame_handle=handle, frame_num=handle.seq)
        self.assertEqual(
            detections, [{"bbox": [1, 2, 3, 4], "confidence": 0.9, "person_id": 0}]
        )

    def test_detect_delivers_the_correct_frame_pixels_via_shared_memory(self):
        """The frame that crosses is the one written into the slot, not
        something reconstructed from the handle's metadata alone - proven
        by making the fake worker record the shape it received and checking
        it matches what was actually written, not just any HxWx3 array."""
        frame = _random_frame(233, 401)  # deliberately odd dimensions
        handle = self.slot.write(frame)
        self.client.detect(camera_id=1, frame_handle=handle, frame_num=handle.seq)
        submit_call = next(c for c in self.gpu_worker.calls if c[0] == "submit_frame")
        self.assertEqual(submit_call[2], (233, 401, 3))

    def test_detect_passes_frame_num_through_for_correlation(self):
        """get_detections' correlation-by-frame_num only works if this
        bridge actually forwards the caller's frame_num rather than, say,
        always using handle.seq or a locally generated counter."""
        frame = _random_frame(64, 64)
        handle = self.slot.write(frame)
        self.client.detect(camera_id=1, frame_handle=handle, frame_num=777)
        get_call = next(c for c in self.gpu_worker.calls if c[0] == "get_detections")
        self.assertEqual(get_call[2], 777)

    def test_detect_honours_submit_frames_drop_report(self):
        """submit_frame returning False is the drop report: the frame
        was discarded and NO reply will ever be produced for it, so waiting
        on get_detections would burn its full timeout per dropped frame -
        under exactly the load conditions where drops happen. The bridge
        must skip the get, the same way camera_worker.py does."""
        socket_path = _free_socket_path("drop")
        gpu_worker = FakeGpuWorker(
            detections=[{"should": "never appear"}], accept_frames=False
        )
        server = GpuWorkerRpcServer(gpu_worker, socket_path=socket_path)
        server.start()
        try:
            client = GpuWorkerRpcClient(socket_path=socket_path, timeout_s=2.0)
            slot = frame_store.CameraFrameSlot(camera_id=4)
            try:
                handle = slot.write(_random_frame(64, 64))
                result = client.detect(camera_id=4, frame_handle=handle, frame_num=handle.seq)
                self.assertEqual(result, [])
                self.assertNotIn("get_detections", [c[0] for c in gpu_worker.calls])
            finally:
                slot.close()
        finally:
            server.stop()
            if os.path.exists(socket_path):
                os.unlink(socket_path)


class EmbedPathTests(unittest.TestCase):
    def setUp(self):
        self.socket_path = _free_socket_path(self._testMethodName)
        self.gpu_worker = FakeGpuWorker(
            embeddings={
                10: {"embedding": [0.1] * 8, "det_score": 0.9},
                20: {"embedding": [0.2] * 8, "det_score": 0.85},
            }
        )
        self.server = GpuWorkerRpcServer(self.gpu_worker, socket_path=self.socket_path)
        self.server.start()
        self.client = GpuWorkerRpcClient(socket_path=self.socket_path, timeout_s=2.0)
        self.roi_slot = frame_store.RoiBatchSlot(camera_id=1)

    def tearDown(self):
        self.server.stop()
        self.roi_slot.close()
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

    def test_embed_round_trips_and_returns_the_real_embeddings(self):
        crops = [_random_frame(64, 32), _random_frame(80, 40)]
        handle = self.roi_slot.write(crops, [10, 20])
        embeddings = self.client.embed(camera_id=1, roi_batch_handle=handle)
        self.assertEqual(set(embeddings.keys()), {10, 20})
        self.assertEqual(embeddings[10]["det_score"], 0.9)

    def test_embed_delivers_the_correct_crop_count_and_track_ids(self):
        crops = [_random_frame(64, 32), _random_frame(80, 40), _random_frame(50, 50)]
        handle = self.roi_slot.write(crops, [10, 20, 30])
        self.client.embed(camera_id=1, roi_batch_handle=handle)
        submit_call = next(c for c in self.gpu_worker.calls if c[0] == "submit_faces")
        self.assertEqual(submit_call[2], 3)  # crop count
        self.assertEqual(submit_call[3], [10, 20, 30])  # track_ids, in order

    def test_embed_of_an_empty_batch_still_completes_the_round_trip(self):
        """submit_faces([], []) is a legitimate 'keep synchronised' call
        (camera_worker.py) when recognition is skipped this cycle - the
        bridge must still complete the round-trip, not short-circuit."""
        handle = self.roi_slot.write([], [])
        embeddings = self.client.embed(camera_id=1, roi_batch_handle=handle)
        self.assertEqual(embeddings, {10: {"embedding": [0.1] * 8, "det_score": 0.9},
                                       20: {"embedding": [0.2] * 8, "det_score": 0.85}})
        self.assertIn("get_embeddings", [c[0] for c in self.gpu_worker.calls])

    def test_embed_uses_the_seq_submit_faces_returned_not_the_handles(self):
        """The regression this suite exists to prevent: get_embeddings must
        be given the seq submit_faces RETURNS (the GPU worker's own
        per-camera counter), never RoiBatchHandle.seq, which is
        RoiBatchSlot's independent counter.
        The two coincide only while both processes restart in lockstep; a
        camera-worker restart resets the slot's counter while the GPU
        worker's keeps counting, and _await_response treats the mismatch as
        newer-reply-supersedes - every embed call would return {}
        permanently from then on. An earlier version of this suite asserted
        the WRONG behavior here (that handle.seq was forwarded), which is
        exactly how the bug survived its first round of tests."""
        handle = self.roi_slot.write([_random_frame(20, 20)], [10])
        self.client.embed(camera_id=1, roi_batch_handle=handle)
        get_call = next(c for c in self.gpu_worker.calls if c[0] == "get_embeddings")
        self.assertEqual(get_call[2], FakeGpuWorker.FACE_SEQ_START + 1)
        self.assertNotEqual(get_call[2], handle.seq)

    def test_embed_when_camera_is_unregistered_returns_empty_without_waiting(self):
        """submit_faces returning None means the camera left the set - no
        reply will come, so calling get_embeddings would burn its full
        timeout for nothing."""
        socket_path = _free_socket_path("unreg")
        gpu_worker = UnregisteredCameraGpuWorker()
        server = GpuWorkerRpcServer(gpu_worker, socket_path=socket_path)
        server.start()
        try:
            client = GpuWorkerRpcClient(socket_path=socket_path, timeout_s=2.0)
            roi_slot = frame_store.RoiBatchSlot(camera_id=5)
            try:
                handle = roi_slot.write([_random_frame(20, 20)], [1])
                result = client.embed(camera_id=5, roi_batch_handle=handle)
                self.assertEqual(result, {})
                self.assertNotIn("get_embeddings", [c[0] for c in gpu_worker.calls])
            finally:
                roi_slot.close()
        finally:
            server.stop()
            if os.path.exists(socket_path):
                os.unlink(socket_path)


class FailureModeTests(unittest.TestCase):
    """detect() must degrade to [] and embed() to {} - never raise, never
    fabricate a plausible-looking result - so a struggling main process
    reads as 'nothing this cycle,' the same as a same-process
    get_detections/get_embeddings timeout already produces today."""

    def test_detect_on_server_down_returns_empty_list_not_raise(self):
        client = GpuWorkerRpcClient(socket_path=_free_socket_path("down1"), timeout_s=0.2)
        slot = frame_store.CameraFrameSlot(camera_id=2)
        try:
            handle = slot.write(_random_frame(64, 64))
            fired = []
            client.on_fallback = lambda: fired.append(True)
            result = client.detect(camera_id=2, frame_handle=handle, frame_num=handle.seq)
            self.assertEqual(result, [])
            self.assertEqual(len(fired), 1)
        finally:
            slot.close()

    def test_embed_on_server_down_returns_empty_dict_not_raise(self):
        client = GpuWorkerRpcClient(socket_path=_free_socket_path("down2"), timeout_s=0.2)
        roi_slot = frame_store.RoiBatchSlot(camera_id=2)
        try:
            handle = roi_slot.write([_random_frame(20, 20)], [1])
            fired = []
            client.on_fallback = lambda: fired.append(True)
            result = client.embed(camera_id=2, roi_batch_handle=handle)
            self.assertEqual(result, {})
            self.assertEqual(len(fired), 1)
        finally:
            roi_slot.close()

    def test_detect_when_frame_slot_is_missing_returns_empty_list(self):
        """The camera's frame slot vanished between write and this read
        (camera removed mid-flight) - GpuWorkerRpcServer._dispatch must
        treat this like GPUInferenceWorker.submit_frame's own tolerance for
        an unregistered camera, not raise."""
        socket_path = _free_socket_path("missing_slot")
        gpu_worker = FakeGpuWorker(detections=[{"should": "never appear"}])
        server = GpuWorkerRpcServer(gpu_worker, socket_path=socket_path)
        server.start()
        try:
            client = GpuWorkerRpcClient(socket_path=socket_path, timeout_s=2.0)
            # A handle naming a slot that was never written - simulates the
            # producer having closed it already.
            phantom_handle = frame_store.FrameHandle(
                camera_id=999, seq=1, height=64, width=64, channels=3
            )
            result = client.detect(camera_id=999, frame_handle=phantom_handle, frame_num=1)
            self.assertEqual(result, [])
            self.assertNotIn("submit_frame", [c[0] for c in gpu_worker.calls])
        finally:
            server.stop()
            if os.path.exists(socket_path):
                os.unlink(socket_path)

    def test_bug_inside_the_real_gpu_worker_degrades_instead_of_raising(self):
        socket_path = _free_socket_path("broken")
        server = GpuWorkerRpcServer(BrokenGpuWorker(), socket_path=socket_path)
        server.start()
        try:
            client = GpuWorkerRpcClient(socket_path=socket_path, timeout_s=2.0)
            slot = frame_store.CameraFrameSlot(camera_id=3)
            try:
                handle = slot.write(_random_frame(64, 64))
                result = client.detect(camera_id=3, frame_handle=handle, frame_num=handle.seq)
                self.assertEqual(result, [])
            finally:
                slot.close()
        finally:
            server.stop()
            if os.path.exists(socket_path):
                os.unlink(socket_path)


if __name__ == "__main__":
    unittest.main()
