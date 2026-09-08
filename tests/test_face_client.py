"""Unit tests for workers/face_client.py.

FaceEmbedClient talks to face-worker over Celery, not a socket, so these
tests fake the task's .apply_async()/AsyncResult surface directly rather than
running a real worker — the same "fake the transport, test the client logic"
approach test_gpu_worker_rpc.py takes for the socket RPC it replaces.

Run: PYTHONPATH=src python -m pytest tests/test_face_client.py
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))


def _roi_handle(camera_id=1, seq=1, track_ids=(5, 9)):
    from workers.frame_store import RoiBatchHandle, RoiHandle

    rois = tuple(
        RoiHandle(track_id=tid, offset=i * 100, height=20, width=10, channels=3)
        for i, tid in enumerate(track_ids)
    )
    return RoiBatchHandle(camera_id=camera_id, seq=seq, segment=seq % 8, instance_id=1, rois=rois)


class FakeAsyncResult:
    def __init__(self, value=None, exc=None):
        self._value = value
        self._exc = exc
        self.get_calls = []
        self.forgotten = False

    def get(self, timeout=None, disable_sync_subtasks=True):
        self.get_calls.append({"timeout": timeout, "disable_sync_subtasks": disable_sync_subtasks})
        if self._exc is not None:
            raise self._exc
        return self._value

    def forget(self):
        self.forgotten = True


class EmbeddingsByTrackTests(unittest.TestCase):
    def test_maps_positional_results_to_the_real_track_ids(self):
        from workers.face_client import embeddings_by_track

        handle = _roi_handle(track_ids=(5, 9))
        results = [{"embedding": [0.1]}, {"embedding": [0.2]}]
        mapping = embeddings_by_track(results, handle)
        self.assertEqual(mapping, {5: {"embedding": [0.1]}, 9: {"embedding": [0.2]}})

    def test_length_mismatch_returns_empty_rather_than_misrouting(self):
        """A short or reordered reply must never be silently zipped against
        the wrong track ids - that would hand one person's embedding to
        another person's identity."""
        from workers.face_client import embeddings_by_track

        handle = _roi_handle(track_ids=(5, 9))
        mapping = embeddings_by_track([{"embedding": [0.1]}], handle)
        self.assertEqual(mapping, {})

    def test_empty_batch_round_trips_to_an_empty_mapping(self):
        from workers.face_client import embeddings_by_track

        handle = _roi_handle(track_ids=())
        self.assertEqual(embeddings_by_track([], handle), {})


class FaceEmbedClientTests(unittest.TestCase):
    def _client_with(self, async_result):
        from workers.face_client import FaceEmbedClient

        client = FaceEmbedClient(expires_s=2.0, timeout_s=2.5)
        self._async_result = async_result
        self._apply_async_calls = []

        def fake_apply_async(kwargs, queue, expires):
            self._apply_async_calls.append({"kwargs": kwargs, "queue": queue, "expires": expires})
            return async_result

        patcher = mock.patch(
            "workers.face_tasks.embed_batch_task.apply_async", side_effect=fake_apply_async
        )
        self.addCleanup(patcher.stop)
        patcher.start()
        return client

    def test_embed_round_trips_and_returns_the_real_embeddings(self):
        handle = _roi_handle(track_ids=(5, 9))
        async_result = FakeAsyncResult(
            value=[{"embedding": [0.1]}, {"embedding": [0.2]}]
        )
        client = self._client_with(async_result)

        result = client.embed(camera_id=1, roi_batch_handle=handle)

        self.assertEqual(result, {5: {"embedding": [0.1]}, 9: {"embedding": [0.2]}})

    def test_embed_dispatches_to_the_face_queue_with_the_configured_expires(self):
        handle = _roi_handle()
        client = self._client_with(FakeAsyncResult(value=[{}, {}]))

        client.embed(camera_id=1, roi_batch_handle=handle)

        self.assertEqual(len(self._apply_async_calls), 1)
        call = self._apply_async_calls[0]
        self.assertEqual(call["queue"], "face")
        self.assertEqual(call["expires"], 2.0)
        self.assertEqual(call["kwargs"]["handle"]["camera_id"], handle.camera_id)
        self.assertEqual(call["kwargs"]["handle"]["seq"], handle.seq)

    def test_get_is_called_with_disable_sync_subtasks_false(self):
        """Required to call .get() from inside a Celery task at all - Celery
        otherwise raises RuntimeError to prevent worker deadlock. This is the
        sanctioned way to do it (see docs/LSO67_FOLLOWUP_QUEUE_DESIGN.md's
        "Verified constraints" section)."""
        handle = _roi_handle()
        async_result = FakeAsyncResult(value=[{}])
        client = self._client_with(async_result)

        client.embed(camera_id=1, roi_batch_handle=handle)

        self.assertEqual(len(async_result.get_calls), 1)
        self.assertEqual(async_result.get_calls[0]["disable_sync_subtasks"], False)
        self.assertEqual(async_result.get_calls[0]["timeout"], 2.5)

    def test_result_is_forgotten_after_a_successful_get(self):
        """Mandatory, not optional: face results carry numpy and pickle to
        tens of KiB each. Never forgetting them fills the result backend."""
        handle = _roi_handle()
        async_result = FakeAsyncResult(value=[{}])
        client = self._client_with(async_result)

        client.embed(camera_id=1, roi_batch_handle=handle)

        self.assertTrue(async_result.forgotten)

    def test_timeout_returns_empty_dict_fires_fallback_and_still_forgets(self):
        handle = _roi_handle()
        async_result = FakeAsyncResult(exc=TimeoutError("no reply"))
        client = self._client_with(async_result)
        fallback_calls = []
        client.on_fallback = lambda: fallback_calls.append(1)

        result = client.embed(camera_id=1, roi_batch_handle=handle)

        self.assertEqual(result, {})
        self.assertEqual(len(fallback_calls), 1)
        self.assertTrue(async_result.forgotten)

    def test_unexpected_exception_degrades_the_same_way_as_a_timeout(self):
        handle = _roi_handle()
        async_result = FakeAsyncResult(exc=RuntimeError("face-worker crashed"))
        client = self._client_with(async_result)

        result = client.embed(camera_id=1, roi_batch_handle=handle)

        self.assertEqual(result, {})
        self.assertTrue(async_result.forgotten)

    def test_length_mismatch_from_the_real_call_also_returns_empty(self):
        handle = _roi_handle(track_ids=(5, 9))
        async_result = FakeAsyncResult(value=[{"embedding": [0.1]}])  # only 1, expected 2
        client = self._client_with(async_result)

        result = client.embed(camera_id=1, roi_batch_handle=handle)

        self.assertEqual(result, {})


if __name__ == "__main__":
    unittest.main()
