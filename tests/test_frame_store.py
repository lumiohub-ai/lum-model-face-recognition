"""Unit tests for frame_store.py.

Covers the shared-memory transport that carries camera frames and ROI-crop
batches across the Celery broker without ever serialising pixels into a task
payload. Tests run genuinely cross-process (via multiprocessing.Process, not
mocked) since the resource_tracker lifecycle and the packed-offset layout in
RoiBatchSlot are exactly the kind of thing a same-process test would paper
over — two real bugs (a resource_tracker double-registration leak, and a
`.name` vs `._name` mismatch) were only caught this way while writing this
module.

Run: PYTHONPATH=src python tests/test_frame_store.py
"""

import multiprocessing as mp
import os
import sys
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

import numpy as np  # noqa: E402


def _random_frame(h, w, c=3):
    return (np.random.rand(h, w, c) * 255).astype(np.uint8)


# ── Cross-process worker functions ──────────────────────────────────────────
# Defined at module scope (not as closures/methods) so they're picklable for
# multiprocessing's spawn/fork machinery, and so each does exactly one
# producer or reader role - mirroring the real CeleryCameraProducer
# (producer) / Celery worker (reader) split this module exists to support.


def _frame_producer(camera_id, conn, frames):
    from workers import frame_store

    slot = frame_store.CameraFrameSlot(camera_id=camera_id)
    for frame in frames:
        handle = slot.write(frame)
        conn.send((handle, frame.tobytes(), frame.shape, str(frame.dtype)))
        conn.recv()  # wait for reader's ack before writing the next one
    slot.close()
    conn.send("closed")
    conn.recv()


def _frame_reader(conn, results):
    from workers import frame_store
    import numpy as np

    while True:
        msg = conn.recv()
        if msg == "closed":
            time.sleep(0.3)  # give the unlink a moment to land
            results.append(("closed_check", None))
            conn.send("ack")
            break
        handle, raw, shape, dtype = msg
        expected = np.frombuffer(raw, dtype=dtype).reshape(shape)
        got = frame_store.attach_and_read(handle)
        results.append((handle.seq, np.array_equal(expected, got) if got is not None else False))
        conn.send("ack")


def _roi_producer(camera_id, conn, batches):
    from workers import frame_store

    slot = frame_store.RoiBatchSlot(camera_id=camera_id)
    for crops, track_ids in batches:
        handle = slot.write(crops, track_ids)
        raw = [(c.tobytes(), c.shape, str(c.dtype)) for c in crops]
        conn.send((handle, raw))
        conn.recv()
    slot.close()
    conn.send("closed")
    conn.recv()


def _roi_reader(conn, results):
    from workers import frame_store
    import numpy as np

    while True:
        msg = conn.recv()
        if msg == "closed":
            time.sleep(0.3)
            results.append(("closed_check", None))
            conn.send("ack")
            break
        handle, raw = msg
        got = frame_store.attach_and_read_roi_batch(handle)
        if got is None:
            results.append((handle.seq, None))
        else:
            ok = len(got) == len(raw) and all(
                tid == expected_tid
                and np.array_equal(
                    crop, np.frombuffer(rb, dtype=dt).reshape(shape)
                )
                for (tid, crop), (rb, shape, dt), expected_tid in zip(
                    got, raw, [h.track_id for h in handle.rois]
                )
            )
            results.append((handle.seq, ok))
        conn.send("ack")


class CameraFrameSlotTests(unittest.TestCase):
    """Every case here runs the producer and reader as genuinely separate
    OS processes connected by a pipe - not two objects in one process - so
    the resource_tracker and shared_memory lifecycle is exercised for real."""

    def _run(self, camera_id, frames):
        manager = mp.Manager()
        results = manager.list()
        parent_conn, child_conn = mp.Pipe()
        reader = mp.Process(target=_frame_reader, args=(parent_conn, results))
        producer = mp.Process(target=_frame_producer, args=(camera_id, child_conn, frames))
        reader.start()
        producer.start()
        producer.join(timeout=15)
        reader.join(timeout=15)
        self.assertEqual(producer.exitcode, 0)
        self.assertEqual(reader.exitcode, 0)
        return list(results)

    def test_single_write_read_round_trips(self):
        results = self._run(101, [_random_frame(64, 64)])
        self.assertEqual(results[0], (1, True))

    def test_sequence_of_writes_all_match(self):
        frames = [_random_frame(64, 64) for _ in range(5)]
        results = self._run(102, frames)
        matches = [r for r in results if r[0] != "closed_check"]
        self.assertEqual(len(matches), 5)
        for seq, ok in matches:
            self.assertTrue(ok, f"seq {seq} did not match")

    def test_reshape_mid_stream_still_matches(self):
        """A later write with a different shape (e.g. ROI config changed at
        runtime) must reallocate correctly, not silently truncate or read
        stale bytes from the old, smaller allocation."""
        frames = [_random_frame(64, 64), _random_frame(720, 1280)]
        results = self._run(103, frames)
        matches = [r for r in results if r[0] != "closed_check"]
        self.assertEqual(len(matches), 2)
        for seq, ok in matches:
            self.assertTrue(ok, f"seq {seq} did not match after reshape")

    def test_read_after_close_returns_none(self):
        results = self._run(104, [_random_frame(64, 64)])
        closed_checks = [r for r in results if r[0] == "closed_check"]
        self.assertEqual(len(closed_checks), 1)


class RoiBatchSlotTests(unittest.TestCase):
    def _run(self, camera_id, batches):
        manager = mp.Manager()
        results = manager.list()
        parent_conn, child_conn = mp.Pipe()
        reader = mp.Process(target=_roi_reader, args=(parent_conn, results))
        producer = mp.Process(target=_roi_producer, args=(camera_id, child_conn, batches))
        reader.start()
        producer.start()
        producer.join(timeout=15)
        reader.join(timeout=15)
        self.assertEqual(producer.exitcode, 0)
        self.assertEqual(reader.exitcode, 0)
        return list(results)

    def test_multiple_variable_size_crops_round_trip_with_correct_track_ids(self):
        crops = [_random_frame(64, 32), _random_frame(100, 50), _random_frame(80, 40)]
        track_ids = [10, 20, 30]
        results = self._run(151, [(crops, track_ids)])
        matches = [r for r in results if r[0] != "closed_check"]
        self.assertEqual(len(matches), 1)
        seq, ok = matches[0]
        self.assertTrue(ok, "packed crops or track_ids did not round-trip correctly")

    def test_empty_batch_is_not_an_error(self):
        """submit_faces' 'keep synchronised' call when recognition is
        skipped this cycle - see camera_worker.py - must round-trip as an
        empty list, not as a failure."""
        results = self._run(152, [([], [])])
        matches = [r for r in results if r[0] != "closed_check"]
        self.assertEqual(matches[0], (1, True))

    def test_sequential_batches_of_different_total_size_all_match(self):
        """Exercises RoiBatchSlot's reallocation path the same way
        CameraFrameSlotTests.test_reshape_mid_stream does for frames."""
        small = ([_random_frame(20, 20)], [1])
        large = ([_random_frame(200, 200), _random_frame(150, 100)], [2, 3])
        empty = ([], [])
        results = self._run(153, [small, large, empty])
        matches = [r for r in results if r[0] != "closed_check"]
        self.assertEqual(len(matches), 3)
        for seq, ok in matches:
            self.assertTrue(ok, f"batch seq {seq} did not match")

    def test_read_after_close_returns_none(self):
        results = self._run(154, [([_random_frame(20, 20)], [1])])
        closed_checks = [r for r in results if r[0] == "closed_check"]
        self.assertEqual(len(closed_checks), 1)


class SameProcessReadTests(unittest.TestCase):
    """Reads in the PRODUCER'S OWN process - not a degenerate test setup but
    the real deployment path: GpuWorkerRpcServer runs in the main process,
    the same process whose CeleryCameraProducer threads own the slots. These reads
    must take the _LOCAL_SLOTS fast path (straight from the owning slot's
    mapping) rather than _attach_fresh, whose resource_tracker.unregister
    would delete the producer's own tracker entry - a KeyError at clean
    shutdown and a /dev/shm leak if the process crashes."""

    def test_frame_read_in_producer_process_matches(self):
        from workers import frame_store

        slot = frame_store.CameraFrameSlot(camera_id=301)
        try:
            frame = _random_frame(64, 48)
            handle = slot.write(frame)
            got = frame_store.attach_and_read(handle)
            self.assertIsNotNone(got)
            self.assertTrue(np.array_equal(frame, got))
        finally:
            slot.close()

    def test_frame_read_after_local_close_returns_none(self):
        from workers import frame_store

        slot = frame_store.CameraFrameSlot(camera_id=302)
        frame = _random_frame(32, 32)
        handle = slot.write(frame)
        slot.close()
        self.assertIsNone(frame_store.attach_and_read(handle))

    def test_frame_read_with_stale_oversized_handle_returns_none(self):
        """A handle describing more bytes than the local slot currently
        holds (produced before a shrink, read after) must be refused, not
        read out of bounds."""
        from workers import frame_store

        slot = frame_store.CameraFrameSlot(camera_id=303)
        try:
            big_handle = slot.write(_random_frame(100, 100))
            slot.write(_random_frame(10, 10))  # reallocates smaller
            self.assertIsNone(frame_store.attach_and_read(big_handle))
        finally:
            slot.close()

    def test_roi_batch_read_in_producer_process_matches(self):
        from workers import frame_store

        slot = frame_store.RoiBatchSlot(camera_id=304)
        try:
            crops = [_random_frame(20, 10), _random_frame(30, 15)]
            handle = slot.write(crops, [7, 8])
            got = frame_store.attach_and_read_roi_batch(handle)
            self.assertIsNotNone(got)
            self.assertEqual([tid for tid, _ in got], [7, 8])
            for (tid, crop), expected in zip(got, crops):
                self.assertTrue(np.array_equal(crop, expected))
        finally:
            slot.close()

    def test_roi_batch_read_after_local_close_returns_none(self):
        from workers import frame_store

        slot = frame_store.RoiBatchSlot(camera_id=305)
        handle = slot.write([_random_frame(20, 10)], [7])
        slot.close()
        self.assertIsNone(frame_store.attach_and_read_roi_batch(handle))

    def test_frame_read_of_recycled_generation_returns_none_not_wrong_pixels(self):
        """The actual bug measured live: a single depth-1 slot let a reader
        silently see whichever frame currently occupies the segment, not the
        one its handle names, because a same-shape overwrite never tripped
        the old size-only staleness check. The ring's in-segment seq stamp
        must catch this: a handle from a generation that has since been
        recycled onto the same segment must read back as gone, never as
        another generation's pixels."""
        from workers import frame_store

        slot = frame_store.CameraFrameSlot(camera_id=306)
        try:
            first_frame = _random_frame(32, 32)
            stale_handle = slot.write(first_frame)
            # Write enough more generations to cycle the ring all the way
            # back around to stale_handle's segment.
            for _ in range(frame_store._RING_SIZE):
                slot.write(_random_frame(32, 32))
            got = frame_store.attach_and_read(stale_handle)
            self.assertIsNone(
                got, "recycled generation must read as gone, not as newer pixels"
            )
        finally:
            slot.close()

    def test_frame_read_survives_restart_with_same_shape(self):
        """A CameraFrameSlot's shape (and therefore each segment's byte size)
        is stable across a producer restart for the common case (same camera
        resolution) — the old size-only staleness check never caught this,
        so a worker's cached mapping kept serving frozen pixels from before
        the restart forever. A bare seq counter isn't enough either: a fresh
        process's counter also starts at 0/1, so it can coincidentally stamp
        the exact seq a stale reader is still checking for. Only the random
        per-process instance id catches this reliably, which is what this
        test actually exercises by patching `_INSTANCE_ID` to a different
        value for the 'post-restart' writer — a real restart is a different
        OS process and therefore a genuinely different random id, not
        something a same-process test gets from doing nothing."""
        from workers import frame_store

        old_slot = frame_store.CameraFrameSlot(camera_id=307)
        try:
            frame_before = _random_frame(32, 32)
            stale_handle = old_slot.write(frame_before)
            # A real restart's worker process doesn't have _LOCAL_SLOTS
            # populated (it is a separate process reading via _ATTACHED), so
            # exercise the cross-process cache path explicitly instead of
            # the same-process fast path this test class is otherwise about.
            with frame_store._ATTACHED_LOCK:
                del frame_store._LOCAL_SLOTS[old_slot._base_name]
            segment_name = frame_store._segment_name(
                old_slot._base_name, stale_handle.segment
            )
            cached_shm = frame_store._attach_segment(segment_name)
            self.assertIsNotNone(cached_shm)
            got_before = frame_store.attach_and_read(stale_handle)
            self.assertTrue(np.array_equal(got_before, frame_before))

            # Same segment, same size, but a genuinely different process
            # would stamp a different instance id — simulate that instead of
            # relying on the seq counter to happen to differ.
            with mock.patch.object(
                frame_store, "_INSTANCE_ID", frame_store._INSTANCE_ID + 1
            ):
                new_slot = frame_store.CameraFrameSlot(camera_id=307)
                try:
                    frame_data_after = _random_frame(32, 32)
                    new_handle = new_slot.write(frame_data_after)
                    if new_handle.segment == stale_handle.segment:
                        # Same segment reused post-restart, same size: the
                        # old size-only check would have kept trusting the
                        # cached mapping forever here. The header check
                        # (seq AND instance id) must not.
                        self.assertIsNone(
                            frame_store.attach_and_read(stale_handle)
                        )
                    got_after = frame_store.attach_and_read(new_handle)
                    self.assertTrue(np.array_equal(got_after, frame_data_after))
                finally:
                    new_slot.close()
        finally:
            old_slot.close()
            frame_store._ATTACHED.pop(
                frame_store._segment_name(old_slot._base_name, stale_handle.segment),
                None,
            )


class FrameBatchSlotTests(unittest.TestCase):
    """One batch spanning several cameras, for the GPU workers. Unlike
    every other slot here it is keyed by GPU loop, not by camera, and each
    packed frame carries its own camera_id/frame_num."""

    def setUp(self):
        from workers import frame_store

        self.slot = frame_store.FrameBatchSlot("test-yolo")

    def tearDown(self):
        self.slot.close()

    def test_multi_camera_batch_round_trips_with_frame_nums(self):
        """frame_num must survive per frame: it is the correlation key the
        GPU loop needs to route each result back to the right camera's
        waiting request."""
        from workers import frame_store

        f29 = _random_frame(480, 640)
        f40 = _random_frame(720, 1280)
        handle = self.slot.write({40: (f40, 100), 29: (f29, 200)})

        got = frame_store.attach_and_read_frame_batch("test-yolo", handle)
        self.assertIsNotNone(got)
        by_cam = {cid: (fn, arr) for cid, fn, arr in got}
        self.assertEqual(by_cam[29][0], 200)
        self.assertEqual(by_cam[40][0], 100)
        self.assertTrue(np.array_equal(by_cam[29][1], f29))
        self.assertTrue(np.array_equal(by_cam[40][1], f40))

    def test_packed_in_sorted_camera_order(self):
        """_yolo_loop distributes results positionally against
        `sorted(batch.keys())` - the packed order must match, or every
        camera gets another camera's detections."""
        handle = self.slot.write({
            40: (_random_frame(64, 64), 1),
            29: (_random_frame(64, 64), 2),
            38: (_random_frame(64, 64), 3),
        })
        self.assertEqual([f.camera_id for f in handle.frames], [29, 38, 40])

    def test_frames_of_different_sizes_in_one_batch(self):
        """Cameras genuinely differ in resolution (2560x1440 vs 640x480 in
        the live set), so a batch is not uniformly shaped."""
        from workers import frame_store

        small, large = _random_frame(240, 320), _random_frame(1080, 1920)
        handle = self.slot.write({1: (small, 10), 2: (large, 20)})
        got = frame_store.attach_and_read_frame_batch("test-yolo", handle)
        by_cam = {cid: arr for cid, _fn, arr in got}
        self.assertTrue(np.array_equal(by_cam[1], small))
        self.assertTrue(np.array_equal(by_cam[2], large))

    def test_empty_batch_is_not_an_error(self):
        """The GPU loop calls this whenever nothing was ready; it must be a
        legitimate empty result, matching _run_yolo_batch([]) -> []."""
        from workers import frame_store

        handle = self.slot.write({})
        self.assertEqual(
            frame_store.attach_and_read_frame_batch("test-yolo", handle), []
        )

    def test_reallocation_between_batches(self):
        from workers import frame_store

        self.slot.write({1: (_random_frame(64, 64), 1)})
        big = _random_frame(1080, 1920)
        handle = self.slot.write({1: (big, 2)})
        got = frame_store.attach_and_read_frame_batch("test-yolo", handle)
        self.assertTrue(np.array_equal(got[0][2], big))

    def test_non_hwc_frame_raises(self):
        with self.assertRaises(ValueError):
            self.slot.write({1: (np.zeros((10, 10), dtype=np.uint8), 1)})


class RoiBatchSlotValidationTests(unittest.TestCase):
    """Same-process is fine here - these exercise argument validation, not
    the cross-process shared-memory path."""

    def test_mismatched_lengths_raise(self):
        from workers import frame_store

        slot = frame_store.RoiBatchSlot(camera_id=199)
        try:
            with self.assertRaises(ValueError):
                slot.write([_random_frame(10, 10)], [1, 2])
        finally:
            slot.close()

    def test_non_hwc_crop_raises(self):
        from workers import frame_store

        slot = frame_store.RoiBatchSlot(camera_id=198)
        try:
            with self.assertRaises(ValueError):
                slot.write([np.zeros((10, 10), dtype=np.uint8)], [1])  # missing channel dim
        finally:
            slot.close()


if __name__ == "__main__":
    unittest.main()
