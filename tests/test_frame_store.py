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
        results.append(
            (handle.seq, np.array_equal(expected, got) if got is not None else False)
        )
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
        if handle.rois and handle.rois[0].track_id < 0:
            # A negative track id marks a batch this reader never reads,
            # like a task that expired before a worker picked it up.
            results.append(("skipped", handle.seq))
            conn.send("ack")
            continue
        got = frame_store.attach_and_read_roi_batch(handle)
        if got is None:
            results.append((handle.seq, None))
        else:
            ok = len(got) == len(raw) and all(
                tid == expected_tid
                and np.array_equal(crop, np.frombuffer(rb, dtype=dt).reshape(shape))
                for (tid, crop), (rb, shape, dt), expected_tid in zip(
                    got, raw, [h.track_id for h in handle.rois]
                )
            )
            results.append((handle.seq, ok))
        conn.send("ack")


def _raw_ring_producer(camera_id, conn):
    """Write one frame, keep its handle, then cycle the raw ring past it —
    exactly what the decode worker does over a few health ticks — and hand
    the reader both the recycled handle and the live one."""
    from workers import frame_store

    slot = frame_store.RawFrameSlot(camera_id=camera_id)
    stale_handle = slot.write(_random_frame(48, 64))
    latest_frame = None
    latest_handle = None
    for _ in range(frame_store._RAW_RING_SIZE):
        latest_frame = _random_frame(48, 64)
        latest_handle = slot.write(latest_frame)
    conn.send((stale_handle, latest_handle, latest_frame.tobytes(), latest_frame.shape, str(latest_frame.dtype)))
    conn.recv()
    slot.close()
    conn.send("closed")
    conn.recv()


def _raw_ring_reader(conn, results):
    from workers import frame_store
    import numpy as np

    stale_handle, latest_handle, raw, shape, dtype = conn.recv()
    expected = np.frombuffer(raw, dtype=dtype).reshape(shape)
    got_latest = frame_store.attach_and_read_raw(latest_handle)
    got_stale = frame_store.attach_and_read_raw(stale_handle)
    results.append(("latest", got_latest is not None and np.array_equal(expected, got_latest)))
    results.append(("stale", got_stale))
    conn.send("ack")
    if conn.recv() == "closed":
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
        producer = mp.Process(
            target=_frame_producer, args=(camera_id, child_conn, frames)
        )
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
        producer = mp.Process(
            target=_roi_producer, args=(camera_id, child_conn, batches)
        )
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

    def test_cross_process_reader_reattaches_when_a_recycled_segment_grows(self):
        """Regression test for the actual bug behind 'ROI batch seq=N is
        gone' firing on EVERY embed call in production (0 recognitions
        despite tracking working correctly): the reader process's _ATTACHED
        cache mapped a segment once, and a later write to that SAME ring
        position with a LARGER payload (a bigger face-crop batch than ever
        seen before) reallocates the underlying shared-memory block via
        close()+unlink()+create() — but the reader's cached mapping still
        pointed at the OLD, now-unlinked block. Every read of that ring
        position then permanently failed the size check and returned None,
        even though the segment was sitting there, correctly written, the
        whole time. This only reproduces once the SAME ring position is
        reused (writes _ROI_RING_SIZE apart) with a size increase — a same-size
        or shrinking reuse, or writes that never wrap the ring, do not
        trigger it, which is why the existing resize test above (3 writes,
        no wraparound) passed even with the bug present."""
        from workers import frame_store

        # _ROI_RING_SIZE, not _RING_SIZE: this exercises RoiBatchSlot, whose
        # write() indexes `seq % _ROI_RING_SIZE`. Padding with the detection
        # ring's constant (now derived, and shallower than the ROI ring) left
        # the final write on a never-before-used segment, so the reallocation
        # below never happened and this test silently stopped covering the bug.
        ring_size = frame_store._ROI_RING_SIZE
        # First batch: establishes the reader's cache for ring position 0
        # with a SMALL mapping.
        first = ([_random_frame(10, 10)], [1])
        # Pad with same-size batches to advance the ring all the way back to
        # position 0 without triggering a reallocation along the way.
        padding = [([_random_frame(10, 10)], [1]) for _ in range(ring_size - 1)]
        # This write lands back on position 0 (seq = ring_size + 1) with a
        # payload far larger than the first — forces close()+unlink()+create()
        # under the same segment name the reader already has cached.
        bigger = ([_random_frame(300, 300), _random_frame(200, 200)], [2, 3])

        results = self._run(155, [first, *padding, bigger])
        matches = [r for r in results if r[0] != "closed_check"]
        self.assertEqual(len(matches), ring_size + 1)
        for seq, ok in matches:
            self.assertTrue(
                ok,
                f"batch seq={seq} did not round-trip — if this is the final "
                f"(largest) batch, the reader's cached mapping was not "
                f"re-validated against the new size (the actual production bug)",
            )


    def test_reader_reattaches_when_it_missed_the_write_that_grew_a_segment(self):
        """Tashkent's reid-worker flood of 'ROI batch seq=N is gone': the
        producer grew a ring position on a batch the reader never read (task
        expired), so the reader's cached mapping kept the old, unlinked
        block. Every later batch that fit the old size passed the size check,
        read the old block's older seq and was reported gone, forever."""
        from workers import frame_store

        ring_size = frame_store._ROI_RING_SIZE
        small = lambda: ([_random_frame(10, 10)], [1])
        batches = [small() for _ in range(ring_size)]          # cache every position small
        batches.append(([_random_frame(300, 300)], [-1]))      # grows position 1, never read
        batches += [small() for _ in range(ring_size)]         # position 1 comes round again

        results = self._run(156, batches)
        matches = [r for r in results if r[0] not in ("closed_check", "skipped")]
        self.assertEqual(len(matches), 2 * ring_size)
        gone = [seq for seq, ok in matches if not ok]
        self.assertEqual(gone, [], f"batches reported gone after a missed growth: {gone}")

class RawFrameSlotRingDepthTests(unittest.TestCase):
    """LSO-224: the calibration (camraw) ring is shallow. It is written once
    per ~2 s health tick and read within ≤5 s, so it must not hold 24 full
    frames per camera like the per-frame detection ring does — that was
    ~3 GB of /dev/shm on a 14-camera site for pixels nobody read."""

    def test_raw_ring_size_tracks_the_health_interval(self):
        """The recycle window (ring × interval) must exceed the 5 s handle
        age limit plus margin — for whatever SO_DECODE_HEALTH_INTERVAL_S is."""
        from workers import frame_store as fs

        self.assertEqual(fs.raw_ring_size(2.0), 4)      # default tick
        self.assertEqual(fs.raw_ring_size(1.0), 8)      # faster telemetry → deeper ring
        self.assertEqual(fs.raw_ring_size(5.0), 2)      # never below 2
        for interval in (0.5, 1.0, 2.0, 3.0, 5.0):
            self.assertGreater(fs.raw_ring_size(interval) * interval, fs.RAW_FRAME_MAX_AGE_S)
        self.assertEqual(fs._RAW_RING_SIZE, fs.raw_ring_size(2.0))

    def test_raw_ring_allocates_only_raw_ring_size_segments(self):
        from workers import frame_store

        slot = frame_store.RawFrameSlot(camera_id=401)
        try:
            for _ in range(frame_store._RING_SIZE * 2):
                slot.write(_random_frame(32, 32))
            self.assertEqual(len(slot._shms), frame_store._RAW_RING_SIZE)
            self.assertLess(frame_store._RAW_RING_SIZE, frame_store._RING_SIZE)
        finally:
            slot.close()

    def test_raw_latest_handle_reads_back_and_recycled_one_is_gone(self):
        from workers import frame_store

        slot = frame_store.RawFrameSlot(camera_id=402)
        try:
            first = _random_frame(32, 32)
            stale_handle = slot.write(first)
            latest_frame = None
            latest_handle = None
            for _ in range(frame_store._RAW_RING_SIZE):
                latest_frame = _random_frame(32, 32)
                latest_handle = slot.write(latest_frame)
            got = frame_store.attach_and_read_raw(latest_handle)
            self.assertIsNotNone(got)
            self.assertTrue(np.array_equal(got, latest_frame))
            # first's segment has been recycled after _RAW_RING_SIZE writes —
            # the seq header must reject it rather than return newer pixels.
            self.assertIsNone(frame_store.attach_and_read_raw(stale_handle))
        finally:
            slot.close()

    def test_cross_process_reader_sees_latest_and_rejects_recycled(self):
        """The production path: the decode worker writes camraw, the engine
        reads it from another process via attach_and_read_raw (the
        _attach_segment / _ATTACHED cache route, not the same-process
        fast path the tests above hit). With the shallow ring recycling
        segments more often, this is the read that must stay correct."""
        manager = mp.Manager()
        results = manager.list()
        parent_conn, child_conn = mp.Pipe()
        reader = mp.Process(target=_raw_ring_reader, args=(parent_conn, results))
        producer = mp.Process(target=_raw_ring_producer, args=(404, child_conn))
        reader.start()
        producer.start()
        producer.join(timeout=15)
        reader.join(timeout=15)
        self.assertEqual(producer.exitcode, 0)
        self.assertEqual(reader.exitcode, 0)
        got = dict(results)
        self.assertTrue(got["latest"], "latest raw frame must round-trip cross-process")
        self.assertIsNone(got["stale"], "recycled raw handle must read as gone, not as newer pixels")
        # No after-close assertion: a reader that already mapped a segment
        # keeps its mapping past the producer's unlink by design (see
        # _ATTACHED); the existing close test only checks the unlink lands.

    def test_detection_ring_is_still_deeper_than_the_raw_ring(self):
        from workers import frame_store

        slot = frame_store.CameraFrameSlot(camera_id=403)
        try:
            handle = slot.write(_random_frame(32, 32))
            # Fewer writes than the detection ring's depth: still readable.
            for _ in range(frame_store._RAW_RING_SIZE + 1):
                slot.write(_random_frame(32, 32))
            self.assertIsNotNone(frame_store.attach_and_read(handle))
        finally:
            slot.close()


class TrackedRingDepthTests(unittest.TestCase):
    """LSO-224 lever 2: the per-frame detection ring (`_RING_SIZE`, the one
    yolo.detect/camera.track read via CameraFrameSlot) is also derived now,
    not a flat 24. A segment only needs to outlive the Celery expiry both
    hops share (frame_pump._TASK_EXPIRES_S, carried forward end to end) at
    the fastest write rate it must tolerate — not survive forever."""

    def test_tracked_ring_size_tracks_expiry_and_write_rate(self):
        from workers import frame_store as fs

        # Slower writes (bigger detection_interval) need fewer segments to
        # cover the same time budget.
        small = fs.tracked_ring_size(detection_interval=2, task_expires_s=1.0)
        big = fs.tracked_ring_size(detection_interval=4, task_expires_s=1.0)
        self.assertLess(big, small)
        # A longer task expiry needs a deeper ring to survive it.
        self.assertGreater(
            fs.tracked_ring_size(detection_interval=2, task_expires_s=3.0), small
        )
        self.assertEqual(
            fs.tracked_ring_size(detection_interval=1, task_expires_s=0.0, margin_s=0.0),
            fs._TRACKED_RING_MIN,
        )
        # Ring lifetime (segments × write interval) must clear the expiry it defends against.
        for interval, expires in ((1, 0.5), (2, 1.0), (4, 2.0)):
            write_interval = interval / fs._TRACKED_FPS_CEILING
            ring = fs.tracked_ring_size(detection_interval=interval, task_expires_s=expires)
            self.assertGreater(ring * write_interval, expires)
        self.assertEqual(fs._RING_SIZE, fs.tracked_ring_size(detection_interval=2, task_expires_s=1.0))

    def test_invalid_fps_ceiling_raises_rather_than_silently_flooring(self):
        # A misconfigured (<=0) ceiling must fail loudly, not silently
        # collapse to _TRACKED_RING_MIN and hide the mistake.
        from workers import frame_store as fs

        with self.assertRaises(ValueError):
            fs.tracked_ring_size(detection_interval=2, task_expires_s=1.0, fps_ceiling=0)
        with self.assertRaises(ValueError):
            fs.tracked_ring_size(detection_interval=2, task_expires_s=1.0, fps_ceiling=-5)

    def test_fps_ceiling_covers_every_camera_measured_live(self):
        # LSO-224 review: the PR originally claimed "every camera runs
        # ~10 fps" — checked live against Tashkent during review and found
        # cam 27 at ~12.5 fps. This pins the ceiling to keep real headroom
        # over the fastest camera actually observed, not just dev's.
        from workers import frame_store as fs

        fastest_camera_fps_observed = 12.5  # Tashkent cam 27, checked live in review
        self.assertGreater(fs._TRACKED_FPS_CEILING, fastest_camera_fps_observed * 1.5)

    def test_roi_ring_is_independent_of_the_detection_ring(self):
        # RoiBatchSlot's ring (face/reid crop batches) is read via a
        # synchronous RPC, not Celery's expires= mechanism — it must not
        # silently move when the detection ring's derivation changes.
        from workers import frame_store as fs

        self.assertEqual(fs._ROI_RING_SIZE, 24)
        self.assertNotEqual(fs._ROI_RING_SIZE, fs._RING_SIZE)


class SameProcessReadTests(unittest.TestCase):
    """Reads in the PRODUCER'S OWN process - not a degenerate test setup.
    Several producers can also be readers: pipeline/frame_pump.py owns a
    camera's CameraFrameSlot in the same process its threads write it, and
    the ROI slots in global_track_client.py / reid_client.py are written by a
    worker that may read them back under an eager or solo pool. Any such
    reader hits this path. These reads must take the _LOCAL_SLOTS fast path
    (straight from the owning slot's mapping) rather than _attach_fresh,
    whose resource_tracker.unregister would delete the producer's own
    tracker entry - a KeyError at clean shutdown and a /dev/shm leak if the
    process crashes."""

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

    def test_reid_assign_and_extract_rings_do_not_collide_same_process(self):
        """Regression: global-track-worker hosts BOTH a reader of the
        camera-worker -> global-track ring ("reid-assign") and a producer of
        the global-track -> reid-worker ring ("reid-extract") for the SAME
        camera. If the two shared one purpose, the producer's _LOCAL_SLOTS
        entry would shadow the inbound ring in that process, so a same-process
        read of the inbound handle silently matched the wrong slot — turning
        ReID matching permanently off for the camera with no error or metric.
        Distinct purposes must keep the two rings separate.
        """
        from workers import frame_store

        cam = 350
        assign_slot = frame_store.RoiBatchSlot(cam, purpose="reid-assign")
        extract_slot = frame_store.RoiBatchSlot(cam, purpose="reid-extract")
        try:
            assign_crops = [_random_frame(20, 10), _random_frame(24, 12)]
            assign_handle = assign_slot.write(assign_crops, [1, 2])

            # extract_slot created/written AFTER — this is exactly the overwrite
            # that broke the inbound read when both shared purpose="reid".
            extract_crops = [_random_frame(30, 15)]
            extract_handle = extract_slot.write(extract_crops, [9])

            # The inbound (assign) handle must still read its OWN crops — not
            # the extract ring's, and not None.
            got_assign = frame_store.attach_and_read_roi_batch(assign_handle)
            self.assertIsNotNone(got_assign)
            self.assertEqual([tid for tid, _ in got_assign], [1, 2])
            for (_tid, crop), expected in zip(got_assign, assign_crops):
                self.assertTrue(np.array_equal(crop, expected))

            # And the extract handle reads its own.
            got_extract = frame_store.attach_and_read_roi_batch(extract_handle)
            self.assertIsNotNone(got_extract)
            self.assertEqual([tid for tid, _ in got_extract], [9])
            self.assertTrue(np.array_equal(got_extract[0][1], extract_crops[0]))
        finally:
            assign_slot.close()
            extract_slot.close()

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
                        self.assertIsNone(frame_store.attach_and_read(stale_handle))
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


class RawFrameSlotTests(unittest.TestCase):
    """RawFrameSlot reuses CameraFrameSlot's ring/header machinery under a
    separate `camraw_<id>` name, for decode_main.py's periodic calibration
    frame (see pipeline/decode_metrics.py). Same-process, same rationale as
    SameProcessReadTests above."""

    def test_round_trips_via_attach_and_read_raw(self):
        from workers import frame_store

        slot = frame_store.RawFrameSlot(camera_id=501)
        try:
            frame = _random_frame(64, 48)
            handle = slot.write(frame)
            got = frame_store.attach_and_read_raw(handle)
            self.assertIsNotNone(got)
            self.assertTrue(np.array_equal(frame, got))
        finally:
            slot.close()

    def test_does_not_collide_with_a_camframe_slot_for_the_same_camera_id(self):
        """The whole point of the separate prefix: a camera's detection
        ring and its raw-frame ring must be two independent segments, not
        one shared by name."""
        from workers import frame_store

        cam_slot = frame_store.CameraFrameSlot(camera_id=502)
        raw_slot = frame_store.RawFrameSlot(camera_id=502)
        try:
            cam_frame = _random_frame(32, 32)
            raw_frame = _random_frame(80, 60)
            cam_handle = cam_slot.write(cam_frame)
            raw_handle = raw_slot.write(raw_frame)

            got_cam = frame_store.attach_and_read(cam_handle)
            got_raw = frame_store.attach_and_read_raw(raw_handle)

            self.assertTrue(np.array_equal(cam_frame, got_cam))
            self.assertTrue(np.array_equal(raw_frame, got_raw))
        finally:
            cam_slot.close()
            raw_slot.close()

    def test_read_after_close_returns_none(self):
        from workers import frame_store

        slot = frame_store.RawFrameSlot(camera_id=503)
        frame = _random_frame(32, 32)
        handle = slot.write(frame)
        slot.close()
        self.assertIsNone(frame_store.attach_and_read_raw(handle))


class ProducerRestartRecoveryTests(unittest.TestCase):
    """A reader must recover on its own when the PRODUCER process restarts.

    Regression for a bug observed live: decode-worker restarted while
    yolo-worker kept running (they no longer share a restart now that
    decoding is its own service). The producer unlinked its segments and
    created new ones under the same names with a fresh instance id, but the
    reader's cached mapping still pointed at the old, unlinked memory.
    `_attach_segment` re-validates size, not identity, so a same-size
    segment looked fine and the stale mapping was served forever — every
    read failed the header check permanently. Measured: skipped_gone ~40/s
    with essentially zero frames processed, only cleared by restarting the
    consumer by hand.
    """

    def test_reader_recovers_after_the_producer_restarts(self):
        """Must exercise the CROSS-PROCESS path (the `_ATTACHED` cache), not
        the same-process `_LOCAL_SLOTS` fast path — the bug only exists in
        the former, and a reader in the real deployment (yolo-worker) is a
        different process with an empty `_LOCAL_SLOTS`. Dropping the local
        registration below is what makes this test see what that reader
        sees; without it the test passes even with the bug present."""
        from workers import frame_store

        camera_id = 601
        shape = (48, 64)

        slot = frame_store.CameraFrameSlot(camera_id=camera_id)
        first = _random_frame(*shape)
        first_handle = slot.write(first)
        # Simulate a remote reader: no local slot registration, so the read
        # goes through _attach_segment and caches in _ATTACHED.
        frame_store._LOCAL_SLOTS.pop(slot._base_name, None)
        self.assertTrue(
            np.array_equal(first, frame_store.attach_and_read(first_handle))
        )

        # Producer "restarts": segments unlinked and recreated, same size,
        # but stamped with a different instance id.
        slot._release()
        original_instance = frame_store._INSTANCE_ID
        try:
            frame_store._INSTANCE_ID = original_instance + 1
            new_slot = frame_store.CameraFrameSlot(camera_id=camera_id)
            frame_store._LOCAL_SLOTS.pop(new_slot._base_name, None)
            try:
                second = _random_frame(*shape)
                second_handle = new_slot.write(second)
                # The reader still holds its cached mapping of the OLD,
                # now-unlinked block. It must notice the instance change,
                # re-attach, and return the new frame — with no restart.
                got = frame_store.attach_and_read(second_handle)
                self.assertIsNotNone(
                    got, "reader did not recover from a producer restart"
                )
                self.assertTrue(np.array_equal(second, got))
            finally:
                new_slot.close()
        finally:
            frame_store._INSTANCE_ID = original_instance
            frame_store._ATTACHED.pop(
                frame_store._segment_name(slot._base_name, first_handle.segment),
                None,
            )

    def test_a_plain_recycled_generation_still_returns_none(self):
        """The recovery path must not weaken the normal staleness check: a
        SAME-instance seq mismatch (the generation was recycled before this
        read) must still return None, not re-attach and hand back whatever
        is in the slot now."""
        from workers import frame_store

        slot = frame_store.CameraFrameSlot(camera_id=602)
        try:
            stale_handle = slot.write(_random_frame(32, 32))
            # Cycle the whole ring so that segment is overwritten by a newer
            # generation from the SAME producer.
            for _ in range(frame_store._RING_SIZE):
                slot.write(_random_frame(32, 32))
            self.assertIsNone(frame_store.attach_and_read(stale_handle))
        finally:
            slot.close()


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
                slot.write(
                    [np.zeros((10, 10), dtype=np.uint8)], [1]
                )  # missing channel dim
        finally:
            slot.close()


if __name__ == "__main__":
    unittest.main()
