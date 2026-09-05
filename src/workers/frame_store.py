"""Shared-memory transport for camera frames crossing the Celery broker.

Redis round-trip cost is payload-size dependent (measured: a 720p frame costs
~15.6ms through Redis, more than a YOLO inference pass) but flat and cheap for
small control messages (~0.3ms). So the frame itself must never be serialised
into a task payload — only a handle (block name + sequence number) travels
through the broker; the pixels live in a `multiprocessing.shared_memory` block
that both the producer (this process) and the worker (a separate process on
the same host) can map directly.

This differs from benchmarks/celery_worker/frame_store.py, which builds one
static corpus decoded once and read many times. Here each camera has a ring of
segments that the producer cycles through every detection cycle. A single
depth-1 slot was tried first and measured live to lose the write-to-execute
race on essentially every batch — by the time a Celery task dequeues, the
producer has already overwritten the one segment with a newer generation, and
a bare size check can't detect a same-shape overwrite. The ring gives each
generation `_RING_SIZE` write-cycles to live instead of one, and the sequence
number is stamped *inside* the segment (not just carried on the handle) so a
reader can verify it actually landed on the generation it was told to read,
not merely that a same-named segment of adequate size currently exists.

The header also stamps a random per-process instance id alongside the seq.
A bare seq is not enough on its own: every producer process's seq counter
starts at 0, so a restarted producer's first write can stamp the exact same
seq value a reader still has cached from before the restart, onto a
same-shape segment recreated under the same name — the seq check would then
pass by coincidence and hand back frozen pre-restart pixels. The instance id
changes on every process start (`os.urandom`, not a counter), so a restart
always produces a header a stale reader cannot mistake for the one it wants,
regardless of what seq value the new process happens to reuse.
"""

from __future__ import annotations

import os
import struct
import threading
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

# How many generations each producer key keeps alive at once. A generation's
# lifetime is _RING_SIZE write-cycles, not one — comfortably longer than
# Celery broker round-trip latency (measured single-segment lifetime: ~20ms,
# far below that latency, which is why every batch lost the race). Bump this
# if warning logs still show a nonzero stale-segment rate under normal load.
_RING_SIZE = 8

# Random per-process id, stamped into every segment's header alongside seq —
# see module docstring for why seq alone can't distinguish generations across
# a producer restart. Unsigned 8 bytes from urandom, not a counter, so a
# restarted process is astronomically unlikely to collide with its own past.
_INSTANCE_ID = int.from_bytes(os.urandom(8), "little")

# 16-byte little-endian header stamped at offset 0 of every segment: the
# (instance_id, seq) pair that owns the payload starting at offset
# _HEADER_SIZE. Unsigned instance_id (Q) since it's a random bit pattern, not
# an ordered quantity; signed seq (q) matches every other seq field in this
# module.
_HEADER_STRUCT = struct.Struct("<Qq")
_HEADER_SIZE = _HEADER_STRUCT.size

_SLOT_NAME_PREFIX = "camframe"

# A second, independent ring per camera for calibration's "grab me a live
# picture" commands (engine.py's capture_frame / test_calibration). The
# camframe ring above holds post-ROI, post-frame-skip crops — wrong for
# calibration, which needs the whole, unmodified frame. Written far less
# often than the detection ring (RawFrameSlot, below) and read as tolerant
# of staleness (a few seconds old is fine), so it costs nothing on the hot
# path and never collides with camframe's name or generation counter.
_RAW_SLOT_NAME_PREFIX = "camraw"

# Worker-side cache: segment name -> attached SharedMemory. One entry per
# ring segment this process has ever read from. Never closed proactively —
# a worker process's segments live as long as the process does.
_ATTACHED: Dict[str, shared_memory.SharedMemory] = {}
_ATTACHED_LOCK = threading.Lock()

# Producer-side registry: producer key (not segment name) -> the live slot
# object that owns it, for readers running IN THE PRODUCER'S OWN PROCESS.
# General-purpose, not tied to any specific caller: any code that reads a
# slot from inside the same process that produced it would otherwise go
# through _attach_fresh, whose resource_tracker.unregister would delete the
# PRODUCER'S OWN tracker entry for the block (the tracker keys by name, one
# entry per process): a spurious KeyError at clean shutdown, and worse, no
# /dev/shm cleanup if this process crashes — the exact leak the tracker
# exists to prevent. A same-process read instead goes straight to the
# owning slot's existing mapping: no second attach, no unregister, and
# faster.
_LOCAL_SLOTS: Dict[str, object] = {}


def _pack_seq_header(seq: int) -> bytes:
    return _HEADER_STRUCT.pack(_INSTANCE_ID, seq)


def _unpack_seq_header(buf) -> Tuple[int, int]:
    return _HEADER_STRUCT.unpack_from(buf, 0)


def _slot_name(camera_id: int) -> str:
    return f"{_SLOT_NAME_PREFIX}_{camera_id}"


def _raw_slot_name(camera_id: int) -> str:
    return f"{_RAW_SLOT_NAME_PREFIX}_{camera_id}"


def _segment_name(base_name: str, segment: int) -> str:
    return f"{base_name}_{segment}"


@dataclass(frozen=True)
class FrameHandle:
    """What actually travels through the Celery broker — no pixels."""

    camera_id: int
    seq: int
    segment: int
    instance_id: int
    height: int
    width: int
    channels: int


class CameraFrameSlot:
    """Producer-side owner of one camera's ring of shared-memory segments.

    One instance per camera, held by the process that reads frames from the
    stream (`CeleryCameraProducer`). Each segment is sized to the camera's
    frame shape on first write to it; a later write with a different shape
    (e.g. ROI config changed at runtime) reallocates every segment lazily as
    the ring cycles through them.
    """

    def __init__(self, camera_id: int, name_prefix: str = _SLOT_NAME_PREFIX):
        self.camera_id = camera_id
        self._base_name = f"{name_prefix}_{camera_id}"
        self._shms: Dict[int, shared_memory.SharedMemory] = {}
        self._shape: Optional[Tuple[int, int, int]] = None
        self._seq = 0
        with _ATTACHED_LOCK:
            _LOCAL_SLOTS[self._base_name] = self

    def _segment_shm(self, segment: int) -> Optional[shared_memory.SharedMemory]:
        return self._shms.get(segment)

    def _ensure_segment(self, segment: int, shape: Tuple[int, int, int]) -> shared_memory.SharedMemory:
        if self._shape != shape:
            # Shape changed: every existing segment is the wrong size, drop
            # them all so each is reallocated the next time it's written.
            self._release()
            self._shape = shape
        shm = self._shms.get(segment)
        nbytes = _HEADER_SIZE + int(np.prod(shape))
        if shm is not None and shm.size >= nbytes:
            return shm
        name = _segment_name(self._base_name, segment)
        with _ATTACHED_LOCK:
            if shm is not None:
                shm.close()
                try:
                    shm.unlink()
                except FileNotFoundError:
                    pass
            try:
                shm = shared_memory.SharedMemory(name=name, create=True, size=nbytes)
            except FileExistsError:
                # A segment from a crashed previous run of this process may
                # still exist under this name — or a second live camera-worker
                # replica may be serving this same camera (which must not
                # happen, see compose.yml's camera-worker note). The two are
                # indistinguishable from here, so log loudly before
                # reclaiming: silently stealing the block would otherwise
                # look identical to a normal first-time allocation. Must
                # close() before unlink(): opening it here registers the
                # attach with this process's resource_tracker, and leaving it
                # open leaks that registration (observed as a spurious
                # KeyError/leak warning at process exit).
                logger.warning(
                    f"Shared-memory segment '{name}' already exists — "
                    f"reclaiming it as leftover from a crashed prior run. If "
                    f"another live process still owns it (e.g. a second "
                    f"camera-worker replica for this camera), this steals "
                    f"its block and corrupts its reads."
                )
                stale = shared_memory.SharedMemory(name=name)
                stale.close()
                stale.unlink()
                shm = shared_memory.SharedMemory(name=name, create=True, size=nbytes)
            self._shms[segment] = shm
        logger.debug(
            f"CameraFrameSlot[cam={self.camera_id}]: allocated segment "
            f"{segment} shape={shape} ({nbytes / 1024:.0f} KiB) as '{name}'"
        )
        return shm

    def write(self, frame: np.ndarray) -> FrameHandle:
        """Copy `frame` into the next ring segment and return a handle.

        Not safe for concurrent writers — one camera's frames are produced
        by exactly one thread/process today, and that invariant must hold for
        this class to be correct. Concurrent *readers* (a worker attaching
        mid-write) are fine: a reader may see a partially-written segment if
        it races a write, which is why the consumer must not trust the
        payload without the header seq matching `handle.seq` — the ring gives
        the previous generation `_RING_SIZE` write-cycles of grace instead of
        being overwritten out from under an in-flight reader immediately.
        """
        shape = frame.shape
        if len(shape) != 3:
            raise ValueError(f"expected an HxWxC frame, got shape {shape}")
        self._seq += 1
        seq = self._seq
        segment = seq % _RING_SIZE
        shm = self._ensure_segment(segment, shape)
        view = np.ndarray(shape, dtype=frame.dtype, buffer=shm.buf, offset=_HEADER_SIZE)
        view[:] = frame
        shm.buf[0:_HEADER_SIZE] = _pack_seq_header(seq)
        return FrameHandle(
            camera_id=self.camera_id,
            seq=seq,
            segment=segment,
            instance_id=_INSTANCE_ID,
            height=shape[0],
            width=shape[1],
            channels=shape[2],
        )

    def _release(self) -> None:
        # Serialised against same-process readers (the _LOCAL_SLOTS fast
        # path in attach_and_read) via _ATTACHED_LOCK: releasing a mapping
        # while such a reader's temporary view is still copying from it
        # raises BufferError in THIS thread (a memoryview with live exports
        # refuses to release) — turning a benign read race into a
        # producer-side crash on reshape/close. Cross-process readers are
        # unaffected either way; they hold their own mapping.
        with _ATTACHED_LOCK:
            for shm in self._shms.values():
                shm.close()
                try:
                    shm.unlink()
                except FileNotFoundError:
                    pass
            self._shms.clear()

    def close(self) -> None:
        """Release every ring segment. Call when the camera is removed."""
        with _ATTACHED_LOCK:
            if _LOCAL_SLOTS.get(self._base_name) is self:
                del _LOCAL_SLOTS[self._base_name]
        self._release()


class RawFrameSlot(CameraFrameSlot):
    """The full, pre-ROI frame — same ring/header machinery as
    CameraFrameSlot, under a separate `camraw_<id>` name so it never
    collides with the detection-frame ring.

    Written by the decode worker on a slow, periodic cadence (not the
    per-detection hot path), for the calibration commands
    (`capture_frame`/`test_calibration`) that need a whole frame rather
    than whatever ROI-cropped, frame-skipped picture the detection ring
    happens to hold. Reuses `FrameHandle` as its transport type — the
    fields are identical, only the ring the handle points at differs.
    """

    def __init__(self, camera_id: int):
        super().__init__(camera_id, name_prefix=_RAW_SLOT_NAME_PREFIX)


def _attach_fresh(name: str) -> Optional[shared_memory.SharedMemory]:
    """Map `name` and unregister it from this process's resource_tracker.

    The producer process owns the block's lifetime (create/unlink); a worker
    only ever maps it read-only. Without unregistering, this process's
    tracker would additionally try to unlink the block on worker shutdown —
    spamming warnings at best, racing the producer's own unlink at worst.
    """
    try:
        shm = shared_memory.SharedMemory(name=name)
    except FileNotFoundError:
        return None
    try:
        from multiprocessing import resource_tracker

        # Must match the string SharedMemory.__init__ actually registered,
        # which is `self._name` (POSIX-prefixed, e.g. "/camframe_7") — NOT
        # the public `.name` property, which strips that prefix. Passing
        # `.name` here silently mismatches the tracker's key: unregister()
        # becomes a no-op on an unknown name, and the entry the *producer's*
        # own SharedMemory.unlink() later tries to remove is now the only
        # remaining reference — surfacing as a spurious KeyError in the
        # tracker's background process when it does eventually get removed.
        # Verified by tracing register()/unregister() calls directly; this
        # is not cosmetic, it is a real double-bookkeeping bug.
        resource_tracker.unregister(shm._name, "shared_memory")  # type: ignore[attr-defined]
    except Exception:
        pass
    return shm


def _attach_segment(name: str, needed: int = 0) -> Optional[shared_memory.SharedMemory]:
    """Worker-side: get a cached mapping for `name`, attaching fresh if
    needed.

    Re-validates size, not just presence: a ring segment's producer can
    reallocate it larger at any write (e.g. a bigger face-crop batch than
    ever seen before) via close()+unlink()+create() under the same name —
    this process's cached mapping from before that reallocation still maps
    the OLD, now-unlinked memory. Without this check, a caller whose handle
    names a byte range past the cached mapping's old (smaller) size would
    hit that boundary and bail out as "gone" forever, never re-attaching to
    see the segment that has been sitting there, correctly written, the
    whole time — this was the actual cause of every embed silently returning
    blank results once any face-crop batch exceeded the first one this
    process ever saw. The header (instance_id, seq) check callers do after
    this call is a separate, correct staleness signal for "this generation
    was recycled before I read it" — it cannot substitute for this, since it
    only runs once the mapping is confirmed large enough to read at all.
    """
    shm = _ATTACHED.get(name)
    if shm is not None and shm.size < needed:
        del _ATTACHED[name]
        shm = None
    if shm is None:
        shm = _attach_fresh(name)
        if shm is None:
            return None
        _ATTACHED[name] = shm
    return shm


def attach_and_read(handle: FrameHandle) -> Optional[np.ndarray]:
    """Worker side: map the camera's ring segment named by `handle.segment`
    and copy out the frame, but only if the segment's stamped header seq
    still matches `handle.seq`.

    Returns None if the segment doesn't exist, is too small, or — the case
    that matters most — has already been recycled by a newer write by the
    time this runs. The caller should treat all three exactly like
    `get_detections`' empty-result case, not as an error worth retrying.

    Returns a **copy**, not a view: the underlying block is being
    concurrently overwritten by the producer, so a zero-copy view would be
    unsafe to use after this call returns. The corpus-mode benchmark could
    return a zero-copy view because its block is immutable after `create()`;
    this one cannot.

    Unlike a bare size check, comparing the header seq also catches a
    same-shape reallocation — e.g. the producer process restarting and
    recreating a segment of the same size a worker already has mapped. The
    fresh process always re-stamps the header on its first write, so a
    worker's stale cached mapping fails the seq check immediately instead of
    silently serving frozen pixels from before the restart.
    """
    return _attach_and_read_ring(handle, _slot_name(handle.camera_id))


def attach_and_read_raw(handle: FrameHandle) -> Optional[np.ndarray]:
    """Same as `attach_and_read`, but for a `RawFrameSlot` handle — the
    `camraw_<id>` ring instead of `camframe_<id>`. See `RawFrameSlot`'s
    docstring for what it's for."""
    return _attach_and_read_ring(handle, _raw_slot_name(handle.camera_id))


def _attach_and_read_ring(handle: FrameHandle, base_name: str) -> Optional[np.ndarray]:
    name = _segment_name(base_name, handle.segment)
    shape = (handle.height, handle.width, handle.channels)
    nbytes = _HEADER_SIZE + int(np.prod(shape))

    with _ATTACHED_LOCK:
        local_slot = _LOCAL_SLOTS.get(base_name)
        if local_slot is not None:
            # Same-process fast path — see _LOCAL_SLOTS. The copy is a
            # single expression so its temporary view releases its buffer
            # export before this lock does; _release() takes the same lock,
            # which is what makes a concurrent reshape/close safe here.
            local_shm = getattr(local_slot, "_segment_shm", lambda _s: None)(handle.segment)
            if local_shm is None or local_shm.size < nbytes:
                return None
            if _unpack_seq_header(local_shm.buf) != (handle.instance_id, handle.seq):
                return None
            frame = np.ndarray(
                shape, dtype=np.uint8, buffer=local_shm.buf, offset=_HEADER_SIZE
            ).copy()
            # Re-check after the copy, not just before: the producer can
            # recycle this same segment (next write, _RING_SIZE cycles later)
            # while the copy above is in flight. The pre-copy check alone
            # can pass and still hand back a torn frame mixing two
            # generations' pixels; a header mismatch now means exactly that
            # happened, so discard it like any other stale read.
            if _unpack_seq_header(local_shm.buf) != (handle.instance_id, handle.seq):
                return None
            return frame

        shm = _attach_segment(name, nbytes)
        if shm is None or shm.size < nbytes:
            return None
        if _unpack_seq_header(shm.buf) != (handle.instance_id, handle.seq):
            # A seq mismatch alone is normal (the generation was recycled
            # before this read). An INSTANCE mismatch is not: it means the
            # producer process restarted, unlinked these segments and made
            # new ones, while this process still holds a mapping of the old,
            # now-unlinked memory. `_attach_segment` can't catch that on its
            # own — it re-validates size, and a same-size segment looks
            # identical — so the stale mapping would be served forever and
            # every read would fail this check permanently.
            #
            # Not hypothetical: observed live when decode-worker restarted
            # while yolo-worker kept running (they no longer share a restart
            # now that decoding is its own service). skipped_gone went to
            # ~40/s with essentially zero frames processed, and only a
            # manual consumer restart cleared it. Drop the mapping and
            # re-attach once, so the next read lands on the new segment.
            if _unpack_seq_header(shm.buf)[0] != handle.instance_id:
                del _ATTACHED[name]
                shm = _attach_segment(name, nbytes)
                if shm is None or shm.size < nbytes:
                    return None
                if _unpack_seq_header(shm.buf) != (handle.instance_id, handle.seq):
                    return None
            else:
                return None
        frame = np.ndarray(
            shape, dtype=np.uint8, buffer=shm.buf, offset=_HEADER_SIZE
        ).copy()
        if _unpack_seq_header(shm.buf) != (handle.instance_id, handle.seq):
            return None

    return frame


def shared_memory_exists(name: str) -> bool:
    """Cheap existence check that never touches this process's
    resource_tracker — a plain filesystem stat, not an attach/detach.

    Deliberately not `shared_memory.SharedMemory(name=...)` + `close()`: that
    pattern registers and unregisters the name with the tracker on every
    call, which is unnecessary overhead on a per-frame hot path and has been
    observed to raise spurious `KeyError`s in the tracker's background
    process when interleaved with a long-lived cached attachment of the same
    name.
    """
    import os

    return os.path.exists(f"/dev/shm/{name}")


_ROI_SLOT_NAME_PREFIX = "camroi"


def _roi_slot_name(camera_id: int) -> str:
    return f"{_ROI_SLOT_NAME_PREFIX}_{camera_id}"


@dataclass(frozen=True)
class RoiHandle:
    """One crop's location within a RoiBatchHandle's packed block."""

    track_id: int
    offset: int  # byte offset into the payload (after the header)
    height: int
    width: int
    channels: int


@dataclass(frozen=True)
class RoiBatchHandle:
    """What actually travels through the Celery broker for submit_faces — no
    pixels, same principle as FrameHandle. `rois` carries each crop's shape
    and packed offset since, unlike frames, crops in one batch are not all
    the same size — there is no single (height, width, channels) to put on
    the handle itself.
    """

    camera_id: int
    seq: int
    segment: int
    instance_id: int
    rois: Tuple[RoiHandle, ...]


class RoiBatchSlot:
    """Producer-side owner of one camera's ring of shared-memory ROI-batch
    segments.

    One instance per camera, mirroring CameraFrameSlot: a single-writer
    ring, one batch in flight per segment (each `write()` call replaces
    whatever the segment's previous generation held `_RING_SIZE` writes
    ago). Owned by `_CameraContext.process_frame` (workers/camera_tasks.py),
    which writes here on every recognition-due detection frame and hands the
    resulting handle to `FaceEmbedClient.embed` (workers/face_client.py).

    All crops in one call are packed into a single contiguous payload —
    variable-size, so packed by running byte offset (after the header) rather
    than a fixed per-segment stride like CameraFrameSlot's single shape. An
    empty batch (`write([], [])`) is a legitimate call, not an error — the
    handle's empty `rois` tuple is what the reader actually branches on,
    matching `embed_batch_task([])` returning `[]` for an empty list without
    needing to inspect the block at all.
    """

    def __init__(self, camera_id: int):
        self.camera_id = camera_id
        self._base_name = _roi_slot_name(camera_id)
        self._shms: Dict[int, shared_memory.SharedMemory] = {}
        self._seq = 0
        with _ATTACHED_LOCK:
            _LOCAL_SLOTS[self._base_name] = self

    def _segment_shm(self, segment: int) -> Optional[shared_memory.SharedMemory]:
        return self._shms.get(segment)

    def _ensure_segment(self, segment: int, nbytes: int) -> shared_memory.SharedMemory:
        shm = self._shms.get(segment)
        alloc = max(nbytes, _HEADER_SIZE)  # SharedMemory requires size > 0
        if shm is not None and shm.size >= alloc:
            return shm
        name = _segment_name(self._base_name, segment)
        with _ATTACHED_LOCK:
            if shm is not None:
                shm.close()
                try:
                    shm.unlink()
                except FileNotFoundError:
                    pass
            try:
                shm = shared_memory.SharedMemory(name=name, create=True, size=alloc)
            except FileExistsError:
                # See CameraFrameSlot._ensure_segment — same crashed-prior-run
                # vs. second-live-replica ambiguity, same loud-warn-then-reclaim
                # handling, same close()-before-unlink() requirement.
                logger.warning(
                    f"Shared-memory segment '{name}' already exists — "
                    f"reclaiming it as leftover from a crashed prior run. If "
                    f"another live process still owns it (e.g. a second "
                    f"camera-worker replica for this camera), this steals "
                    f"its block and corrupts its reads."
                )
                stale = shared_memory.SharedMemory(name=name)
                stale.close()
                stale.unlink()
                shm = shared_memory.SharedMemory(name=name, create=True, size=alloc)
            self._shms[segment] = shm
        logger.debug(
            f"RoiBatchSlot[cam={self.camera_id}]: allocated segment {segment} "
            f"{alloc} bytes as '{name}'"
        )
        return shm

    def write(
        self, person_rois: List[np.ndarray], track_ids: List[int]
    ) -> RoiBatchHandle:
        """Pack `person_rois` into the next ring segment and return a handle.
        An empty list is a legitimate call, not an error — see the class
        docstring's note on `write([], [])`.
        """
        if len(person_rois) != len(track_ids):
            raise ValueError(
                f"person_rois ({len(person_rois)}) and track_ids "
                f"({len(track_ids)}) must be the same length"
            )

        self._seq += 1
        seq = self._seq
        segment = seq % _RING_SIZE
        total_bytes = sum(int(np.prod(r.shape)) for r in person_rois)
        shm = self._ensure_segment(segment, _HEADER_SIZE + total_bytes)

        rois: List[RoiHandle] = []
        offset = 0
        for roi, track_id in zip(person_rois, track_ids):
            shape = roi.shape
            if len(shape) != 3:
                raise ValueError(f"expected an HxWxC crop, got shape {shape}")
            nbytes = int(np.prod(shape))
            view = np.ndarray(
                shape, dtype=np.uint8, buffer=shm.buf, offset=_HEADER_SIZE + offset
            )
            view[:] = roi
            rois.append(
                RoiHandle(
                    track_id=track_id,
                    offset=offset,
                    height=shape[0],
                    width=shape[1],
                    channels=shape[2],
                )
            )
            offset += nbytes

        shm.buf[0:_HEADER_SIZE] = _pack_seq_header(seq)
        return RoiBatchHandle(
            camera_id=self.camera_id,
            seq=seq,
            segment=segment,
            instance_id=_INSTANCE_ID,
            rois=tuple(rois),
        )

    def _release(self) -> None:
        # Same _ATTACHED_LOCK serialisation as CameraFrameSlot._release,
        # for the same BufferError-on-reshape/close reason.
        with _ATTACHED_LOCK:
            for shm in self._shms.values():
                shm.close()
                try:
                    shm.unlink()
                except FileNotFoundError:
                    pass
            self._shms.clear()

    def close(self) -> None:
        """Release every ring segment. Call when the camera is removed."""
        with _ATTACHED_LOCK:
            if _LOCAL_SLOTS.get(self._base_name) is self:
                del _LOCAL_SLOTS[self._base_name]
        self._release()


def attach_and_read_roi_batch(
    handle: RoiBatchHandle,
) -> Optional[List[Tuple[int, np.ndarray]]]:
    """Worker side: map the camera's ROI-batch ring segment named by
    `handle.segment` and copy out every crop it names, as `(track_id, crop)`
    pairs in the same order `handle.rois` lists them — but only if the
    segment's stamped header seq still matches `handle.seq`.

    Returns `[]` for a legitimately empty batch (see RoiBatchSlot.write), and
    `None` if the segment doesn't exist, is too small, or has already been
    recycled by a newer write — same None-means-gone convention as
    `attach_and_read`, so a caller can treat any of these exactly like a
    missing frame slot.
    """
    if not handle.rois:
        return []

    name = _segment_name(_roi_slot_name(handle.camera_id), handle.segment)
    needed = _HEADER_SIZE + max(
        r.offset + int(np.prod((r.height, r.width, r.channels))) for r in handle.rois
    )

    with _ATTACHED_LOCK:
        local_slot = _LOCAL_SLOTS.get(_roi_slot_name(handle.camera_id))
        if local_slot is not None:
            # Same-process fast path — see _LOCAL_SLOTS and the matching
            # block in attach_and_read. Each copy is a single expression so
            # no view outlives its statement; _release() takes this same
            # lock, making a concurrent reallocation/close safe.
            local_shm = getattr(local_slot, "_segment_shm", lambda _s: None)(handle.segment)
            if local_shm is None or local_shm.size < needed:
                return None
            if _unpack_seq_header(local_shm.buf) != (handle.instance_id, handle.seq):
                return None
            crops = [
                (
                    r.track_id,
                    np.ndarray(
                        (r.height, r.width, r.channels),
                        dtype=np.uint8,
                        buffer=local_shm.buf,
                        offset=_HEADER_SIZE + r.offset,
                    ).copy(),
                )
                for r in handle.rois
            ]
            # Re-check after copying every crop: the producer can recycle
            # this segment mid-copy (see attach_and_read's matching check),
            # handing back a batch mixing two generations' pixels.
            if _unpack_seq_header(local_shm.buf) != (handle.instance_id, handle.seq):
                return None
            return crops

        shm = _attach_segment(name, needed)
        if shm is None or shm.size < needed:
            return None
        if _unpack_seq_header(shm.buf) != (handle.instance_id, handle.seq):
            # Same producer-restart case attach_and_read handles — see its
            # comment. Applies here too: camera-worker owns these segments
            # and face-worker reads them, so a camera-worker restart while
            # face-worker keeps running would otherwise leave face-worker
            # permanently reading unlinked memory and returning no
            # embeddings at all.
            if _unpack_seq_header(shm.buf)[0] != handle.instance_id:
                del _ATTACHED[name]
                shm = _attach_segment(name, needed)
                if shm is None or shm.size < needed:
                    return None
                if _unpack_seq_header(shm.buf) != (handle.instance_id, handle.seq):
                    return None
            else:
                return None

        crops: List[Tuple[int, np.ndarray]] = []
        for r in handle.rois:
            shape = (r.height, r.width, r.channels)
            view = np.ndarray(
                shape, dtype=np.uint8, buffer=shm.buf, offset=_HEADER_SIZE + r.offset
            )
            crops.append((r.track_id, view.copy()))
        if _unpack_seq_header(shm.buf) != (handle.instance_id, handle.seq):
            return None

    return crops

