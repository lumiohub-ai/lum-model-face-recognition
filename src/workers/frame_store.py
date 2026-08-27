"""Shared-memory transport for camera frames crossing the Celery broker.

Redis round-trip cost is payload-size dependent (measured: a 720p frame costs
~15.6ms through Redis, more than a YOLO inference pass) but flat and cheap for
small control messages (~0.3ms). So the frame itself must never be serialised
into a task payload — only a handle (block name + sequence number) travels
through the broker; the pixels live in a `multiprocessing.shared_memory` block
that both the producer (this process) and the worker (a separate process on
the same host) can map directly.

This differs from benchmarks/celery_worker/frame_store.py, which builds one
static corpus decoded once and read many times. Here each camera has one slot
that the producer overwrites every detection cycle — a single-writer,
single-slot ring of depth 1, not a corpus. A worker reads whatever is
currently in the slot; staleness is caught by comparing the sequence number
the task payload carries against what's actually written, the same
correlation discipline LSO-138 added to the in-process GPU reply queues.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

_SLOT_NAME_PREFIX = "camframe"

# Worker-side cache: block name -> attached SharedMemory. One entry per
# camera slot this process has ever read from. Never closed proactively —
# a worker process's slots live as long as the process does.
_ATTACHED: Dict[str, shared_memory.SharedMemory] = {}
_ATTACHED_LOCK = threading.Lock()

# Producer-side registry: block name -> the live slot object that owns it,
# for readers running IN THE PRODUCER'S OWN PROCESS. This is not a
# hypothetical: the GpuWorkerRpcServer (gpu_worker_rpc.py) runs in the main
# process — the same process whose CameraWorker threads own the frame slots —
# so its reads would otherwise go through _attach_fresh, whose
# resource_tracker.unregister would delete the PRODUCER'S OWN tracker entry
# for the block (the tracker keys by name, one entry per process): a spurious
# KeyError at clean shutdown, and worse, no /dev/shm cleanup if this process
# crashes — the exact leak the tracker exists to prevent. A same-process read
# instead goes straight to the owning slot's existing mapping: no second
# attach, no unregister, and faster.
_LOCAL_SLOTS: Dict[str, object] = {}


def _slot_name(camera_id: int) -> str:
    return f"{_SLOT_NAME_PREFIX}_{camera_id}"


@dataclass(frozen=True)
class FrameHandle:
    """What actually travels through the Celery broker — no pixels."""

    camera_id: int
    seq: int
    height: int
    width: int
    channels: int


class CameraFrameSlot:
    """Producer-side owner of one camera's shared-memory slot.

    One instance per camera, held by the process that reads frames from the
    stream (today: `CameraWorker`). Sized to the camera's frame shape on
    first write; a later write with a different shape (e.g. ROI config
    changed at runtime) reallocates the block.
    """

    def __init__(self, camera_id: int):
        self.camera_id = camera_id
        self._name = _slot_name(camera_id)
        self._shm: Optional[shared_memory.SharedMemory] = None
        self._shape: Optional[Tuple[int, int, int]] = None
        self._seq = 0
        with _ATTACHED_LOCK:
            _LOCAL_SLOTS[self._name] = self

    def _ensure_capacity(self, shape: Tuple[int, int, int]) -> None:
        if self._shm is not None and self._shape == shape:
            return
        self._release()
        nbytes = int(np.prod(shape))
        try:
            # A slot from a crashed previous run of this process may still
            # exist under this name. Must close() before unlink(): opening
            # it here registers the attach with this process's
            # resource_tracker, and leaving it open leaks that registration
            # (observed as a spurious KeyError/leak warning at process exit).
            stale = shared_memory.SharedMemory(name=self._name)
            stale.close()
            stale.unlink()
        except FileNotFoundError:
            pass
        self._shm = shared_memory.SharedMemory(
            name=self._name, create=True, size=nbytes
        )
        self._shape = shape
        logger.debug(
            f"CameraFrameSlot[cam={self.camera_id}]: allocated {shape} "
            f"({nbytes / 1024:.0f} KiB) as '{self._name}'"
        )

    def write(self, frame: np.ndarray) -> FrameHandle:
        """Copy `frame` into the slot and return a handle for the task payload.

        Not safe for concurrent writers — one camera's frames are produced
        by exactly one thread/process today, and that invariant must hold for
        this class to be correct. Concurrent *readers* (a worker attaching
        mid-write) are fine: a reader may see a partially-written frame if
        it races a write, which is why the consumer must not trust it
        without the sequence number matching what it expected — the same
        reasoning as LSO-138's request/response correlation.
        """
        shape = frame.shape
        if len(shape) != 3:
            raise ValueError(f"expected an HxWxC frame, got shape {shape}")
        self._ensure_capacity(shape)
        shm = self._shm
        assert shm is not None  # _ensure_capacity always allocates or reuses
        view = np.ndarray(shape, dtype=frame.dtype, buffer=shm.buf)
        view[:] = frame
        self._seq += 1
        return FrameHandle(
            camera_id=self.camera_id,
            seq=self._seq,
            height=shape[0],
            width=shape[1],
            channels=shape[2],
        )

    def _release(self) -> None:
        # Serialised against same-process readers (the _LOCAL_SLOTS fast
        # path in attach_and_read) via _ATTACHED_LOCK: releasing this
        # mapping while such a reader's temporary view is still copying
        # from it raises BufferError in THIS thread (a memoryview with live
        # exports refuses to release) — turning a benign read race into a
        # producer-side crash on reshape/close. Cross-process readers are
        # unaffected either way; they hold their own mapping.
        with _ATTACHED_LOCK:
            if self._shm is not None:
                self._shm.close()
                try:
                    self._shm.unlink()
                except FileNotFoundError:
                    pass
                self._shm = None
                self._shape = None

    def close(self) -> None:
        """Release the slot. Call when the camera is removed."""
        with _ATTACHED_LOCK:
            if _LOCAL_SLOTS.get(self._name) is self:
                del _LOCAL_SLOTS[self._name]
        self._release()


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


def attach_and_read(handle: FrameHandle) -> Optional[np.ndarray]:
    """Worker side: map the camera's slot and copy out the frame it names.

    Returns None if the slot doesn't exist (camera removed/never started, or
    removed since this worker last attached) — the caller should treat this
    exactly like `get_detections`' empty-result case, not as an error worth
    retrying.

    Returns a **copy**, not a view: the underlying block is being
    concurrently overwritten by the producer, so a zero-copy view would be
    unsafe to use after this call returns. The corpus-mode benchmark could
    return a zero-copy view because its block is immutable after `create()`;
    this one cannot.

    Staleness (a worker reading a frame *older* than the one it expected) is
    the caller's responsibility, the same as `get_detections`' seq-matching
    in `gpu_worker.py`: `handle.seq` is what the caller compares against, not
    something this function can judge on its own. What this function does
    guard is a **freed or resized** block: `CameraFrameSlot.close()` or a
    shape-changing `write()` unlinks the old block by name, but cannot reach
    into a worker's already-mapped memory to invalidate it — an
    unlinked-but-still-mapped segment stays readable on Linux, so a cached
    mapping is re-validated on every read rather than trusted indefinitely.
    """
    name = _slot_name(handle.camera_id)
    shape = (handle.height, handle.width, handle.channels)
    nbytes = int(np.prod(shape))

    with _ATTACHED_LOCK:
        local_slot = _LOCAL_SLOTS.get(name)
        if local_slot is not None:
            # Same-process fast path — see _LOCAL_SLOTS. The copy is a
            # single expression so its temporary view releases its buffer
            # export before this lock does; _release() takes the same lock,
            # which is what makes a concurrent reshape/close safe here.
            local_shm = getattr(local_slot, "_shm", None)
            if local_shm is None or local_shm.size < nbytes:
                return None
            return np.ndarray(shape, dtype=np.uint8, buffer=local_shm.buf).copy()

        shm = _ATTACHED.get(name)

        if shm is not None and shm.size < nbytes:
            # Producer reallocated to a larger shape since this worker last
            # attached (e.g. ROI config changed). Stop trusting this mapping.
            del _ATTACHED[name]
            shm = None

        if shm is None:
            shm = _attach_fresh(name)
            if shm is None:
                return None
            _ATTACHED[name] = shm

        view = np.ndarray(shape, dtype=np.uint8, buffer=shm.buf)
        frame = view.copy()

    # Re-validate the name still resolves *after* copying, outside the lock:
    # a producer that closed the slot mid-read leaves the copy above reading
    # garbage or a torn frame, which this catches after the fact rather than
    # preventing (preventing it would need a producer-side lock per read,
    # defeating the point of a lock-free single-writer slot). A stale-name
    # result is discarded; the caller's next call will get a clean None.
    if not shared_memory_exists(name):
        with _ATTACHED_LOCK:
            _ATTACHED.pop(name, None)
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
    offset: int  # byte offset into the block
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
    rois: Tuple[RoiHandle, ...]


class RoiBatchSlot:
    """Producer-side owner of one camera's shared-memory ROI-batch slot.

    One instance per camera, mirroring CameraFrameSlot but for `submit_faces`:
    a single-writer, single-batch-in-flight-per-camera block, matching
    GPUInferenceWorker.submit_faces' own one-batch-in-flight-per-camera
    semantics (each call replaces whatever the previous seq's batch was).

    All crops in one call are packed into a single contiguous block —
    variable-size, so packed by running byte offset rather than a fixed
    per-slot stride like CameraFrameSlot's single shape. An empty batch
    (`write([], [])`, submit_faces' "keep synchronised" call when recognition
    is skipped this cycle — see camera_worker.py) allocates a zero-size
    block; the handle's empty `rois` tuple is what the reader actually
    branches on, matching `_run_arcface_batch([])` returning `[]` for an
    empty list without needing to inspect the block at all.
    """

    def __init__(self, camera_id: int):
        self.camera_id = camera_id
        self._name = _roi_slot_name(camera_id)
        self._shm: Optional[shared_memory.SharedMemory] = None
        self._capacity: int = 0
        self._seq = 0
        with _ATTACHED_LOCK:
            _LOCAL_SLOTS[self._name] = self

    def _ensure_capacity(self, nbytes: int) -> None:
        if self._shm is not None and self._capacity >= nbytes:
            return
        self._release()
        alloc = max(nbytes, 1)  # SharedMemory requires size > 0 even for an empty batch
        try:
            # See CameraFrameSlot._ensure_capacity — same crashed-prior-run
            # cleanup, same close()-before-unlink() requirement.
            stale = shared_memory.SharedMemory(name=self._name)
            stale.close()
            stale.unlink()
        except FileNotFoundError:
            pass
        self._shm = shared_memory.SharedMemory(
            name=self._name, create=True, size=alloc
        )
        self._capacity = alloc
        logger.debug(
            f"RoiBatchSlot[cam={self.camera_id}]: allocated {alloc} bytes "
            f"as '{self._name}'"
        )

    def write(
        self, person_rois: List[np.ndarray], track_ids: List[int]
    ) -> RoiBatchHandle:
        """Pack `person_rois` into the slot and return a handle for the task
        payload. Mirrors GPUInferenceWorker.submit_faces' contract: an empty
        list is a legitimate call (keeps the sequence counter — and, over
        RPC, the round-trip — synchronised even when this cycle skips
        recognition), not an error.
        """
        if len(person_rois) != len(track_ids):
            raise ValueError(
                f"person_rois ({len(person_rois)}) and track_ids "
                f"({len(track_ids)}) must be the same length"
            )

        total_bytes = sum(int(np.prod(r.shape)) for r in person_rois)
        self._ensure_capacity(total_bytes)
        shm = self._shm
        assert shm is not None  # _ensure_capacity always allocates or reuses

        rois: List[RoiHandle] = []
        offset = 0
        for roi, track_id in zip(person_rois, track_ids):
            shape = roi.shape
            if len(shape) != 3:
                raise ValueError(f"expected an HxWxC crop, got shape {shape}")
            nbytes = int(np.prod(shape))
            view = np.ndarray(
                shape, dtype=np.uint8, buffer=shm.buf, offset=offset
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

        self._seq += 1
        return RoiBatchHandle(
            camera_id=self.camera_id, seq=self._seq, rois=tuple(rois)
        )

    def _release(self) -> None:
        # Same _ATTACHED_LOCK serialisation as CameraFrameSlot._release,
        # for the same BufferError-on-reshape/close reason.
        with _ATTACHED_LOCK:
            if self._shm is not None:
                self._shm.close()
                try:
                    self._shm.unlink()
                except FileNotFoundError:
                    pass
                self._shm = None
                self._capacity = 0

    def close(self) -> None:
        """Release the slot. Call when the camera is removed."""
        with _ATTACHED_LOCK:
            if _LOCAL_SLOTS.get(self._name) is self:
                del _LOCAL_SLOTS[self._name]
        self._release()


def attach_and_read_roi_batch(
    handle: RoiBatchHandle,
) -> Optional[List[Tuple[int, np.ndarray]]]:
    """Worker side: map the camera's ROI-batch slot and copy out every crop
    it names, as `(track_id, crop)` pairs in the same order `handle.rois`
    lists them.

    Returns `[]` for a legitimately empty batch (see RoiBatchSlot.write),
    and `None` if the slot doesn't exist at all — same None-means-gone
    convention as `attach_and_read`, so a caller can treat a missing ROI
    slot exactly like a missing frame slot.
    """
    if not handle.rois:
        return []

    name = _roi_slot_name(handle.camera_id)
    needed = max(r.offset + int(np.prod((r.height, r.width, r.channels))) for r in handle.rois)

    with _ATTACHED_LOCK:
        local_slot = _LOCAL_SLOTS.get(name)
        if local_slot is not None:
            # Same-process fast path — see _LOCAL_SLOTS and the matching
            # block in attach_and_read. Each copy is a single expression so
            # no view outlives its statement; _release() takes this same
            # lock, making a concurrent reallocation/close safe.
            local_shm = getattr(local_slot, "_shm", None)
            if local_shm is None or local_shm.size < needed:
                return None
            return [
                (
                    r.track_id,
                    np.ndarray(
                        (r.height, r.width, r.channels),
                        dtype=np.uint8,
                        buffer=local_shm.buf,
                        offset=r.offset,
                    ).copy(),
                )
                for r in handle.rois
            ]

        shm = _ATTACHED.get(name)

        if shm is not None and shm.size < needed:
            del _ATTACHED[name]
            shm = None

        if shm is None:
            shm = _attach_fresh(name)
            if shm is None:
                return None
            _ATTACHED[name] = shm

        crops: List[Tuple[int, np.ndarray]] = []
        for r in handle.rois:
            shape = (r.height, r.width, r.channels)
            view = np.ndarray(
                shape, dtype=np.uint8, buffer=shm.buf, offset=r.offset
            )
            crops.append((r.track_id, view.copy()))

    if not shared_memory_exists(name):
        # Same post-copy re-validation as attach_and_read, and for the same
        # reason: a producer that reallocated or closed the slot mid-read
        # leaves the copies above reading garbage or a torn batch.
        with _ATTACHED_LOCK:
            _ATTACHED.pop(name, None)
        return None

    return crops


_FRAME_BATCH_SLOT_NAME_PREFIX = "framebatch"


def _frame_batch_slot_name(loop_name: str) -> str:
    return f"{_FRAME_BATCH_SLOT_NAME_PREFIX}_{loop_name}"


@dataclass(frozen=True)
class BatchedFrameHandle:
    """One frame's location within a FrameBatchHandle's packed block.

    Carries `camera_id` and `frame_num` per frame because — unlike every
    other slot type here — a single batch spans multiple cameras, and the
    consumer must be able to route each result back to the camera that
    submitted it (GPUInferenceWorker._yolo_loop distributes by camera).
    """

    camera_id: int
    frame_num: int
    offset: int  # byte offset into the block
    height: int
    width: int
    channels: int


@dataclass(frozen=True)
class FrameBatchHandle:
    """What travels through the Celery broker for one cross-camera batch."""

    seq: int
    frames: Tuple[BatchedFrameHandle, ...]


class FrameBatchSlot:
    """Producer-side owner of one GPU loop's cross-camera frame batch.

    LSO-67 Stage 2. Keyed by GPU **loop** name (there is one YOLO loop and
    one ArcFace loop), NOT by camera — the whole point is that one batch
    holds frames from several cameras, which is what makes the batched GPU
    call worth ~1.8x over per-frame calls.

    Why CameraFrameSlot cannot be reused for this, despite also holding
    frames: it is one continuously-overwritten slot per camera, so by the
    time a worker reads camera X's slot the producer may already have
    written a newer frame there — a batch assembled from those handles
    would silently mix frames from different instants. The batch that
    `_collect_frames` hands over holds arrays already copied *out* of those
    slots, so there is nothing for a per-camera handle to point at anyway.
    This slot takes its own copy of exactly the frames in one batch, and
    that copy is stable until the next batch replaces it.

    Single writer (the GPU loop thread), one batch in flight at a time,
    overwritten per batch — same discipline as RoiBatchSlot, which this
    otherwise mirrors (offset-packed, variable-size, capacity-reusing).
    """

    def __init__(self, loop_name: str):
        self.loop_name = loop_name
        self._name = _frame_batch_slot_name(loop_name)
        self._shm: Optional[shared_memory.SharedMemory] = None
        self._capacity: int = 0
        self._seq = 0
        with _ATTACHED_LOCK:
            _LOCAL_SLOTS[self._name] = self

    def _ensure_capacity(self, nbytes: int) -> None:
        if self._shm is not None and self._capacity >= nbytes:
            return
        self._release()
        alloc = max(nbytes, 1)  # SharedMemory requires size > 0
        try:
            # See CameraFrameSlot._ensure_capacity — same crashed-prior-run
            # cleanup, same close()-before-unlink() requirement.
            stale = shared_memory.SharedMemory(name=self._name)
            stale.close()
            stale.unlink()
        except FileNotFoundError:
            pass
        self._shm = shared_memory.SharedMemory(
            name=self._name, create=True, size=alloc
        )
        self._capacity = alloc
        logger.debug(
            f"FrameBatchSlot[{self.loop_name}]: allocated {alloc / 1024:.0f} KiB "
            f"as '{self._name}'"
        )

    def write(
        self, batch: Dict[int, Tuple[np.ndarray, int]]
    ) -> FrameBatchHandle:
        """Pack one cross-camera batch and return its handle.

        `batch` is exactly what `GPUInferenceWorker._collect_frames()`
        returns: `{camera_id: (frame, frame_num)}`. Iterated in sorted
        camera-id order so the packed order is deterministic and matches
        `_yolo_loop`'s existing `cam_ids = sorted(batch.keys())` — the
        results come back as a list positionally aligned to that order.

        An empty batch is legitimate (the loop simply had nothing ready) and
        produces an empty `frames` tuple; the reader branches on that
        without touching the block, matching `_run_yolo_batch([])` → `[]`.
        """
        cam_ids = sorted(batch.keys())
        total_bytes = sum(int(np.prod(batch[cid][0].shape)) for cid in cam_ids)
        self._ensure_capacity(total_bytes)
        shm = self._shm
        assert shm is not None  # _ensure_capacity always allocates or reuses

        frames: List[BatchedFrameHandle] = []
        offset = 0
        for cam_id in cam_ids:
            frame, frame_num = batch[cam_id]
            shape = frame.shape
            if len(shape) != 3:
                raise ValueError(f"expected an HxWxC frame, got shape {shape}")
            nbytes = int(np.prod(shape))
            view = np.ndarray(shape, dtype=np.uint8, buffer=shm.buf, offset=offset)
            view[:] = frame
            frames.append(
                BatchedFrameHandle(
                    camera_id=cam_id,
                    frame_num=frame_num,
                    offset=offset,
                    height=shape[0],
                    width=shape[1],
                    channels=shape[2],
                )
            )
            offset += nbytes

        self._seq += 1
        return FrameBatchHandle(seq=self._seq, frames=tuple(frames))

    def _release(self) -> None:
        # Same _ATTACHED_LOCK serialisation as the other slots, for the same
        # BufferError-on-reshape/close reason.
        with _ATTACHED_LOCK:
            if self._shm is not None:
                self._shm.close()
                try:
                    self._shm.unlink()
                except FileNotFoundError:
                    pass
                self._shm = None
                self._capacity = 0

    def close(self) -> None:
        """Release the slot. Call when the GPU loop stops."""
        with _ATTACHED_LOCK:
            if _LOCAL_SLOTS.get(self._name) is self:
                del _LOCAL_SLOTS[self._name]
        self._release()


def attach_and_read_frame_batch(
    loop_name: str, handle: FrameBatchHandle
) -> Optional[List[Tuple[int, int, np.ndarray]]]:
    """Worker side: map the batch block and copy out every frame it names, as
    `(camera_id, frame_num, frame)` triples in packed order.

    Returns `[]` for a legitimately empty batch, and `None` if the slot does
    not exist at all — same None-means-gone convention as the other readers
    here, so a caller can treat a vanished batch slot exactly like a vanished
    frame or ROI slot.

    `loop_name` is a separate argument rather than a handle field because the
    handle describes *what is in* the batch, while the slot name identifies
    *which producer* owns it — the worker knows which queue it consumes and
    therefore which loop's slot to read.
    """
    if not handle.frames:
        return []

    name = _frame_batch_slot_name(loop_name)
    needed = max(
        f.offset + int(np.prod((f.height, f.width, f.channels)))
        for f in handle.frames
    )

    with _ATTACHED_LOCK:
        local_slot = _LOCAL_SLOTS.get(name)
        if local_slot is not None:
            # Same-process fast path — see _LOCAL_SLOTS. Each copy is a
            # single expression so no view outlives its statement.
            local_shm = getattr(local_slot, "_shm", None)
            if local_shm is None or local_shm.size < needed:
                return None
            return [
                (
                    f.camera_id,
                    f.frame_num,
                    np.ndarray(
                        (f.height, f.width, f.channels),
                        dtype=np.uint8,
                        buffer=local_shm.buf,
                        offset=f.offset,
                    ).copy(),
                )
                for f in handle.frames
            ]

        shm = _ATTACHED.get(name)

        if shm is not None and shm.size < needed:
            del _ATTACHED[name]
            shm = None

        if shm is None:
            shm = _attach_fresh(name)
            if shm is None:
                return None
            _ATTACHED[name] = shm

        out: List[Tuple[int, int, np.ndarray]] = []
        for f in handle.frames:
            view = np.ndarray(
                (f.height, f.width, f.channels),
                dtype=np.uint8,
                buffer=shm.buf,
                offset=f.offset,
            )
            out.append((f.camera_id, f.frame_num, view.copy()))

    if not shared_memory_exists(name):
        # Same post-copy re-validation as the other readers: a producer that
        # reallocated or closed the slot mid-read leaves the copies above
        # reading garbage or a torn batch.
        with _ATTACHED_LOCK:
            _ATTACHED.pop(name, None)
        return None

    return out
