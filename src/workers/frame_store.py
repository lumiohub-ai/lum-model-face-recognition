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
from typing import Dict, Optional, Tuple

import numpy as np
from loguru import logger

_SLOT_NAME_PREFIX = "camframe"

# Worker-side cache: block name -> attached SharedMemory. One entry per
# camera slot this process has ever read from. Never closed proactively —
# a worker process's slots live as long as the process does.
_ATTACHED: Dict[str, shared_memory.SharedMemory] = {}
_ATTACHED_LOCK = threading.Lock()


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
