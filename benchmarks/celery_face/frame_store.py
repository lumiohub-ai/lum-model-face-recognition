""".
Shared-memory frame store — identical mechanism to
benchmarks/celery_worker/frame_store.py, copied rather than imported so this
benchmark stays fully independent of celery_worker's package.

The `path` mode makes every task re-decode a JPEG from disk. Here the driver
decodes the corpus ONCE into one shared block. A task payload carries only the
block name and a few frame indices, so the broker moves a handle instead of
pixels. Workers attach once per process and take a zero-copy numpy view.

Used for two distinct corpora here (unlike the YOLO benchmark's single one):
full frames for detect_and_align, and pre-aligned 112x112 crops for
embed_batch — see build_corpus()/build_crop_corpus() in run_bench.py.
"""

import numpy as np
from multiprocessing import shared_memory

_ATTACHED = {}  # name -> (SharedMemory, ndarray). Cached per worker process.


def create(frames, name="facebench_frames"):
    """Decode `frames` (paths) into one shared block. Driver side."""
    import cv2

    first = cv2.imread(frames[0])
    shape = (len(frames),) + first.shape
    nbytes = int(np.prod(shape))

    try:  # a previous crashed run may have left the block behind
        shared_memory.SharedMemory(name=name).unlink()
    except FileNotFoundError:
        pass

    shm = shared_memory.SharedMemory(name=name, create=True, size=nbytes)
    arr = np.ndarray(shape, dtype=np.uint8, buffer=shm.buf)
    for i, p in enumerate(frames):
        arr[i] = cv2.imread(p)
    return shm, shape


def create_from_arrays(crops, name="facebench_crops"):
    """Same as create(), but from in-memory arrays rather than file paths.

    Used for the aligned-crop corpus: crops come from running
    detect_and_align once offline, not from disk.
    """
    shape = (len(crops),) + crops[0].shape
    nbytes = int(np.prod(shape))

    try:
        shared_memory.SharedMemory(name=name).unlink()
    except FileNotFoundError:
        pass

    shm = shared_memory.SharedMemory(name=name, create=True, size=nbytes)
    arr = np.ndarray(shape, dtype=np.uint8, buffer=shm.buf)
    for i, c in enumerate(crops):
        arr[i] = c
    return shm, shape


def attach(name, shape):
    """Attach read-only and return a view. Worker side, cached per process."""
    hit = _ATTACHED.get(name)
    if hit is not None:
        return hit[1]

    shm = shared_memory.SharedMemory(name=name)
    # The driver owns this block. Without unregistering, each worker's
    # resource_tracker would try to unlink it at shutdown and spam warnings.
    try:
        from multiprocessing import resource_tracker
        resource_tracker.unregister(shm._name, "shared_memory")
    except Exception:
        pass

    arr = np.ndarray(tuple(shape), dtype=np.uint8, buffer=shm.buf)
    _ATTACHED[name] = (shm, arr)
    return arr
