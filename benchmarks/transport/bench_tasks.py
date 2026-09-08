"""Worker-side half of the transport comparison.

Every task does the same thing: note when it arrived, turn whatever it was
given into a real (720, 1280, 3) uint8 array, and report what that cost. The
only difference between cells is *how the pixels got here* — which is the
entire question.

CRITICAL: no cv2/numpy-heavy import at module scope beyond numpy itself. The
parent process imports this module to register tasks; keeping heavy imports in
function bodies matches celery_worker/bench_tasks.py:1-8 and keeps worker
start-up ~2s instead of ~30s.

On timing: `t_recv` is time.time(), not perf_counter, because it is compared
against a timestamp taken in the *driver* process. Both processes are on one
host and read the same kernel CLOCK_REALTIME, so the subtraction is valid with
no calibration. This breaks the moment the worker runs in a container — see the
locality assertion in run_bench.py.
"""

import base64
import os
import threading
import time

import numpy as np

from transport.bench_app import app

SHAPE = (720, 1280, 3)
_ATTACHED = {}  # name -> (SharedMemory, ndarray), cached per worker process


def _checksum(arr):
    """Cheap content check. Not a hash — a hash of 2.64 MiB would cost more
    than the transport we are trying to measure. Summing a strided sample
    catches truncation, wrong shape and byte-order mangling, which is what we
    need to prove the payload actually arrived intact."""
    return int(arr[::64, ::64, :].sum())


def _attach(name, shape):
    """Attach to the driver's block, cached per process."""
    from multiprocessing import shared_memory

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


def _result(t_recv, decode_ms, arr, nbytes):
    return {
        "t_recv": t_recv,
        "decode_ms": decode_ms,
        "nbytes": nbytes,
        "shape_ok": tuple(arr.shape) == SHAPE,
        "checksum": _checksum(arr),
        "pid": os.getpid(),
        "tid": threading.get_ident(),
    }


@app.task(name="xport.probe")
def probe():
    """Round-trip smoke test; also proves the worker is alive before timing."""
    return {"pid": os.getpid(), "t_recv": time.time()}


@app.task(name="xport.shm")
def shm(name, idx, shape, t_sent, copy=False):
    """Pixels via shared memory — the design under defence.

    The payload is a name plus two ints (~100 B). `copy=False` takes a
    zero-copy view, which is what production does (src/workers/frame_store.py).
    `copy=True` materialises an owned array, shm's honest worst case for a
    consumer that cannot hold a view.
    """
    t_recv = time.time()
    t0 = time.perf_counter()
    store = _attach(name, shape)
    arr = np.array(store[idx]) if copy else store[idx]
    t1 = time.perf_counter()
    # nbytes is what crossed the BROKER, not what the frame weighs: for this
    # cell that is the handle (a name plus two ints), which is the whole point.
    wire = len(name) + 32
    return _result(t_recv, (t1 - t0) * 1e3, arr, wire)


@app.task(name="xport.b64")
def b64(payload, t_sent):
    """Pixels as a base64 string through the json serializer.

    This is the rejected design measured under production's real serializer
    (src/workers/celery_app.py:75 pins task_serializer='json', and json cannot
    hold bytes). kombu has already json-decoded the body by the time this runs,
    so decode_ms covers only b64 -> ndarray; the json half is attributed by the
    kombu microbench in run_bench.py.
    """
    t_recv = time.time()
    t0 = time.perf_counter()
    raw = base64.b64decode(payload)
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(SHAPE)
    t1 = time.perf_counter()
    return _result(t_recv, (t1 - t0) * 1e3, arr, len(raw))


@app.task(name="xport.raw")
def raw(payload, t_sent):
    """Pixels as raw bytes through the pickle serializer.

    Steelman: the cheapest way to get pixels across a broker, skipping base64
    entirely. If shm still wins here, the argument does not rest on json being
    a poor choice.
    """
    t_recv = time.time()
    t0 = time.perf_counter()
    arr = np.frombuffer(payload, dtype=np.uint8).reshape(SHAPE)
    t1 = time.perf_counter()
    return _result(t_recv, (t1 - t0) * 1e3, arr, len(payload))


@app.task(name="xport.jpeg")
def jpeg(payload, t_sent):
    """Pixels as a JPEG through the pickle serializer.

    Steelman: the smallest payload (~806 KiB vs 2.64 MiB), and the option a
    reviewer will raise. It trades wire bytes for codec CPU, and imdecode is
    the expensive half — which lands in decode_ms where it can be seen.
    """
    import cv2

    t_recv = time.time()
    t0 = time.perf_counter()
    arr = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    t1 = time.perf_counter()
    return _result(t_recv, (t1 - t0) * 1e3, arr, len(payload))


@app.task(name="xport.control")
def control(payload, t_sent):
    """A ~100-byte payload that decodes to nothing.

    The floor: everything Celery costs per task regardless of payload. Every
    pixel number is reported as a delta over this, which is what "the cost of
    sending a frame" actually means.
    """
    t_recv = time.time()
    return {
        "t_recv": t_recv,
        "decode_ms": 0.0,
        "nbytes": len(payload),
        "shape_ok": True,
        "checksum": 0,
        "pid": os.getpid(),
        "tid": threading.get_ident(),
    }
