"""Celery tasks for the face-detection benchmark.

CRITICAL: no ``lum_vision``/``insightface``/``onnxruntime`` import at module
scope. The parent process imports this module to register tasks, and a
premature GPU/CUDA touch there poisons ``fork()`` for the prefork pool. Every
heavy import lives inside a function body — same rule as
benchmarks/celery_worker/bench_tasks.py.
"""

import os
import threading
import time

from celery.signals import worker_process_init

from celery_face.bench_app import app


@worker_process_init.connect
def _load_model_in_child(**_kwargs):
    """Fires post-fork in each prefork child. Solo/threads pools do not emit
    this, which is why every task also calls ``ensure_loaded()``."""
    from celery_face import model_holder
    model_holder.ensure_loaded()


@app.task(name="facebench.probe")
def probe():
    """Round-trip smoke test — proves the broker path works, touches no GPU."""
    return {"pid": os.getpid(), "tid": threading.get_ident()}


@app.task(name="facebench.warmup")
def warmup(path, deadline):
    """Build/load this slot's detector, then hold the slot until ``deadline``.

    Holding matters: with prefetch_multiplier=1 a blocked slot cannot take
    another warmup task, so a group of `concurrency` warmups is forced to land
    one-per-slot. Without the hold, one fast thread drains the whole group and
    the rest are still cold when the clock starts.
    """
    import cv2
    from celery_face import model_holder

    model_holder.ensure_loaded()
    frame = cv2.imread(path)
    for _ in range(5):
        model_holder.detect_and_align(frame)
    time.sleep(max(0.0, deadline - time.time()))
    return model_holder.identity()


@app.task(name="facebench.detect_shm")
def detect_shm(shm_name, idxs, shape):
    """detect_and_align on frames read from shared memory. Always one frame
    per call — SCRFD has no batch path (see model_holder.py)."""
    from celery_face import frame_store, model_holder

    model_holder.ensure_loaded()

    t0 = time.perf_counter()
    store = frame_store.attach(shm_name, shape)
    frames = [store[i] for i in idxs]
    t1 = time.perf_counter()
    results = [model_holder.detect_and_align(f) for f in frames]
    t2 = time.perf_counter()

    return {
        "n": len(idxs),
        "read_ms": (t1 - t0) * 1e3,
        "infer_ms": (t2 - t1) * 1e3,
        "nfaces": sum(len(r) for r in results),
        "pid": os.getpid(),
        "tid": threading.get_ident(),
    }


@app.task(name="facebench.detect")
def detect(paths):
    """detect_and_align on frames read from local disk."""
    import cv2
    from celery_face import model_holder

    model_holder.ensure_loaded()

    t0 = time.perf_counter()
    frames = [cv2.imread(p) for p in paths]
    t1 = time.perf_counter()
    results = [model_holder.detect_and_align(f) for f in frames]
    t2 = time.perf_counter()

    return {
        "n": len(paths),
        "read_ms": (t1 - t0) * 1e3,
        "infer_ms": (t2 - t1) * 1e3,
        "nfaces": sum(len(r) for r in results),
        "pid": os.getpid(),
        "tid": threading.get_ident(),
    }


@app.task(name="facebench.embed_shm")
def embed_shm(shm_name, idxs, shape):
    """embed_batch on a batch of pre-aligned crops from shared memory.

    idxs already has exactly `batch` entries — batching is real here (LSO-117),
    so unlike detect_shm this is genuinely one GPU call for the whole list.
    """
    from celery_face import frame_store, model_holder

    model_holder.ensure_loaded()

    t0 = time.perf_counter()
    store = frame_store.attach(shm_name, shape)
    crops = [store[i] for i in idxs]
    t1 = time.perf_counter()
    embeddings = model_holder.embed_batch(crops)
    t2 = time.perf_counter()

    return {
        "n": len(idxs),
        "read_ms": (t1 - t0) * 1e3,
        "infer_ms": (t2 - t1) * 1e3,
        "nfaces": int(embeddings.shape[0]),
        "pid": os.getpid(),
        "tid": threading.get_ident(),
    }
