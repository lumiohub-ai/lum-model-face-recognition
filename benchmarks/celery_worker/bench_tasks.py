"""Celery tasks for the spike.

CRITICAL: no ``torch`` / ``ultralytics`` / ``lum_vision`` import at module
scope. The parent process imports this module to register tasks, and any CUDA
touch there poisons ``fork()`` for the prefork pool. Every heavy import lives
inside a function body.
"""

import os
import threading
import time

from celery.signals import worker_process_init

from celery_worker.bench_app import app


@worker_process_init.connect
def _load_model_in_child(**_kwargs):
    """Fires post-fork in each prefork child. Solo/threads pools do not emit
    this, which is why every task also calls ``ensure_loaded()``."""
    from celery_worker import model_holder
    model_holder.ensure_loaded()


@app.task(name="yolobench.probe")
def probe():
    """Round-trip smoke test — proves the broker path works, touches no GPU."""
    return {"pid": os.getpid(), "tid": threading.get_ident()}


@app.task(name="yolobench.warmup")
def warmup(path, deadline):
    """Build this slot's predictor, then hold the slot until ``deadline``.

    Holding matters: with prefetch_multiplier=1 a blocked slot cannot take
    another warmup task, so a group of `concurrency` warmups is forced to land
    one-per-slot. Without the hold, one fast thread drains the whole group and
    the rest are still cold when the clock starts.
    """
    import cv2
    from celery_worker import model_holder

    model_holder.ensure_loaded()
    frame = cv2.imread(path)
    for _ in range(5):
        model_holder.infer([frame])
    time.sleep(max(0.0, deadline - time.time()))
    return model_holder.identity()


@app.task(name="yolobench.infer_shm")
def infer_shm(shm_name, idxs, shape):
    """Same work as `infer`, but frames come from shared memory.

    The payload is a name plus a few ints, so the broker carries a handle
    rather than pixels — and there is no JPEG decode in the task at all.
    """
    from celery_worker import frame_store, model_holder

    model_holder.ensure_loaded()

    t0 = time.perf_counter()
    store = frame_store.attach(shm_name, shape)
    frames = [store[i] for i in idxs]
    t1 = time.perf_counter()
    results = model_holder.infer(frames)
    t2 = time.perf_counter()

    speed = results[0].speed
    return {
        "n": len(idxs),
        "read_ms": (t1 - t0) * 1e3,
        "infer_ms": (t2 - t1) * 1e3,
        "pre_ms": speed["preprocess"],
        "inf_ms": speed["inference"],
        "post_ms": speed["postprocess"],
        "ndet": sum(len(r.boxes) for r in results),
        "pid": os.getpid(),
        "tid": threading.get_ident(),
    }


@app.task(name="yolobench.infer")
def infer(paths):
    """Detect on a batch of frames read from local disk."""
    import cv2
    from celery_worker import model_holder

    model_holder.ensure_loaded()

    t0 = time.perf_counter()
    frames = [cv2.imread(p) for p in paths]
    t1 = time.perf_counter()
    results = model_holder.infer(frames)
    t2 = time.perf_counter()

    speed = results[0].speed
    return {
        "n": len(paths),
        "read_ms": (t1 - t0) * 1e3,
        "infer_ms": (t2 - t1) * 1e3,
        "pre_ms": speed["preprocess"],
        "inf_ms": speed["inference"],
        "post_ms": speed["postprocess"],
        "ndet": sum(len(r.boxes) for r in results),
        "pid": os.getpid(),
        "tid": threading.get_ident(),
    }
