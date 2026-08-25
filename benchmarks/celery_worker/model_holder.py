"""Shared-model machinery — the point of the whole spike.

One YOLO ``nn.Module`` lives per process. Each thread gets its own
``DetectionPredictor`` bound to that same module, so N threads cost N sets of
pre/post-processing state but only ONE copy of the weights on the GPU.

Why per-thread predictors instead of a shared ``YOLO`` object: ultralytics'
``Model.predict()`` rebuilds ``self.predictor.args`` via ``get_cfg()`` on every
call, and ``BasePredictor.stream_inference`` wraps its whole body in
``self._lock``. A shared ``YOLO`` is therefore already fully serialised, and the
lock convoy on top makes it slower than a single thread. Binding a fresh
predictor to an already-loaded module avoids both: ``setup_model`` ->
``AutoBackend`` -> ``PyTorchBackend.load_model`` takes the
``isinstance(weight, nn.Module)`` branch, whose ``weight.to(device)`` is an
in-place no-op on a module already resident on that device. No copy.

Nothing here imports torch/ultralytics/lum_vision at module scope.
"""

import os
import threading

CONF = float(os.environ.get("YOLOBENCH_CONF", "0.45"))
IOU = float(os.environ.get("YOLOBENCH_IOU", "0.45"))
SHARE = os.environ.get("YOLOBENCH_SHARE", "per_thread")

_MODEL = None      # DetectionModel nn.Module, shared by every thread in this process
_YOLO = None       # raw ultralytics YOLO wrapper, used only by the naive_shared arm
_TLS = threading.local()
_LOAD_LOCK = threading.Lock()
# Predictor construction must be serialised even though inference is not.
# setup_model() -> AutoBackend -> PyTorchBackend.load_model() calls
# BaseModel.fuse(), whose is_fused() guard makes a *sequential* second call a
# no-op but is not atomic: two threads both see an unfused module, both enter
# fuse(), and the loser dies on `delattr(m, "bn")`. Observed, not theoretical.
_PRED_LOCK = threading.Lock()
_naive_warm = False


def ensure_loaded():
    """Load the model once per process. Idempotent, thread-safe.

    Called from ``worker_process_init`` (which fires post-fork, so the prefork
    children each load their own) and again at the top of every task, because
    the solo and threads pools do not emit that signal.
    """
    global _MODEL, _YOLO
    if _MODEL is not None:
        return
    with _LOAD_LOCK:
        if _MODEL is not None:
            return
        from lum_vision.person_tracking.detector import PersonDetector

        det = PersonDetector(
            model_size=os.environ.get("YOLOBENCH_MODEL_SIZE", "s"),
            model_version=os.environ.get("YOLOBENCH_MODEL_VERSION", "yolo26"),
            confidence_threshold=CONF,
            iou_threshold=IOU,
            # Explicit device is mandatory: PersonDetector's default path calls
            # torch.cuda.is_available(), which initialises CUDA in whichever
            # process reaches it first and poisons fork() for the prefork pool.
            device=os.environ.get("YOLOBENCH_DEVICE", "cuda"),
            # Absolute path is mandatory: src/config/vision.py uses a
            # CWD-relative Path("volumes/models"), which a subprocess would
            # resolve somewhere else.
            weights_dir=os.environ["YOLOBENCH_WEIGHTS"],
        )
        _YOLO = det.model
        _MODEL = det.model.model


def get_predictor():
    """One predictor per thread, all bound to the single shared module."""
    p = getattr(_TLS, "predictor", None)
    if p is not None:
        return p

    import numpy as np
    from ultralytics.models.yolo.detect import DetectionPredictor

    with _PRED_LOCK:
        p = DetectionPredictor(overrides=dict(
            conf=CONF, iou=IOU, device=os.environ.get("YOLOBENCH_DEVICE", "cuda"),
            batch=1, save=False, mode="predict", rect=True, embed=None,
            verbose=False, task="detect",
        ))
        p.setup_model(model=_MODEL, verbose=False)
        # Burn cuDNN autotune on this thread so the first real task is not an
        # outlier. Inside the lock so autotune does not race either.
        p(source=[np.zeros((640, 640, 3), dtype=np.uint8)], stream=False)

    _TLS.predictor = p
    return p


def infer(frames):
    """Run detection on a list of BGR frames. Returns ultralytics Results."""
    if SHARE == "naive_shared":
        # The naive arm builds its predictor lazily inside Model.predict(), so
        # its FIRST call hits the same non-atomic fuse() race. Serialise only
        # that first call; every later call stays unguarded, which is the whole
        # point of this arm — it measures ultralytics' own internal lock convoy.
        global _naive_warm
        if not _naive_warm:
            with _PRED_LOCK:
                if not _naive_warm:
                    _YOLO(frames, conf=CONF, iou=IOU, verbose=False,
                          device=os.environ.get("YOLOBENCH_DEVICE", "cuda"))
                    _naive_warm = True
        return _YOLO(frames, conf=CONF, iou=IOU, verbose=False,
                     device=os.environ.get("YOLOBENCH_DEVICE", "cuda"))
    return get_predictor()(source=frames, stream=False)


def identity():
    """Who am I — used to prove threads share one module and one process."""
    return {
        "pid": os.getpid(),
        "tid": threading.get_ident(),
        "module_id": id(_MODEL),
        "predictor_id": id(getattr(_TLS, "predictor", None)),
        "share": SHARE,
    }
