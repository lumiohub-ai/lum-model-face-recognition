"""Shared-model machinery for the face-detection benchmark.

One FaceDetector (InsightFace FaceAnalysis, ONNXRuntime-backed) lives per
process, shared by every thread in that process — no per-thread predictor
object, unlike benchmarks/celery_worker/model_holder.py's YOLO design.

Two things were verified before writing this file (see
/tmp/.../step0_concurrency_check.py, run 2026-08-25):

1. One shared FaceDetector is safe under concurrent threads: detect_and_align
   and embed_batch gave identical results (face counts; embedding cosine
   similarity 1.000000) across 4 concurrent threads vs. a single-threaded
   reference, and throughput scaled 2.07x at 4 threads vs 1 — so, unlike
   YOLO's ultralytics wrapper, there's no internal lock forcing serialization
   and no fuse()-style race to guard against. A single module-level instance,
   loaded once per process, is enough.

2. SCRFD (the detector inside buffalo_l — already the model in use, not
   RetinaFace) cannot batch. Its ONNX graph (det_10g.onnx) declares a FIXED
   batch dimension of 1 on the input (not a dynamic axis — only H/W are
   dynamic), and its outputs are 2D ([N_anchors, C]), not 3D with a leading
   batch dim. Forcing a batch=2 input throws
   onnxruntime.InvalidArgument ("Got: 2 Expected: 1"). This is a model-file
   limitation, not a wrapper-loop limitation: batching it for real (LSO-118)
   would mean re-exporting the ONNX graph with a dynamic batch axis, not just
   writing a batched call path around the existing one. detect_and_align
   stays batch=1 in every cell below for that reason.

embed_batch's recognition model (w600k_r50.onnx) DOES have a dynamic batch
axis, which is why it already batches (LSO-117) and why the embed cells here
sweep batch size.

Nothing here imports lum_vision at module scope — see bench_tasks.py for why.
"""

import os
import threading

_DETECTOR = None  # FaceDetector, shared by every thread in this process
_LOAD_LOCK = threading.Lock()


def ensure_loaded():
    """Load the model once per process. Idempotent, thread-safe.

    Called from ``worker_process_init`` (fires post-fork, so prefork children
    each load their own) and again at the top of every task, because the solo
    and threads pools do not emit that signal.
    """
    global _DETECTOR
    if _DETECTOR is not None:
        return _DETECTOR
    with _LOAD_LOCK:
        if _DETECTOR is not None:
            return _DETECTOR
        from lum_vision.face_detection.detector import FaceDetector

        _DETECTOR = FaceDetector(
            gpu_id=int(os.environ.get("FACEBENCH_GPU_ID", "0")),
            model_name=os.environ.get("FACEBENCH_MODEL_NAME", "buffalo_l"),
            padding_percent=0.0,  # matches the offline notebook's setting
            # Absolute path is mandatory: a subprocess started with a
            # different CWD would otherwise resolve a relative model_root
            # somewhere else.
            model_root=os.environ.get("FACEBENCH_MODEL_ROOT") or None,
        )
    return _DETECTOR


def detect_and_align(frame):
    """Detect + align faces in one frame. No batch path exists (see module
    docstring) — always one frame per call."""
    detector = ensure_loaded()
    return detector.detect_and_align(frame)


def embed_batch(crops):
    """Embed a batch of aligned 112x112 crops in one GPU/ORT call."""
    detector = ensure_loaded()
    return detector.embed_batch(crops)


def identity():
    """Who am I — used to prove threads in one process share one detector."""
    return {
        "pid": os.getpid(),
        "tid": threading.get_ident(),
        "detector_id": id(_DETECTOR),
    }
