"""YOLO person detection as a Celery task (LSO-67 Stage 2).

The `_run_yolo_batch` + `_parse_yolo_result` pair moved here from
`pipeline/gpu_worker.py` essentially verbatim; `main.py` now dispatches to
this task instead of calling the model in-process. What did *not* move is
everything around them — the per-camera queues, the cross-camera batch
collector, and LSO-138's request/response correlation all stay in
`GPUInferenceWorker`, which still owns the batching that makes a batched GPU
call worth ~1.8x over per-frame calls.

Frames arrive as a `FrameBatchHandle` (shared memory), never as pixels: a
720p frame costs ~15.6 ms through a broker versus ~0.3 ms as a handle, more
than the inference itself. Results go back through the broker directly —
they are already plain dicts of scalars and lists (`_parse_yolo_result`
converts ultralytics `Results` before anything leaves the function), so
unlike the face results they carry no numpy at all.

No torch/ultralytics import at module scope — the parent imports this module
to register the task, and a CUDA touch there poisons `fork()`.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

from celery.signals import worker_process_init
from loguru import logger

from workers import model_holder
from workers.celery_app import celery


@worker_process_init.connect
def _load_model_in_child(**_kwargs):
    """Fires post-fork in each prefork child. The solo and threads pools do
    not emit this, which is why the task body also calls ensure_*_loaded()."""
    model_holder.ensure_person_detector_loaded()


def _parse_yolo_result(result) -> List[Dict]:
    """Convert a YOLO result object to a list of detection dicts.

    Moved verbatim from GPUInferenceWorker._parse_yolo_result. This is what
    keeps ultralytics' `Results` object from ever crossing the broker — the
    output is plain dicts/lists/floats.
    """
    detections = []
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return detections
    for idx in range(len(boxes)):
        cls_id = int(boxes.cls[idx].cpu().numpy())
        if cls_id != 0:
            continue
        bbox = boxes.xyxy[idx].cpu().numpy().tolist()
        conf = float(boxes.conf[idx].cpu().numpy())
        detections.append(
            {
                "bbox": bbox,
                "confidence": conf,
                "keypoints": None,
                "person_id": idx,
            }
        )
    return detections


@celery.task(name="yolo.detect_batch", queue="yolo")
def detect_batch_task(handle: Dict[str, Any]) -> List[List[Dict]]:
    """Run YOLO over one cross-camera batch.

    `handle` is a `FrameBatchHandle` flattened to a plain dict — task
    payloads stay JSON-serialised (see celery_app.py), so the dataclass is
    reconstructed here, the same boundary conversion Stage 1 established for
    `FrameHandle`.

    Returns detections **positionally aligned to `handle.frames`**, which
    `FrameBatchSlot.write` packs in sorted-camera-id order. That alignment is
    the contract `GPUInferenceWorker._yolo_loop` relies on to route each
    result back to the camera that submitted it — it must not be reordered.

    On any failure this returns one empty detection list per frame rather
    than raising: a camera missing detections for one cycle ages its tracks
    by a frame, which the tracker already tolerates, whereas an exception
    would surface as a task failure and leave the caller waiting out its
    timeout for nothing.
    """
    from workers.frame_store import BatchedFrameHandle, FrameBatchHandle

    batch_handle = FrameBatchHandle(
        seq=handle["seq"],
        frames=tuple(BatchedFrameHandle(**f) for f in handle["frames"]),
    )

    n_frames = len(batch_handle.frames)
    if n_frames == 0:
        return []

    detector = model_holder.ensure_person_detector_loaded()

    from workers.frame_store import attach_and_read_frame_batch

    packed = attach_and_read_frame_batch("yolo", batch_handle)
    if packed is None:
        # The producer's slot vanished (loop stopped, or reallocated
        # mid-flight). Same "nothing to do" answer the in-process path gives
        # when a camera is no longer registered.
        logger.warning(
            f"yolo.detect_batch: frame batch seq={batch_handle.seq} is gone "
            f"— returning {n_frames} empty results"
        )
        return [[] for _ in range(n_frames)]

    frames = [frame for _cam_id, _frame_num, frame in packed]

    try:
        t0 = time.time()
        results = detector.model(
            frames,
            conf=detector.confidence_threshold,
            iou=detector.iou_threshold,
            verbose=False,
            device=detector.device,
        )
        duration_ms = (time.time() - t0) * 1000
        logger.debug(
            f"yolo.detect_batch: seq={batch_handle.seq} batch={len(frames)} "
            f"took {duration_ms:.1f}ms"
        )
        return [_parse_yolo_result(r) for r in results]
    except Exception as e:
        logger.exception(f"YOLO batch inference failed: {e}")
        return [[] for _ in frames]
