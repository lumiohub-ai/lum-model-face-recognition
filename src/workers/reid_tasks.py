"""Body ReID embedding extraction as a Celery task.

Wraps `lum_vision.BodyReidExtractor` (person_tracking/reid.py) the same way
`face_tasks.py` wraps `FaceDetector`: model loading lives in model_holder.py,
this module only owns the task boundary and the shared-memory transport.

Called from `RemoteGlobalTrackManager._run_assign`
(workers/global_track_adapter.py) — the background thread Step 1 introduced
for assign_global_id — via `workers.reid_client.ReidExtractClient`. Replaces
the in-process `_extract_body_embedding` call that used to run inside the
same process as GlobalTrackManager, back when the register and the ReID
model shared one container. See docs/GLOBAL_TRACKING.md's "final long-term
solution" section for why they're split.

Crops travel by shared memory (`workers.frame_store.RoiBatchSlot`,
`purpose="reid"`), not pickled through the broker — same reasoning as
face_tasks.py: frame_store.py's own docstring measured ~15.6ms for a 720p
frame through Redis, and a person crop is real pixel data of the same
category, not a small control message. One call carries exactly one crop
(RemoteGlobalTrackManager submits one assign_global_id per track, not a
per-frame batch — see global_track_adapter.py's docstring on why), so the
batch here is always length 1; RoiBatchSlot/RoiBatchHandle are reused as-is
rather than building a second single-crop primitive, since they already
handle any batch length including 1.

No boxmot/torch import at module scope — the parent imports this module to
register the task, and a CUDA touch there poisons `fork()`, same constraint
as face_tasks.py and model_holder.py.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from celery.signals import worker_process_init
from loguru import logger

from workers import model_holder
from workers.celery_app import celery


@worker_process_init.connect
def _load_model_in_child(**_kwargs):
    """Preload the ReID model post-fork, but ONLY in a worker that actually
    serves this queue — see the matching guard in yolo_tasks.py/face_tasks.py
    for why the check is needed and why the worker has to declare it via an
    env var rather than the handler inferring it."""
    import os

    if "reid" in os.environ.get("SO_WORKER_PRELOAD", "").split(","):
        model_holder.ensure_reid_extractor_loaded()


@celery.task(name="reid.extract_batch", queue="reid", track_started=False)
def extract_batch_task(handle: Dict[str, Any]) -> List[Optional[List[float]]]:
    """Extract a body ReID embedding for each crop in a `RoiBatchHandle`.

    `handle` is a `RoiBatchHandle` flattened to a plain dict (task payloads
    stay JSON — see celery_app.py), reconstructed here the same way
    `face_tasks.embed_batch_task` does for its own handle.

    Returns one embedding per crop, **positionally aligned to `handle.rois`**
    — `None` for a crop that failed extraction, never a shorter or reordered
    list; a caller matching results back to tracks by position depends on
    this exactly the same way face_tasks.py's callers do. Each embedding is
    returned as a plain list of floats, not a numpy array — task results are
    pickled here (result_serializer='pickle', celery_app.py), which *can*
    carry numpy, but a plain list keeps this result identical whether it
    crosses the broker or is read in-process, and avoids a caller needing to
    know that reid.extract_batch specifically returns arrays where every
    other task field is a scalar.

    Never raises: on failure every crop gets None, matching
    BodyReidExtractor.extract's own "no embedding available" contract
    (person_tracking/reid.py) — a blank result degrades assign_global_id
    exactly the way a missing crop already does (falls through to "create a
    new global track"), not a task failure the caller has to handle
    specially.
    """
    from workers.frame_store import (
        RoiBatchHandle,
        RoiHandle,
        attach_and_read_roi_batch,
    )

    batch_handle = RoiBatchHandle(
        camera_id=handle["camera_id"],
        seq=handle["seq"],
        segment=handle["segment"],
        instance_id=handle["instance_id"],
        rois=tuple(RoiHandle(**r) for r in handle["rois"]),
        purpose=handle.get("purpose", "reid"),
    )

    n_rois = len(batch_handle.rois)
    if n_rois == 0:
        return []

    packed = attach_and_read_roi_batch(batch_handle)
    if packed is None:
        logger.warning(
            f"reid.extract_batch: ROI batch seq={batch_handle.seq} is gone — "
            f"returning {n_rois} blank results"
        )
        return [None] * n_rois

    extractor = model_holder.ensure_reid_extractor_loaded()

    results: List[Optional[List[float]]] = []
    for _track_id, crop in packed:
        try:
            embedding = extractor.extract(crop)
        except Exception as e:
            logger.warning(f"reid.extract_batch: extraction failed for one crop: {e}")
            embedding = None
        results.append(embedding.tolist() if embedding is not None else None)

    logger.debug(
        f"reid.extract_batch: seq={batch_handle.seq} crops={n_rois} "
        f"extracted={sum(1 for r in results if r is not None)}"
    )
    return results
