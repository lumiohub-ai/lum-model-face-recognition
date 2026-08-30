"""SCRFD detection + ArcFace embedding as a Celery task.

`_run_arcface_batch` moved here from `pipeline/gpu_worker.py` essentially
verbatim — including, deliberately, its one-crop-per-call embedding loop.
That loop is not an oversight to tidy up: batching all crops into a single
`embed_batch` call shipped as 0.3.0 and **halved production FPS**, because
ORT's CUDA EP re-plans on every input-shape change (~80ms at a varying batch
size versus ~2.6ms at a fixed one). The comment marking that is carried
across with the code.

What stays in `GPUInferenceWorker`: the per-camera queues, the cross-camera
ROI collection, the request/response correlation, and the scatter of
results back to the camera that submitted each crop.

Crops arrive as a `RoiBatchHandle` (shared memory) rather than pixels
through the broker. Results come back through the broker as pickled dicts —
they carry numpy in three places (`embedding`, `face_image`, and
`face_bbox`, whose elements are numpy.int64), which is why the app's
`result_serializer` is pickle; see celery_app.py.

No insightface/onnxruntime import at module scope — the parent imports this
module to register the task, and a CUDA touch there poisons `fork()`.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from celery.signals import worker_process_init
from loguru import logger

from workers import model_holder
from workers.celery_app import celery


@worker_process_init.connect
def _load_model_in_child(**_kwargs):
    """Preload the face models post-fork, but ONLY in a worker that actually
    serves this queue — see the matching guard in yolo_tasks.py for why the
    check is needed and why the worker has to declare it via an env var
    rather than the handler inferring it."""
    import os

    if "face" in os.environ.get("SO_WORKER_PRELOAD", "").split(","):
        model_holder.ensure_face_detector_loaded()


@celery.task(name="face.embed_batch", queue="face")
def embed_batch_task(handle: Dict[str, Any]) -> List[Dict]:
    """Detect + embed faces for one flat, cross-camera batch of person ROIs.

    `handle` is a `RoiBatchHandle` flattened to a plain dict (task payloads
    stay JSON — see celery_app.py), reconstructed here the same way
    `camera_tasks` does for `FrameHandle`.

    Returns one result dict per ROI, **positionally aligned to the ROIs in
    the handle**. `_arcface_loop` scatters those back to cameras by index, so
    this ordering is a hard contract — a reordered or short list would hand
    one person's embedding to another person's track.

    Never raises: on failure every ROI gets the default "no face" dict, which
    is exactly what the in-process version produced for an ROI whose
    detection failed. Downstream treats `face_detected` as "has a usable
    embedding", so a blank result degrades to "recognition didn't advance
    this cycle" rather than corrupting identity.
    """
    from lum_vision.face_detection import frontality, pitch

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
    )

    n_rois = len(batch_handle.rois)
    if n_rois == 0:
        return []

    def _blank_results() -> List[Dict]:
        return [
            {
                "embedding": None,
                "face_image": None,
                "face_detected": False,
                "det_score": 0.0,
            }
            for _ in range(n_rois)
        ]

    packed = attach_and_read_roi_batch(batch_handle)
    if packed is None:
        logger.warning(
            f"face.embed_batch: ROI batch seq={batch_handle.seq} is gone — "
            f"returning {n_rois} blank results"
        )
        return _blank_results()

    person_rois = [crop for _tid, crop in packed]
    face_detector = model_holder.ensure_face_detector_loaded()

    try:
        t0 = time.time()
        results: List[Dict] = []
        # (result_idx, face) for every ROI that had >=1 detected face with
        # landmarks; parallel to crops so embed_batch()'s output lines up.
        faces_to_embed: List[Any] = []
        crops: List[Any] = []

        # Timed separately from embedding below: detection is still N
        # per-ROI calls (SCRFD has no batch path), so this number won't
        # move with batch size the way embedding does.
        det_t0 = time.time()
        for i, roi in enumerate(person_rois):
            result: Dict = {
                "embedding": None,
                "face_image": None,
                "face_detected": False,
                "det_score": 0.0,
            }
            results.append(result)
            if roi is None or roi.size == 0:
                continue
            try:
                ## Detect + align faces in the ROI (no embedding yet - that
                ## happens once, batched, after this loop over all ROIs).
                faces = face_detector.detect_and_align(roi)
                if faces and faces[0].aligned_crop is not None:
                    faces_to_embed.append((i, faces[0]))
                    crops.append(faces[0].aligned_crop)
                # A face with no landmarks (aligned_crop is None) can't be
                # embedded - result stays the default no-face dict.
            except Exception as e:
                logger.debug(f"Face detection error on ROI: {e}")
        det_ms_total = (time.time() - det_t0) * 1000

        if crops:
            embed_t0 = time.time()
            # One crop per call, deliberately: ORT's CUDA EP re-plans on every
            # input-shape change, so a batch size that varies cycle-to-cycle
            # costs ~80ms vs ~2.6ms at a fixed shape. Batching all crops in one
            # call shipped as 0.3.0 and halved prod FPS. Don't reintroduce it.
            embeddings: List[Any] = []
            last_error: Optional[Exception] = None
            for crop in crops:
                try:
                    out = face_detector.embed_batch([crop])
                except Exception as e:
                    last_error = e
                    out = None
                embeddings.append(out[0] if out is not None and len(out) > 0 else None)
            failed = sum(1 for e in embeddings if e is None)
            if failed:
                # One line per cycle, not per face: a hard embedder failure
                # (CUDA OOM, model unloaded) would otherwise log once per face
                # per cycle across every camera.
                reason = last_error if last_error is not None else "embedder returned no result"
                logger.warning(
                    f"Face embedding failed for {failed}/{len(crops)} faces "
                    f"this cycle: {reason}"
                )
            embed_ms_total = (time.time() - embed_t0) * 1000

            for (i, face), embedding in zip(faces_to_embed, embeddings):
                if embedding is None:
                    # Failed embedding stays "no face", as before 0.3.0 —
                    # downstream treats face_detected as "has an embedding".
                    continue
                roi = person_rois[i]
                # face.bbox / face.kps are already in ROI coordinates; the
                # detector's internal padding is undone before it returns.
                x1, y1, x2, y2 = face.bbox.astype(int)
                face_crop = roi[max(0, y1):y2, max(0, x1):x2]

                kps = None
                if hasattr(face, "kps") and face.kps is not None:
                    kps = face.kps.astype(int).tolist()

                results[i] = {
                    "embedding": embedding,
                    "face_image": face_crop if face_crop.size > 0 else None,
                    "face_detected": True,
                    "det_score": (
                        float(face.det_score)
                        if hasattr(face, "det_score")
                        else 0.0
                    ),
                    "face_bbox": [x1, y1, x2, y2],
                    "face_landmarks": kps,
                    # Orientation proxies from the package (single source of
                    # truth); the unrecognized-case gate reads these.
                    "frontality": frontality(kps),
                    "pitch": pitch(kps),
                }
        else:
            embed_ms_total = 0.0

        logger.debug(
            f"face.embed_batch: seq={batch_handle.seq} rois={n_rois} "
            f"faces={len(crops)} det={det_ms_total:.1f}ms "
            f"embed={embed_ms_total:.1f}ms total={(time.time() - t0) * 1000:.1f}ms"
        )
        return results
    except Exception as e:
        logger.exception(f"ArcFace batch inference failed: {e}")
        return _blank_results()
