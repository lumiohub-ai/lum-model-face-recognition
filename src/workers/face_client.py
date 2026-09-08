"""Synchronous client for the `face.embed_batch` Celery task.

Replaces `GpuWorkerRpcClient.embed`'s Unix-socket round trip: this hop no
longer goes through the GPU RPC middleman (that whole layer is deleted by
docs/LSO67_FOLLOWUP_QUEUE_DESIGN.md's plan — see its §"Changes by file" /
`src/workers/face_client.py` entry). Face inference already runs on its own
Celery worker (`face-worker`, queue `face`); this client just calls it
directly and waits, the same way `camera_tasks.process_frame` already waits
on this exact RPC today — no new blocking semantics, only a different
transport underneath the wait.

Why a plain `.get()` and not a Batches task like `yolo.detect`: SCRFD has no
batch path and ArcFace embedding is deliberately one crop per call (LSO-117 —
batching regressed production FPS because onnxruntime's CUDA EP re-plans on
every input-shape change), so cross-request batching on the face leg buys
nothing on the GPU side. Batches would only add flush-window latency to a
call this method is already blocking on, plus manual `mark_as_done`
bookkeeping celery-batches doesn't provide for free. See
`docs/LSO67_FOLLOWUP_QUEUE_DESIGN.md`'s "Changes by file" / face_tasks.py
entry for the full reasoning.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any, Callable, Dict, Optional

from loguru import logger

from workers.frame_store import RoiBatchHandle

# Ordering mirrors gpu_worker_rpc.py's DEFAULT_TIMEOUT_S rationale: the task's
# own `expires` must be strictly below the client's `.get()` timeout, or the
# client's timeout can race a legitimate-but-late server-side expiry and log
# a spurious transport failure for what was really an expected drop.
DEFAULT_EXPIRES_S = float(os.environ.get("SO_FACE_EMBED_EXPIRES_S", "2.0"))
DEFAULT_TIMEOUT_S = float(os.environ.get("SO_FACE_EMBED_TIMEOUT_S", "2.5"))


def embeddings_by_track(
    results: list, handle: RoiBatchHandle
) -> Dict[int, Dict[str, Any]]:
    """Map `embed_batch_task`'s positionally-aligned result list back to
    `{track_id: result}`, using the real track_id each RoiHandle carries.

    Unlike the deleted GPU-dispatcher's `_arcface_loop` (which flattened
    ROIs across cameras and used throwaway `range(n_rois)` position tags,
    because one batch spanned many cameras), this client's batches are
    already scoped to a single camera's `RoiBatchSlot.write(rois, track_ids)`
    call — so `handle.rois[i].track_id` IS the real, correct track id, not a
    tag needing a separate lookup table.
    """
    if len(results) != len(handle.rois):
        # A short or reordered reply would hand one person's embedding to
        # another person's track. Refuse it rather than guess.
        logger.error(
            f"face.embed_batch returned {len(results)} results for "
            f"{len(handle.rois)} ROIs (camera_id={handle.camera_id}, "
            f"seq={handle.seq}) — discarding to avoid misrouting identities"
        )
        return {}
    return {roi.track_id: result for roi, result in zip(handle.rois, results)}


class FaceEmbedClient:
    """Worker-side: submit a camera's ROI batch to `face-worker` and block
    for the embeddings. Never raises — degrades to `{}`, matching what a
    same-process timeout or the old GPU-RPC embed() already did on failure.
    """

    def __init__(
        self,
        expires_s: float = DEFAULT_EXPIRES_S,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ):
        self._expires_s = expires_s
        self._timeout_s = timeout_s
        # Same convention as gpu_rpc.GpuRpcClient.on_fallback and
        # gpu_worker_rpc.GpuWorkerRpcClient.on_fallback: a struggling
        # face-worker should be visible as a failure-rate metric, not only
        # inferable from every camera's recognition going quiet.
        self.on_fallback: Optional[Callable[[], None]] = None

    def embed(
        self, camera_id: int, roi_batch_handle: RoiBatchHandle
    ) -> Dict[int, Dict[str, Any]]:
        from workers.face_tasks import embed_batch_task

        async_result = embed_batch_task.apply_async(
            kwargs={"handle": dataclasses.asdict(roi_batch_handle)},
            queue="face",
            expires=self._expires_s,
        )
        try:
            results = async_result.get(
                timeout=self._timeout_s, disable_sync_subtasks=False
            )
        except Exception as e:
            logger.warning(
                f"face_client: embed(camera_id={camera_id}, "
                f"seq={roi_batch_handle.seq}) failed: {type(e).__name__}: {e} "
                f"— no embeddings this cycle"
            )
            if self.on_fallback is not None:
                self.on_fallback()
            return {}
        finally:
            # Mandatory, not cosmetic: face results carry numpy (embedding,
            # face_image) and pickle to ~70 KiB each; at ~15 results/s with
            # result_expires=3600 left unforgotten, that is ~3.8 GB/h sitting
            # in Redis DB 1 for results nobody will ever read again after
            # this .get() returns. See docs/LSO67_FOLLOWUP_QUEUE_DESIGN.md's
            # "Backpressure / timeouts" section.
            async_result.forget()

        return embeddings_by_track(results, roi_batch_handle)
