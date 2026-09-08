"""YOLO person detection as a Celery Batches task.

Batching moved here from `pipeline/gpu_batch_dispatcher.py`'s
`GPUInferenceWorker` — that whole component (per-camera queues, cross-camera
batch collection, request/response correlation over a Unix-socket RPC) is
deleted by docs/LSO67_FOLLOWUP_QUEUE_DESIGN.md's plan. `celery-batches`'
`Batches` base class buffers requests arriving on this worker's own queue and
flushes them together, replacing that whole in-process collector with a
library primitive.

Verified in the go/no-go spike (benchmarks/celery_batches_spike/README.md,
"Step 0" in the design doc): batching here is a side effect of the worker
being busy inside the real model call, not primarily of `flush_interval`'s
timer — that timer only matters for topping up partial batches during
genuinely idle gaps. No tuning was needed for 10 cameras / 7.5Hz / real YOLO
latency; re-run the spike before assuming these numbers hold at a much
higher camera count.

Frames arrive as a `FrameHandle` (shared memory) per request, never as
pixels: measured at ~3.05 ms/frame through shared memory versus ~19.8 ms
through the broker even in the best case (JPEG; production's actual
json+base64 payload measures ~175 ms) — see benchmarks/transport/README.md
for the harness and full numbers. Detections for each request are
forwarded on to `camera.track` on that camera's own pinned queue — this task
never returns a result to a caller (`ignore_result=True`); the next hop is
itself a `send_task` call, not a return value.

Two Celery/celery-batches behaviors this task's body works around, both
confirmed against the installed celery_batches source and the spike:
  - Batches tasks do NOT honour `expires` (no `revoked()` call anywhere in
    celery_batches' Strategy). Every request therefore carries an explicit
    `deadline` (epoch seconds) checked in this task's body — both on the way
    in (drop stale detect requests before running the model) and re-applied
    on the way out (forward `camera.track` with the REMAINING budget, not a
    fresh window).
  - Batch size is capped by the worker's prefetch count, since Batches only
    acks after a flush completes. yolo-worker's compose command therefore
    passes `--prefetch-multiplier=32` — this is load-bearing, not a
    performance tweak; at the default multiplier=1 every batch is size 1.

No torch/ultralytics import at module scope — the parent imports this module
to register the task, and a CUDA touch there poisons `fork()`.
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable, Dict, List

from celery.signals import worker_process_init
from celery_batches import Batches, SimpleRequest
from loguru import logger

from workers import model_holder
from workers.celery_app import celery

# Matches the values validated in the spike (benchmarks/celery_batches_spike/
# bench_tasks.py) — see the module docstring's batching-mechanism note before
# assuming a change here changes achieved batch size in isolation.
FLUSH_EVERY = int(os.environ.get("SO_YOLO_FLUSH_EVERY", "8"))
FLUSH_INTERVAL_S = float(os.environ.get("SO_YOLO_FLUSH_INTERVAL_S", "0.010"))


@worker_process_init.connect
def _load_model_in_child(**_kwargs):
    """Preload YOLO post-fork, but ONLY in a worker that actually serves this
    queue.

    `worker_process_init` is app-global: every task module in the app's
    `include` list is imported by every worker, so without this guard the
    face worker and the camera worker would each also load YOLO. Measured
    before the guard existed: all three workers sat at ~654 MiB having
    loaded both models, instead of only the one they use.

    Celery gives the handler no way to know which `-Q` queues this process
    serves (the signal carries no queue context, and `app.amqp.queues` is not
    narrowed at this point), so the worker declares it explicitly via
    SO_WORKER_PRELOAD. Absent that, nothing is preloaded and the models load
    lazily on first task instead — correct either way, just with a slower
    first task.
    """
    if "yolo" in os.environ.get("SO_WORKER_PRELOAD", "").split(","):
        model_holder.ensure_person_detector_loaded()


def _parse_yolo_result(result) -> List[Dict]:
    """Convert a YOLO result object to a list of detection dicts.

    Unchanged from the pre-Batches version. This is what keeps ultralytics'
    `Results` object from ever crossing the broker — the output is plain
    dicts/lists/floats.
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


class _BatchStats:
    """One INFO line per N batches — the e2e verification check
    (docs/LSO67_FOLLOWUP_QUEUE_DESIGN.md's "End-to-end verification" section)
    greps for `avg_batch=` in the worker log. Deliberately not per-batch
    (that stays at DEBUG): this runs on every flush, potentially every ~10ms
    under load, so an INFO line per batch would itself become log-volume
    noise on the exact path being measured for latency.

    avg_batch/avg_ms cover only the last `log_every` flushes, not the
    process's lifetime — a lifetime average dilutes any real change (a
    fresh camera reconnecting, a stall starting) more and more the longer
    the worker has been up, so it drifts slowly toward the truth instead of
    showing it. skipped_expired/skipped_gone stay lifetime-cumulative on
    purpose: those are meant to answer "is this still happening", and a
    flat cumulative count across log lines is exactly what shows a problem
    has stopped, not just gone quiet for one window.
    """

    def __init__(self, log_every: int = 100):
        self._log_every = log_every
        self._n = 0
        self._window_batch = 0
        self._window_ms = 0.0
        self._skipped_expired = 0
        self._skipped_gone = 0
        # Constructed lazily on first record(), not here: this class is
        # instantiated at module scope, which runs in the PARENT before fork.
        # Touching Redis there would share one connection across every child.
        self._reporter = None

    def record(self, batch_size: int, wall_ms: float, skipped_expired: int, skipped_gone: int) -> None:
        self._n += 1
        self._window_batch += batch_size
        self._window_ms += wall_ms
        self._skipped_expired += skipped_expired
        self._skipped_gone += skipped_gone
        self._publish(batch_size, wall_ms)
        if self._n % self._log_every == 0:
            logger.info(
                f"yolo.detect: batches={self._n} avg_batch={self._window_batch / self._log_every:.2f} "
                f"avg_ms={self._window_ms / self._log_every:.1f} "
                f"skipped_expired={self._skipped_expired} skipped_gone={self._skipped_gone}"
            )
            self._window_batch = 0
            self._window_ms = 0.0

    def _publish(self, batch_size: int, wall_ms: float) -> None:
        """Feed the dashboard's yolo latency gauge (pipeline/infer_metrics.py).

        `wall_ms` is the whole `run_detect_batch` call — shm attach and the
        forwarding dispatch included, not just `detector.model(...)`. That is
        the number worth watching from outside (it is what the frame actually
        waits through) but it is NOT comparable to the old in-process
        `record_yolo_ms`, which timed the model call alone; the gauge is named
        for the batch, not the model, to keep that honest.

        Both per-batch and per-frame are published because they answer
        different questions: per-batch tracks whether the GPU call is slowing
        down, per-frame tracks what each frame costs once batching has
        amortised it.
        """
        if batch_size <= 0:
            return  # an all-skipped flush times the skipping, not inference
        if self._reporter is None:
            from pipeline.infer_metrics import InferLatencyReporter

            self._reporter = InferLatencyReporter("yolo")
        self._reporter.record(
            batch_ms=wall_ms,
            batch_count=1,
            frame_ms=wall_ms,
            frame_count=batch_size,
        )


_stats = _BatchStats()


def _default_dispatch(kwargs: Dict[str, Any], queue: str, expires: float) -> None:
    celery.send_task("camera.track", kwargs=kwargs, queue=queue, expires=expires)


def run_detect_batch(
    requests: List[SimpleRequest],
    detector,
    dispatch: Callable[..., None] = _default_dispatch,
    now: Callable[[], float] = time.time,
) -> None:
    """The actual batching logic, factored out of `detect_task` so tests can
    call it directly with a fake detector and a recording `dispatch`,
    without going through Celery's Batches machinery at all.

    Per request: reads a FrameHandle out of shared memory, runs ONE model
    call over every surviving frame in the batch, then forwards each
    request's detections on to `camera.track` on its own camera's queue.
    Never raises: a model exception degrades every live request in the batch
    to `[]` detections rather than losing the batch's frames entirely — the
    tracker already tolerates one frame with no detections (it ages the
    existing tracks), which is a better failure mode than a task exception
    that would surface nowhere useful, since this task is ignore_result.
    """
    from workers.frame_store import FrameHandle, attach_and_read

    t0 = time.monotonic()
    t_now = now()

    live_requests: List[SimpleRequest] = []
    live_frames = []
    n_skipped_expired = 0
    n_skipped_gone = 0

    for req in requests:
        deadline = req.kwargs.get("deadline")
        if deadline is not None and t_now > deadline:
            n_skipped_expired += 1
            continue
        frame = attach_and_read(FrameHandle(**req.kwargs["frame_handle"]))
        if frame is None:
            # Same "nothing to do" case camera.track's own attach_and_read
            # would hit if forwarded — no point paying that hop just to
            # rediscover the frame is already gone.
            n_skipped_gone += 1
            continue
        live_requests.append(req)
        live_frames.append(frame)

    if live_frames:
        try:
            results = detector.model(
                live_frames,
                conf=detector.confidence_threshold,
                iou=detector.iou_threshold,
                verbose=False,
                device=detector.device,
            )
            detections_per_request = [_parse_yolo_result(r) for r in results]
        except Exception as e:
            logger.exception(f"yolo.detect: batch inference failed: {e}")
            detections_per_request = [[] for _ in live_frames]
    else:
        detections_per_request = []

    for req, detections in zip(live_requests, detections_per_request):
        camera_id = req.kwargs["camera_id"]
        deadline = req.kwargs.get("deadline")
        remaining = (deadline - now()) if deadline is not None else None
        if remaining is not None and remaining <= 0:
            n_skipped_expired += 1
            continue
        next_queue = req.kwargs.get("next_queue") or f"cam.{camera_id}"
        dispatch(
            kwargs={
                "camera_id": camera_id,
                "frame_handle": req.kwargs["frame_handle"],
                "frame_num": req.kwargs["frame_num"],
                "detections": detections,
                "deadline": deadline,
            },
            queue=next_queue,
            expires=remaining,
        )

    wall_ms = (time.monotonic() - t0) * 1000.0
    _stats.record(len(live_requests), wall_ms, n_skipped_expired, n_skipped_gone)


@celery.task(
    base=Batches,
    name="yolo.detect",
    queue="yolo",
    ignore_result=True,
    flush_every=FLUSH_EVERY,
    flush_interval=FLUSH_INTERVAL_S,
)
def detect_task(requests: List[SimpleRequest]) -> None:
    detector = model_holder.ensure_person_detector_loaded()
    run_detect_batch(requests, detector)
