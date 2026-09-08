"""The Batches task under test.

Stands in for `yolo.detect` from the real design: sleeps ~20ms per batch
(stand-in for one YOLO forward pass) rather than doing anything with a real
model, so this spike is pure Celery/kombu/celery-batches behavior with zero
GPU or lum_vision dependency.

Every request carries a `deadline` (epoch seconds) exactly like the real
design's frame_pump -> yolo.detect hop. Requests past their deadline are
skipped, never "processed late" — this is the in-body substitute for
`expires`, which celery-batches does not honor (see docs/LSO67_FOLLOWUP_QUEUE_DESIGN.md
Sec 4, "Verified constraints").

Batch outcomes are appended to a JSONL file rather than kept in a module-level
list: the worker runs in its own OS process, so an in-memory list here would
never be visible to run_bench.py's driver process.
"""

import json
import os
import time
from typing import List

from celery_batches import Batches, SimpleRequest

from celery_batches_spike.bench_app import app

# Matches the plan's proposed defaults (SO_YOLO_FLUSH_EVERY / SO_YOLO_FLUSH_INTERVAL_S).
FLUSH_EVERY = 8
FLUSH_INTERVAL = 0.010

# Stand-in for one YOLO forward pass over a batch of frames. Sized so that at
# 7.5Hz per "camera" the model call is comparable to the real ~20-45ms budget.
# Override to 0 to isolate the flush-timer's own granularity from the effect
# of a solo-pool worker being busy "in the model" when new requests arrive.
SIMULATED_MODEL_MS = float(os.environ.get("BATCHBENCH_MODEL_MS", "20.0"))

OUTCOMES_PATH = os.environ.get(
    "BATCHBENCH_OUTCOMES_PATH",
    os.path.join(os.path.dirname(__file__), "outcomes.jsonl"),
)


@app.task(
    base=Batches,
    name="batchbench.detect",
    queue="batchbench",
    ignore_result=True,
    flush_every=FLUSH_EVERY,
    flush_interval=FLUSH_INTERVAL,
)
def detect_task(requests: List[SimpleRequest]) -> None:
    t0 = time.monotonic()
    now = time.time()

    live = []
    n_expired = 0
    for req in requests:
        deadline = req.kwargs.get("deadline")
        if deadline is not None and now > deadline:
            n_expired += 1
            continue
        live.append(req)

    # Stand-in for `detector.model(frames, ...)` -- ONE call regardless of
    # how many requests are in this flush. This is the throughput claim under
    # test: batch_size > 1 means fewer of these expensive calls per frame.
    if live:
        time.sleep(SIMULATED_MODEL_MS / 1000.0)

    wall_ms = (time.monotonic() - t0) * 1000.0
    record = {
        "batch_size": len(live),
        "wall_ms": round(wall_ms, 3),
        "n_expired": n_expired,
        "flushed_at": time.time(),
    }
    # Append mode + one JSON object per line: safe under a single-writer
    # solo-pool worker, and lets the driver tail the file live if it wants to.
    with open(OUTCOMES_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")
