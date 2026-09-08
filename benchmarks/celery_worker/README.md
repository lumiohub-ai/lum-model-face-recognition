# Celery YOLO worker spike

Does a **single YOLO instance shared by N threads** give us worker-level
concurrency without multiplying VRAM — and how does it compare to N processes
that each hold their own copy?

This is a spike, deliberately small. It exercises **only** YOLO through Celery,
in isolation from the application. It imports `lum_vision`, `ultralytics` and
`celery`, and **nothing from `src/`**.

## Why this lives in `benchmarks/`, not `tests/`

This is not a test. It needs a GPU, a live Redis and minutes of wall time, and
it asserts nothing — it measures. `pytest` must never collect it, so it sits
outside `tests/` entirely rather than relying on a missing `test_` prefix.

The package is `celery_worker`, deliberately **not** `workers`: `src/workers` is
the production Celery package (`workers.detection_tasks`, `workers.embedding_tasks`)
and two importable `workers.*` packages disambiguated only by `PYTHONPATH`
ordering is a trap.

It also does **not** do the `sys.path.insert(0, ".../src")` that the real tests
do. Importing `src/` pulls in `config.settings` → the stale `.env` (which points
at port 6400) → sqlalchemy and psycopg2, none of which are installed in `.venv`.
Instead `benchmarks/` goes on `PYTHONPATH` so `celery -A celery_worker.bench_app`
resolves.

## Setup

```bash
.venv/bin/pip install 'celery~=5.4.0' 'redis~=5.0.1'
docker run --rm -d --name yolobench-redis -p 6401:6379 redis:7-alpine
```

**Port 6401 matters.** 6379 is another developer's `sostack-redis-1`; 6400 is
this repo's own `so-face-oybek` stack. A throwaway container is also cleaner
than a spare db index, because Redis Pub/Sub — which carries Celery's pidbox —
is *not* database-scoped.

## Run

```bash
PYTHONPATH=benchmarks .venv/bin/python benchmarks/celery_worker/run_bench.py                # all cells
PYTHONPATH=benchmarks .venv/bin/python benchmarks/celery_worker/run_bench.py --cells 2,4    # a subset
docker rm -f yolobench-redis                                               # teardown
```

Results land in `output/results.json` (gitignored) plus a table on stdout.

## Layout

| file | role |
|---|---|
| `bench_app.py` | Isolated Celery app `yolobench`. Own broker/backend/queue/exchange. |
| `model_holder.py` | **The core.** One shared `nn.Module`, one `DetectionPredictor` per thread. |
| `bench_tasks.py` | `yolobench.infer` / `warmup` / `probe`. No heavy imports at module scope. |
| `run_bench.py` | Driver: corpus, worker spawn, warmup barrier, load, NVML sampling, teardown. |

## The mechanism

`PersonDetector` in `lum_vision` is a **loader only** — it has no `detect()`
method. Production reaches into `.model` directly (`src/pipeline/gpu_worker.py`).

Ultralytics' `Model.predict()` rebuilds `self.predictor.args` via `get_cfg()` on
every call, and `BasePredictor.stream_inference` wraps its whole body in
`self._lock`. So a **shared `YOLO` object is already fully serialised**, and the
lock convoy on top makes it *slower than a single thread*.

The design that works: share only the `nn.Module`, give each thread its own
predictor. `setup_model` → `AutoBackend` → `PyTorchBackend.load_model` takes the
`isinstance(weight, nn.Module)` branch, whose `weight.to(device)` is an in-place
no-op on an already-resident module. Verified at runtime: `threads -c4` reports
**1 pid, 1 module id, 4 distinct predictor ids**.

## Three traps this harness hit, so you don't have to

1. **Concurrent predictor construction races on `fuse()`.** `BaseModel.fuse()`
   is guarded by `is_fused()`, which makes a *sequential* second call a no-op but
   is **not atomic**. Two threads both see an unfused module, both enter
   `fuse()`, and the loser dies on `delattr(m, "bn")`. `model_holder._PRED_LOCK`
   serialises construction; inference stays lock-free.
2. **`PersonDetector(device=None)` calls `torch.cuda.is_available()`**, which
   initialises CUDA in whichever process reaches it first and poisons `fork()`
   for the prefork pool. Always pass `device="cuda"` explicitly, and never
   import torch/ultralytics/lum_vision at module scope in `bench_tasks.py`.
3. **One warmup task warms one slot.** The warmup group holds each slot until a
   shared deadline; with `prefetch_multiplier=1` that forces one warmup per slot.
   Without the hold, one fast thread drains the group and the rest are still cold
   when the clock starts.

## Results (RTX 3080 Ti, 1280x720 frames, 1200 frames/cell)

| cell | pool | -c | B | tasks/s | **frames/s** | p50 ms | VRAM MiB | GPU pids |
|---|---|---|---|---|---|---|---|---|
| 1 solo | solo | 1 | 1 | 82.1 | 82.1 | 48 | 478 | 1 |
| 2 threads | threads | 4 | 1 | 167.2 | **167.2** | 93 | **716** | **1** |
| 3 prefork | prefork | 4 | 1 | 206.6 | 206.6 | 76 | 1912 | 4 |
| 4 threads+batch | threads | 4 | 6 | 49.2 | **295.5** | 312 | **1328** | **1** |
| 5 prefork+batch | prefork | 4 | 6 | 53.1 | **318.3** | 291 | 2512 | 4 |
| 6 naive shared | threads | 4 | 1 | 65.0 | 65.0 | 257 | 664 | 1 |
| 7 prod prefetch | threads | 4 | 1 | **0.5** | 0.5 | **30047** | 732 | 1 |
| 8 solo, shm | solo | 1 | 1 | 118.0 | 118.0 | 33 | 478 | 1 |
| **9 threads+batch, shm** | threads | 4 | 6 | 70.7 | **424.2** | 213 | **1360** | **1** |
| **10 prefork+batch, shm** | prefork | 4 | 6 | 76.5 | **459.1** | 198 | 2512 | 4 |

Incumbent for comparison: the in-process path does **~390 fps** on one thread by
batching ~6 cross-camera frames into a single forward pass.

**What it says.**

1. **The shared-model design works.** Cell 2 holds 4 concurrent workers on
   **one** GPU process at 716 MiB and 2.04x solo throughput. Cell 3 buys 1.24x
   more throughput for **2.7x the VRAM** and 4 CUDA contexts. With batching the
   gap closes further: cell 4 reaches 93% of cell 5's throughput on **53% of the
   VRAM**. Sharing one instance across threads is the better trade here.
2. **Batching dominates concurrency.** B=1 -> B=6 is worth ~1.8x on both pools;
   4-way concurrency is worth ~2.0-2.5x.
3. **With frames passed by handle, Celery workers BEAT the in-process incumbent**
   — 424 fps on threads (1.09x) and 459 fps on prefork (1.18x). See below; the
   earlier `path`-mode conclusion that Celery was a regression was an artifact
   of this harness, not a property of Celery.
4. **Never share a `YOLO` object across threads.** Cell 6 (65 fps) is *slower
   than a single process* (cell 1, 82 fps) — ultralytics' internal lock plus a
   `get_cfg()` rebuild per call. It is also the arm that crashes on `fuse()` if
   its first call is not serialised.
5. **The threads pool cannot use production's delivery semantics.** Cell 7 is
   the same worker as cell 2 with `worker_prefetch_multiplier=1`: throughput
   collapses **330x** and p50 latency goes to **30 seconds**.

## Celery's real broker cost is ~1.3 ms per task

`mode=path` makes every task re-decode a JPEG. At batch 6 that is ~26 ms of CPU
per task — more than the inference — and it dominated the first round of
results. It is an artifact of the harness, not a cost of Celery: a real pipeline
already has the frame in memory.

`mode=shm` (`frame_store.py`) decodes the corpus once into one shared block and
sends only a name plus a few indices, so the broker moves a **handle** rather
than pixels. Isolating the overhead — note `work` uses ultralytics' *per-image*
speed times the batch, plus the task's own read time:

| cell | B | service ms | work ms | **broker ms** | per frame |
|---|---|---|---|---|---|
| solo, path | 1 | 12.2 | 10.8 | **1.3** | 1.33 |
| solo, shm | 1 | 8.5 | 7.2 | **1.3** | 1.27 |
| threads, shm | 6 | 56.6 | 51.6 | **5.0** | 0.83 |
| prefork, shm | 6 | 52.3 | 48.3 | **4.0** | 0.66 |

**Celery costs ~1.3 ms per task and it does not depend on payload mode** —
identical for path and shm at B=1. Amortised over a batch of 6 it is
**0.6–0.8 ms per frame**.

That is why the shared-memory cells beat the in-process incumbent: GPU
utilisation was only 22–35% in the `path` cells and rises to 56–60% here. The
monolith was limited by the GIL and by JPEG decode, not by the GPU — so moving
work into separate workers unlocks headroom that was being wasted.

**Consequence for a worker-based architecture:** the broker is cheap enough to
build on, *provided frames move by handle rather than as data*. Serialising
pixels through Redis is what makes it expensive, and that is avoidable.

## The threads-pool prefetch trap

Measured directly (200 tasks, 16 driver threads in flight):

| pool | prefetch | acks_late | tasks/s |
|---|---|---|---|
| prefork | 1 | True | 144.0 |
| threads | 1 | True | 1.1 |
| threads | 1 | False | 1.7 |
| threads | 4 | True | 3.0 |
| threads | 4 | False | 161.3 |
| threads | 16 | True | 149.7 |

The threads pool stalls whenever **in-flight depth >= the prefetch window**,
recovering only on a slow periodic tick; prefork is immune at every setting.
This matters for the migration: production runs `worker_prefetch_multiplier=1`
with `task_acks_late=True` (`src/workers/celery_app.py`), which is exactly the
broken quadrant. A threads-pool worker must size its prefetch window above
expected in-flight depth — which forfeits the bounded-memory and fair-dispatch
properties that `prefetch_multiplier=1` exists to provide.

## Reading the numbers

- **`frames_per_s = tasks_per_s × batch`** is the number comparable to the
  incumbent in-process path, which does **390 fps** by batching ~6 cross-camera
  frames into one forward pass (`gpu_worker.py:220-265`).
- `mode=path` pays a real `cv2.imread` per frame (~4 ms warm, ~6 ms cold at
  1280×720). That is ~30% of per-task time and is *not* an artifact — a
  distributed pipeline must get pixels to the worker somehow. Bench numbers are
  correspondingly lower than pure-inference microbenchmarks.
- Celery's own overhead measured ~2 ms/task (service time 15.2 ms vs 13.2 ms of
  in-task work, solo pool).

## Caveats

- **A batched Celery task needs producer-side cross-camera batching** —
  accumulate frames from N cameras for up to T ms, then enqueue one task. That
  reintroduces the coupling Celery was meant to remove and adds T ms of latency.
  An architectural consequence, not a benchmark result.
- **Backpressure is lost.** `gpu_worker.py`'s `maxsize=2` queues drop frames
  under load; Celery queues are unbounded, so overload shows up as diverging
  latency instead. A migration needs an explicit drop policy (`task_expires`, or
  a producer-side queue-depth check).
- **VRAM must be read per-PID via NVML.** `torch.cuda.memory_allocated()`
  excludes the ~300–600 MiB per-process CUDA context, which is the entire point
  of comparing threads against prefork.
- Other GPU tenants on this box (`comfyui`, `gpustack-worker`, `ollama`) would
  poison the VRAM readings. The driver asserts VRAM returns to baseline between
  cells and aborts otherwise.
- `lum-model-vision` is installed **editable from a working tree**, not the
  `requirements.txt`-pinned tag; `results.json` records both git revisions.

## Not covered (deliberately)

The Celery event stream, open-loop rate pacing, golden detection-hash
verification, `threads -c8`.

`mode=bytes` payloads (pixels through the broker instead of shared memory) *are*
now covered — see [`benchmarks/transport/`](../transport/README.md), a
separate harness built specifically to measure that comparison head-to-head.
Corrected context: this corpus is 720p, not 2560×1440; a 720p JPEG q90 frame
is ~806 KiB before base64, and base64 inflates that by 33% again. Raw pixels
through json+base64 (this app's actual serializer) measured ~175 ms/frame;
JPEG (the cheapest broker option) ~19.8 ms/frame; shared memory ~3.05 ms/frame
including both the producer's copy-in and the consumer's copy-out.
