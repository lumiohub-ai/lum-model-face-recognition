# celery-batches spike — LSO-67 follow-up go/no-go

Answers, with real numbers, the "Step 0" gate in
[`docs/LSO67_FOLLOWUP_QUEUE_DESIGN.md`](../../docs/LSO67_FOLLOWUP_QUEUE_DESIGN.md) before any of
that plan's steps 1–6 are implemented. **Result: GO** — celery-batches on `--pool=solo` behaves
correctly and batches usefully, with one mechanism correction to the plan's own assumption (see
below). No fallback needed.

## How to reproduce

```bash
docker run --rm -d --name lso67batch-redis -p 6402:6379 redis:7-alpine

cd benchmarks
PYTHONPATH=. python celery_batches_spike/run_bench.py                    # main timing run
PYTHONPATH=. python celery_batches_spike/run_bench.py --kill-mid-batch --seconds 20

docker stop lso67batch-redis
```

Port 6402 is dedicated to this spike — distinct from 6379 (another developer's stack), 6400
(this repo's own dev stack), and 6401 (`benchmarks/celery_worker`'s own throwaway Redis).

## Results (10 simulated cameras, 7.5 Hz, 30s, `--pool=solo --prefetch-multiplier=32`, `flush_every=8`, `flush_interval=10ms`, 20ms simulated model call)

| Check | Result |
|---|---|
| Batch size achieved | **mean 2.17**, histogram `{1: 2, 2: 798, 3: 170}` — beats run C's 1.6–1.7 without run C's tracker-affinity corruption, since this design keeps camera→worker affinity separately (§1 of the design doc) |
| `unacked` after settle | `0` — acks complete correctly under `task_acks_late=True` |
| Queue length after settle | `0` — no stuck/leaked messages |
| Past-deadline requests | Correctly skipped, not processed late (0 in the steady-state run; 11 confirmed skipped after a `kill -9` mid-batch + redelivery, see below) |
| `kill -9` mid-batch → redelivery | Worker killed mid-run, restarted; Redis redelivered the unacked in-flight messages; the restarted worker correctly identified and skipped the ones past their `deadline` (`n_expired: 11` in that run's outcomes) rather than reprocessing them as fresh |

## Finding: the 10ms `flush_interval` is not the batching mechanism at this load — worker business is

This is a real correction to the design plan's mental model, not just a confirming data point.

The plan assumed `flush_interval` was the primary lever controlling batch size ("does the 10ms
timer actually fire near 10ms"). Measured behavior is different:

- **With the 20ms simulated-model sleep** (representative of real YOLO inference time): batch
  size mean 2.17, inter-flush gap mean **~31ms** — three times the nominal 10ms.
- **With the simulated model sleep set to 0** (`BATCHBENCH_MODEL_MS=0`): batch size collapses to
  **exactly 1.0**, and the inter-flush gap drops to ~11-14ms, i.e. genuinely close to the nominal
  10ms timer.

Reading `celery_batches/__init__.py`'s `Strategy.task_message_handler` explains why:
`_do_flush` — and therefore the task body's `time.sleep(SIMULATED_MODEL_MS)` — runs **inline, on
the worker's single main loop**, exactly as `--pool=solo` promises. While a flush's model call is
running, the worker cannot service the flush timer or drain the buffer; every request that
arrives during that window queues up, and the moment the call returns, the accumulated backlog
either hits `flush_every` or the timer fires almost immediately. **Batching on `--pool=solo`
here is a side effect of the worker being busy in the model call, not of the timer window
independently accumulating idle time.** The `flush_interval` timer only matters for topping up
partial batches during genuinely idle gaps (confirmed by the `BATCHBENCH_MODEL_MS=0` run, where
it's the only mechanism left and does land near 10ms).

**Practical consequence for the real design:** this means `flush_every`/`flush_interval` tuning
should be reasoned about together with the real YOLO inference time, not independently — a
faster model (shorter busy window) accumulates a smaller batch per cycle; a slower model
accumulates more. At 10 cameras / 7.5Hz / ~20-40ms real YOLO latency this lands comfortably in
the useful range (batch 2-3) without any tuning beyond the plan's proposed defaults. If camera
count grows significantly, re-run this spike at the new arrival rate before assuming the same
batch sizes hold — the mechanism scales with *load relative to model latency*, not `flush_every`
alone.

## Files

| File | Role |
|---|---|
| `bench_app.py` | isolated Celery app, no imports from `src/` (matches `benchmarks/celery_worker`'s pattern) |
| `bench_tasks.py` | the `Batches` task under test; appends one JSON line per flush to `outcomes.jsonl` (module-level state can't cross the worker's process boundary back to the driver) |
| `run_bench.py` | driver: starts a real worker process + producer thread, then summarizes `outcomes.jsonl` |

Package is named `celery_batches_spike`, not `celery_batches` — the latter is the installed
PyPI library; a same-named local package would shadow it on `sys.path`.
