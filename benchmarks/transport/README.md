# Shared memory vs. the Redis broker, for one 720p frame

## The question

The pipeline moves frames between processes through `multiprocessing.shared_memory`, sending
only a handle (segment name + index) through Celery. The rejected alternative was to put the
pixels in the task payload.

This harness measures both, on the same frames, in the same run.

## Why it exists

Five places in the repo assert:

> a 720p frame costs ~15.6 ms through a broker versus ~0.3 ms as a handle

That number predates every benchmark in this directory. It first appears in commit `b2f6eb5`
(2026-08-26), three days before the harness next door was written, and a search of every blob in
the object database — including dangling and unreachable ones — finds only copies of the
sentence, never the measurement. It was taken outside version control.

Worse, the data that *is* committed does not support the architectural claim.
`../celery_worker/output/results.json` compares `mode=shm` (0.01–0.02 ms) against `mode=path`
(3.61–26.0 ms) — but `path` is *re-decoding JPEGs from disk*, which nobody ever proposed. The
alternative actually argued was pixels-in-the-payload, and no committed cell measured it.

So a reviewer could fairly say: *you proved shared memory beats an option no one suggested.*
This harness closes that gap.

## What it measures

Six cells, batch=1, `--pool=solo --concurrency=1`, submitted serially so queueing delay is never
mistaken for transport:

| cell | payload on the wire | role |
|---|---|---|
| `0-control` | ~100 B, no pixels | the per-task floor Celery costs regardless |
| `1-shm` | handle (~49 B); producer copies frame in, worker copies it out | **the design being defended** — what production actually does (`frame_store.py:223`, `:391`) |
| `2-shm-view` | handle (~49 B); producer copies frame in, worker takes a zero-copy view | shm's best case, if a consumer can work directly off the view |
| `3-json-b64` | base64 pixels, 3.52 MiB | **the rejected design**, under production's real serializer |
| `4-pickle-raw` | raw pixel bytes, 2.64 MiB | steelman: cheapest way to put pixels on a broker |
| `5-jpeg` | JPEG q90, ~0.31 MiB | steelman: smallest payload, pays codec CPU instead |

Cell 1, not cell 2, is what production does: every consumer of a `CameraFrameSlot`
(`frame_store.py:391`, `:432`, `:706`) copies out rather than holding a view. An earlier version
of this benchmark measured only the zero-copy case for the "production" cell, which understated
shared memory's true cost by skipping a real copy the pipeline always pays — see the note in
`run_bench.py`'s `make_payload()` for how that was caught and fixed.

Cells 4 and 5 exist so the result cannot be dismissed as a strawman. Cell 5 is the objection a
reviewer will actually raise ("just JPEG it") — it trades wire bytes for `imdecode` CPU, and the
harness attributes that explicitly rather than hiding it in a total.

Production pins `task_serializer='json'` (`src/workers/celery_app.py:75`), and JSON cannot carry
bytes — so cell 3, the base64 one, is the honest representation of the rejected design. Cells 4
and 5 need `pickle`, which this app accepts but does not default to.

**No model, no GPU.** The question is transport; inference would only add variance.

## Timing

Celery's `.get()` polls the result backend, and that latency is **not** transport. Reported
components, none of which include it:

- `encode_ms` — driver, serialization only, before `send_task`
- `publish_ms` — driver, around `send_task` alone
- `delivery_ms` — worker `t_recv` minus driver `t_sent`
- `decode_ms` — worker, deserialization into an array

`transport_ms` is their sum. `roundtrip_ms` is recorded too but flagged, never headlined.

`delivery_ms` subtracts a timestamp taken in one process from one taken in another. That is
valid **only** because both are on one host and read the same kernel `CLOCK_REALTIME`; the driver
asserts the worker is a local subprocess and refuses to run otherwise. Containerise the worker
and this number becomes meaningless.

Every pixel cell is also reported as a **delta over the control floor** — that difference, not
the absolute total, is what sending the pixels actually costs.

## Discipline

The first attempt at these numbers was casual, and re-running the same measurement minutes later
gave answers up to 4× apart. That is why the harness:

- discards 50 warmup iterations (JIT, allocator, connection pool, Redis buffers)
- reports mean/p50/p95/stdev/n per component, never a bare mean
- **interleaves the cells** round-robin, so a transient CPU spike hits every cell instead of
  fabricating a difference in whichever one it landed in
- runs the whole suite twice and **flags any cell differing >15% between runs as unstable**
- **aborts if the 1-minute load average is above `--max-load`** rather than silently producing
  contaminated numbers — stop the dev stack before running
- verifies a checksum per task, so it cannot be timing a truncated or elided payload (JPEG is
  exempt: it is lossy by design)

A `kombu.serialization` microbench runs alongside, because by the time a task body executes kombu
has already decoded the message — the serializer cost is invisible from inside the task, and it
is the dominant term for the base64 cell.

## Running it

```bash
# the numbers are only defensible on an idle machine
docker compose -p sostack-oybek down

# throwaway broker on 6401, bound to loopback only.
# NOT 6380 (this repo's dev stack) and NOT 6379 (a teammate's).
docker run --rm -d --name xportbench-redis -p 127.0.0.1:6401:6379 redis:7-alpine

PYTHONPATH=benchmarks .venv/bin/python benchmarks/transport/run_bench.py

docker rm -f xportbench-redis
```

Reuses the 200-frame corpus at `../celery_worker/output/frames/` (verified `(720, 1280, 3)`)
rather than re-extracting, so results stay comparable with the run next door. That directory is
gitignored (only `results.json` is tracked, same convention as `celery_face/`), so on a fresh
clone it won't exist yet — run `benchmarks/celery_worker/run_bench.py` once first to build it
from `cam1.mp4`. That video is itself gitignored and not provided by this repo; supply any real
1280x720 clip at the repo root under that name before either benchmark can run.

Results land in `output/results.json` with full provenance: repo revision, host, CPU, library
versions, load average at start, and the tasks/warmup counts.

## Scope

This holds for processes on **one host**. Shared memory is not available across machines, so a
multi-host pipeline faces a genuinely different trade-off — the tension already noted in
`docs/PIPELINE_FOR_REVIEW.md:748` and `:978`. The claim proved here is *"shared memory wins
decisively for same-host transport"*, which is narrower and more useful than *"shm always wins"*.
