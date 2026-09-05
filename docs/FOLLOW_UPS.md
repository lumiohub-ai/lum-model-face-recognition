# Follow-ups and known issues

Running list of work deferred, problems observed but not yet fixed, and decisions we
consciously postponed. Add to this rather than letting things live only in commit messages
or someone's memory.

**This file is the source of truth for what's outstanding.** The roadmap in
[`PIPELINE_FOR_REVIEW.md`](PIPELINE_FOR_REVIEW.md) is a summary for review, and refers to
the item numbers used here.

Items are ordered **by when you'd actually do them**, not by size — several are blocked by
earlier ones, and those dependencies are stated. Numbers are stable identifiers: if an item
is done, mark it Done and leave the number retired rather than renumbering the rest.

Each entry says **what**, **why it matters**, **why it wasn't done now**, and **what would
tell us it's time**.

Last updated: 2026-09-05

---

## Legend

| Field | Meaning |
|---|---|
| **Blocks** | Other items that cannot be decided until this is done |
| **Blocked by** | Must wait for the named item |
| **Trigger** | The observable condition that says "do this now" |

---

# Before this branch merges

## 1. ~~Two internal docs describe the deleted architecture~~ — DONE (2026-09-05)

**Was:** `CELERY_MIGRATION.md` and `PIPELINE_OVERVIEW.md` documented `GPUInferenceWorker`,
the `camera_frames` queue, two RPC sockets, and linked to `gpu_batch_dispatcher.py` /
`gpu_worker_rpc.py` — files this branch deletes. Sequenced deliberately after
end-to-end testing, since real results might change what the docs should say. They did.

**Done:** both rewritten against the architecture that actually runs — six processes,
`decode-worker` owning the cameras, one RPC socket, batching inside `yolo-worker`, the
three shared-memory rings, and lease-based camera ownership. Every `../src/...` link
verified to resolve.

Three things the rewrite corrected that were not merely stale but *wrong*:

- `PIPELINE_OVERVIEW.md` §5 explained small batch sizes as camera threads drifting out of
  sync, and predicted a self-reinforcing feedback loop. Live testing disproved it — the
  real causes were `--prefetch-multiplier=1` capping batches at 1, decode starved of CPU,
  and ~51% duplicate frames. That section now explains batch size as an *output* of
  arrival rate vs. GPU speed, and says plainly that a small batch is not a problem when
  nothing is being dropped.
- Both docs asserted "do not scale these services" based on correlation logic in
  `gpu_batch_dispatcher.py`, which no longer exists. The real remaining constraint is
  narrower: `camera-worker` must stay one consumer per `cam.<id>` because tracker state
  lives in its memory. `decode-worker` scales freely.
- The shared-memory section listed four defences; there are now five. The added one
  (re-attach when the producer restarted) came directly from a live incident.

---

## 2. Run 0 — the pre-migration baseline measurement

**What:** A before/after latency comparison against the pre-LSO-67 architecture. Without
it we cannot say whether any of this made the system faster.

**Why it matters:** It is the only thing that turns "we rebuilt it" into "here is what it
bought." It also decides item 7 — whether the low GPU batch size is a problem or spare
capacity.

**Status:** Fully prepared, never run. Git worktree at
`/home/oybek/workspace/run0-baseline/` (commit `986cae4`), timing patch applied,
`run0_summarize.py` written, `RUN0_INSTRUCTIONS.md` has the verified mount paths.

**Why not now:** Needs exclusive use of the stack for a clean run.

**Blocks:** item 8. (Items 5 and 7 were settled by direct measurement on 2026-09-05
instead — see item 7 for the numbers.)

**Trigger:** Next window where the stack is free. Roughly an afternoon.

---

# Before this reaches a production site

## 3. No sustained-stability run

**What:** The longest verified continuous run is under an hour. Nothing is known about
behaviour over days — slow memory growth in the shared-memory rings, socket or connection
leaks, gradual drift in tracked-identity counts.

**Why it matters:** These are exactly the failure classes a one-hour run cannot surface,
and this runs production sites (an airport, a railway).

**Why not now:** Not yet needed; the branch is not merged.

**Trigger:** Before any production deployment. Cheap — leave it running and re-check the
same metrics after 24–48 hours.

---

## 4. ~~Four cameras did not connect~~ — RESOLVED, not a bug

**What it looked like:** 6 of 10 configured cameras connected at engine startup (29, 36, 37,
38, 39, 40). Cameras 30, 31, 32, 33 did not.

**What it actually is (confirmed 2026-09-05):** those 4 cameras are online and streaming —
`org_humblebee.cameras.last_heartbeat_at` is current for all ten — but have
`application: []` in the database, while the 6 that loaded have `["attendance"]` and/or
`["activity"]`. `load_cameras_from_db` (`src/config/camera_loader.py:93-109`) filters by
`application` by design. These 4 cameras were correctly excluded; they were never assigned to
this pipeline.

**Consequence for item 2:** the prior end-to-end run's ~60% camera count was the DB's real
configured load, not a partial system. **No longer a blocker for the baseline measurement** —
unless the intent is specifically to test at 10 cameras, in which case someone needs to decide
whether to assign `attendance`/`activity` to 30-33 first, which is a data change, not a code
fix.

---

# Next, each as its own PR

## 5. ~~Move video decoding into its own process~~ — DONE (2026-09-05)

**Status: code complete, 259 tests passing, and verified against 6 real cameras on
`so.stack`.** Four bugs were found and fixed during that run (lease churn at cold start,
51% duplicate frames, renewal starved by the claim loop, consumers stuck on a dead shm
segment). Measured results are in item 7. Implemented on
`refactor/lso-67-camera-frame-store`:

- `src/decode_main.py` — the decode-worker daemon (not a Celery worker).
- `src/pipeline/camera_lease.py` — `CameraLeaseManager`, the claim/renew/
  release lease scheme (see below "was" text for why, kept for context).
- `src/pipeline/decode_metrics.py` — the Redis bridge for stream health and
  the raw-frame handle, replacing the direct `StreamHandler` calls
  `engine.py`'s dashboard gauges and calibration commands used to make.
- `src/workers/frame_store.py` — new `RawFrameSlot`/`attach_and_read_raw`,
  a second `camraw_<id>` ring for calibration's full, pre-ROI frame.
- `src/pipeline/engine.py` — `StreamManager`, `CeleryCameraProducer`
  construction, and `_restart_camera_stream` all removed; the three
  couplings below are rewired to the new bridge.
- `compose.yml` / `compose.override.yml` — one `decode-worker` service
  (identical, interchangeable replicas — no `-a`/`-b` naming), person-
  tracking's CPU limit lowered from 4.0/2.0 to 2.0/1.0 (unmeasured in
  isolation — re-check on the target host).
- Tests: `test_camera_lease.py` (19), `test_decode_worker.py` (7, including
  the LSO-155 stream_url-restart regression, moved here from
  `test_engine_reload.py` since that's decode_main's job now), plus 3 new
  `RawFrameSlot` cases in `test_frame_store.py`.

**What changed from the original plan, based on real review:**
- **`camera_id % N` rejected** — it reshuffles nearly every camera when the
  replica count changes. Replaced with the Redis lease scheme (below).
- **Auto-scaling considered, then explicitly dropped** — a controller with
  Docker socket access was designed, then cut: real, separate
  infrastructure work (its own image, tuned scale timers, a pre-production
  security review), disproportionate to what's needed now. The lease
  scheme doesn't foreclose adding it later. Fixed pool, sized **N+1** by a
  human for the target host — not exact-fit — so one crash doesn't strand
  cameras.

**Still open before this is done:**
- Not yet run against real cameras / real infra (only unit-tested with
  fakes). This is the next step, not a nice-to-have — the design's central
  claim (identity-lookup fallbacks stop happening) can only be confirmed
  live.
- Recording (`save_video`) is not ported — `engine.py` now logs a warning
  and ignores the flag rather than silently doing nothing. Follow-up if
  recording is actually used.
- `SO_DECODE_WORKER_CAPACITY`/replica-count defaults in `compose.yml` are
  placeholders (capacity 6, 1 replica) — must be sized against the real
  target host's `nproc` and measured per-camera decode cost before
  deployment, per the N+1 guidance above.

### Original problem statement, kept for context

**What:** Decoding RTSP streams costs ~1 full CPU core per 1440p camera and runs as one
thread per camera inside `person-tracking` — the same process that serves the identity
register (`GlobalTrackManager`) over a socket.

**Why it matters:** Decode saturates the CPU, and the identity lookup is a *blocking call
with a timeout* competing for the same GIL. When it loses that race the lookup times out,
the track falls back to a local-only ID, and **the person is never recognised** — no
attendance record. A correctness failure, not a slowdown.

We already mitigated the symptom once (commit `52fae59`: persistent RPC connections
removed ~550–600 connection setups/sec; `SO_GPU_RPC_TIMEOUT_S` loosened 250 ms → 1.0 s).
**The underlying CPU cost is untouched.**

**Why this shape:** the problem is *co-location*, not decode speed. Moving decode to its own
process gives it its own core and leaves the main process free to answer identity questions
quickly. Same library, same reconnect handling, writing into the shared memory it already
uses. Small, relative to what this branch already did — and it becomes the natural home for
GPU decoding (item 9) later.

**Blocked by:** item 2, softly — worth confirming decode is the constraint before committing,
though this is cheap enough that the risk of doing it anyway is low.

**Decision (2026-09-05):** proceeding ahead of item 2. Live evidence already exists: a real
`GlobalTrackManager` RPC fallback (`TimeoutError`, falling back to a local-only ID) was
observed on camera 38 with only 6 of 10 cameras running — see the note in item 4's history.
The baseline measurement still matters for judging items 7 and 8, but item 5's own case does
not depend on it.

**Trigger:** Adding cameras to this host, or identity-lookup fallbacks reappearing in
`camera-worker` logs. (Already true — see above.)

---

## 6. Automatic camera-to-worker reassignment (the cut reconciler)

**What:** `cam.<id>` queues are statically assigned via each worker's `-Q` list in compose.
A dead `camera-worker` means its cameras are dark until an operator restarts the container.

**Consequence today:** Adding a camera in the DB is picked up by the engine, but no worker
consumes its new queue until someone edits config and restarts. An operational step, not
automatic.

**Why not now:** Deliberately cut after review. It was the highest-risk, highest-complexity
part of the original design — polling the control plane, a hysteresis state machine for
missed broadcasts, a purge-before-assign race, and a new failure class (two workers on one
camera) that doesn't exist today. It bought 15–30 s automatic recovery where an operator
already closes the same window. On production sites, boring and verifiable beat clever and
self-healing.

**Team position (2026-09-03):** wanted in the near future, as its **own PR**. It was cut to
keep an already-large change reviewable, not because it isn't needed.

**Build it as specified** in `LSO67_FOLLOWUP_QUEUE_DESIGN.md` §Shelved rather than
improvising a simpler version: the hysteresis handling is what stops a busy worker being
mistaken for a dead one, which would put two workers on one camera — the exact corruption
per-camera queues exist to prevent.

**Independent of item 5** — these two can run in parallel if there are people for both.

**Trigger:** Observed `camera-worker` crash frequency high enough that manual restarts
become an operational burden, or camera count growing past what one operator can manage.

---

# Answered by the live run on 2026-09-05

## 7. ~~GPU batching is not actually batching yet~~ — RESOLVED, nothing is wrong

**Was:** `yolo.detect` grouped frames correctly (mechanism verified), but the observed
batch size in production was ~1. Two explanations were open, with opposite responses:
decode couldn't supply frames fast enough (→ fix item 5), or the GPU was genuinely fast
enough that a queue never formed (→ nothing is wrong).

**The answer: both, in that order.** Decode was the constraint, and it was worse than
suspected — `StreamHandler.read()` was returning the same frame twice on average (185
reads/s, 90 distinct), so roughly half of all decode work and half of every batch slot was
spent on duplicates. Fixing that (frame-sequence stamping, commit `5798371`) plus moving
decode into its own process (item 5) changed the measured picture on `so.stack` with 6 real
cameras:

| | Before | After |
|---|---|---|
| `yolo.detect` batch size | 0.42 | **2.15** |
| Frames dropped | 84/s | **0** |
| `person-tracking` CPU | 612% | 13.7% |
| decode CPU | 547% | 122% |
| Identity RPC fallbacks | present under load | **zero** |
| `yolo` queue depth | — | 0–2 |
| GPU utilisation | — | 31% |

2.15 is inside the design's 2–4 target and matches the go/no-go spike's 2.17. With the
queue empty, the GPU at 31%, and zero drops, **there is no backlog left to batch** — the
remaining headroom is capacity for more cameras, not a defect.

**Consequence for the two untried levers** (a second `yolo-worker` replica; raising
`_RING_SIZE`): still not pulled, and now for a positive reason rather than for lack of
information. Neither helps a queue that is already empty. Revisit only if queue depth
climbs with camera count.

**Note on item 2:** this no longer blocks on Run 0. Run 0 still matters for the end-to-end
before/after latency claim, but the batching question it was going to decide is decided.

---

# Later, when the trigger fires

## 8. GPU decoding (NVDEC)

**What:** Decode on the GPU's dedicated decoder chip instead of the CPU. CPU cost drops to
near zero and the frame arrives already in GPU memory, removing the one remaining copy in
the transport path.

**Why it matters:** Raises the per-server camera ceiling substantially — decode CPU is
currently the tightest limit on how many cameras fit on one host.

**Why not now — bigger than it sounds.** The obvious route ("keep OpenCV, use its GPU
decoder") does not work with what we install. Verified on our own environment:

```
cv2.__version__          4.11.0
has cudacodec            False
CUDA devices             0
```

`opencv-python-headless` ships CPU-only wheels. Getting OpenCV's GPU decoder means
compiling OpenCV from source against the CUDA toolkit and NVIDIA's Video Codec SDK and
maintaining that custom build in our image — or switching to NVIDIA's own decoder bindings,
or driving FFmpeg's hardware decoder directly. Any route **replaces the video layer**,
including the RTSP reconnect handling we have invested in.

**Do item 5 first.** It fixes the failure we actually observe at a fraction of the effort,
and a decode worker is where a GPU decoder would live anyway. Nothing in item 5 has to be
undone to get here.

**Trigger:** The host running out of CPU — i.e. wanting more cameras per server than decode
can feed.

---

## 9. GPU workers are not pinned to a specific GPU

**What:** `yolo-worker` and `face-worker` request `capabilities: [gpu]` with no `device_ids`
in `compose.yml`.

**Why it matters:** On a single-GPU host (today) this is correct — they share the one GPU,
which batching is designed around. On a **multi-GPU host**, both would still land on the
same default GPU, leaving the second idle.

**Why not now:** Every current deployment is single-GPU. No observed impact.

**Trigger:** Before provisioning any server with more than one GPU. Fix is additive — set
`device_ids` per service — not a redesign.

---

## 10. GPU latency metrics have no publisher

**What:** `MetricsCollector.record_yolo_ms` / `record_arcface_ms` are kept as API but have
no callers — the timings now happen worker-side, in a different process.

**Why it matters:** Dashboard gauges that read these are blank.

**Why not now:** Deliberate. Deleting the API alongside its old call sites would have
silently removed the metric; keeping it marks the gap.

**Fix:** Publish worker-side timings via a Redis hash the collector reads.

---

## 11. An identity service, for multi-machine deployment

**What:** `GlobalTrackManager` is an object in one process behind a local Unix socket. To
run cameras across more than one server, it must become a service reachable over the
network — the one piece of genuinely shared state.

**Why it matters:** It is the only thing preventing multi-server deployment. Everything
else shards cleanly: each host runs a self-contained stack for its own cameras, and frames
never cross machines.

**Why it's hard, specifically:** `assign_global_id` runs a numpy similarity search over the
entire embedding gallery. That does not decompose into Redis operations — it cannot simply
"go in Redis." It needs to stay one service owning the gallery, exposed over the network.
The RPC surface is already narrow and well-defined, which helps. The hard parts are latency
(a network call replacing a 0.16 ms local one) and it becoming a single point of failure
for every host rather than one.

**Also needed:** camera-to-host assignment becomes fleet-wide — the same problem as item 6,
one level up. Good argument for building item 6 first, at single-host scale, and learning
from it.

**Trigger:** A second server being seriously considered. Not before.

---

# Watch list — not broken, but worth knowing

## `celery-batches` is a small third-party dependency

372 lines, outside Celery itself. It replaced ~1,900 lines of our own code, which is a good
trade — but if it goes unmaintained it becomes ours. The fallback is documented in
`LSO67_FOLLOWUP_QUEUE_DESIGN.md`: an in-process collector (dict behind a lock, ~40 lines).
No action needed; noted so the dependency is a known choice rather than a surprise.

## `--prefetch-multiplier=32` on `yolo-worker` is load-bearing

Not a performance tweak. `Batches` only acks after a flush, so the buffer can never exceed
the prefetch count — at the app-wide default of 1, **every batch is size 1** and batching
silently does nothing. We shipped this bug to the test stack once and caught it via
`avg_batch=0.00` plus a 33k-message backlog. If batch size ever reads ~1.0 again, check
this flag first.

## Batches drains oldest-first, not newest-first

`celery_batches._do_flush` consumes its buffer FIFO — the oldest queued frame is processed
first. For live video the opposite would be better: a 900 ms-old frame is nearly worthless
next to the fresh one behind it.

In practice the deadline check in `run_detect_batch` covers this: stale frames are dropped
before the model runs, so we approximate newest-first by discarding rather than reordering.
Adequate at current load. If we ever want *true* newest-first under sustained overload, that
is our code to write — the library will not do it.

## The same `cam.<id>` must never appear in two workers' `-Q` lists

That reintroduces the exact double-tracker corruption per-camera queues exist to prevent
(measured: global track count climbing 33 → 45 and never settling). When moving a camera
between workers, restart **both** the losing and gaining service.

---

# Done

- **Per-camera queues + consumer-side batching** (LSO-67 follow-up) — 4 commits on
  `refactor/lso-67-camera-frame-store`. GPU RPC middleman deleted, ~1,400 net lines removed,
  223/223 tests passing, verified on real hardware.
- **Identity-lookup starvation under decode load** — commit `52fae59`. Symptom fixed; root
  cause is item 5 above.
- **Missing `--prefetch-multiplier=32` in `so.stack`** — found live via `avg_batch=0.00` and
  a 33k backlog; fixed, backlog self-cleared to ~50 in seconds.
