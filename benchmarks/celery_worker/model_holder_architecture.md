# model_holder.py — architecture

Traces actual execution: what runs, in what order, under which lock, on
which thread. Each diagram maps to specific lines in
[`benchmarks/celery_worker/model_holder.py`](../../../workspace/so.model-face-recognition/benchmarks/celery_worker/model_holder.py).

## Process state

What lives in this process, and who is allowed to touch it.

```mermaid
flowchart TB
    subgraph proc["one Celery worker process"]
        MODEL["_MODEL\nDetectionModel nn.Module\non GPU · written once"]
        YOLO["_YOLO\nraw ultralytics wrapper\nnaive_shared arm only"]
        LOCK1["_LOAD_LOCK\nguards first-time load"]
        LOCK2["_PRED_LOCK\nguards predictor construction"]

        subgraph tls["_TLS  (threading.local)"]
            TA["thread A → predictor"]
            TB["thread B → predictor"]
            TC["thread C → predictor"]
            TD["thread D → predictor"]
        end
    end

    TA -- "setup_model(model=_MODEL)\nno copy" --> MODEL
    TB -- "setup_model(model=_MODEL)" --> MODEL
    TC -- "setup_model(model=_MODEL)" --> MODEL
    TD -- "setup_model(model=_MODEL)" --> MODEL

    style MODEL fill:#3a2c17,stroke:#e0973f,color:#f0dcc0
    style tls fill:#17322c,stroke:#5cbcab,color:#d7ece7
```

**The rule the whole file exists to enforce:** `_MODEL` is written exactly
once and read by everyone. `_TLS` is written by every thread, once each,
and never shared.

---

## `ensure_loaded()` — line 40

Called from `worker_process_init` (fires post-fork) and again at the top of
every task, because `solo`/`threads` pools never emit that signal.

```mermaid
sequenceDiagram
    participant A as thread A
    participant L as _LOAD_LOCK
    participant M as _MODEL (module global)
    participant B as thread B

    A->>M: read _MODEL
    Note over A: None → keep going
    A->>L: acquire()
    activate L
    A->>M: read _MODEL (recheck)
    Note over A: still None → really load
    A->>A: PersonDetector(...)
    A->>M: _MODEL = det.model.model

    B->>M: read _MODEL
    Note over B: None → keep going
    B->>L: acquire() — BLOCKS
    A->>L: release()
    deactivate L
    L->>B: acquired
    activate L
    B->>M: read _MODEL (recheck)
    Note over B: now set → skip load entirely
    B->>L: release()
    deactivate L
```

Two checks, one lock (lines 48–52):

```python
if _MODEL is not None:      # cheap check, no lock — the common case after warmup
    return
with _LOAD_LOCK:
    if _MODEL is not None:  # someone else won the race while we waited
        return
    ...load...
```

Without the second check, two threads that both pass the first check would
both load the model — wasted GPU memory, and a `_MODEL` reassignment mid-use.

---

## `get_predictor()` — line 73

Runs on **every task**, every thread. Cheap after the first call.

```mermaid
flowchart TD
    Start(["get_predictor() called"]) --> Check{"_TLS has\na predictor?"}
    Check -- "yes (typical)" --> Return(["return it — no lock touched"])
    Check -- "no (first task\non this thread)" --> Acquire["acquire _PRED_LOCK"]
    Acquire --> Build["DetectionPredictor(overrides)"]
    Build --> Setup["setup_model(model=_MODEL)\n→ AutoBackend\n→ PyTorchBackend.load_model\n→ weight.to(device) — no-op, no copy"]
    Setup --> Warm["run 1 dummy inference\n(burns cuDNN autotune)"]
    Warm --> Release["release _PRED_LOCK"]
    Release --> Store["_TLS.predictor = p"]
    Store --> Return2(["return p"])

    style Check fill:#f0dcc0,stroke:#c97a2b,color:#1c1a17
    style Acquire fill:#f5dcd8,stroke:#b8453a,color:#1c1a17
    style Release fill:#f5dcd8,stroke:#b8453a,color:#1c1a17
```

**Why `_PRED_LOCK` exists at all** (comment at lines 31–35): `setup_model`
calls `BaseModel.fuse()`, guarded by `is_fused()`. That guard is safe for
**sequential** calls but not **concurrent** ones — two threads can both see
"not fused yet," both enter `fuse()`, and the second one crashes on
`delattr(m, "bn")` because the first already removed it. Observed while
building this harness, not theoretical.

Once a thread has its predictor cached in `_TLS`, it never touches the lock
again — inference itself is lock-free.

---

## `infer()` — line 97, the branch that is actually being measured

```mermaid
flowchart TD
    Start(["infer(frames)"]) --> Which{"SHARE == ?"}

    Which -- "naive_shared" --> WarmCheck{"_naive_warm?"}
    WarmCheck -- "no" --> NLock["acquire _PRED_LOCK\n(first call only)"]
    NLock --> NWarm["_YOLO(frames) — warms it\nset _naive_warm = True"]
    NWarm --> NRelease["release _PRED_LOCK"]
    NRelease --> NCall["_YOLO(frames)\nunguarded from here on"]
    WarmCheck -- "yes" --> NCall
    NCall --> NResult(["Results\n— but every call re-enters\nYOLO's OWN internal _lock"])

    Which -- "per_thread\n(default)" --> PCall["get_predictor()(frames)\nthread's own predictor"]
    PCall --> PResult(["Results\n— no shared lock at all"])

    style NCall fill:#f5dcd8,stroke:#b8453a,color:#1c1a17
    style PCall fill:#d7ece7,stroke:#2f7d70,color:#1c1a17
    style NResult fill:#f5dcd8,stroke:#b8453a,color:#1c1a17
    style PResult fill:#d7ece7,stroke:#2f7d70,color:#1c1a17
```

The `naive_shared` branch (lines 99–112) only serializes its **first** call
— after that it calls `_YOLO(frames)` directly, unguarded by our code. But
`_YOLO` is ultralytics' own `Model.__call__`, and internally **that** locks
around every single inference (`BasePredictor.stream_inference` wraps its
body in `self._lock` — see the module docstring at lines 7–15). So four
threads on `naive_shared` still queue up on every call, just on a lock we
didn't write and can't see from here.

The `per_thread` branch (line 113) never touches a shared lock after
warmup — each thread's predictor is entirely its own.

**What this predicts, and what was measured:**

| branch | expected | measured |
|---|---|---|
| `per_thread`, 4 threads | parallel inference | **167 fps** |
| `naive_shared`, 4 threads | serialized by ultralytics' internal lock | **65 fps** — slower than 1 thread (82 fps) |

---

## Putting it together — one task, start to finish

```mermaid
sequenceDiagram
    participant Task as Celery task (bench_tasks.infer)
    participant MH as model_holder
    participant TLS as _TLS (this thread)
    participant GPU as _MODEL (shared)

    Task->>MH: ensure_loaded()
    Note over MH: no-op after first call\non this process
    Task->>MH: infer(frames)
    MH->>MH: get_predictor()
    alt first call on this thread
        MH->>TLS: build + setup_model(_MODEL) + warm
        Note right of MH: under _PRED_LOCK
    else already warmed
        MH->>TLS: read cached predictor
        Note right of MH: no lock
    end
    MH->>GPU: predictor(frames) — forward pass
    GPU-->>MH: Results
    MH-->>Task: Results
```

This is the path every one of the 1200 frames in a benchmark cell takes.
The only per-task cost after warmup is the forward pass itself — everything
above it (`ensure_loaded`, the `_TLS` lookup) is a few nanoseconds once the
process has run a handful of tasks.
