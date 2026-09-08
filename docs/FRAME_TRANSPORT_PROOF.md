# Why camera frames travel through shared memory, not Redis

A plain-language walkthrough of one architectural decision and the benchmark that proves it.
No prior knowledge of this system assumed.

## The setup, in one picture

This app watches office cameras with two things happening in **separate processes**:

- One process pulls video from the camera and decodes each frame into a picture.
- Other processes (running AI models) need to look at that picture to find people and faces.

Separate processes cannot normally see each other's memory — a picture decoded in process A does
not just "exist" for process B to read. Something has to move it across.

```mermaid
flowchart LR
    CAM["📷 Camera<br/>(RTSP video)"] --> DEC["Decode process<br/>turns video into a picture"]
    DEC -.->|"picture must get here<br/>somehow"| AI["AI process<br/>looks for people/faces"]
```

Two ways to bridge that gap were on the table:

1. **Send the actual picture** through the message queue (Redis) that already connects these
   processes for small coordination messages.
2. **Put the picture in shared memory** — a piece of RAM both processes can directly point at —
   and send only a short note through the queue saying *where* to look.

## The two designs, side by side

```mermaid
flowchart TB
    subgraph A["Option 1 — picture through the queue"]
        direction LR
        A1["Decode process"] -->|"whole picture,<br/>~2.6 MB"| A2["Redis"] -->|"whole picture"| A3["AI process"]
    end
    subgraph B["Option 2 — picture through shared memory"]
        direction LR
        B1["Decode process"] -->|"writes picture"| B2["Shared memory<br/>(RAM both sides see)"]
        B1 -->|"tiny note,<br/>~50 bytes"| B3["Redis"] -->|"tiny note"| B4["AI process"]
        B4 -->|"reads picture"| B2
    end
```

Option 2 is what this app actually does. This document is about proving that choice was right —
not by assertion, but by measuring both options and comparing.

## Why this needed proving at all

The code and docs already claimed shared memory was faster, with a specific number: "~15.6 ms
for a picture through Redis vs ~0.3 ms for a note." That number turned out to have a problem —
**nobody could find where it came from.** It first appeared in a commit message, three days
*before* any benchmark existed in this repo that could have measured it. Searching the entire
project history — every commit, every deleted branch, everything — turned up only copies of the
sentence, never a measurement.

Worse, the benchmark that *did* exist compared shared memory against the wrong opponent: reading
JPEG files off a hard disk. Nobody ever proposed reading frames off disk — the real alternative
being weighed was Redis. So the existing proof didn't actually prove the thing it claimed to.

## What was measured, and where

A new, self-contained benchmark: **[`benchmarks/transport/`](../benchmarks/transport/)**

| file | what it does |
|---|---|
| [`bench_app.py`](../benchmarks/transport/bench_app.py) | sets up an isolated test version of the message queue, separate from the real one |
| [`bench_tasks.py`](../benchmarks/transport/bench_tasks.py) | the receiving side — six small jobs, one per way of sending a picture |
| [`run_bench.py`](../benchmarks/transport/run_bench.py) | the driver: sends 500 real pictures through each method, times every step, checks nothing got corrupted along the way |
| [`README.md`](../benchmarks/transport/README.md) | the full method and reasoning, for anyone who wants to re-run it |
| [`output/results.json`](../benchmarks/transport/output/results.json) | the actual numbers from the last run, with a timestamp and machine details |

It used 200 real video frames already sitting in this repo (`benchmarks/celery_worker/output/frames/`)
— the same frames a real camera would produce, not synthetic test data.

## The comparison it ran

```mermaid
flowchart LR
    F["One real camera frame"] --> M1["Shared memory<br/>(what production does)"]
    F --> M2["Redis, as a compressed JPEG<br/>(the cheapest way to send it)"]
    F --> M3["Redis, as raw pixel bytes"]
    F --> M4["Redis, the way this app<br/>actually formats messages"]

    M1 --> R1["3.05 ms"]
    M2 --> R2["19.8 ms"]
    M3 --> R3["94.9 ms"]
    M4 --> R4["174.8 ms"]

    style R1 fill:#1a7f37,color:#fff
    style R2 fill:#9a6700,color:#fff
    style R3 fill:#9a6700,color:#fff
    style R4 fill:#cf222e,color:#fff
```

Every one of these times includes the **whole trip**: preparing the picture to send, handing it
to the queue, the queue delivering it, and the receiving side unpacking it — not just one leg of
the journey.

### What's actually inside that 3.05 ms for shared memory

Splitting shared memory's number into its parts matters, because most of it isn't copying at all:

| step | time |
|---|---|
| Copy the picture **into** shared memory | 0.29 ms |
| Send the note through the queue, both directions | ~2.2 ms |
| Copy the picture **out of** shared memory | 0.29 ms |
| **Total** | **3.05 ms** |

The actual memory copying — the thing "shared memory" refers to — costs about **0.6 ms**. The
other ~2.2 ms is Redis/Celery overhead for delivering *any* message, no matter how small; the
50-byte note pays nearly all of that too. That overhead isn't a shared-memory cost — it's the
fixed price of using this queue at all, and every method on this page pays some version of it.

### Why the picture is "heavy" even though it's just a raw array, not a compressed image

The picture is stored as a plain grid of numbers (a NumPy array) — not a JPEG, no compression.
That's true on **both** sides of this comparison: the "raw pixels" and "this app's real format"
rows above send the exact same uncompressed array shared memory does. So the format is not
mismatched between the two options; a 720×1280 color picture is just large on its own —
**2.64 MB**, however you refer to it.

What makes sending it through Redis expensive is a separate problem: Redis's messages are text
(JSON), and raw binary bytes can't go into text directly. So the picture first has to be
translated into text-safe form (base64), which **inflates it another 33%** and is pure CPU work
that has nothing to do with the picture's actual content. Shared memory skips this translation
entirely — it copies the same raw bytes both options are working with, with no extra step.

## The result

| method | time per picture | how much slower than shared memory |
|---|---|---|
| **Shared memory** (what this app does) | **3.05 ms** | — |
| Redis, best-case (JPEG) | 19.8 ms | **6.5× slower** |
| Redis, raw pixels | 94.9 ms | 31× slower |
| Redis, this app's real message format | 174.8 ms | **57× slower** |

**Shared memory wins by 6.5× even against the cheapest possible way to use Redis, and by 57×
against how this app would actually have to format the message.** The original decision holds —
and now there's a number anyone can re-check by running the benchmark themselves.

## Why the gap is that big

It comes down to how much work each side does per picture:

```mermaid
flowchart TD
    subgraph shm["Shared memory: ~3 ms"]
        s1["Copy picture into shared RAM"] --> s2["Send a 50-byte note"] --> s3["Copy picture back out"]
    end
    subgraph redis["Redis, real format: ~175 ms"]
        r1["Turn picture into text<br/>(base64 encoding)"] --> r2["Wrap it in a message<br/>and send the whole thing"] --> r3["Unwrap the message"] --> r4["Turn the text back<br/>into a picture"]
    end
```

Shared memory only ever copies raw bytes twice. Sending a picture through Redis has to first
translate it into a text-safe format (because that's how these messages are normally encoded),
which is the expensive step — and that step doesn't exist at all on the shared-memory side.

## One thing this does *not* claim

**It only holds when both processes are on the same computer.** Shared memory doesn't work
across machines — if this system ever needs to split across multiple computers, sending frames
between them would need a different answer than either option here. That question is
deliberately out of scope for this benchmark.

## How to check this yourself

```bash
docker run --rm -d --name xportbench-redis -p 127.0.0.1:6401:6379 redis:7-alpine
PYTHONPATH=benchmarks .venv/bin/python benchmarks/transport/run_bench.py
docker rm -f xportbench-redis
```

Full method, including how the timing was kept honest (checksums, shuffled test order, two full
runs compared against each other for consistency) is in
[`benchmarks/transport/README.md`](../benchmarks/transport/README.md).
