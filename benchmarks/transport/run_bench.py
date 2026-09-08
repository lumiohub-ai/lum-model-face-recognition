#!/usr/bin/env python
"""Shared memory vs. the Redis broker, for moving one 720p frame.

Answers one architectural question: is it actually cheaper to put pixels in
shared memory and send a handle, than to put the pixels in the task payload?

The repo asserts it is (~15.6 ms vs ~0.3 ms per frame) in five places, but that
number predates every benchmark here and no committed harness ever measured a
pixel payload — celery_worker/ only ever compared shm against re-reading JPEGs
from disk, which nobody proposed. This measures the comparison that was
actually argued.

No GPU, no model: the question is transport, so inference would only add noise.

    docker run --rm -d --name xportbench-redis -p 127.0.0.1:6401:6379 redis:7-alpine
    PYTHONPATH=benchmarks .venv/bin/python benchmarks/transport/run_bench.py
    docker rm -f xportbench-redis
"""

import argparse
import base64
import json
import os
import platform
import signal
import socket
import statistics as st
import subprocess
import sys
import time
from multiprocessing import shared_memory

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, os.path.join(REPO, "benchmarks"))

import numpy as np  # noqa: E402
import psutil  # noqa: E402

from transport.bench_app import app  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "output")
# Reuse celery_worker's corpus rather than re-extracting: 200 real frames at
# exactly (720,1280,3), already verified, and sharing them keeps this run
# comparable with the committed results.json next door.
FRAMES_DIR = os.path.join(REPO, "benchmarks/celery_worker/output/frames")
VENV_CELERY = os.path.join(REPO, ".venv/bin/celery")
SHM_NAME = "xportbench_frames"
SHAPE = (720, 1280, 3)
# A small ring the producer overwrites, mirroring production's per-camera ring
# (frame_store.py _RING_SIZE) rather than a static corpus. The size is not
# load-bearing here — in-flight depth is 1 — but writing into a rotating slot
# keeps cache behaviour honest instead of rewriting one hot buffer forever.
_RING_SLOTS = 24

# Ordered so the control floor is established first. Every pixel cell is
# reported as a delta over it.
CELLS = [
    dict(cell="0-control",    task="xport.control", ser="json",   what="~100 B, no pixels"),
    # Cell 1 is the one that must match production. The producer really does
    # copy the decoded frame into the ring (frame_store.py:223 `view[:] = frame`)
    # and the consumer really does .copy() out of it (frame_store.py:391,432).
    # Timing only the handle would flatter shm by ~0.9 ms/frame and measure a
    # pipeline nobody runs.
    dict(cell="1-shm",        task="xport.shm",     ser="json",   what="handle + producer write + consumer copy (production path)"),
    dict(cell="2-shm-view",   task="xport.shm",     ser="json",   what="handle + producer write, zero-copy read (best case)"),
    dict(cell="3-json-b64",   task="xport.b64",     ser="json",   what="base64 pixels (production serializer)"),
    dict(cell="4-pickle-raw", task="xport.raw",     ser="pickle", what="raw pixel bytes"),
    dict(cell="5-jpeg",       task="xport.jpeg",    ser="pickle", what="JPEG q90 bytes"),
]


# ---------------------------------------------------------------- corpus


def load_corpus(n):
    import cv2

    if not os.path.isdir(FRAMES_DIR):
        raise SystemExit(
            f"corpus missing: {FRAMES_DIR}\n"
            "Run benchmarks/celery_worker/run_bench.py once to build it, or point\n"
            "FRAMES_DIR at another directory of 1280x720 jpgs."
        )
    paths = sorted(p for p in os.listdir(FRAMES_DIR) if p.endswith(".jpg"))[:n]
    frames = [cv2.imread(os.path.join(FRAMES_DIR, p)) for p in paths]
    bad = [p for p, f in zip(paths, frames) if f is None or f.shape != SHAPE]
    if bad:
        raise SystemExit(f"corpus frames are not {SHAPE}: {bad[:3]}")
    return frames


# ---------------------------------------------------------------- worker


def spawn(cell):
    """A LOCAL subprocess, deliberately.

    delivery_ms subtracts a driver timestamp from a worker timestamp. That is
    only meaningful because both processes read the same kernel clock, which
    holds for a local subprocess and does NOT hold across a container boundary.
    """
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("SO_CELERY", "SO_REDIS"))}  # never inherit prod pointers
    env.update(
        PYTHONPATH=os.path.join(REPO, "benchmarks"),
        OMP_NUM_THREADS="2",
        OPENCV_NUM_THREADS="1",
        # No GPU is touched, but an inherited device list could still make
        # a stray cv2/numpy build initialise one.
        CUDA_VISIBLE_DEVICES="",
    )
    argv = [VENV_CELERY, "-A", "transport.bench_app", "worker",
            "--loglevel=WARNING", "--pool=solo", "--queues=xportbench",
            f"--hostname=xp%d@{cell}",
            "--without-gossip", "--without-mingle", "--without-heartbeat"]
    log = open(os.path.join(OUT, "worker.log"), "a")
    return subprocess.Popen(argv, cwd=REPO, env=env, stdout=log,
                            stderr=subprocess.STDOUT, start_new_session=True), log


def terminate(proc):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=20)
    except (subprocess.TimeoutExpired, ProcessLookupError):
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=10)
        except Exception:
            pass


def reap_stale_workers():
    killed = []
    for p in psutil.process_iter(["pid", "cmdline"]):
        cmd = " ".join(p.info.get("cmdline") or [])
        if "transport.bench_app" in cmd and p.info["pid"] != os.getpid():
            try:
                p.kill()
                killed.append(p.info["pid"])
            except psutil.Error:
                pass
    if killed:
        print(f"reaped stale workers: {killed}")
        time.sleep(3)


def wait_ready(timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        if app.control.inspect(timeout=1).ping():
            return True
        time.sleep(0.5)
    return False


# ---------------------------------------------------------------- payloads


def make_payload(cell, frames, i, ring=None):
    """Build one task's args. Encode cost is timed by the caller.

    For the shm cells "encode" is the producer-side copy into shared memory —
    the real `view[:] = frame` that frame_store.py:223 performs on every frame.
    Pre-loading the corpus and sending only an index would measure a pipeline
    that does not exist, and would understate shm by ~0.9 ms/frame.
    """
    f = frames[i % len(frames)]
    if cell["task"] == "xport.control":
        return ("x" * 100,)
    if cell["task"] == "xport.shm":
        slot = i % _RING_SLOTS
        ring[slot][:] = f  # the production write, inside the timed region
        return (SHM_NAME, slot, [_RING_SLOTS] + list(SHAPE))
    if cell["task"] == "xport.b64":
        return (base64.b64encode(f.tobytes()).decode(),)
    if cell["task"] == "xport.raw":
        return (f.tobytes(),)
    if cell["task"] == "xport.jpeg":
        import cv2
        ok, enc = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise SystemExit("cv2.imencode failed")
        return (enc.tobytes(),)
    raise SystemExit(f"unknown task {cell['task']}")


def expected_checksum(frames, i):
    f = frames[i % len(frames)]
    return int(f[::64, ::64, :].sum())


# ---------------------------------------------------------------- one sample


def one_task(cell, frames, i, verify, ring=None):
    """Submit one frame and return its component timings, in ms."""
    t_enc0 = time.perf_counter()
    args = make_payload(cell, frames, i, ring=ring)
    t_enc1 = time.perf_counter()

    t_sent = time.time()
    t_pub0 = time.perf_counter()
    async_res = app.send_task(
        cell["task"], args=args + (t_sent,),
        # Cell 1 copies out, as production does; cell 2 keeps the view.
        kwargs={"copy": True} if cell["cell"] == "1-shm" else None,
        queue="xportbench", serializer=cell["ser"],
    )
    t_pub1 = time.perf_counter()
    out = async_res.get(timeout=120)
    t_done = time.perf_counter()

    if verify:
        if not out["shape_ok"]:
            raise SystemExit(f"{cell['cell']}: worker decoded the wrong shape")
        # JPEG is lossy, so its checksum legitimately differs; every other
        # cell must reproduce the driver's bytes exactly.
        if cell["task"] not in ("xport.control", "xport.jpeg"):
            exp = expected_checksum(frames, i)
            if out["checksum"] != exp:
                raise SystemExit(
                    f"{cell['cell']}: payload corrupted "
                    f"(checksum {out['checksum']} != {exp})")

    return {
        "encode_ms": (t_enc1 - t_enc0) * 1e3,
        "publish_ms": (t_pub1 - t_pub0) * 1e3,
        "delivery_ms": (out["t_recv"] - t_sent) * 1e3,
        "decode_ms": out["decode_ms"],
        "roundtrip_ms": (t_done - t_pub0) * 1e3,
        "nbytes": out["nbytes"],
    }


def summarise(samples, key):
    xs = sorted(s[key] for s in samples)
    return {
        "mean": round(st.mean(xs), 3),
        "p50": round(st.median(xs), 3),
        "p95": round(xs[min(len(xs) - 1, int(0.95 * len(xs)))], 3),
        "stdev": round(st.stdev(xs), 3) if len(xs) > 1 else 0.0,
        "n": len(xs),
    }


# ---------------------------------------------------------------- kombu probe


def kombu_microbench(frames, n=200):
    """Attribute the serializer half, which a task body cannot see.

    By the time a task runs, kombu has already decoded the message body — so
    xport.b64's decode_ms covers only b64->ndarray, missing the json step that
    dominates. This measures dumps/loads directly, under the same warmup and
    percentile discipline as the cells.
    """
    from kombu.serialization import dumps, enable_insecure_serializers, loads

    # kombu blocks pickle deserialization by default. The worker is allowed it
    # via accept_content; this driver-side probe needs the same permission to
    # time the pickle path. Scope: this process, this benchmark, data it just
    # serialized itself.
    enable_insecure_serializers(["pickle"])

    f = frames[0]
    b64 = base64.b64encode(f.tobytes()).decode()
    body_json = {"f": b64, "shape": list(SHAPE)}
    handle = {"shm": SHM_NAME, "idx": 0, "shape": [len(frames)] + list(SHAPE)}

    def timed(fn, warmup=20):
        for _ in range(warmup):
            fn()
        xs = []
        for _ in range(n):
            t = time.perf_counter()
            fn()
            xs.append((time.perf_counter() - t) * 1e3)
        xs.sort()
        return {"p50": round(st.median(xs), 4),
                "p95": round(xs[int(0.95 * len(xs))], 4),
                "n": n}

    ct_j, en_j, bd_j = dumps(body_json, "json")
    ct_h, en_h, bd_h = dumps(handle, "json")
    ct_p, en_p, bd_p = dumps(f.tobytes(), "pickle")
    return {
        "json_dumps_frame": {**timed(lambda: dumps(body_json, "json")),
                             "body_kib": round(len(bd_j) / 1024, 1)},
        "json_loads_frame": timed(lambda: loads(bd_j, ct_j, en_j)),
        "json_dumps_handle": {**timed(lambda: dumps(handle, "json")),
                              "body_bytes": len(bd_h)},
        "json_loads_handle": timed(lambda: loads(bd_h, ct_h, en_h)),
        "pickle_dumps_frame": {**timed(lambda: dumps(f.tobytes(), "pickle")),
                               "body_kib": round(len(bd_p) / 1024, 1)},
        "pickle_loads_frame": timed(lambda: loads(bd_p, ct_p, en_p)),
    }


# ---------------------------------------------------------------- main


def run_suite(cells, frames, n_tasks, warmup, verify, ring):
    """Round-robin the cells rather than draining each in turn.

    A transient CPU spike during a sequential run poisons whichever cell it
    landed in and leaves the others clean, which silently fabricates a
    difference. Interleaving spreads any such disturbance across all cells.

    A fixed order is not enough, though: whichever cell runs right after
    another cell in every single iteration inherits *that neighbour's*
    after-effects on every sample, never a random one. Measured proof of
    this, twice: with a fixed order, the control cell came out with
    delivery_ms ~2x every shm cell's (tight, 0% spread across two runs — a
    real bias, not noise). Merely rotating the *start* position was not
    enough either — over a short cell list every cell still has the same
    fixed predecessor most of the time, just offset — and the warning still
    fired on the second attempt. A full shuffle each iteration is what
    actually randomises neighbours.
    """
    import random

    rng = random.Random(1234)  # fixed seed: run is reproducible, not just its stats
    samples = {c["cell"]: [] for c in cells}
    total = warmup + n_tasks
    for i in range(total):
        order = list(cells)
        rng.shuffle(order)
        for c in order:
            s = one_task(c, frames, i, verify=verify and i >= warmup, ring=ring)
            if i >= warmup:  # discard warmup: JIT, allocator, redis buffers
                samples[c["cell"]].append(s)
        if i and i % 50 == 0:
            print(f"    {i}/{total}")
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=500, help="measured tasks per cell")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--corpus", type=int, default=200)
    ap.add_argument("--runs", type=int, default=2,
                    help="full suite repetitions; 2+ exposes run-to-run drift")
    ap.add_argument("--max-load", type=float, default=4.0,
                    help="abort if 1-min load average exceeds this")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip per-task checksum (it costs ~0.1ms on the driver)")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)

    load1 = os.getloadavg()[0]
    if load1 > args.max_load:
        raise SystemExit(
            f"1-min load average is {load1:.1f} (limit {args.max_load}).\n"
            "Stop the dev stack first: docker compose -p sostack-oybek down\n"
            "Measuring under load produces numbers that cannot be defended.")

    reap_stale_workers()
    frames = load_corpus(args.corpus)
    print(f"corpus: {len(frames)} frames at {frames[0].shape}, "
          f"{frames[0].nbytes / 1024**2:.2f} MiB each")

    # A ring the producer writes into per frame — NOT a pre-loaded corpus.
    # Production copies every decoded frame in (frame_store.py:223), so that
    # copy belongs inside the measurement.
    try:
        shared_memory.SharedMemory(name=SHM_NAME).unlink()
    except FileNotFoundError:
        pass
    ring_shape = (_RING_SLOTS,) + SHAPE
    nbytes = int(np.prod(ring_shape))
    shm = shared_memory.SharedMemory(name=SHM_NAME, create=True, size=nbytes)
    ring = np.ndarray(ring_shape, dtype=np.uint8, buffer=shm.buf)
    print(f"shared ring {SHM_NAME}: {ring_shape}, {shm.size / 1024**2:.0f} MiB "
          f"(producer writes one slot per frame, as production does)")

    proc = log = None
    try:
        proc, log = spawn("xport")
        if not wait_ready():
            raise SystemExit(f"worker never became ready — see {OUT}/worker.log")

        ping = app.send_task("xport.probe", queue="xportbench").get(timeout=30)
        # delivery_ms is only meaningful if both clocks are the same clock.
        if not psutil.pid_exists(ping["pid"]):
            raise SystemExit("worker is not a local process — delivery_ms would be invalid")
        print(f"worker pid {ping['pid']} (local)\n")

        runs = []
        for r in range(args.runs):
            print(f"### run {r + 1}/{args.runs}  ({args.tasks} tasks/cell "
                  f"after {args.warmup} warmup, interleaved)")
            samples = run_suite(CELLS, frames, args.tasks, args.warmup,
                                verify=not args.no_verify, ring=ring)
            runs.append({
                c["cell"]: {
                    "what": c["what"],
                    "serializer": c["ser"],
                    "wire_bytes": samples[c["cell"]][0]["nbytes"],
                    **{k: summarise(samples[c["cell"]], k) for k in
                       ("encode_ms", "publish_ms", "delivery_ms", "decode_ms",
                        "roundtrip_ms")},
                    "transport_ms": summarise(
                        [{"transport_ms": s["encode_ms"] + s["publish_ms"]
                          + s["delivery_ms"] + s["decode_ms"]}
                         for s in samples[c["cell"]]], "transport_ms"),
                } for c in CELLS})

        print("\nkombu serializer microbench...")
        kombu = kombu_microbench(frames)
    finally:
        if proc:
            terminate(proc)
        if log:
            log.close()
        shm.close()
        shm.unlink()

    meta = {
        "measured_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "repo_rev": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                   capture_output=True, text=True).stdout.strip(),
        "host": socket.gethostname(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": psutil.cpu_count(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "celery": __import__("celery").__version__,
        "redis_py": __import__("redis").__version__,
        "frame_shape": list(SHAPE),
        "frame_bytes": int(frames[0].nbytes),
        "loadavg_1min_at_start": round(load1, 2),
        "worker": "local subprocess (delivery_ms valid: shared kernel clock)",
        "tasks_per_cell": args.tasks,
        "warmup_discarded": args.warmup,
        "note": "Transport only. No model, no GPU: the question is how pixels "
                "move between processes, so inference would only add variance.",
    }
    path = os.path.join(OUT, "results.json")
    with open(path, "w") as f:
        json.dump({"meta": meta, "runs": runs, "kombu": kombu}, f, indent=2)

    report(runs, kombu, path)


def report(runs, kombu, path):
    first = runs[0]
    floor = first["0-control"]["transport_ms"]["p50"]
    cheapest = min(first[c["cell"]]["transport_ms"]["p50"] for c in CELLS)
    if floor > cheapest + 0.5:
        # The control cell should be the cheapest thing measured — it does no
        # work. If something beats it, the floor is contaminated (this really
        # happened: a fixed cell order let 0-control inherit queueing latency
        # from whichever cell ran last each round — see run_suite's docstring)
        # and "net" would be comparing against a number that overstates the
        # true per-task overhead.
        print(f"\n!! WARNING: control floor ({floor:.2f} ms) exceeds the cheapest "
              f"measured cell ({cheapest:.2f} ms) by >0.5ms.")
        print("   The floor is probably contaminated (see run_suite() docstring). "
              "'net' below is not trustworthy; use absolute transport_ms instead.")

    print(f"\n{'cell':14} {'wire':>10} {'enc':>7} {'pub':>7} {'deliv':>7} "
          f"{'dec':>7} {'TOTAL':>8} {'net':>8}")
    print("-" * 80)
    for c in CELLS:
        r = first[c["cell"]]
        net = r["transport_ms"]["p50"] - floor
        wire = r["wire_bytes"]
        wire_s = f"{wire/1024**2:.2f} MiB" if wire > 8192 else f"{wire} B"
        print(f"{c['cell']:14} {wire_s:>10} "
              f"{r['encode_ms']['p50']:>7.2f} {r['publish_ms']['p50']:>7.2f} "
              f"{r['delivery_ms']['p50']:>7.2f} {r['decode_ms']['p50']:>7.2f} "
              f"{r['transport_ms']['p50']:>8.2f} "
              f"{net:>8.2f}{'' if c['cell'] != '0-control' else ' (floor)'}")
    print("\nAll figures are p50 ms per frame. 'net' subtracts the control floor:\n"
          "that difference is what sending the pixels actually costs.")

    # Stability gate across runs.
    if len(runs) > 1:
        print(f"\n{'cell':14} " + " ".join(f"{'run'+str(i+1):>9}" for i in range(len(runs)))
              + f" {'spread':>9}")
        unstable = []
        for c in CELLS:
            vals = [r[c["cell"]]["transport_ms"]["p50"] for r in runs]
            lo, hi = min(vals), max(vals)
            spread = (hi - lo) / lo * 100 if lo > 0 else 0.0
            if spread > 15:
                unstable.append(c["cell"])
            print(f"{c['cell']:14} " + " ".join(f"{v:>9.2f}" for v in vals)
                  + f" {spread:>8.0f}%")
        if unstable:
            print(f"\n!! UNSTABLE (>15% between runs): {', '.join(unstable)}")
            print("   Treat these as indicative only. Re-run on an idle machine.")
        else:
            print("\nAll cells stable within 15% across runs.")

    shm_p50 = first["1-shm"]["transport_ms"]["p50"]
    b64_p50 = first["3-json-b64"]["transport_ms"]["p50"]
    best_broker = min((first[c["cell"]]["transport_ms"]["p50"], c["cell"])
                      for c in CELLS if c["cell"] in
                      ("3-json-b64", "4-pickle-raw", "5-jpeg"))
    if shm_p50 > 0:
        print(f"\nHEADLINE (per 720p frame, p50, incl. producer copy into shm):")
        print(f"  shared memory (production path) : {shm_p50:>7.2f} ms")
        print(f"  Redis payload, json+base64      : {b64_p50:>7.2f} ms "
              f"= {b64_p50 / shm_p50:.1f}x")
        print(f"  Redis, best case ({best_broker[1]:<12})  : {best_broker[0]:>7.2f} ms "
              f"= {best_broker[0] / shm_p50:.1f}x")
    print(f"\nkombu serializer, p50 ms:")
    for k, v in kombu.items():
        extra = v.get("body_kib") or v.get("body_bytes")
        unit = "KiB" if "body_kib" in v else ("B" if "body_bytes" in v else "")
        print(f"  {k:22} {v['p50']:>8.4f}   {str(extra) + ' ' + unit if extra else ''}")

    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
