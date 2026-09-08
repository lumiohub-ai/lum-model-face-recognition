"""Go/no-go spike driver for celery-batches (LSO-67 follow-up).

Runs a real Celery worker process (--pool=solo, matching the production
camera-worker/yolo-worker pool choice) against a throwaway Redis, and a
producer thread simulating N cameras submitting detect_task requests at a
fixed rate, each carrying a `deadline` kwarg.

This answers, with real numbers, the questions
docs/LSO67_FOLLOWUP_QUEUE_DESIGN.md "Step 0" requires before any of the
plan's steps 1-6 are implemented:

  1. What batch sizes does celery-batches actually achieve under --pool=solo
     with flush_every=8 / flush_interval=10ms, at a load comparable to 10
     cameras @ 7.5Hz (75 req/s)?
  2. Does the 10ms flush_interval timer actually fire near 10ms on the kombu
     hub, or is real granularity much coarser?
  3. Do unacked messages actually reach zero after the run (task_acks_late
     bookkeeping is not silently leaking)?
  4. Are expired (past-deadline) requests skipped rather than processed late?

Usage:
    # terminal 1:
    docker run --rm -d --name lso67batch-redis -p 6402:6379 redis:7-alpine
    # terminal 2:
    PYTHONPATH=benchmarks python benchmarks/celery_batches_spike/run_bench.py

    # separately, to check point 4 (redelivery after a crash):
    PYTHONPATH=benchmarks python benchmarks/celery_batches_spike/run_bench.py \\
        --kill-mid-batch --seconds 20
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import subprocess
import sys
import time
from dataclasses import dataclass

from celery_batches_spike.bench_app import app, BROKER
from celery_batches_spike.bench_tasks import detect_task, FLUSH_INTERVAL, OUTCOMES_PATH

N_CAMERAS = 10
RATE_HZ = 7.5
RUN_SECONDS = 30
# How long a request is allowed to sit before it's considered "expired" --
# mirrors the real design's producer-side `deadline = now + _TASK_EXPIRES_S`.
DEADLINE_BUDGET_S = 1.0


@dataclass
class ProducerStats:
    sent: int
    send_errors: int


def _producer(stop_at: float, rate_hz: float, n_cameras: int) -> ProducerStats:
    interval = 1.0 / (rate_hz * n_cameras)
    sent = 0
    errors = 0
    next_cam = 0
    while time.time() < stop_at:
        camera_id = next_cam
        next_cam = (next_cam + 1) % n_cameras
        deadline = time.time() + DEADLINE_BUDGET_S
        try:
            detect_task.apply_async(
                kwargs={
                    "camera_id": camera_id,
                    "frame_num": sent,
                    "deadline": deadline,
                },
                queue="batchbench",
                expires=DEADLINE_BUDGET_S,
            )
            sent += 1
        except Exception as exc:  # noqa: BLE001 - spike harness, report and continue
            errors += 1
            print(f"send error: {exc!r}", file=sys.stderr)
        time.sleep(interval)
    return ProducerStats(sent=sent, send_errors=errors)


def _redis_cli(*args: str) -> str:
    """Query the spike broker.

    Prefers a real `redis-cli` on the host if present; falls back to `docker
    exec` into the throwaway broker container (this dev box has no redis-cli
    binary installed), and finally to the `redis` Python client if neither
    shell tool is available.
    """
    hostport = BROKER.split("//", 1)[1].split("/", 1)[0]
    host, port = hostport.split(":")

    container = os.environ.get("BATCHBENCH_REDIS_CONTAINER", "lso67batch-redis")
    for cmd in (
        ["redis-cli", "-h", host, "-p", port, *args],
        ["docker", "exec", container, "redis-cli", *args],
    ):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                return result.stdout.strip()
        except FileNotFoundError:
            continue

    try:
        import redis as redis_py

        r = redis_py.Redis(host=host, port=int(port))
        return str(r.execute_command(*args))
    except Exception as exc:  # noqa: BLE001 - best-effort diagnostic only
        return f"<redis query failed: {exc!r}>"


def _run_worker(argv: list) -> None:
    app.worker_main(argv)


def _start_worker() -> multiprocessing.Process:
    worker_argv = [
        "worker",
        "--pool=solo",
        "--prefetch-multiplier=32",
        "--loglevel=WARNING",
        "--queues=batchbench",
        "--without-gossip",
        "--without-mingle",
        "--without-heartbeat",
    ]
    proc = multiprocessing.Process(target=_run_worker, args=(worker_argv,), daemon=True)
    proc.start()
    return proc


def _summarize_outcomes(path: str) -> None:
    if not os.path.exists(path):
        print(f"no outcomes file at {path} -- did the worker process any batches?")
        return
    sizes, walls, gaps, expired = [], [], [], 0
    prev_ts = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sizes.append(rec["batch_size"])
            walls.append(rec["wall_ms"])
            expired += rec["n_expired"]
            ts = rec["flushed_at"]
            if prev_ts is not None:
                gaps.append((ts - prev_ts) * 1000.0)
            prev_ts = ts

    if not sizes:
        print("outcomes file exists but is empty.")
        return

    n = len(sizes)
    mean_size = sum(sizes) / n
    mean_wall = sum(walls) / n
    mean_gap = sum(gaps) / len(gaps) if gaps else float("nan")
    p50_gap = sorted(gaps)[len(gaps) // 2] if gaps else float("nan")

    print(f"--- batch outcomes ({n} flushes) ---")
    print(f"batch size: mean={mean_size:.2f}  min={min(sizes)}  max={max(sizes)}  "
          f"histogram={_histogram(sizes)}")
    print(f"flush wall time: mean={mean_wall:.1f}ms")
    print(f"inter-flush gap (proxy for real flush_interval, target ~{FLUSH_INTERVAL*1000:.0f}ms): "
          f"mean={mean_gap:.1f}ms  p50={p50_gap:.1f}ms")
    print(f"expired (past-deadline) requests skipped: {expired}")


def _histogram(values: list) -> dict:
    hist: dict = {}
    for v in values:
        hist[v] = hist.get(v, 0) + 1
    return dict(sorted(hist.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cameras", type=int, default=N_CAMERAS)
    parser.add_argument("--rate-hz", type=float, default=RATE_HZ)
    parser.add_argument("--seconds", type=int, default=RUN_SECONDS)
    parser.add_argument(
        "--kill-mid-batch", action="store_true",
        help="SIGKILL the worker partway through, to test deadline-based "
             "redelivery handling. Run this separately from the main timing run.",
    )
    args = parser.parse_args()

    if os.path.exists(OUTCOMES_PATH):
        os.remove(OUTCOMES_PATH)

    print(f"=== celery-batches spike: {args.cameras} cameras @ {args.rate_hz}Hz, "
          f"{args.seconds}s ===")
    print(f"broker: {BROKER}")
    print(f"flush_every={detect_task.flush_every} flush_interval={FLUSH_INTERVAL}s")
    print()

    worker_proc = _start_worker()
    print(f"worker pid={worker_proc.pid}, warming up 3s...")
    time.sleep(3.0)

    if args.kill_mid_batch:
        stop_at = time.time() + args.seconds
        prod_proc = multiprocessing.Process(
            target=_producer, args=(stop_at, args.rate_hz, args.cameras)
        )
        prod_proc.start()
        time.sleep(args.seconds / 2)
        print(f"SIGKILL worker pid={worker_proc.pid} mid-run...")
        worker_proc.kill()
        worker_proc.join()
        time.sleep(1.0)
        print("restarting worker to drain redelivered messages...")
        worker_proc = _start_worker()
        time.sleep(3.0)
        prod_proc.join()
        time.sleep(2.0)
        worker_proc.terminate()
        worker_proc.join(timeout=5)
        _summarize_outcomes(OUTCOMES_PATH)
        print()
        print("Check above: any flush after the restart with n_expired > 0 confirms")
        print("redelivered (post-crash) messages past their deadline were skipped,")
        print("not reprocessed as if fresh.")
        return

    stop_at = time.time() + args.seconds
    t_start = time.time()
    stats = _producer(stop_at, args.rate_hz, args.cameras)
    t_send_done = time.time()

    # Let the last flush + acks settle.
    time.sleep(max(1.0, FLUSH_INTERVAL * 5))

    unacked = _redis_cli("llen", "unacked")
    queue_len = _redis_cli("llen", "batchbench")

    worker_proc.terminate()
    worker_proc.join(timeout=5)

    print()
    print("--- producer ---")
    print(f"sent: {stats.sent}  errors: {stats.send_errors}  "
          f"wall: {t_send_done - t_start:.1f}s")
    print()
    print("--- broker state after settle ---")
    print(f"LLEN unacked: {unacked!r}  (want 0 or empty -- acks completed)")
    print(f"LLEN batchbench (still-queued): {queue_len!r}  (want 0)")
    print()
    _summarize_outcomes(OUTCOMES_PATH)


if __name__ == "__main__":
    main()
