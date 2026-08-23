#!/usr/bin/env python
"""Celery YOLO worker spike — driver.

Answers one question: if a single YOLO instance is shared by N threads, what
happens to throughput, latency and VRAM, and how does that compare to N
processes each holding their own copy?

Not a unit test — needs a GPU, a live Redis and minutes of wall time. The name
has no ``test_`` prefix so ``pytest tests/`` will not collect it.

    docker run --rm -d --name yolobench-redis -p 6401:6379 redis:7-alpine
    PYTHONPATH=tests .venv/bin/python tests/workers/run_bench.py
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from statistics import median

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, os.path.join(REPO, "tests"))

import psutil  # noqa: E402
import pynvml  # noqa: E402
from celery import group  # noqa: E402

from workers.bench_app import app  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "output")
FRAMES_DIR = os.path.join(OUT, "frames")
WEIGHTS = os.path.join(REPO, "volumes/models/weights")
VENV_CELERY = os.path.join(REPO, ".venv/bin/celery")

# All cells keep production's task_acks_late=True. They also use a prefetch
# window wider than the driver's in-flight depth, because the threads pool
# stalls to ~1 task/s whenever in-flight >= the prefetch window (see README);
# prefork is immune. Cell 7 pins production's prefetch=1 to document that.
CELLS = [
    dict(cell="1-solo-b1",        pool="solo",    conc=1, share="per_thread",   batch=1),
    dict(cell="2-threads4-b1",    pool="threads", conc=4, share="per_thread",   batch=1),
    dict(cell="3-prefork4-b1",    pool="prefork", conc=4, share="per_thread",   batch=1),
    dict(cell="4-threads4-b6",    pool="threads", conc=4, share="per_thread",   batch=6),
    dict(cell="5-prefork4-b6",    pool="prefork", conc=4, share="per_thread",   batch=6),
    dict(cell="6-threads4-naive", pool="threads", conc=4, share="naive_shared", batch=1),
    # Production delivery semantics on the threads pool. Expect a collapse.
    dict(cell="7-threads4-prod-prefetch", pool="threads", conc=4, share="per_thread",
         batch=1, prefetch=1, frames=60),
    # Frames by shared-memory handle instead of a file path. Isolates Celery's
    # real broker cost from the JPEG decode the path mode forces into each task.
    dict(cell="8-solo-b1-shm",     pool="solo",    conc=1, share="per_thread", batch=1, mode="shm"),
    dict(cell="9-threads4-b6-shm", pool="threads", conc=4, share="per_thread", batch=6, mode="shm"),
    dict(cell="10-prefork4-b6-shm", pool="prefork", conc=4, share="per_thread", batch=6, mode="shm"),
]
SHM_NAME = "yolobench_frames"
DEFAULT_PREFETCH = 16


# ---------------------------------------------------------------- corpus


def build_corpus(n_frames, video):
    """Extract frames from a real video, cached across runs.

    notebooks/input/ holds only 7 images; a loop that short sits entirely in
    page cache and would measure zero real I/O.
    """
    os.makedirs(FRAMES_DIR, exist_ok=True)
    existing = sorted(f for f in os.listdir(FRAMES_DIR) if f.endswith(".jpg"))
    if len(existing) >= n_frames:
        return [os.path.join(FRAMES_DIR, f) for f in existing[:n_frames]]

    import cv2

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    paths, i = [], 0
    while len(paths) < n_frames:
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop short clips
            ok, frame = cap.read()
            if not ok:
                break
        p = os.path.join(FRAMES_DIR, f"f{i:05d}.jpg")
        cv2.imwrite(p, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        paths.append(p)
        i += 1
    cap.release()
    print(f"corpus: {len(paths)} frames in {FRAMES_DIR}")
    return paths


# ---------------------------------------------------------------- GPU probe


class GpuSampler:
    """Per-PID VRAM, filtered to the worker process tree.

    torch.cuda.memory_allocated() would NOT do: it excludes the ~300-600 MiB
    per-process CUDA context, which is the entire point of comparing threads
    against prefork.
    """

    def __init__(self, root_pid, interval=0.5):
        self.root_pid, self.interval = root_pid, interval
        self.per_pid, self.max_total, self.utils = {}, 0, []
        self._stop = False
        self._t = None

    def _tree(self):
        try:
            proc = psutil.Process(self.root_pid)
            return {self.root_pid} | {c.pid for c in proc.children(recursive=True)}
        except psutil.Error:
            return {self.root_pid}

    def _loop(self, handle):
        while not self._stop:
            tree = self._tree()
            total = 0
            for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
                if p.pid in tree and p.usedGpuMemory:
                    mib = p.usedGpuMemory / 2 ** 20
                    self.per_pid[p.pid] = max(self.per_pid.get(p.pid, 0), mib)
                    total += mib
            self.max_total = max(self.max_total, total)
            try:
                self.utils.append(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
            except pynvml.NVMLError:
                pass
            time.sleep(self.interval)

    def start(self):
        import threading
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        self._t = threading.Thread(target=self._loop, args=(h,), daemon=True)
        self._t.start()

    def stop(self):
        self._stop = True
        if self._t:
            self._t.join(timeout=3)


def _lum_vision_rev() -> str:
    """Git rev of the installed lum_vision, wherever it lives on this machine.

    Returns "" if it is not a git worktree (e.g. a wheel install) — the field is
    provenance, so an empty value is honest rather than fatal.
    """
    try:
        import lum_vision
        pkg_file = lum_vision.__file__
        if not pkg_file:
            return ""
        pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(pkg_file)))
        out = subprocess.run(["git", "-C", pkg_dir, "rev-parse", "HEAD"],
                             capture_output=True, text=True)
        return out.stdout.strip()
    except Exception:
        return ""


def gpu_used_mib():
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    return pynvml.nvmlDeviceGetMemoryInfo(h).used / 2 ** 20


# ---------------------------------------------------------------- worker


def spawn(pool, conc, share, cell, prefetch):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("SO_CELERY", "SO_REDIS"))}  # never inherit prod pointers
    env.update(
        PYTHONPATH=os.path.join(REPO, "tests"),
        CUDA_VISIBLE_DEVICES="0",
        YOLOBENCH_WEIGHTS=WEIGHTS,
        YOLOBENCH_SHARE=share,
        YOLOBENCH_PREFETCH=str(prefetch),
        YOLOBENCH_ACKS_LATE="1",
        OMP_NUM_THREADS="2",
        OPENCV_NUM_THREADS="1",
        YOLO_VERBOSE="false",
    )
    argv = [VENV_CELERY, "-A", "workers.bench_app", "worker",
            "--loglevel=WARNING", f"--pool={pool}", "--queues=yolobench",
            # %d not %h: a reused node name lets a slow-dying worker from the
            # previous cell consume this cell's tasks.
            f"--hostname=yb%d@{cell}",
            "--without-gossip", "--without-mingle", "--without-heartbeat"]
    if pool != "solo":
        argv.append(f"--concurrency={conc}")
    if pool == "prefork":
        argv.append("-Ofair")

    log = open(os.path.join(OUT, f"{cell}.worker.log"), "w")
    return subprocess.Popen(argv, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True), log


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
    """Kill leftover bench workers from an interrupted run.

    An orphan still holds ~1 GiB of VRAM and would trip the next cell's
    baseline check (or, worse, silently consume its tasks).
    """
    killed = []
    for proc in psutil.process_iter(["pid", "cmdline"]):
        cmd = " ".join(proc.info.get("cmdline") or [])
        if "workers.bench_app" in cmd and proc.info["pid"] != os.getpid():
            try:
                proc.kill()
                killed.append(proc.info["pid"])
            except psutil.Error:
                pass
    if killed:
        print(f"reaped stale bench workers: {killed}")
        time.sleep(5)


def wait_ready(timeout=120):
    end = time.time() + timeout
    while time.time() < end:
        if app.control.inspect(timeout=1).ping():
            return True
        time.sleep(0.5)
    return False


# ---------------------------------------------------------------- one cell


def run_cell(cell, pool, conc, share, batch, frames, total_frames, prefetch,
             mode="path", shm_shape=None):
    print(f"\n### {cell}: pool={pool} -c{conc} share={share} B={batch} "
          f"prefetch={prefetch} mode={mode}")
    baseline = gpu_used_mib()
    # Here we spawn the worker before the warmup, so that the warmup tasks are included in
    proc, log = spawn(pool, conc, share, cell, prefetch)
    sampler = GpuSampler(proc.pid)
    try:
        if not wait_ready():
            raise SystemExit(f"{cell}: worker never became ready — see {cell}.worker.log")

        # Warmup barrier: one task per slot, each holding its slot until a shared
        # deadline. With prefetch_multiplier=1 this forces every thread/child to
        # build its own predictor; without the hold one fast slot drains the group.
        slots = conc if pool != "solo" else 1
        deadline = time.time() + 15
        ids = group(app.signature("yolobench.warmup", args=(frames[0], deadline),
                                  queue="yolobench") for _ in range(slots)
                    ).apply_async().get(timeout=300)
        print(f"    warmed {len(ids)} slots | pids={len({i['pid'] for i in ids})} "
              f"modules={len({i['module_id'] for i in ids})} "
              f"predictors={len({i['predictor_id'] for i in ids})}")

        sampler.start()
        n_tasks = max(1, total_frames // batch)
        if mode == "shm":
            # Payload is a name plus a few ints — the broker moves a handle.
            payloads = [[(t * batch + k) % len(frames) for k in range(batch)]
                        for t in range(n_tasks)]
            task_name = "yolobench.infer_shm"
            mk_args = lambda p: (SHM_NAME, p, list(shm_shape))
        else:
            payloads = [[frames[(t * batch + k) % len(frames)] for k in range(batch)]
                        for t in range(n_tasks)]
            task_name = "yolobench.infer"
            mk_args = lambda p: (p,)

        lat = []

        def one(p):
            t0 = time.perf_counter()
            out = app.send_task(task_name, args=mk_args(p), queue="yolobench").get(timeout=300)
            lat.append((time.perf_counter() - t0) * 1e3)
            return out

        # Closed loop: one driver thread per outstanding request.
        inflight = 4 * (conc if pool != "solo" else 1)
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=inflight) as ex:
            outs = list(ex.map(one, payloads))
        wall = time.perf_counter() - t0
        sampler.stop()

        lat.sort()
        pick = lambda q: lat[min(len(lat) - 1, int(len(lat) * q))]
        return {
            "cell": cell, "pool": pool, "concurrency": conc, "share": share,
            "batch": batch, "prefetch": prefetch, "mode": mode,
            "tasks": n_tasks, "frames": n_tasks * batch, "wall_s": round(wall, 2),
            "tasks_per_s": round(n_tasks / wall, 1),
            "frames_per_s": round(n_tasks * batch / wall, 1),
            "lat_ms_p50": round(median(lat), 1), "lat_ms_p95": round(pick(0.95), 1),
            "read_ms_mean": round(sum(o["read_ms"] for o in outs) / len(outs), 2),
            "inf_ms_mean": round(sum(o["inf_ms"] for o in outs) / len(outs), 2),
            "pre_ms_mean": round(sum(o["pre_ms"] for o in outs) / len(outs), 2),
            "post_ms_mean": round(sum(o["post_ms"] for o in outs) / len(outs), 2),
            "ndet_total": sum(o["ndet"] for o in outs),
            "worker_pids": len({o["pid"] for o in outs}),
            "worker_threads": len({(o["pid"], o["tid"]) for o in outs}),
            "vram_mib_total": round(sampler.max_total, 1),
            "vram_mib_per_pid": {str(k): round(v, 1) for k, v in sampler.per_pid.items()},
            "n_gpu_pids": len(sampler.per_pid),
            "gpu_util_mean": round(sum(sampler.utils) / len(sampler.utils), 1) if sampler.utils else None,
        }
    finally:
        sampler.stop()
        terminate(proc)
        log.close()
        # A worker that ignored SIGTERM leaks VRAM into the next cell and would
        # silently corrupt its numbers.
        for _ in range(30):
            if gpu_used_mib() <= baseline + 50:
                break
            time.sleep(1)
        else:
            raise SystemExit(f"{cell}: VRAM did not return to baseline "
                             f"({gpu_used_mib():.0f} vs {baseline:.0f} MiB) — aborting")


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="", help="comma-separated cell name prefixes")
    ap.add_argument("--frames", type=int, default=1200, help="frames processed per cell")
    ap.add_argument("--corpus", type=int, default=200)
    ap.add_argument("--video", default=os.path.join(REPO, "cam1.mp4"))
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    pynvml.nvmlInit()
    reap_stale_workers()
    frames = build_corpus(args.corpus, args.video)

    wanted = [c.strip() for c in args.cells.split(",") if c.strip()]
    cells = [c for c in CELLS if not wanted or any(c["cell"].startswith(w) for w in wanted)]

    meta = {
        "repo_rev": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                   capture_output=True, text=True).stdout.strip(),
        # Derived from where the package is actually installed, not hardcoded:
        # lum-model-vision is an editable install from a working tree, so its
        # version number does not identify the code that ran.
        "lum_vision_rev": _lum_vision_rev(),
        "gpu": pynvml.nvmlDeviceGetName(pynvml.nvmlDeviceGetHandleByIndex(0)),
        "frames_per_cell": args.frames,
    }
    path = os.path.join(OUT, "results.json")

    # Merge into whatever is already on disk, and write after EVERY cell: a
    # crash in a late cell must not discard the ones that already succeeded.
    by_cell = {}
    if os.path.exists(path):
        try:
            by_cell = {r["cell"]: r for r in json.load(open(path))["results"]}
        except (ValueError, KeyError):
            pass

    shm = shm_shape = None # shared memory handle and shape, if any cell uses shm mode
    if any(c.get("mode") == "shm" for c in cells):
        from workers import frame_store
        shm, shm_shape = frame_store.create(frames, SHM_NAME)
        print(f"shared block {SHM_NAME}: {shm_shape}, "
              f"{shm.size / 2 ** 20:.0f} MiB (decoded once, reused by every task)")

    for c in cells:
        r = run_cell(c["cell"], c["pool"], c["conc"], c["share"], c["batch"], frames,
                     c.get("frames", args.frames), c.get("prefetch", DEFAULT_PREFETCH),
                     mode=c.get("mode", "path"), shm_shape=shm_shape)
        by_cell[r["cell"]] = r
        with open(path, "w") as f:
            json.dump({"meta": meta, "results": [by_cell[k] for k in sorted(by_cell)]},
                      f, indent=2)
        print("   ", json.dumps({k: r[k] for k in
                                 ("tasks_per_s", "frames_per_s", "lat_ms_p50",
                                  "vram_mib_total", "n_gpu_pids")}))

    results = [by_cell[k] for k in sorted(by_cell)]

    hdr = f"\n{'cell':<24} {'pool':<8} {'-c':>3} {'B':>2} {'mode':<5} {'tasks/s':>8} {'frames/s':>9} " \
          f"{'p50ms':>7} {'read':>6} {'VRAM MiB':>9} {'pids':>5} {'util%':>6}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(f"{r['cell']:<24} {r['pool']:<8} {r['concurrency']:>3} {r['batch']:>2} "
              f"{r.get('mode','path'):<5} {r['tasks_per_s']:>8} {r['frames_per_s']:>9} "
              f"{r['lat_ms_p50']:>7} {r['read_ms_mean']:>6} {r['vram_mib_total']:>9} "
              f"{r['n_gpu_pids']:>5} {str(r['gpu_util_mean']):>6}")
    if shm is not None:
        shm.close()
        shm.unlink()

    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
