#!/usr/bin/env python
"""Celery face-detection benchmark — driver.

Answers two questions, mirroring benchmarks/celery_worker/run_bench.py's
method so results are comparable:

  1. detect_and_align throughput under real Celery/thread contention — is
     detection still the dominant cost the offline notebook found (93% of
     ArcFace-stage time), or does that change under concurrent load? Always
     batch=1: SCRFD has no batch path (see model_holder.py's docstring).
  2. embed_batch throughput at different batch sizes under contention — this
     is where LSO-117's batching payoff (2.5x offline) should show up, or not,
     once real concurrency is involved.

Not a unit test — needs a GPU, a live Redis and minutes of wall time. The name
has no ``test_`` prefix so ``pytest tests/`` will not collect it.

    docker run --rm -d --name facebench-redis -p 6402:6379 redis:7-alpine
    PYTHONPATH=benchmarks .venv/bin/python benchmarks/celery_face/run_bench.py
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
sys.path.insert(0, os.path.join(REPO, "benchmarks"))
LUM_VISION_SRC = os.environ.get(
    "FACEBENCH_LUM_VISION_SRC", "/home/oybek/workspace/lum-model-vision/src"
)
if os.path.isdir(LUM_VISION_SRC):
    sys.path.insert(0, LUM_VISION_SRC)

import psutil  # noqa: E402
import pynvml  # noqa: E402
from celery import group  # noqa: E402

from celery_face.bench_app import app  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "output")
FRAMES_DIR = os.path.join(OUT, "frames")
CROPS_DIR = os.path.join(OUT, "crops")
VENV_CELERY = os.path.join(REPO, ".venv/bin/celery")

# detect cells: batch is always 1 (SCRFD has no batch path). embed cells sweep
# batch size, since embed_batch's recognition model does have a dynamic batch
# axis and LSO-117 already batches it.
CELLS = [
    dict(cell="1-solo-detect",          task="detect", pool="solo",    conc=1, batch=1),
    dict(cell="2-threads4-detect",      task="detect", pool="threads", conc=4, batch=1),
    dict(cell="3-prefork4-detect",      task="detect", pool="prefork", conc=4, batch=1),
    dict(cell="4-solo-embed-b1",        task="embed",  pool="solo",    conc=1, batch=1),
    dict(cell="5-threads4-embed-b1",    task="embed",  pool="threads", conc=4, batch=1),
    dict(cell="6-threads4-embed-b8",    task="embed",  pool="threads", conc=4, batch=8),
    dict(cell="7-threads4-embed-b32",   task="embed",  pool="threads", conc=4, batch=32),
    dict(cell="8-prefork4-embed-b8",    task="embed",  pool="prefork", conc=4, batch=8),
    # solo + batch=8/32: isolates what batching alone is worth with zero
    # concurrency, so the b1->b8/b32 gains in cells 6/7 can be separated from
    # concurrency gains — same reasoning as the YOLO benchmark's cell 11.
    dict(cell="9-solo-embed-b8",        task="embed",  pool="solo",    conc=1, batch=8),
    dict(cell="10-solo-embed-b32",      task="embed",  pool="solo",    conc=1, batch=32),
]
SHM_FRAMES_NAME = "facebench_frames"
SHM_CROPS_NAME = "facebench_crops"
DEFAULT_PREFETCH = 16


# ---------------------------------------------------------------- corpus


def build_corpus(n_frames, video):
    """Extract frames from a real video, cached across runs. Same source
    (cam1.mp4) as the YOLO benchmark's corpus, for comparable conditions."""
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
    print(f"frame corpus: {len(paths)} frames in {FRAMES_DIR}")
    return paths


def build_crop_corpus(frame_paths, min_crops):
    """Run detect_and_align on the frame corpus once, offline, and cache the
    usable aligned crops (kps present) to disk as .npy — the corpus embed
    cells actually measure.

    Mirrors the offline notebook's "500 ROIs -> 362 usable crops" step. This
    is NOT part of the timed benchmark: it runs in the driver process, once,
    before any cell starts.
    """
    os.makedirs(CROPS_DIR, exist_ok=True)
    existing = sorted(f for f in os.listdir(CROPS_DIR) if f.endswith(".npy"))
    if len(existing) >= min_crops:
        import numpy as np
        return [np.load(os.path.join(CROPS_DIR, f)) for f in existing[:min_crops]]

    import cv2
    from celery_face import model_holder

    detector = model_holder.ensure_loaded()
    crops = []
    for p in frame_paths:
        frame = cv2.imread(p)
        for face in detector.detect_and_align(frame):
            if face.aligned_crop is not None:
                crops.append(face.aligned_crop)
        if len(crops) >= min_crops:
            break

    if len(crops) < min_crops:
        print(f"warning: only found {len(crops)} usable crops "
              f"(wanted {min_crops}) from {len(frame_paths)} frames; "
              "extend the frame corpus with --corpus if this matters")

    import numpy as np
    for i, c in enumerate(crops):
        np.save(os.path.join(CROPS_DIR, f"c{i:05d}.npy"), c)
    print(f"crop corpus: {len(crops)} aligned 112x112 crops in {CROPS_DIR}")
    return crops


# ---------------------------------------------------------------- GPU probe


class GpuSampler:
    """Per-PID VRAM, filtered to the worker process tree.

    torch.cuda.memory_allocated() would NOT do here either: it excludes the
    per-process CUDA/ORT context, which is the entire point of comparing
    threads against prefork.
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
    """Git rev of the lum_vision checkout this run used, wherever it lives.

    Returns "" if it is not a distinct git worktree — the field is
    provenance, so an empty value is honest rather than fatal.
    """
    try:
        import lum_vision
        pkg_file = lum_vision.__file__
        if not pkg_file:
            return ""
        pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(pkg_file)))
        root = subprocess.run(["git", "-C", pkg_dir, "rev-parse", "--show-toplevel"],
                              capture_output=True, text=True)
        if root.returncode != 0:
            return ""
        if os.path.realpath(root.stdout.strip()) == os.path.realpath(REPO):
            return ""
        out = subprocess.run(["git", "-C", pkg_dir, "rev-parse", "HEAD"],
                             capture_output=True, text=True)
        return out.stdout.strip()
    except Exception:
        return ""


def gpu_used_mib():
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    return pynvml.nvmlDeviceGetMemoryInfo(h).used / 2 ** 20


# ---------------------------------------------------------------- worker


def spawn(pool, conc, cell, prefetch):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("SO_CELERY", "SO_REDIS"))}  # never inherit prod pointers
    env.update(
        PYTHONPATH=os.path.join(REPO, "benchmarks") + os.pathsep + LUM_VISION_SRC,
        FACEBENCH_GPU_ID="0",
        FACEBENCH_PREFETCH=str(prefetch),
        FACEBENCH_ACKS_LATE="1",
        OMP_NUM_THREADS="2",
        OPENCV_NUM_THREADS="1",
    )
    argv = [VENV_CELERY, "-A", "celery_face.bench_app", "worker",
            "--loglevel=WARNING", f"--pool={pool}", "--queues=facebench",
            # %d not %h: a reused node name lets a slow-dying worker from the
            # previous cell consume this cell's tasks.
            f"--hostname=fb%d@{cell}",
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
    """Kill leftover bench workers from an interrupted run."""
    killed = []
    for proc in psutil.process_iter(["pid", "cmdline"]):
        cmd = " ".join(proc.info.get("cmdline") or [])
        if "celery_face.bench_app" in cmd and proc.info["pid"] != os.getpid():
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


def run_cell(cell, task, pool, conc, batch, warmup_path, total_units, prefetch,
             frame_shape, crop_shape):
    print(f"\n### {cell}: task={task} pool={pool} -c{conc} B={batch} prefetch={prefetch}")
    baseline = gpu_used_mib()
    proc, log = spawn(pool, conc, cell, prefetch)
    sampler = GpuSampler(proc.pid)
    try:
        if not wait_ready():
            raise SystemExit(f"{cell}: worker never became ready — see {cell}.worker.log")

        slots = conc if pool != "solo" else 1
        deadline = time.time() + 15
        ids = group(app.signature("facebench.warmup", args=(warmup_path, deadline),
                                  queue="facebench") for _ in range(slots)
                    ).apply_async().get(timeout=300)
        print(f"    warmed {len(ids)} slots | pids={len({i['pid'] for i in ids})} "
              f"detectors={len({i['detector_id'] for i in ids})}")

        sampler.start()

        if task == "detect":
            corpus_len = frame_shape[0]
            n_tasks = total_units  # one frame per task, batch is always 1
            payloads = [[t % corpus_len] for t in range(n_tasks)]
            task_name = "facebench.detect_shm"
            shape = frame_shape
        else:
            corpus_len = crop_shape[0]
            n_tasks = max(1, total_units // batch)
            payloads = [[(t * batch + k) % corpus_len for k in range(batch)]
                        for t in range(n_tasks)]
            task_name = "facebench.embed_shm"
            shape = crop_shape

        shm_name = SHM_FRAMES_NAME if task == "detect" else SHM_CROPS_NAME
        lat = []

        def one(p):
            t0 = time.perf_counter()
            out = app.send_task(task_name, args=(shm_name, p, list(shape)),
                                queue="facebench").get(timeout=300)
            lat.append((time.perf_counter() - t0) * 1e3)
            return out

        inflight = 4 * (conc if pool != "solo" else 1)
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=inflight) as ex:
            outs = list(ex.map(one, payloads))
        wall = time.perf_counter() - t0
        sampler.stop()

        lat.sort()
        pick = lambda q: lat[min(len(lat) - 1, int(len(lat) * q))]
        n_items = sum(o["n"] for o in outs)
        return {
            "cell": cell, "task": task, "pool": pool, "concurrency": conc,
            "batch": batch, "prefetch": prefetch,
            "tasks": n_tasks, "items": n_items, "wall_s": round(wall, 2),
            "tasks_per_s": round(n_tasks / wall, 1),
            "items_per_s": round(n_items / wall, 1),
            "lat_ms_p50": round(median(lat), 1), "lat_ms_p95": round(pick(0.95), 1),
            "read_ms_mean": round(sum(o["read_ms"] for o in outs) / len(outs), 2),
            "infer_ms_mean": round(sum(o["infer_ms"] for o in outs) / len(outs), 2),
            "nfaces_total": sum(o["nfaces"] for o in outs),
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
    ap.add_argument("--cells", default="",
                    help="comma-separated selectors: an exact cell number (1, 8) "
                         "or a substring of the name (solo, embed)")
    ap.add_argument("--units", type=int, default=600,
                    help="frames processed per detect cell, or crops per embed cell")
    ap.add_argument("--corpus", type=int, default=200, help="frame corpus size")
    ap.add_argument("--crops", type=int, default=300, help="crop corpus size")
    ap.add_argument("--video", default=os.path.join(REPO, "cam1.mp4"))
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    pynvml.nvmlInit()
    reap_stale_workers()
    frame_paths = build_corpus(args.corpus, args.video)

    wanted = [c.strip() for c in args.cells.split(",") if c.strip()]

    def selected(name: str) -> bool:
        head = name.split("-", 1)[0]
        return any(w == head if w.isdigit() else w in name for w in wanted)

    cells = [c for c in CELLS if not wanted or selected(c["cell"])]
    need_crops = any(c["task"] == "embed" for c in cells)

    crops = build_crop_corpus(frame_paths, args.crops) if need_crops else []

    meta = {
        "repo_rev": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                   capture_output=True, text=True).stdout.strip(),
        "lum_vision_rev": _lum_vision_rev(),
        "gpu": pynvml.nvmlDeviceGetName(pynvml.nvmlDeviceGetHandleByIndex(0)),
        "units_per_cell": args.units,
    }
    path = os.path.join(OUT, "results.json")

    by_cell = {}
    if os.path.exists(path):
        try:
            by_cell = {r["cell"]: r for r in json.load(open(path))["results"]}
        except (ValueError, KeyError):
            pass

    from celery_face import frame_store

    frame_shm = frame_shape = None
    crop_shm = crop_shape = None
    if any(c["task"] == "detect" for c in cells):
        frame_shm, frame_shape = frame_store.create(frame_paths, SHM_FRAMES_NAME)
        print(f"shared block {SHM_FRAMES_NAME}: {frame_shape}, "
              f"{frame_shm.size / 2 ** 20:.0f} MiB")
    if need_crops:
        crop_shm, crop_shape = frame_store.create_from_arrays(crops, SHM_CROPS_NAME)
        print(f"shared block {SHM_CROPS_NAME}: {crop_shape}, "
              f"{crop_shm.size / 2 ** 20:.0f} MiB")

    warmup_path = frame_paths[0]

    try:
        for c in cells:
            r = run_cell(c["cell"], c["task"], c["pool"], c["conc"], c["batch"],
                         warmup_path, args.units, c.get("prefetch", DEFAULT_PREFETCH),
                         frame_shape, crop_shape)
            by_cell[r["cell"]] = r
            with open(path, "w") as f:
                json.dump({"meta": meta, "results": [by_cell[k] for k in sorted(by_cell)]},
                          f, indent=2)
            print("   ", json.dumps({k: r[k] for k in
                                     ("tasks_per_s", "items_per_s", "lat_ms_p50",
                                      "vram_mib_total", "n_gpu_pids")}))
    finally:
        if frame_shm is not None:
            frame_shm.close()
            frame_shm.unlink()
        if crop_shm is not None:
            crop_shm.close()
            crop_shm.unlink()

    results = [by_cell[k] for k in sorted(by_cell)]

    hdr = f"\n{'cell':<22} {'task':<7} {'pool':<8} {'-c':>3} {'B':>3} {'tasks/s':>8} " \
          f"{'items/s':>8} {'p50ms':>7} {'VRAM MiB':>9} {'pids':>5} {'util%':>6}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(f"{r['cell']:<22} {r['task']:<7} {r['pool']:<8} {r['concurrency']:>3} "
              f"{r['batch']:>3} {r['tasks_per_s']:>8} {r['items_per_s']:>8} "
              f"{r['lat_ms_p50']:>7} {r['vram_mib_total']:>9} "
              f"{r['n_gpu_pids']:>5} {str(r['gpu_util_mean']):>6}")

    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
